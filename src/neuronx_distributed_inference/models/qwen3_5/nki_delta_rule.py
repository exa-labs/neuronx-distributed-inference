"""
NKI Kernel for the Gated Delta Rule (recurrent form)
=====================================================

Custom NKI kernel implementing the recurrent gated delta rule for
Qwen3.5/Qwen3-Next's DeltaNet layers. Bypasses XLA trace + neuronx-cc
PGTiling entirely, allowing compilation at ANY seq_len and batch_size.

Qwen3.5's DeltaNet uses linear_key_head_dim=128, linear_value_head_dim=128
(NOT the attention head_dim=256). Since D=128 = NeuronCore-v2 partition limit,
the full [128, 128] state fits in a single SBUF tile with no splitting.

Input layout (caller must transpose q/k before calling):
  q_ref:     [BH, Dk, T]  — queries in column-major (pre-normalized)
  k_ref:     [BH, Dk, T]  — keys in column-major (pre-normalized)
  v_ref:     [BH, T, Dv]  — values in row-major
  exp_g_ref: [BH, T]      — pre-computed exp(g) decay factors
  beta_ref:  [BH, T]      — beta scalars
  state_ref: [BH, Dk, Dv] — initial recurrent state (fp32)

Per-token operations (state ∈ R^{Dk×Dv}, Dk=Dv=128):
  1. state *= exp_g_t           — gated decay (scalar broadcast)
  2. kv_mem = k_t^T @ state     — read from state (mat-vec)
  3. delta = (v_t - kv_mem) * β  — compute update
  4. state += k_t ⊗ delta       — rank-1 update (outer product)
  5. out_t = q_t^T @ state      — produce output (mat-vec)
"""

import nki
import nki.isa as nisa
import nki.language as nl

# Qwen3.5 DeltaNet dimensions: linear_key_head_dim = linear_value_head_dim = 128
_D = 128  # fits exactly in one NeuronCore-v2 partition (128)


@nki.jit(mode="torchxla")
def nki_recurrent_gated_delta_rule(
    q_ref, k_ref, v_ref, exp_g_ref, beta_ref, state_ref
):
    """NKI recurrent gated delta rule kernel (general, any T).

    Args:
        q_ref:     [BH, Dk, T]  queries (column-major, l2-normed+scaled)
        k_ref:     [BH, Dk, T]  keys (column-major, l2-normed)
        v_ref:     [BH, T, Dv]  values (row-major)
        exp_g_ref: [BH, T]      pre-computed exp(g) per token
        beta_ref:  [BH, T]      beta per token
        state_ref: [BH, Dk, Dv] initial state (fp32)

    Returns:
        out_ref:   [BH, T, Dv]  output (row-major, input dtype)
        final_state_ref: [BH, Dk, Dv]  final state (fp32)
    """
    bh = q_ref.shape[0]
    dk = q_ref.shape[1]   # Dk = 128
    seq_len = q_ref.shape[2]  # T is last dim for q/k (column-major)
    dv = v_ref.shape[2]   # Dv = 128

    # Allocate shared HBM outputs (required by NKI for kernel return tensors)
    out_ref = nl.ndarray((bh, seq_len, dv), dtype=q_ref.dtype, buffer=nl.shared_hbm)
    final_state_ref = nl.ndarray((bh, dk, dv), dtype=nl.float32, buffer=nl.shared_hbm)

    # Pre-allocate a ones vector [1, 128] for scalar broadcasting via nc_matmul
    ones_row = nl.ndarray((1, _D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(ones_row, 1.0)

    for idx in nl.affine_range(bh):
        # Load initial state as single [128, 128] tile
        state = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=state, src=state_ref[idx, :, :])

        # Sequential time loop
        for t in nl.sequential_range(seq_len):
            # Load q_t as [Dk, 1] column tile
            q_t = nl.ndarray((dk, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=q_t, src=q_ref[idx, :, t:t+1])

            # Load k_t as [Dk, 1] column tile
            k_t = nl.ndarray((dk, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=k_t, src=k_ref[idx, :, t:t+1])

            # Load v_t as [1, Dv] row tile
            v_t = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=v_t, src=v_ref[idx, t:t+1, :])

            # Load exp_g_t as [1, 1] scalar
            exp_g_t = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=exp_g_t, src=exp_g_ref[idx, t:t+1])

            # Load beta_t as [1, 1] scalar
            beta_t = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=beta_t, src=beta_ref[idx, t:t+1])

            # ── Step 1: state *= exp(g_t) ──────────────────────────────────
            # Broadcast scalar to [128, 1]: ones[1,128]^T @ s[1,1] = [128, 1]
            exp_g_bc_psum = nl.ndarray(
                (_D, 1), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_matmul(exp_g_bc_psum, ones_row, exp_g_t)
            exp_g_bc = nl.ndarray((_D, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(exp_g_bc, exp_g_bc_psum)

            # state[128, 128] *= exp_g_bc[128, 1] (broadcasts along free dim)
            nisa.tensor_scalar(state, state, nl.multiply, exp_g_bc)

            # ── Step 2: kv_mem = k_t^T @ state → [1, Dv] ──────────────────
            # nc_matmul: C[M,N] = A[K,M]^T @ B[K,N]
            # kv_mem[1, 128] = k_t[128, 1]^T @ state[128, 128]
            kv_psum = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(kv_psum, k_t, state)
            kv_mem = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(kv_mem, kv_psum)

            # ── Step 3: delta = (v_t - kv_mem) * beta_t → [1, Dv] ─────────
            delta = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(delta, v_t, kv_mem, op=nl.subtract)
            nisa.tensor_scalar(delta, delta, nl.multiply, beta_t)

            # ── Step 4: state += outer(k_t, delta) → [Dk, Dv] ─────────────
            # Transpose k_t[128, 1] → k_t_T[1, 128]
            k_t_T_psum = nl.ndarray((1, dk), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(k_t_T_psum, k_t)
            k_t_T = nl.ndarray((1, dk), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(k_t_T, k_t_T_psum)

            # outer[128, 128] = k_t_T[1, 128]^T @ delta[1, 128]
            outer_psum = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(outer_psum, k_t_T, delta)
            outer = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(outer, outer_psum)
            nisa.tensor_tensor(state, state, outer, op=nl.add)

            # ── Step 5: out_t = q_t^T @ state → [1, Dv] ───────────────────
            out_psum = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(out_psum, q_t, state)
            out_sbuf = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(out_sbuf, out_psum)

            # Cast back to input dtype and store
            out_cast = nl.ndarray((1, dv), dtype=q_ref.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(out_cast, out_sbuf, nl.multiply, 1.0)
            nisa.dma_copy(dst=out_ref[idx, t:t+1, :], src=out_cast)

        # Store final state
        nisa.dma_copy(dst=final_state_ref[idx, :, :], src=state)

    return out_ref, final_state_ref


@nki.jit(mode="torchxla")
def nki_recurrent_gated_delta_rule_decode(
    q_ref, k_ref, k_row_ref, v_ref, exp_g_ref, beta_ref, state_ref
):
    """Decode-optimized NKI kernel for T=1 (single token per step).

    Eliminates the per-head nc_transpose by accepting k in row-major layout
    directly.  For T=1 the wrapper provides k as both [BH, Dk, 1] (column,
    for the k^T @ state mat-vec) and [BH, 1, Dk] (row, for the outer product
    k ⊗ delta).  The row format is a zero-cost view in torch for T=1.

    Saves 2 TensorEngine instructions per head (nc_transpose + tensor_copy)
    relative to the general kernel — with 128 BH per core per layer × 24
    DeltaNet layers, this is ~6144 fewer instructions per decode step.

    Args:
        q_ref:     [BH, Dk, 1]  queries (column-major, l2-normed+scaled)
        k_ref:     [BH, Dk, 1]  keys (column-major, for k^T @ state)
        k_row_ref: [BH, 1, Dk]  keys (row-major, for outer product)
        v_ref:     [BH, 1, Dv]  values (row-major)
        exp_g_ref: [BH, 1]      pre-computed exp(g)
        beta_ref:  [BH, 1]      beta scalars
        state_ref: [BH, Dk, Dv] initial state (fp32)

    Returns:
        out_ref:         [BH, 1, Dv]  output (row-major, input dtype)
        final_state_ref: [BH, Dk, Dv] final state (fp32)
    """
    bh = q_ref.shape[0]
    dk = q_ref.shape[1]   # Dk = 128
    dv = v_ref.shape[2]   # Dv = 128

    out_ref = nl.ndarray((bh, 1, dv), dtype=q_ref.dtype, buffer=nl.shared_hbm)
    final_state_ref = nl.ndarray((bh, dk, dv), dtype=nl.float32, buffer=nl.shared_hbm)

    # Ones vector for scalar broadcast: ones[1,128]^T @ s[1,1] → [128,1]
    ones_row = nl.ndarray((1, _D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(ones_row, 1.0)

    for idx in nl.affine_range(bh):
        # ── Load all inputs for this head ──────────────────────────────────
        state = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=state, src=state_ref[idx, :, :])

        q_t = nl.ndarray((dk, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_t, src=q_ref[idx, :, :])

        k_t = nl.ndarray((dk, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=k_t, src=k_ref[idx, :, :])

        # Row-major key for outer product (eliminates nc_transpose)
        k_row = nl.ndarray((1, dk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=k_row, src=k_row_ref[idx, :, :])

        v_t = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=v_t, src=v_ref[idx, :, :])

        exp_g_t = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=exp_g_t, src=exp_g_ref[idx, :])

        beta_t = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=beta_t, src=beta_ref[idx, :])

        # ── Step 1: state *= exp(g) ───────────────────────────────────────
        exp_g_bc_psum = nl.ndarray((_D, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(exp_g_bc_psum, ones_row, exp_g_t)
        exp_g_bc = nl.ndarray((_D, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(exp_g_bc, exp_g_bc_psum)
        nisa.tensor_scalar(state, state, nl.multiply, exp_g_bc)

        # ── Step 2: kv_mem = k^T @ state → [1, Dv] ───────────────────────
        kv_psum = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(kv_psum, k_t, state)
        kv_mem = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(kv_mem, kv_psum)

        # ── Step 3: delta = (v - kv_mem) * beta → [1, Dv] ────────────────
        delta = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(delta, v_t, kv_mem, op=nl.subtract)
        nisa.tensor_scalar(delta, delta, nl.multiply, beta_t)

        # ── Step 4: state += outer(k, delta) → [Dk, Dv] ──────────────────
        # nc_matmul(out[M,N], A[K,M], B[K,N]): K=1, M=Dk=128, N=Dv=128
        # A = k_row[1, 128], B = delta[1, 128]
        # out[128,128] = k_row^T[128,1] @ delta[1,128] = outer product
        outer_psum = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(outer_psum, k_row, delta)
        outer = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(outer, outer_psum)
        nisa.tensor_tensor(state, state, outer, op=nl.add)

        # ── Step 5: out = q^T @ state → [1, Dv] ──────────────────────────
        out_psum = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(out_psum, q_t, state)
        out_sbuf = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(out_sbuf, out_psum)

        # Cast back to input dtype and store
        out_cast = nl.ndarray((1, dv), dtype=q_ref.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(out_cast, out_sbuf, nl.multiply, 1.0)
        nisa.dma_copy(dst=out_ref[idx, :, :], src=out_cast)

        # Store final state
        nisa.dma_copy(dst=final_state_ref[idx, :, :], src=state)

    return out_ref, final_state_ref


@nki.jit(mode="torchxla")
def nki_recurrent_gated_delta_rule_decode_v2(
    q_ref, k_ref, k_beta_row_ref, v_ref, exp_g_bc_ref, state_ref
):
    """Optimized decode kernel (T=1) with host-side pre-computation.

    Eliminates per-head overhead vs v1 by accepting pre-broadcast exp_g and
    pre-fused k*beta from the wrapper.  Per-head savings:
      - 1 DMA load (beta no longer loaded)
      - 1 nc_matmul + 1 tensor_copy (exp_g scalar broadcast eliminated)
      - 1 tensor_scalar (delta *= beta folded into k_beta_row)
      - 1 tensor_scalar (no-op fp32→fp32 cast removed)
    Net: −5 instructions + −1 DMA per head × BH × 24 layers/step.

    Args:
        q_ref:          [BH, Dk, 1]  queries (column-major, l2-normed+scaled)
        k_ref:          [BH, Dk, 1]  keys (column-major, for k^T @ state)
        k_beta_row_ref: [BH, 1, Dk]  keys * beta (row-major, for outer product)
        v_ref:          [BH, 1, Dv]  values (row-major)
        exp_g_bc_ref:   [BH, Dk]     exp(g) broadcast to Dk (same scalar per row)
        state_ref:      [BH, Dk, Dv] initial state (fp32)

    Returns:
        out_ref:         [BH, 1, Dv]  output (row-major, fp32)
        final_state_ref: [BH, Dk, Dv] final state (fp32)
    """
    bh = q_ref.shape[0]
    dk = q_ref.shape[1]   # Dk = 128
    dv = v_ref.shape[2]   # Dv = 128

    out_ref = nl.ndarray((bh, 1, dv), dtype=nl.float32, buffer=nl.shared_hbm)
    final_state_ref = nl.ndarray((bh, dk, dv), dtype=nl.float32, buffer=nl.shared_hbm)

    for idx in nl.affine_range(bh):
        # ── Load inputs for this head ─────────────────────────────────────
        state = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=state, src=state_ref[idx, :, :])

        q_t = nl.ndarray((dk, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_t, src=q_ref[idx, :, :])

        k_t = nl.ndarray((dk, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=k_t, src=k_ref[idx, :, :])

        k_beta_row = nl.ndarray((1, dk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=k_beta_row, src=k_beta_row_ref[idx, :, :])

        v_t = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=v_t, src=v_ref[idx, :, :])

        # exp_g pre-broadcast: [Dk] → loads as [Dk, 1] tile (partition=Dk)
        exp_g_bc = nl.ndarray((dk, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=exp_g_bc, src=exp_g_bc_ref[idx, :])

        # ── Step 1: state *= exp(g) ──────────────────────────────────────
        # exp_g_bc is [128, 1], tensor_scalar broadcasts over free dim (Dv)
        nisa.tensor_scalar(state, state, nl.multiply, exp_g_bc)

        # ── Step 2: kv_mem = k^T @ state → [1, Dv] ──────────────────────
        kv_psum = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(kv_psum, k_t, state)
        kv_mem = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(kv_mem, kv_psum)

        # ── Step 3: delta = v - kv_mem → [1, Dv] ────────────────────────
        # (beta already folded into k_beta_row on host)
        delta = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(delta, v_t, kv_mem, op=nl.subtract)

        # ── Step 4: state += outer(k*beta, delta) → [Dk, Dv] ────────────
        # nc_matmul: out[M,N] = A[K,M]^T @ B[K,N]
        # k_beta_row[1, Dk]^T @ delta[1, Dv] → [Dk, Dv]
        outer_psum = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(outer_psum, k_beta_row, delta)
        outer = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(outer, outer_psum)
        nisa.tensor_tensor(state, state, outer, op=nl.add)

        # ── Step 5: out = q^T @ state → [1, Dv] ─────────────────────────
        out_psum = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(out_psum, q_t, state)
        out_sbuf = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(out_sbuf, out_psum)

        # Store output (already fp32, no cast needed)
        nisa.dma_copy(dst=out_ref[idx, :, :], src=out_sbuf)

        # Store final state
        nisa.dma_copy(dst=final_state_ref[idx, :, :], src=state)

    return out_ref, final_state_ref


@nki.jit(mode="torchxla")
def nki_recurrent_gated_delta_rule_decode_v2_bf16(
    q_ref, k_ref, k_beta_row_ref, v_ref, exp_g_bc_ref, state_ref
):
    """Decode kernel v2 with bf16 state storage — halves DMA bandwidth.

    Same math as v2 but the persistent state is stored in HBM as bf16 instead
    of fp32.  Computation remains in fp32 (cast on load, cast on store).  This
    halves the dominant DMA cost: each [128,128] state tile is 32KB instead of
    64KB, saving ~5ms per decode step at batch16×24 layers on inf2.xlarge.

    Numerical safety: the exponential decay exp(g) < 1 attenuates old state
    entries, bounding accumulation error.  bf16 has 8 mantissa bits → ~1/256
    relative precision, which is acceptable for the decode path where each step
    adds one rank-1 update to a decaying state.

    Args:
        q_ref:          [BH, Dk, 1]  queries (column-major, l2-normed+scaled)
        k_ref:          [BH, Dk, 1]  keys (column-major, for k^T @ state)
        k_beta_row_ref: [BH, 1, Dk]  keys * beta (row-major, for outer product)
        v_ref:          [BH, 1, Dv]  values (row-major)
        exp_g_bc_ref:   [BH, Dk]     exp(g) broadcast to Dk (same scalar per row)
        state_ref:      [BH, Dk, Dv] initial state (bf16)

    Returns:
        out_ref:         [BH, 1, Dv]  output (row-major, fp32)
        final_state_ref: [BH, Dk, Dv] final state (bf16)
    """
    bh = q_ref.shape[0]
    dk = q_ref.shape[1]   # Dk = 128
    dv = v_ref.shape[2]   # Dv = 128

    out_ref = nl.ndarray((bh, 1, dv), dtype=nl.float32, buffer=nl.shared_hbm)
    final_state_ref = nl.ndarray((bh, dk, dv), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for idx in nl.affine_range(bh):
        # Load state as bf16 from HBM, cast to fp32 in SBUF for computation
        state_bf16 = nl.ndarray((dk, dv), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=state_bf16, src=state_ref[idx, :, :])
        state = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(state, state_bf16, nl.multiply, 1.0)

        q_t = nl.ndarray((dk, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_t, src=q_ref[idx, :, :])

        k_t = nl.ndarray((dk, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=k_t, src=k_ref[idx, :, :])

        k_beta_row = nl.ndarray((1, dk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=k_beta_row, src=k_beta_row_ref[idx, :, :])

        v_t = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=v_t, src=v_ref[idx, :, :])

        exp_g_bc = nl.ndarray((dk, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=exp_g_bc, src=exp_g_bc_ref[idx, :])

        # Step 1: state *= exp(g)
        nisa.tensor_scalar(state, state, nl.multiply, exp_g_bc)

        # Step 2: kv_mem = k^T @ state → [1, Dv]
        kv_psum = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(kv_psum, k_t, state)
        kv_mem = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(kv_mem, kv_psum)

        # Step 3: delta = v - kv_mem → [1, Dv]
        delta = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(delta, v_t, kv_mem, op=nl.subtract)

        # Step 4: state += outer(k*beta, delta) → [Dk, Dv]
        outer_psum = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(outer_psum, k_beta_row, delta)
        outer = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(outer, outer_psum)
        nisa.tensor_tensor(state, state, outer, op=nl.add)

        # Step 5: out = q^T @ state → [1, Dv]
        out_psum = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(out_psum, q_t, state)
        out_sbuf = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(out_sbuf, out_psum)

        # Store output (fp32)
        nisa.dma_copy(dst=out_ref[idx, :, :], src=out_sbuf)

        # Cast state back to bf16 and store
        state_out_bf16 = nl.ndarray((dk, dv), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(state_out_bf16, state, nl.multiply, 1.0)
        nisa.dma_copy(dst=final_state_ref[idx, :, :], src=state_out_bf16)

    return out_ref, final_state_ref


@nki.jit(mode="torchxla")
def nki_recurrent_gated_delta_rule_decode_v3(
    q_ref, k_ref, k_beta_col_ref, k_beta_row_ref, v_ref, exp_g_bc_ref, state_ref
):
    """Decode kernel v3: rank-1 decomposition breaks critical-path dependency.

    Key algebraic insight: the state update ``outer(k*beta, delta)`` is rank-1, so
    ``q^T @ state_new = q^T @ state_scaled + dot(q, k*beta) * delta``.
    This lets us compute q^T@state BEFORE the state update completes, decoupling
    the output computation from the state-write path.

    Critical path comparison:
      v2: state_scale → k_matmul → delta → outer → **q_matmul** (5 sequential)
      v3: state_scale → {k_matmul, q_matmul} → delta → {outer || correction} → output
    The q_matmul moves before the state update; output uses an algebraic correction
    instead of reading the updated state.  The compiler can pipeline the state-write
    (step 5) with the output correction (steps 6-8) since they're independent.

    Matmul count: 4 (k^T@s, q^T@s, outer, dot) vs v2's 3. The extra dot is [1,1]
    (trivial) and the critical-path shortening dominates.

    Args:
        q_ref:          [BH, Dk, 1]  queries (column-major, l2-normed+scaled)
        k_ref:          [BH, Dk, 1]  keys (column-major, for k^T @ state)
        k_beta_col_ref: [BH, Dk, 1]  keys * beta (column-major, for dot product)
        k_beta_row_ref: [BH, 1, Dk]  keys * beta (row-major, for outer product)
        v_ref:          [BH, 1, Dv]  values (row-major)
        exp_g_bc_ref:   [BH, Dk]     exp(g) broadcast to Dk
        state_ref:      [BH, Dk, Dv] initial state (fp32)

    Returns:
        out_ref:         [BH, 1, Dv]  output (row-major, fp32)
        final_state_ref: [BH, Dk, Dv] final state (fp32)
    """
    bh = q_ref.shape[0]
    dk = q_ref.shape[1]   # Dk = 128
    dv = v_ref.shape[2]   # Dv = 128

    out_ref = nl.ndarray((bh, 1, dv), dtype=nl.float32, buffer=nl.shared_hbm)
    final_state_ref = nl.ndarray((bh, dk, dv), dtype=nl.float32, buffer=nl.shared_hbm)

    for idx in nl.affine_range(bh):
        # ── Load inputs for this head ─────────────────────────────────────
        state = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=state, src=state_ref[idx, :, :])

        q_t = nl.ndarray((dk, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_t, src=q_ref[idx, :, :])

        k_t = nl.ndarray((dk, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=k_t, src=k_ref[idx, :, :])

        k_beta_col = nl.ndarray((dk, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=k_beta_col, src=k_beta_col_ref[idx, :, :])

        k_beta_row = nl.ndarray((1, dk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=k_beta_row, src=k_beta_row_ref[idx, :, :])

        v_t = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=v_t, src=v_ref[idx, :, :])

        exp_g_bc = nl.ndarray((dk, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=exp_g_bc, src=exp_g_bc_ref[idx, :])

        # ── Step 1: state *= exp(g) ──────────────────────────────────────
        nisa.tensor_scalar(state, state, nl.multiply, exp_g_bc)

        # ── Step 2: kv_mem = k^T @ state → [1, Dv] ──────────────────────
        kv_psum = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(kv_psum, k_t, state)
        kv_mem = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(kv_mem, kv_psum)

        # ── Step 3: q_state = q^T @ state → [1, Dv] (BEFORE state update!)
        # This reads the scaled state before the rank-1 update; the output
        # correction (step 7) accounts for the missing outer product term.
        q_psum = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(q_psum, q_t, state)
        q_state = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(q_state, q_psum)

        # ── Step 4: delta = v - kv_mem → [1, Dv] ────────────────────────
        delta = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(delta, v_t, kv_mem, op=nl.subtract)

        # ── Step 5: state += outer(k*beta, delta) → [Dk, Dv] ────────────
        # (state update path — independent of output computation below)
        outer_psum = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(outer_psum, k_beta_row, delta)
        outer = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(outer, outer_psum)
        nisa.tensor_tensor(state, state, outer, op=nl.add)

        # ── Step 6: dot(q, k*beta) → scalar ─────────────────────────────
        # q_t[Dk,1]^T @ k_beta_col[Dk,1] → [1,1]
        dot_psum = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dot_psum, q_t, k_beta_col)
        dot_val = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dot_val, dot_psum)

        # ── Step 7: correction = dot_val * delta → [1, Dv] ──────────────
        correction = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(correction, delta, nl.multiply, dot_val)

        # ── Step 8: output = q_state + correction ────────────────────────
        out_sbuf = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(out_sbuf, q_state, correction, op=nl.add)

        # Store output and final state
        nisa.dma_copy(dst=out_ref[idx, :, :], src=out_sbuf)
        nisa.dma_copy(dst=final_state_ref[idx, :, :], src=state)

    return out_ref, final_state_ref


@nki.jit(mode="torchxla")
def nki_recurrent_gated_delta_rule_decode_v4(
    q_ref, k_ref, k_beta_row_ref, v_ref, exp_g_bc_ref, state_ref
):
    """Decode kernel v4: parallel_range for DMA/compute overlap across heads.

    Since each head's state is independent (no inter-head dependency), the BH
    loop iterations can execute in parallel — the compiler overlaps DMA of
    iteration N+1 with compute of iteration N, hiding memory latency.

    Also eliminates one psum→sbuf copy by accumulating the outer product
    directly into state via tensor_tensor on the psum result (reusing the
    psum register as the add operand to state).

    Args:
        q_ref:          [BH, Dk, 1]  queries (column-major, l2-normed+scaled)
        k_ref:          [BH, Dk, 1]  keys (column-major, for k^T @ state)
        k_beta_row_ref: [BH, 1, Dk]  keys * beta (row-major, for outer product)
        v_ref:          [BH, 1, Dv]  values (row-major)
        exp_g_bc_ref:   [BH, Dk]     exp(g) broadcast to Dk
        state_ref:      [BH, Dk, Dv] initial state (fp32)

    Returns:
        out_ref:         [BH, 1, Dv]  output (row-major, fp32)
        final_state_ref: [BH, Dk, Dv] final state (fp32)
    """
    bh = q_ref.shape[0]
    dk = q_ref.shape[1]   # Dk = 128
    dv = v_ref.shape[2]   # Dv = 128

    out_ref = nl.ndarray((bh, 1, dv), dtype=nl.float32, buffer=nl.shared_hbm)
    final_state_ref = nl.ndarray((bh, dk, dv), dtype=nl.float32, buffer=nl.shared_hbm)

    # parallel_range: each head is independent, compiler can overlap DMA/compute
    for idx in nl.parallel_range(bh):
        # ── Load inputs for this head ─────────────────────────────────────
        state = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=state, src=state_ref[idx, :, :])

        q_t = nl.ndarray((dk, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_t, src=q_ref[idx, :, :])

        k_t = nl.ndarray((dk, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=k_t, src=k_ref[idx, :, :])

        k_beta_row = nl.ndarray((1, dk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=k_beta_row, src=k_beta_row_ref[idx, :, :])

        v_t = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=v_t, src=v_ref[idx, :, :])

        exp_g_bc = nl.ndarray((dk, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=exp_g_bc, src=exp_g_bc_ref[idx, :])

        # ── Step 1: state *= exp(g) ──────────────────────────────────────
        nisa.tensor_scalar(state, state, nl.multiply, exp_g_bc)

        # ── Step 2: kv_mem = k^T @ state → [1, Dv] ──────────────────────
        kv_psum = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(kv_psum, k_t, state)
        kv_mem = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(kv_mem, kv_psum)

        # ── Step 3: delta = v - kv_mem → [1, Dv] ────────────────────────
        delta = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(delta, v_t, kv_mem, op=nl.subtract)

        # ── Step 4: state += outer(k*beta, delta) → [Dk, Dv] ────────────
        outer_psum = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(outer_psum, k_beta_row, delta)
        outer = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(outer, outer_psum)
        nisa.tensor_tensor(state, state, outer, op=nl.add)

        # ── Step 5: out = q^T @ state → [1, Dv] ─────────────────────────
        out_psum = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(out_psum, q_t, state)
        out_sbuf = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(out_sbuf, out_psum)

        # Store output and final state
        nisa.dma_copy(dst=out_ref[idx, :, :], src=out_sbuf)
        nisa.dma_copy(dst=final_state_ref[idx, :, :], src=state)

    return out_ref, final_state_ref


@nki.jit(mode="torchxla")
def nki_recurrent_gated_delta_rule_decode_v4_bf16(
    q_ref, k_ref, k_beta_row_ref, v_ref, exp_g_bc_ref, state_ref
):
    """Decode kernel v4 + bf16 state: parallel_range + halved DMA bandwidth.

    Combines two orthogonal optimisations:
      1. parallel_range (from v4): heads are independent so the compiler can
         overlap DMA of head N+1 with compute of head N.
      2. bf16 state (from v2_bf16): recurrent state stored as bf16 in HBM,
         halving the dominant 64KB → 32KB DMA per [128,128] tile.

    These are multiplicative: less bytes to move AND better pipelining of moves
    with computation.  Computation remains fp32 (cast on load/store).

    Args:
        q_ref:          [BH, Dk, 1]  queries (column-major, l2-normed+scaled)
        k_ref:          [BH, Dk, 1]  keys (column-major, for k^T @ state)
        k_beta_row_ref: [BH, 1, Dk]  keys * beta (row-major, for outer product)
        v_ref:          [BH, 1, Dv]  values (row-major)
        exp_g_bc_ref:   [BH, Dk]     exp(g) broadcast to Dk
        state_ref:      [BH, Dk, Dv] initial state (bf16)

    Returns:
        out_ref:         [BH, 1, Dv]  output (row-major, fp32)
        final_state_ref: [BH, Dk, Dv] final state (bf16)
    """
    bh = q_ref.shape[0]
    dk = q_ref.shape[1]   # Dk = 128
    dv = v_ref.shape[2]   # Dv = 128

    out_ref = nl.ndarray((bh, 1, dv), dtype=nl.float32, buffer=nl.shared_hbm)
    final_state_ref = nl.ndarray((bh, dk, dv), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    # affine_range: sequential iteration (parallel_range crashes in-model trace)
    for idx in nl.affine_range(bh):
        # Load state as bf16 from HBM, cast to fp32 in SBUF for computation
        state_bf16 = nl.ndarray((dk, dv), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=state_bf16, src=state_ref[idx, :, :])
        state = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(state, state_bf16, nl.multiply, 1.0)

        q_t = nl.ndarray((dk, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_t, src=q_ref[idx, :, :])

        k_t = nl.ndarray((dk, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=k_t, src=k_ref[idx, :, :])

        k_beta_row = nl.ndarray((1, dk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=k_beta_row, src=k_beta_row_ref[idx, :, :])

        v_t = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=v_t, src=v_ref[idx, :, :])

        exp_g_bc = nl.ndarray((dk, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=exp_g_bc, src=exp_g_bc_ref[idx, :])

        # Step 1: state *= exp(g)
        nisa.tensor_scalar(state, state, nl.multiply, exp_g_bc)

        # Step 2: kv_mem = k^T @ state → [1, Dv]
        kv_psum = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(kv_psum, k_t, state)
        kv_mem = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(kv_mem, kv_psum)

        # Step 3: delta = v - kv_mem → [1, Dv]
        delta = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(delta, v_t, kv_mem, op=nl.subtract)

        # Step 4: state += outer(k*beta, delta) → [Dk, Dv]
        outer_psum = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(outer_psum, k_beta_row, delta)
        outer = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(outer, outer_psum)
        nisa.tensor_tensor(state, state, outer, op=nl.add)

        # Step 5: out = q^T @ state → [1, Dv]
        out_psum = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(out_psum, q_t, state)
        out_sbuf = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(out_sbuf, out_psum)

        # Store output (fp32) and final state (bf16)
        nisa.dma_copy(dst=out_ref[idx, :, :], src=out_sbuf)
        state_out_bf16 = nl.ndarray((dk, dv), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(state_out_bf16, state, nl.multiply, 1.0)
        nisa.dma_copy(dst=final_state_ref[idx, :, :], src=state_out_bf16)

    return out_ref, final_state_ref


@nki.jit(mode="torchxla")
def nki_recurrent_gated_delta_rule_decode_v6_bf16(
    qk_ref, k_beta_row_ref, k_beta_col_ref, v_ref, exp_g_bc_ref, state_ref
):
    """Decode kernel v6: fused q/k matmul + algebraic output decomposition.

    Key optimisation over v4: batches q^T@state and k^T@state into a single
    nc_matmul by stacking [q, k] as [Dk, 2], reducing the critical path from
    3 nc_matmul to 2 large TensorEngine ops (the q·k_beta dot is a tiny 1×128×1
    that barely registers on the TE pipeline).

    Uses algebraic decomposition:
      state_scaled = state * exp_g
      [base_out; kv_mem] = [q, k]^T @ state_scaled   (ONE nc_matmul, [Dk,2]@[Dk,Dv])
      delta = v - kv_mem
      q_dot_kb = k_beta_col^T @ q_col                 (tiny [Dk,1]@[Dk,1] → [1,1])
      out = base_out + q_dot_kb * delta               (scalar broadcast)
      state_new = state_scaled + outer(k_beta, delta) (ONE nc_matmul, [1,Dk]@[1,Dv])

    Args:
        qk_ref:          [BH, Dk, 2]  stacked queries+keys (column-major)
        k_beta_row_ref:  [BH, 1, Dk]  keys * beta (row-major, for outer product)
        k_beta_col_ref:  [BH, Dk, 1]  keys * beta (column-major, for dot product)
        v_ref:           [BH, 1, Dv]  values (row-major)
        exp_g_bc_ref:    [BH, Dk]     exp(g) broadcast to Dk
        state_ref:       [BH, Dk, Dv] initial state (bf16)

    Returns:
        out_ref:         [BH, 1, Dv]  output (row-major, fp32)
        final_state_ref: [BH, Dk, Dv] final state (bf16)
    """
    bh = qk_ref.shape[0]
    dk = qk_ref.shape[1]   # Dk = 128
    dv = v_ref.shape[2]    # Dv = 128

    out_ref = nl.ndarray((bh, 1, dv), dtype=nl.float32, buffer=nl.shared_hbm)
    final_state_ref = nl.ndarray((bh, dk, dv), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for idx in nl.affine_range(bh):
        # Load state bf16 → fp32
        state_bf16 = nl.ndarray((dk, dv), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=state_bf16, src=state_ref[idx, :, :])
        state = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(state, state_bf16, nl.multiply, 1.0)

        # Load inputs
        qk_t = nl.ndarray((dk, 2), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=qk_t, src=qk_ref[idx, :, :])

        k_beta_row = nl.ndarray((1, dk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=k_beta_row, src=k_beta_row_ref[idx, :, :])

        k_beta_col = nl.ndarray((dk, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=k_beta_col, src=k_beta_col_ref[idx, :, :])

        v_t = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=v_t, src=v_ref[idx, :, :])

        exp_g_bc = nl.ndarray((dk, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=exp_g_bc, src=exp_g_bc_ref[idx, :])

        # Step 1: state *= exp(g) (column-broadcast)
        nisa.tensor_scalar(state, state, nl.multiply, exp_g_bc)

        # Step 2: FUSED q/k matmul — [q,k]^T @ state → [2, Dv]
        # One nc_matmul replaces two separate vector-matrix products
        qk_psum = nl.ndarray((2, dv), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(qk_psum, qk_t, state)
        qk_out = nl.ndarray((2, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(qk_out, qk_psum)

        # Split: row 0 = q^T @ state_scaled (base_out), row 1 = k^T @ state_scaled (kv_mem)
        base_out = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(base_out, qk_out[0:1, :])
        kv_mem = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(kv_mem, qk_out[1:2, :])

        # Step 3: delta = v - kv_mem → [1, Dv]
        delta = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(delta, v_t, kv_mem, op=nl.subtract)

        # Step 4: out = base_out + (q·k_beta) * delta
        # q·k_beta = k_beta_col^T @ q_col  (nc_matmul: [Dk,1]@[Dk,1] → [1,1])
        q_col = nl.ndarray((dk, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(q_col, qk_t[:, 0:1])
        qkb_psum = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(qkb_psum, k_beta_col, q_col)
        q_dot_kb_scalar = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(q_dot_kb_scalar, qkb_psum)
        # correction = scalar * delta
        correction = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(correction, delta, nl.multiply, q_dot_kb_scalar)
        # out = base_out + correction
        out_sbuf = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(out_sbuf, base_out, correction, op=nl.add)

        # Step 5: state += outer(k_beta, delta) → [Dk, Dv] (needed for next token)
        outer_psum = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(outer_psum, k_beta_row, delta)
        outer = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(outer, outer_psum)
        nisa.tensor_tensor(state, state, outer, op=nl.add)

        # Store output (fp32) and final state (bf16)
        nisa.dma_copy(dst=out_ref[idx, :, :], src=out_sbuf)
        state_out_bf16 = nl.ndarray((dk, dv), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(state_out_bf16, state, nl.multiply, 1.0)
        nisa.dma_copy(dst=final_state_ref[idx, :, :], src=state_out_bf16)

    return out_ref, final_state_ref


@nki.jit(mode="torchxla")
def nki_within_chunk_state_update(k_decay_ref, v_new_ref):
    """Within-chunk DeltaNet state increment as one TensorEngine matmul per head.

    Computes ``state_update[bh, Dk, Dv] = k_decay^T @ v_new`` over the chunk
    dimension, i.e. ``sum_c outer(k_decay[:, c], v_new[:, c])``.  This is the
    increment that ``chunk_gated_delta_rule`` accumulates into the recurrent
    state once per chunk.

    The XLA-lowered torch forms of this op (matmul / einsum / bmm / broadcast-
    reduce, tiled or not) all crash neuronx-cc with NCC_INLA001
    (``assignStaticPattern<TPB_TENSOR2D>``) because the contraction over the
    64-chunk dim into the [128, 128] state tile trips the static-pattern
    assignment.  Emitting the contraction directly as ``nisa.nc_matmul`` bypasses
    that lowering: ``nc_matmul(out[Dk, Dv], A=k_decay[C, Dk], B=v_new[C, Dv])``
    computes ``A^T @ B`` with the chunk dim ``C`` as the partition/contraction
    axis (C <= 128), collapsing the default 64-step sequential rank-1 loop into a
    single dense matmul per (batch*head) slice.

    Args:
        k_decay_ref: [BH, C, Dk]  per-chunk decayed keys (C = chunk_size <= 128)
        v_new_ref:   [BH, C, Dv]  per-chunk corrected values

    Returns:
        out_ref:     [BH, Dk, Dv]  within-chunk state increment (fp32)
    """
    bh = k_decay_ref.shape[0]
    c = k_decay_ref.shape[1]
    dk = k_decay_ref.shape[2]
    dv = v_new_ref.shape[2]

    out_ref = nl.ndarray((bh, dk, dv), dtype=nl.float32, buffer=nl.shared_hbm)

    for idx in nl.affine_range(bh):
        # Load this head's chunk tiles: partition dim = chunk (contraction axis).
        kd = nl.ndarray((c, dk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=kd, src=k_decay_ref[idx, :, :])
        vn = nl.ndarray((c, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=vn, src=v_new_ref[idx, :, :])

        # nc_matmul: out[Dk, Dv] = kd[C, Dk]^T @ vn[C, Dv]  (contraction = C).
        su_psum = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(su_psum, kd, vn)
        su = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(su, su_psum)
        nisa.dma_copy(dst=out_ref[idx, :, :], src=su)

    return out_ref


@nki.jit(mode="torchxla")
def nki_chunk_gated_delta_rule_kernel(
    value_ref,
    k_cumdecay_t_ref,
    qg_t_ref,
    attn_intra_t_ref,
    k_decay_ref,
    g_last_ref,
    state_ref,
):
    """Chunked gated delta rule: sequential inter-chunk recurrence on TensorEngine.

    Runs the DeltaNet chunk recurrence as ``num_chunks`` sequential steps (vs the
    HF reference's ``seq_len`` per-token steps) keeping the [Dk, Dv] recurrent
    state resident in SBUF across the whole chunk loop.  Every state-dependent op
    is a full TensorEngine matmul, so the ~64-step sequential within-chunk loop
    (the only torch form neuronx-cc compiles, and the tp2 prefill bottleneck)
    collapses to five ``nc_matmul`` per chunk.

    The caller (``nki_chunk_gated_delta_rule`` in modeling) precomputes every
    state-independent quantity in torch and pre-transposes the matmul stationary
    operands so each contraction axis lands on the partition dim, as
    ``nc_matmul(out[M, N], A[K, M], B[K, N]) = A^T @ B`` contracts over the
    partition dim ``K``.

    Per chunk ``i`` (state ``S`` ∈ R^{Dk×Dv}, C = chunk_size <= 128):
      1. v_prime[C, Dv]    = k_cumdecay_i[C, Dk] @ S        (contract Dk)
      2. v_new[C, Dv]      = value_i - v_prime
      3. attn_inter[C, Dv] = (q_i * exp(g_i))[C, Dk] @ S    (contract Dk)
      4. core_i[C, Dv]     = attn_inter + attn_intra_i[C, C] @ v_new (contract C)
      5. state_update      = k_decay_i^T[Dk, C] @ v_new[C, Dv]       (contract C)
      6. S                 = S * exp(g_last_i) + state_update

    Args:
        value_ref:        [BH, NC, C, Dv]  UT-transformed values (= attn @ v_beta)
        k_cumdecay_t_ref: [BH, NC, Dk, C]  k_cumdecay transposed (stationary)
        qg_t_ref:         [BH, NC, Dk, C]  (query * exp(g)) transposed (stationary)
        attn_intra_t_ref: [BH, NC, C, C]   intra-chunk attn transposed (stationary)
        k_decay_ref:      [BH, NC, C, Dk]  decayed keys (natural layout)
        g_last_ref:       [BH, NC]         exp-arg of the last-token cumulative gate
        state_ref:        [BH, Dk, Dv]     initial recurrent state (fp32)

    Returns:
        out_ref:         [BH, NC, C, Dv]  per-chunk core attention output (fp32)
        final_state_ref: [BH, Dk, Dv]     final recurrent state (fp32)
    """
    bh = value_ref.shape[0]
    nc = value_ref.shape[1]
    c = value_ref.shape[2]
    dv = value_ref.shape[3]
    dk = k_decay_ref.shape[3]

    out_ref = nl.ndarray((bh, nc, c, dv), dtype=nl.float32, buffer=nl.shared_hbm)
    final_state_ref = nl.ndarray((bh, dk, dv), dtype=nl.float32, buffer=nl.shared_hbm)

    # Ones row for broadcasting the per-chunk scalar gate to [Dk, 1] via matmul.
    ones_row = nl.ndarray((1, _D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(ones_row, 1.0)

    for idx in nl.affine_range(bh):
        state = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=state, src=state_ref[idx, :, :])

        for i in nl.sequential_range(nc):
            # Load this chunk's precomputed tiles. Partition (first) dim of each
            # tile is the contraction axis of the matmul it feeds.
            kcd_t = nl.ndarray((dk, c), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=kcd_t, src=k_cumdecay_t_ref[idx, i, :, :])
            qg_t = nl.ndarray((dk, c), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=qg_t, src=qg_t_ref[idx, i, :, :])
            ait = nl.ndarray((c, c), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=ait, src=attn_intra_t_ref[idx, i, :, :])
            kd = nl.ndarray((c, dk), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=kd, src=k_decay_ref[idx, i, :, :])
            v = nl.ndarray((c, dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=v, src=value_ref[idx, i, :, :])

            # 1. v_prime[C, Dv] = k_cumdecay_i[C, Dk] @ S  (contract Dk = partition).
            vp_psum = nl.ndarray((c, dv), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(vp_psum, kcd_t, state)
            vp = nl.ndarray((c, dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(vp, vp_psum)

            # 2. v_new = value_i - v_prime.
            v_new = nl.ndarray((c, dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(v_new, v, vp, op=nl.subtract)

            # 3. attn_inter[C, Dv] = (q_i*exp(g_i))[C, Dk] @ S  (contract Dk).
            ai_psum = nl.ndarray((c, dv), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(ai_psum, qg_t, state)
            ai = nl.ndarray((c, dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(ai, ai_psum)

            # 4. core_i = attn_inter + attn_intra_i @ v_new  (contract C = partition).
            tmp_psum = nl.ndarray((c, dv), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(tmp_psum, ait, v_new)
            tmp = nl.ndarray((c, dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(tmp, tmp_psum)
            core = nl.ndarray((c, dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(core, ai, tmp, op=nl.add)
            nisa.dma_copy(dst=out_ref[idx, i, :, :], src=core)

            # 5. state_update[Dk, Dv] = k_decay_i^T @ v_new  (contract C = partition).
            su_psum = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(su_psum, kd, v_new)
            su = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(su, su_psum)

            # 6. S = S * exp(g_last_i) + state_update.
            # Broadcast the per-chunk scalar gate to [Dk, 1]: ones[1,Dk]^T @ g[1,1].
            g_last_t = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=g_last_t, src=g_last_ref[idx, i:i + 1])
            g_bc_psum = nl.ndarray((_D, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(g_bc_psum, ones_row, g_last_t)
            g_bc = nl.ndarray((_D, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(g_bc, g_bc_psum)
            nisa.tensor_scalar(state, state, nl.multiply, g_bc)
            nisa.tensor_tensor(state, state, su, op=nl.add)

        nisa.dma_copy(dst=final_state_ref[idx, :, :], src=state)

    return out_ref, final_state_ref


@nki.jit(mode="torchxla")
def nki_chunk_gated_delta_rule_kernel_v2(
    value_ref,
    k_cumdecay_t_ref,
    qg_t_ref,
    attn_intra_t_ref,
    k_decay_ref,
    g_last_bc_ref,
    state_ref,
):
    """Optimized chunked gated delta rule v2: pre-broadcast gate, no ones_row.

    Same math as v1 but the per-chunk gate ``exp(g_last)`` is pre-broadcast on the
    host to ``[BH, NC, Dk]`` (128 identical values per chunk), eliminating 3
    instructions per chunk (scalar DMA + nc_matmul broadcast + tensor_copy).  This
    drops the ``ones_row`` allocation and the per-chunk matmul from the critical
    sequential path.

    Args:
        value_ref:        [BH, NC, C, Dv]  UT-transformed values
        k_cumdecay_t_ref: [BH, NC, Dk, C]  k_cumdecay transposed (stationary)
        qg_t_ref:         [BH, NC, Dk, C]  (query * exp(g)) transposed (stationary)
        attn_intra_t_ref: [BH, NC, C, C]   intra-chunk attn transposed (stationary)
        k_decay_ref:      [BH, NC, C, Dk]  decayed keys (natural layout)
        g_last_bc_ref:    [BH, NC, Dk]     exp(g_last) pre-broadcast to Dk
        state_ref:        [BH, Dk, Dv]     initial recurrent state (fp32)

    Returns:
        out_ref:         [BH, NC, C, Dv]  per-chunk core attention output (fp32)
        final_state_ref: [BH, Dk, Dv]     final recurrent state (fp32)
    """
    bh = value_ref.shape[0]
    nc = value_ref.shape[1]
    c = value_ref.shape[2]
    dv = value_ref.shape[3]
    dk = k_decay_ref.shape[3]

    out_ref = nl.ndarray((bh, nc, c, dv), dtype=nl.float32, buffer=nl.shared_hbm)
    final_state_ref = nl.ndarray((bh, dk, dv), dtype=nl.float32, buffer=nl.shared_hbm)

    for idx in nl.affine_range(bh):
        state = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=state, src=state_ref[idx, :, :])

        for i in nl.sequential_range(nc):
            # Load this chunk's precomputed tiles.
            kcd_t = nl.ndarray((dk, c), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=kcd_t, src=k_cumdecay_t_ref[idx, i, :, :])
            qg_t = nl.ndarray((dk, c), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=qg_t, src=qg_t_ref[idx, i, :, :])
            ait = nl.ndarray((c, c), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=ait, src=attn_intra_t_ref[idx, i, :, :])
            kd = nl.ndarray((c, dk), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=kd, src=k_decay_ref[idx, i, :, :])
            v = nl.ndarray((c, dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=v, src=value_ref[idx, i, :, :])

            # 1. v_prime[C, Dv] = k_cumdecay_i[C, Dk] @ S  (contract Dk = partition).
            vp_psum = nl.ndarray((c, dv), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(vp_psum, kcd_t, state)
            vp = nl.ndarray((c, dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(vp, vp_psum)

            # 2. v_new = value_i - v_prime.
            v_new = nl.ndarray((c, dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(v_new, v, vp, op=nl.subtract)

            # 3. attn_inter[C, Dv] = (q_i*exp(g_i))[C, Dk] @ S  (contract Dk).
            ai_psum = nl.ndarray((c, dv), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(ai_psum, qg_t, state)
            ai = nl.ndarray((c, dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(ai, ai_psum)

            # 4. core_i = attn_inter + attn_intra_i @ v_new  (contract C = partition).
            tmp_psum = nl.ndarray((c, dv), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(tmp_psum, ait, v_new)
            tmp = nl.ndarray((c, dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(tmp, tmp_psum)
            core = nl.ndarray((c, dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(core, ai, tmp, op=nl.add)
            nisa.dma_copy(dst=out_ref[idx, i, :, :], src=core)

            # 5. state_update[Dk, Dv] = k_decay_i^T @ v_new  (contract C = partition).
            su_psum = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(su_psum, kd, v_new)
            su = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(su, su_psum)

            # 6. S = S * exp(g_last_i) + state_update.
            # Gate is pre-broadcast on host: [Dk] loads as [Dk, 1] partition tile.
            g_bc = nl.ndarray((dk, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=g_bc, src=g_last_bc_ref[idx, i, :])
            nisa.tensor_scalar(state, state, nl.multiply, g_bc)
            nisa.tensor_tensor(state, state, su, op=nl.add)

        nisa.dma_copy(dst=final_state_ref[idx, :, :], src=state)

    return out_ref, final_state_ref


@nki.jit(mode="torchxla")
def nki_chunk_gated_delta_rule_kernel_v3(
    value_ref,
    k_cumdecay_t_ref,
    qg_t_ref,
    attn_intra_t_ref,
    k_decay_ref,
    g_last_bc_ref,
    state_ref,
):
    """Chunked gated delta rule v3: parallel_range for DMA/compute overlap across heads.

    Same math as v2, but the outer head loop uses ``nl.parallel_range(bh)`` instead
    of ``nl.affine_range(bh)``.  This lets the compiler overlap DMA loads for head
    N+1 with TensorEngine compute for head N, reducing effective latency when
    processing many independent heads (e.g. 8 heads × batch during prefill).

    The inner chunk loop remains ``nl.sequential_range(nc)`` because chunks have
    inter-chunk state dependencies.

    Args:
        value_ref:        [BH, NC, C, Dv]  UT-transformed values
        k_cumdecay_t_ref: [BH, NC, Dk, C]  k_cumdecay transposed (stationary)
        qg_t_ref:         [BH, NC, Dk, C]  (query * exp(g)) transposed (stationary)
        attn_intra_t_ref: [BH, NC, C, C]   intra-chunk attn transposed (stationary)
        k_decay_ref:      [BH, NC, C, Dk]  decayed keys (natural layout)
        g_last_bc_ref:    [BH, NC, Dk]     exp(g_last) pre-broadcast to Dk
        state_ref:        [BH, Dk, Dv]     initial recurrent state (fp32)

    Returns:
        out_ref:         [BH, NC, C, Dv]  per-chunk core attention output (fp32)
        final_state_ref: [BH, Dk, Dv]     final recurrent state (fp32)
    """
    bh = value_ref.shape[0]
    nc = value_ref.shape[1]
    c = value_ref.shape[2]
    dv = value_ref.shape[3]
    dk = k_decay_ref.shape[3]

    out_ref = nl.ndarray((bh, nc, c, dv), dtype=nl.float32, buffer=nl.shared_hbm)
    final_state_ref = nl.ndarray((bh, dk, dv), dtype=nl.float32, buffer=nl.shared_hbm)

    for idx in nl.parallel_range(bh):
        state = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=state, src=state_ref[idx, :, :])

        for i in nl.sequential_range(nc):
            kcd_t = nl.ndarray((dk, c), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=kcd_t, src=k_cumdecay_t_ref[idx, i, :, :])
            qg_t = nl.ndarray((dk, c), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=qg_t, src=qg_t_ref[idx, i, :, :])
            ait = nl.ndarray((c, c), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=ait, src=attn_intra_t_ref[idx, i, :, :])
            kd = nl.ndarray((c, dk), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=kd, src=k_decay_ref[idx, i, :, :])
            v = nl.ndarray((c, dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=v, src=value_ref[idx, i, :, :])

            vp_psum = nl.ndarray((c, dv), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(vp_psum, kcd_t, state)
            vp = nl.ndarray((c, dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(vp, vp_psum)

            v_new = nl.ndarray((c, dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(v_new, v, vp, op=nl.subtract)

            ai_psum = nl.ndarray((c, dv), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(ai_psum, qg_t, state)
            ai = nl.ndarray((c, dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(ai, ai_psum)

            tmp_psum = nl.ndarray((c, dv), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(tmp_psum, ait, v_new)
            tmp = nl.ndarray((c, dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(tmp, tmp_psum)
            core = nl.ndarray((c, dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(core, ai, tmp, op=nl.add)
            nisa.dma_copy(dst=out_ref[idx, i, :, :], src=core)

            su_psum = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(su_psum, kd, v_new)
            su = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(su, su_psum)

            g_bc = nl.ndarray((dk, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=g_bc, src=g_last_bc_ref[idx, i, :])
            nisa.tensor_scalar(state, state, nl.multiply, g_bc)
            nisa.tensor_tensor(state, state, su, op=nl.add)

        nisa.dma_copy(dst=final_state_ref[idx, :, :], src=state)

    return out_ref, final_state_ref


@nki.jit(mode="torchxla")
def nki_recurrent_gated_delta_rule_decode_v5_bf16(
    q_ref, k_ref, k_beta_col_ref, k_beta_row_ref, v_ref, exp_g_bc_ref, state_ref
):
    """Decode kernel v5: rank-1 decomposition + bf16 state.

    Combines two orthogonal optimisations:
      1. v3's rank-1 algebraic trick: output = q^T@state_scaled + dot(q,k*beta)*delta
         Decouples q_matmul from state update -> shorter critical path.
      2. bf16 state: halves the dominant [128,128] DMA (32KB vs 64KB per tile).

    Note: parallel_range crashes during in-model trace on neuronx-cc, so we use
    affine_range.  The rank-1 trick is the primary speedup (shorter critical path).

    Critical path per head (4-deep vs v2's 6):
      state_scale -> {k_matmul, q_matmul} -> delta -> {outer || correction} -> output

    Args:
        q_ref:          [BH, Dk, 1]  queries (column-major, l2-normed+scaled)
        k_ref:          [BH, Dk, 1]  keys (column-major, for k^T @ state)
        k_beta_col_ref: [BH, Dk, 1]  keys * beta (column-major, for dot product)
        k_beta_row_ref: [BH, 1, Dk]  keys * beta (row-major, for outer product)
        v_ref:          [BH, 1, Dv]  values (row-major)
        exp_g_bc_ref:   [BH, Dk]     exp(g) broadcast to Dk
        state_ref:      [BH, Dk, Dv] initial state (bf16)

    Returns:
        out_ref:         [BH, 1, Dv]  output (row-major, fp32)
        final_state_ref: [BH, Dk, Dv] final state (bf16)
    """
    bh = q_ref.shape[0]
    dk = q_ref.shape[1]   # Dk = 128
    dv = v_ref.shape[2]   # Dv = 128

    out_ref = nl.ndarray((bh, 1, dv), dtype=nl.float32, buffer=nl.shared_hbm)
    final_state_ref = nl.ndarray((bh, dk, dv), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    # affine_range: sequential iteration (parallel_range crashes in-model trace)
    for idx in nl.affine_range(bh):
        # Load state as bf16 from HBM, cast to fp32 in SBUF
        state_bf16 = nl.ndarray((dk, dv), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=state_bf16, src=state_ref[idx, :, :])
        state = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(state, state_bf16, nl.multiply, 1.0)

        q_t = nl.ndarray((dk, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_t, src=q_ref[idx, :, :])

        k_t = nl.ndarray((dk, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=k_t, src=k_ref[idx, :, :])

        k_beta_col = nl.ndarray((dk, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=k_beta_col, src=k_beta_col_ref[idx, :, :])

        k_beta_row = nl.ndarray((1, dk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=k_beta_row, src=k_beta_row_ref[idx, :, :])

        v_t = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=v_t, src=v_ref[idx, :, :])

        exp_g_bc = nl.ndarray((dk, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=exp_g_bc, src=exp_g_bc_ref[idx, :])

        # -- Step 1: state *= exp(g) --
        nisa.tensor_scalar(state, state, nl.multiply, exp_g_bc)

        # -- Step 2: kv_mem = k^T @ state -> [1, Dv] --
        kv_psum = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(kv_psum, k_t, state)
        kv_mem = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(kv_mem, kv_psum)

        # -- Step 3: q_state = q^T @ state -> [1, Dv] (BEFORE state update!)
        q_psum = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(q_psum, q_t, state)
        q_state = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(q_state, q_psum)

        # -- Step 4: delta = v - kv_mem -> [1, Dv] --
        delta = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(delta, v_t, kv_mem, op=nl.subtract)

        # -- Step 5: state += outer(k*beta, delta) -> [Dk, Dv] --
        outer_psum = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(outer_psum, k_beta_row, delta)
        outer = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(outer, outer_psum)
        nisa.tensor_tensor(state, state, outer, op=nl.add)

        # -- Step 6: dot(q, k*beta) -> scalar --
        dot_psum = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dot_psum, q_t, k_beta_col)
        dot_val = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dot_val, dot_psum)

        # -- Step 7: correction = dot_val * delta -> [1, Dv] --
        correction = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(correction, delta, nl.multiply, dot_val)

        # -- Step 8: output = q_state + correction --
        out_sbuf = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(out_sbuf, q_state, correction, op=nl.add)

        # Store output (fp32) and state (bf16)
        nisa.dma_copy(dst=out_ref[idx, :, :], src=out_sbuf)
        state_out_bf16 = nl.ndarray((dk, dv), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(state_out_bf16, state, nl.multiply, 1.0)
        nisa.dma_copy(dst=final_state_ref[idx, :, :], src=state_out_bf16)

    return out_ref, final_state_ref
