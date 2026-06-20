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
    """NKI recurrent gated delta rule kernel.

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
