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
