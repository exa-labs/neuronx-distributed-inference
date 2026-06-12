"""
NKI Kernel for the Gated Delta Rule (recurrent form)
=====================================================

Custom NKI kernel implementing the recurrent gated delta rule for
Qwen3.5/Qwen3-Next's DeltaNet layers. Bypasses XLA trace + neuronx-cc
PGTiling entirely, allowing compilation at ANY seq_len and batch_size.

Input layout (caller must transpose q/k before calling):
  q_ref:     [BH, D, T]  — queries in column-major (pre-normalized)
  k_ref:     [BH, D, T]  — keys in column-major (pre-normalized)
  v_ref:     [BH, T, D]  — values in row-major
  exp_g_ref: [BH, T]     — pre-computed exp(g) decay factors
  beta_ref:  [BH, T]     — beta scalars
  state_ref: [BH, D, D]  — initial recurrent state (fp32)

The kernel processes one (batch, head) pair per NeuronCore iteration
and steps sequentially through T tokens maintaining state in SBUF.

Per-token operations (state ∈ R^{Dk×Dv}, Dk=Dv=D=256):
  1. state *= exp_g_t           — gated decay (scalar broadcast)
  2. kv_mem = k_t^T @ state     — read from state (mat-vec)
  3. delta = (v_t - kv_mem) * β  — compute update
  4. state += k_t ⊗ delta       — rank-1 update (outer product)
  5. out_t = q_t^T @ state      — produce output (mat-vec)

Uses nisa.* (ISA-level) APIs for mode='torchxla' compatibility.
Tiling: D=256 split into 2×128 on partition axis for state/q/k.
"""

import nki
import nki.isa as nisa
import nki.language as nl

# ── Qwen3.5-4B constants ─────────────────────────────────────────────────────
_D = 256       # head_dim (Dk = Dv)
_D_HALF = 128  # NeuronCore-v2 partition limit


@nki.jit(mode="torchxla")
def nki_recurrent_gated_delta_rule(
    q_ref, k_ref, v_ref, exp_g_ref, beta_ref, state_ref
):
    """NKI recurrent gated delta rule kernel.

    Args:
        q_ref:     [BH, D, T]  queries (column-major, l2-normed+scaled)
        k_ref:     [BH, D, T]  keys (column-major, l2-normed)
        v_ref:     [BH, T, D]  values (row-major)
        exp_g_ref: [BH, T]     pre-computed exp(g) per token
        beta_ref:  [BH, T]     beta per token
        state_ref: [BH, D, D]  initial state (fp32)

    Returns:
        out_ref:   [BH, T, D]  output (row-major, input dtype)
        final_state_ref: [BH, D, D]  final state (fp32)
    """
    bh = q_ref.shape[0]
    seq_len = q_ref.shape[2]  # T is last dim for q/k (column-major)

    # Allocate HBM outputs
    out_ref = nl.ndarray((bh, seq_len, _D), dtype=q_ref.dtype, buffer=nl.hbm)
    final_state_ref = nl.ndarray((bh, _D, _D), dtype=nl.float32, buffer=nl.hbm)

    # Pre-allocate a ones vector [1, 128] for scalar broadcasting via nc_matmul
    # ones^T @ scalar[1,1] = [128, 1] filled with scalar
    ones_row = nl.ndarray((1, _D_HALF), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(ones_row, 1.0)

    for idx in nl.affine_range(bh):
        # ── Load initial state: 2 tiles of [128, 256] ────────────────────
        state_top = nl.ndarray((_D_HALF, _D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=state_top, src=state_ref[idx, 0:_D_HALF, :])

        state_bot = nl.ndarray((_D_HALF, _D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=state_bot, src=state_ref[idx, _D_HALF:_D, :])

        # ── Sequential time loop ─────────────────────────────────────────
        for t in nl.sequential_range(seq_len):
            # Load q_t and k_t as [128, 1] column tiles (top/bot halves)
            q_top = nl.ndarray((_D_HALF, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=q_top, src=q_ref[idx, 0:_D_HALF, t:t+1])
            q_bot = nl.ndarray((_D_HALF, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=q_bot, src=q_ref[idx, _D_HALF:_D, t:t+1])

            k_top = nl.ndarray((_D_HALF, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=k_top, src=k_ref[idx, 0:_D_HALF, t:t+1])
            k_bot = nl.ndarray((_D_HALF, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=k_bot, src=k_ref[idx, _D_HALF:_D, t:t+1])

            # Load v_t as [1, D] row tile
            v_tile = nl.ndarray((1, _D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=v_tile, src=v_ref[idx, t:t+1, :])

            # Load exp_g_t as [1, 1] scalar tile
            exp_g_tile = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=exp_g_tile, src=exp_g_ref[idx, t:t+1])

            # Load beta_t as [1, 1] scalar tile
            beta_tile = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=beta_tile, src=beta_ref[idx, t:t+1])

            # ── Step 1: state *= exp(g_t) ────────────────────────────────
            # Broadcast scalar to [128, 1] via nc_matmul: ones[1,128]^T @ s[1,1]
            exp_g_bc_psum = nl.ndarray(
                (_D_HALF, 1), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_matmul(exp_g_bc_psum, ones_row, exp_g_tile)
            exp_g_bc = nl.ndarray((_D_HALF, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(exp_g_bc, exp_g_bc_psum)

            # Multiply state by broadcast scalar (broadcasts along free dim)
            nisa.tensor_scalar(
                state_top, state_top, nl.multiply, exp_g_bc
            )
            nisa.tensor_scalar(
                state_bot, state_bot, nl.multiply, exp_g_bc
            )

            # ── Step 2: kv_mem = state^T @ k  →  [1, D] ─────────────────
            # nc_matmul: C[M,N] = A[K,M]^T @ B[K,N]
            # C[1, 256] = k_top[128, 1]^T @ state_top[128, 256]  (K=128)
            kv_top_psum = nl.ndarray((1, _D), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(kv_top_psum, k_top, state_top)

            kv_bot_psum = nl.ndarray((1, _D), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(kv_bot_psum, k_bot, state_bot)

            # Sum top + bot in SBUF
            kv_top_sbuf = nl.ndarray((1, _D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(kv_top_sbuf, kv_top_psum)
            kv_bot_sbuf = nl.ndarray((1, _D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(kv_bot_sbuf, kv_bot_psum)

            kv_mem = nl.ndarray((1, _D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(kv_mem, kv_top_sbuf, kv_bot_sbuf, op=nl.add)

            # ── Step 3: delta = (v - kv_mem) * beta  →  [1, D] ──────────
            delta = nl.ndarray((1, _D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(delta, v_tile, kv_mem, op=nl.subtract)
            # beta_tile is [1, 1], tensor_scalar broadcasts along F=256
            nisa.tensor_scalar(delta, delta, nl.multiply, beta_tile)

            # ── Step 4: state += outer(k, delta) ─────────────────────────
            # outer(k_top[128], delta[256]) → [128, 256]
            # Transpose k_top[128, 1] → [1, 128] for nc_matmul K=1 path
            k_top_T_psum = nl.ndarray(
                (1, _D_HALF), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_transpose(k_top_T_psum, k_top)
            k_top_T = nl.ndarray((1, _D_HALF), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(k_top_T, k_top_T_psum)

            # nc_matmul: outer[128, 256] = k_top_T[1, 128]^T @ delta[1, 256]
            outer_top_psum = nl.ndarray(
                (_D_HALF, _D), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_matmul(outer_top_psum, k_top_T, delta)
            outer_top = nl.ndarray((_D_HALF, _D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(outer_top, outer_top_psum)
            nisa.tensor_tensor(state_top, state_top, outer_top, op=nl.add)

            # Same for bottom half
            k_bot_T_psum = nl.ndarray(
                (1, _D_HALF), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_transpose(k_bot_T_psum, k_bot)
            k_bot_T = nl.ndarray((1, _D_HALF), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(k_bot_T, k_bot_T_psum)

            outer_bot_psum = nl.ndarray(
                (_D_HALF, _D), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_matmul(outer_bot_psum, k_bot_T, delta)
            outer_bot = nl.ndarray((_D_HALF, _D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(outer_bot, outer_bot_psum)
            nisa.tensor_tensor(state_bot, state_bot, outer_bot, op=nl.add)

            # ── Step 5: out_t = state^T @ q  →  [1, D] ──────────────────
            out_top_psum = nl.ndarray((1, _D), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(out_top_psum, q_top, state_top)

            out_bot_psum = nl.ndarray((1, _D), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(out_bot_psum, q_bot, state_bot)

            out_top_sbuf = nl.ndarray((1, _D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(out_top_sbuf, out_top_psum)
            out_bot_sbuf = nl.ndarray((1, _D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(out_bot_sbuf, out_bot_psum)

            out_t = nl.ndarray((1, _D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(out_t, out_top_sbuf, out_bot_sbuf, op=nl.add)

            # Cast back to input dtype and store
            out_cast = nl.ndarray((1, _D), dtype=q_ref.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(out_cast, out_t, nl.multiply, 1.0)
            nisa.dma_copy(dst=out_ref[idx, t:t+1, :], src=out_cast)

        # ── Store final state ────────────────────────────────────────────
        nisa.dma_copy(dst=final_state_ref[idx, 0:_D_HALF, :], src=state_top)
        nisa.dma_copy(dst=final_state_ref[idx, _D_HALF:_D, :], src=state_bot)

    return out_ref, final_state_ref
