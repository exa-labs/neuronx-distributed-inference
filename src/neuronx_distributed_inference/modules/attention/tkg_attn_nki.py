"""From-scratch fused token-generation attention kernel (batched over the grid).

Fuses only the attention core for one decode step -- QK^T, mask, online softmax
over [prior || active], P.V -- with fp32 accumulators and GQA broadcast of the
single KV head across query heads. The QKV projection and q/k/v RMSNorms
(including Gemma 4's data-dependent v_layernorm) stay in model code outside the
kernel, so HF numerical equivalence is preserved by construction. A leading
batch dim is dispatched over the NKI launch grid so it is called once per
attention layer from ``NeuronAttentionBase.compute_for_token_gen``.

Layouts (one TP shard, q_len == 1, single KV head):
  Q    : [B, H, D]
  Kp   : [B, Sp, D]   prior (cached) keys
  Vp   : [B, Sp, D]   prior (cached) values
  Ka   : [B, 1, D]    active-token key
  Va   : [B, 1, D]    active-token value
  mask : [B, H, Sp]   1 keep / 0 drop (numeric, causal + sliding-window)
  out  : [B, H, D]
"""
import math

try:  # standalone nki wheel (Neuron SDK >= 2.24)
    import nki
    import nki.language as nl
except ImportError:  # bundled frontend (Neuron SDK 2.22)
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl

P = 128  # Tensor Engine partition / free-dim tile limit
NEG = 30000.0  # additive mask magnitude (bf16-safe "-inf")


@nki.jit
def tkg_attention_kernel_batched(Q, Kp, Vp, Ka, Va, mask, inv_scale):
    """Fused single-step flash-decode attention, one batch element per grid program."""
    b = nl.program_id(0)
    B, H, D = Q.shape
    Sp = Kp.shape[1]
    dt = Q.dtype
    nD = math.ceil(D / P)
    nS = math.ceil(Sp / P)

    out = nl.ndarray((B, H, D), dtype=dt, buffer=nl.shared_hbm)

    q_tiles = []
    for i in nl.static_range(nD):
        d0, d1 = i * P, min(D, (i + 1) * P)
        q_tiles.append(nl.load(Q[b, :, d0:d1]))                    # [H, d]

    m = nl.full((H, 1), -NEG, dtype=nl.float32, buffer=nl.sbuf)
    l = nl.zeros((H, 1), dtype=nl.float32, buffer=nl.sbuf)
    acc = nl.zeros((H, D), dtype=nl.float32, buffer=nl.sbuf)

    blocks = [(Kp, Vp, mask, j * P, min(Sp, (j + 1) * P)) for j in range(nS)]
    blocks.append((Ka, Va, None, 0, 1))

    for Kb, Vb, Mb, s0, s1 in blocks:
        sB = s1 - s0
        s = nl.zeros((H, sB), dtype=nl.float32, buffer=nl.sbuf)
        for i in nl.static_range(nD):
            d0, d1 = i * P, min(D, (i + 1) * P)
            kT = nl.transpose(nl.load(Kb[b, s0:s1, d0:d1]))        # [d, sB]
            s[...] = nl.add(s, nl.matmul(q_tiles[i], kT))          # [H, sB]
        s[...] = nl.multiply(s, inv_scale)
        if Mb is not None:
            mb = nl.load(Mb[b, :, s0:s1])                          # [H, sB]
            s[...] = nl.add(s, nl.multiply(nl.subtract(mb, 1.0), NEG))
        m_new = nl.maximum(m, nl.max(s, axis=1))                    # [H,1]
        alpha = nl.exp(nl.subtract(m, m_new))                       # [H,1]
        p = nl.exp(nl.subtract(s, m_new))                           # [H,sB]
        l[...] = nl.add(nl.multiply(l, alpha), nl.sum(p, axis=1))
        for i in nl.static_range(nD):
            d0, d1 = i * P, min(D, (i + 1) * P)
            pv = nl.matmul(p, nl.load(Vb[b, s0:s1, d0:d1]))        # [H, d]
            acc[:, d0:d1] = nl.add(nl.multiply(acc[:, d0:d1], alpha), pv)
        m[...] = m_new

    inv_l = nl.reciprocal(l)
    for i in nl.static_range(nD):
        d0, d1 = i * P, min(D, (i + 1) * P)
        o = nl.multiply(acc[:, d0:d1], inv_l)                       # [H, d]
        nl.store(out[b, :, d0:d1], o.astype(dt))
    return out
