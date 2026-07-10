"""From-scratch fused token-generation attention kernel (batched in-kernel).

Fuses only the attention core for one decode step -- QK^T, mask, online softmax
over [prior || active], P.V -- with fp32 accumulators and GQA broadcast of the
single KV head across query heads. The QKV projection and q/k/v RMSNorms
(including Gemma 4's data-dependent v_layernorm) stay in model code outside the
kernel, so HF numerical equivalence is preserved by construction. The leading
batch dim is iterated inside the kernel with ``nl.affine_range`` (the standalone
nki wheel, SDK >= 2.24, has no SPMD launch grid; ``kernel[n]`` selects LNC), so
it is called once per attention layer from
``NeuronAttentionBase.compute_for_token_gen``.

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


def _transpose_sbuf(x):
    """Transpose ``x`` and land the result in SBUF.

    ``nl.transpose`` emits into PSUM and ``nl.copy`` preserves the source buffer,
    so a matmul stationary operand (which must live in SBUF) needs an explicit
    SBUF destination; assigning through ``[...]`` honours the destination buffer.
    """
    xt = nl.transpose(x)                                   # [x1, x0] in PSUM
    # fp32 so it can serve as a Tensor-Engine matmul stationary (dst dtype must
    # be fp32/bf16) and so QK^T / P.V accumulate in fp32 (HF equivalence).
    dst = nl.ndarray((xt.shape[0], xt.shape[1]), dtype=nl.float32, buffer=nl.sbuf)
    dst[...] = xt
    return dst


def _load_f32(ref):
    """Load an HBM tile into fp32 SBUF (matmul requires both operands fp32)."""
    x = nl.load(ref)
    dst = nl.ndarray((x.shape[0], x.shape[1]), dtype=nl.float32, buffer=nl.sbuf)
    dst[...] = x
    return dst


def _attend_block(qT_tiles, Kb, Vb, Mb, b, s0, s1, nD, D, inv_scale, m, l, acc, has_mask):
    """Fold one key/value block into the running online-softmax state in place.

    ``Kb``/``Vb`` are ``[B, S, D]`` HBM refs and ``[s0:s1]`` selects this block's
    rows; ``Mb`` is the ``[B, H, S]`` numeric mask (only when ``has_mask``). The
    state tensors ``m`` (running max), ``l`` (running denom) and ``acc`` (running
    weighted value sum) are SBUF buffers mutated through ``[...]`` so the caller
    sees the update. Kept as an explicit helper (not a packed-tuple loop) because
    the NKI backend cannot trace a list comprehension of tensor-bearing tuples.
    """
    sB = s1 - s0
    s = nl.zeros((qT_tiles[0].shape[1], sB), dtype=nl.float32, buffer=nl.sbuf)  # [H, sB]
    for i in nl.static_range(nD):
        d0, d1 = i * P, min(D, (i + 1) * P)
        # Kb[b] is [sB, d]; transpose to [d, sB] so the contract dim (d) is on
        # partitions for both operands.
        kT = _transpose_sbuf(nl.load(Kb[b, s0:s1, d0:d1]))        # [d, sB]
        s[...] = nl.add(s, nl.matmul(qT_tiles[i], kT, transpose_x=True))  # [H, sB]
    s[...] = nl.multiply(s, inv_scale)
    if has_mask:
        mb = nl.load(Mb[b, :, s0:s1])                            # [H, sB]
        s[...] = nl.add(s, nl.multiply(nl.subtract(mb, 1.0), NEG))
    m_new = nl.maximum(m, nl.max(s, axis=1, keepdims=True))       # [H,1]
    alpha = nl.exp(nl.subtract(m, m_new))                        # [H,1]
    p = nl.exp(nl.subtract(s, m_new))                            # [H,sB]
    l[...] = nl.add(nl.multiply(l, alpha), nl.sum(p, axis=1, keepdims=True))
    # P.V: contract over sB. Stationary = p.T [sB, H] in SBUF, moving = Vb
    # [sB, d] loaded directly (partition = sB).
    pT = _transpose_sbuf(p)                                      # [sB, H]
    for i in nl.static_range(nD):
        d0, d1 = i * P, min(D, (i + 1) * P)
        pv = nl.matmul(pT, _load_f32(Vb[b, s0:s1, d0:d1]), transpose_x=True)  # [H, d]
        acc[:, d0:d1] = nl.add(nl.multiply(acc[:, d0:d1], alpha), pv)
    m[...] = m_new


@nki.jit
def tkg_attention_kernel_batched(Q, Kp, Vp, Ka, Va, mask, inv_scale):
    """Fused single-step flash-decode attention over all batch elements."""
    B, H, D = Q.shape
    Sp = Kp.shape[1]
    dt = Q.dtype
    nD = math.ceil(D / P)
    nS = math.ceil(Sp / P)

    out = nl.ndarray((B, H, D), dtype=dt, buffer=nl.shared_hbm)

    for b in nl.affine_range(B):
        # matmul(x, y, transpose_x=True) computes x.T @ y with the stationary
        # operand x kept in SBUF (contract dim on partitions). We therefore hold
        # Q transposed as [d, H] tiles so QK^T needs no per-block transpose of Q.
        qT_tiles = []
        for i in nl.static_range(nD):
            d0, d1 = i * P, min(D, (i + 1) * P)
            qT_tiles.append(_transpose_sbuf(nl.load(Q[b, :, d0:d1])))  # [d, H]

        m = nl.full((H, 1), -NEG, dtype=nl.float32, buffer=nl.sbuf)
        l = nl.zeros((H, 1), dtype=nl.float32, buffer=nl.sbuf)
        acc = nl.zeros((H, D), dtype=nl.float32, buffer=nl.sbuf)

        # Prior (cached) key/value blocks, tiled to the Tensor-Engine free-dim
        # limit, then the single active-token block. Explicit loop + call rather
        # than a packed-tuple list so the NKI backend can trace every operand.
        for j in nl.static_range(nS):
            s0, s1 = j * P, min(Sp, (j + 1) * P)
            _attend_block(qT_tiles, Kp, Vp, mask, b, s0, s1, nD, D, inv_scale,
                          m, l, acc, has_mask=True)
        _attend_block(qT_tiles, Ka, Va, mask, b, 0, 1, nD, D, inv_scale,
                      m, l, acc, has_mask=False)

        inv_l = nl.reciprocal(l)
        for i in nl.static_range(nD):
            d0, d1 = i * P, min(D, (i + 1) * P)
            o = nl.multiply(acc[:, d0:d1], inv_l, dtype=dt)        # [H, d]
            nl.store(out[b, :, d0:d1], o)
    return out
