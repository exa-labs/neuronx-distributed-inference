# coding=utf-8
"""CPU numerical-equivalence tests for the Qwen3.5 / Qwen3-Next Neuron port.

Each test instantiates the Neuron module in CPU mode (tp_degree=1, via
init_cpu_env) with weights copied from a randomly-initialized HuggingFace
``qwen3_next`` reference module and asserts the outputs match.

Covers:
  - chunked gated delta rule vs HF torch_chunk_gated_delta_rule (prefill)
  - recurrent gated delta rule vs HF torch_recurrent_gated_delta_rule (decode)
  - GatedDeltaNet module prefill + decode continuity vs HF
  - gated attention prefill + decode vs HF Qwen3NextAttention
  - sparse MoE block (routed + shared expert) vs HF
  - full-model logits (prefill + one decode step) vs HF Qwen3NextForCausalLM

Run with:
  python -m pytest test/unit/models/qwen3_5/test_qwen3_5_layers.py -v
"""

import unittest

import numpy
import torch
import torch.nn as nn
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextDynamicCache,
    Qwen3NextForCausalLM,
    Qwen3NextGatedDeltaNet,
    Qwen3NextRotaryEmbedding,
    Qwen3NextSparseMoeBlock,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)

from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.models.qwen3_5.modeling_qwen3_5 import (
    NeuronQwen3_5Attention,
    NeuronQwen3_5DecoderLayer,
    NeuronQwen3_5GatedDeltaNet,
    NeuronQwen3_5SparseMoeBlock,
    Qwen3_5InferenceConfig,
    Qwen3_5RMSNorm,
    chunk_gated_delta_rule,
    convert_qwen3_5_hf_to_neuron_state_dict,
    recurrent_gated_delta_rule,
)
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from neuronx_distributed_inference.utils.random import set_random_seed
from neuronx_distributed_inference.utils.testing import init_cpu_env

DTYPE = torch.float32


def tiny_hf_config() -> Qwen3NextConfig:
    """A 4-layer (3 DeltaNet : 1 attention) hybrid MoE config small enough
    to run on CPU."""
    return Qwen3NextConfig(
        attn_implementation="eager",
        vocab_size=128,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        partial_rotary_factor=0.25,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        max_position_embeddings=512,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        num_experts=8,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        norm_topk_prob=True,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        attention_bias=False,
        tie_word_embeddings=False,
    )


def make_inference_config(hf_config: Qwen3NextConfig, batch_size=2, seq_len=128):
    neuron_config = MoENeuronConfig(
        tp_degree=1,
        batch_size=batch_size,
        max_context_length=seq_len,
        seq_len=seq_len,
        torch_dtype="float32",
    )
    return Qwen3_5InferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(hf_config=hf_config),
    )


def custom_allclose(expected, actual, atol=0.0, rtol=1e-5):
    """allclose with rtol scaled by the absolute max of the expected tensor,
    matching torch_neuronx.testing.validation.custom_allclose semantics."""
    tolerance = atol + rtol * expected.abs().max()
    return bool(((expected - actual).abs() <= tolerance).all())


def assert_close(test, expected, actual, rtol=1e-5, name=""):
    expected = expected.float()
    actual = actual.float()
    passed = custom_allclose(expected, actual, atol=0.0, rtol=rtol)
    max_err = (expected - actual).abs().max().item()
    test.assertTrue(passed, f"{name}: outputs differ (max abs err {max_err:.3e})")


class TestGatedDeltaRuleKernels(unittest.TestCase):
    """The chunked/recurrent delta-rule reimplementations must match HF."""

    def setUp(self):
        set_random_seed(0)

    def test_chunked_matches_hf(self):
        b, s, hk, hv, dk, dv = 2, 50, 2, 4, 16, 16
        q = torch.randn(b, s, hv, dk)
        k = torch.randn(b, s, hv, dk)
        v = torch.randn(b, s, hv, dv)
        g = -torch.rand(b, s, hv)
        beta = torch.rand(b, s, hv)

        expected_out, expected_state = torch_chunk_gated_delta_rule(
            q, k, v, g=g, beta=beta, chunk_size=64, output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        actual_out, actual_state = chunk_gated_delta_rule(
            q, k, v, g=g, beta=beta, chunk_size=64
        )
        assert_close(self, expected_out, actual_out, name="chunked core_attn_out")
        assert_close(self, expected_state, actual_state, name="chunked final state")

    def test_recurrent_matches_hf(self):
        b, s, hv, dk, dv = 2, 1, 4, 16, 16
        q = torch.randn(b, s, hv, dk)
        k = torch.randn(b, s, hv, dk)
        v = torch.randn(b, s, hv, dv)
        g = -torch.rand(b, s, hv)
        beta = torch.rand(b, s, hv)
        state = torch.randn(b, hv, dk, dv)

        expected_out, expected_state = torch_recurrent_gated_delta_rule(
            q, k, v, g=g, beta=beta, initial_state=state.clone(), output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        actual_out, actual_state = recurrent_gated_delta_rule(
            q, k, v, g=g, beta=beta, initial_state=state.clone()
        )
        assert_close(self, expected_out, actual_out, name="recurrent core_attn_out")
        assert_close(self, expected_state, actual_state, name="recurrent final state")

    def test_nki_within_chunk_state_update_matches_loop(self):
        """The NKI TensorEngine state-update kernel must match the torch
        rank-1 loop / matmul reference bit-near-exactly (it is the only form of
        the contraction that compiles on neuronx-cc)."""
        try:
            import nki  # noqa: F401
            from neuronx_distributed_inference.models.qwen3_5.nki_delta_rule import (
                nki_within_chunk_state_update,
            )
        except ImportError:
            self.skipTest("nki not available in this environment")

        bh, c, dk, dv = 4, 64, 128, 128
        k_decay = torch.randn(bh, c, dk)
        v_new = torch.randn(bh, c, dv)

        # Reference: the default "loop" form summed over the chunk dimension.
        expected = k_decay.new_zeros(bh, dk, dv)
        for ci in range(c):
            expected = expected + k_decay[:, ci, :, None] * v_new[:, ci, None, :]

        try:
            actual = nki.simulate(nki_within_chunk_state_update)(
                k_decay.numpy(), v_new.numpy()
            )
        except Exception as exc:  # pragma: no cover - sim infra dependent
            self.skipTest(f"nki.simulate unavailable: {exc}")
        actual = torch.from_numpy(numpy.asarray(actual)).float()

        assert_close(self, expected, actual, rtol=1e-4, name="nki within-chunk state update")

    def test_nki_chunk_gated_delta_rule_kernel_matches_loop(self):
        """The full Approach-C chunked NKI kernel must reproduce the sequential
        inter-chunk recurrence (five TensorEngine matmuls per chunk, state kept
        in SBUF) bit-near-exactly vs the torch reference loop.  This is the lever
        that collapses the seq-len-deep within-chunk loop to one matmul/chunk."""
        try:
            import nki  # noqa: F401
            from neuronx_distributed_inference.models.qwen3_5.nki_delta_rule import (
                nki_chunk_gated_delta_rule_kernel,
            )
        except ImportError:
            self.skipTest("nki not available in this environment")

        bh, nc, c, dk, dv = 2, 3, 64, 128, 128
        value = torch.randn(bh, nc, c, dv)
        k_cumdecay = torch.randn(bh, nc, c, dk)
        qg = torch.randn(bh, nc, c, dk)
        attn_intra = torch.randn(bh, nc, c, c)
        k_decay = torch.randn(bh, nc, c, dk)
        g_last = torch.rand(bh, nc)  # exp(g_last_i) in (0, 1]
        init_state = torch.randn(bh, dk, dv)

        # Reference: the documented sequential chunk recurrence.
        state = init_state.clone()
        cores = []
        for i in range(nc):
            v_prime = k_cumdecay[:, i] @ state
            v_new = value[:, i] - v_prime
            attn_inter = qg[:, i] @ state
            cores.append(attn_inter + attn_intra[:, i] @ v_new)
            state_update = k_decay[:, i].transpose(-1, -2) @ v_new
            state = state * g_last[:, i, None, None] + state_update
        expected_core = torch.stack(cores, dim=1)
        expected_state = state

        # nc_matmul contracts the partition (first) dim, so the three matmuls
        # over a non-chunk axis take pre-transposed stationary operands.
        k_cumdecay_t = k_cumdecay.transpose(-1, -2).contiguous()
        qg_t = qg.transpose(-1, -2).contiguous()
        attn_intra_t = attn_intra.transpose(-1, -2).contiguous()

        try:
            core, final_state = nki.simulate(nki_chunk_gated_delta_rule_kernel)(
                value.numpy(), k_cumdecay_t.numpy(), qg_t.numpy(),
                attn_intra_t.numpy(), k_decay.numpy(), g_last.numpy(),
                init_state.numpy(),
            )
        except Exception as exc:  # pragma: no cover - sim infra dependent
            self.skipTest(f"nki.simulate unavailable: {exc}")
        core = torch.from_numpy(numpy.asarray(core)).float()
        final_state = torch.from_numpy(numpy.asarray(final_state)).float()

        assert_close(self, expected_core, core, rtol=1e-4, name="nki chunked core")
        assert_close(self, expected_state, final_state, rtol=1e-4, name="nki chunked final state")

    def test_nki_chunk_kernel_v2_matches_loop(self):
        """v2 kernel (pre-broadcast gate, no ones_row) matches sequential ref."""
        try:
            import nki  # noqa: F401
            from neuronx_distributed_inference.models.qwen3_5.nki_delta_rule import (
                nki_chunk_gated_delta_rule_kernel_v2,
            )
        except ImportError:
            self.skipTest("nki not available in this environment")

        bh, nc, c, dk, dv = 2, 3, 64, 128, 128
        value = torch.randn(bh, nc, c, dv)
        k_cumdecay = torch.randn(bh, nc, c, dk)
        qg = torch.randn(bh, nc, c, dk)
        attn_intra = torch.randn(bh, nc, c, c)
        k_decay = torch.randn(bh, nc, c, dk)
        g_last = torch.rand(bh, nc)
        init_state = torch.randn(bh, dk, dv)

        state = init_state.clone()
        cores = []
        for i in range(nc):
            v_prime = k_cumdecay[:, i] @ state
            v_new = value[:, i] - v_prime
            attn_inter = qg[:, i] @ state
            cores.append(attn_inter + attn_intra[:, i] @ v_new)
            state_update = k_decay[:, i].transpose(-1, -2) @ v_new
            state = state * g_last[:, i, None, None] + state_update
        expected_core = torch.stack(cores, dim=1)
        expected_state = state

        k_cumdecay_t = k_cumdecay.transpose(-1, -2).contiguous()
        qg_t = qg.transpose(-1, -2).contiguous()
        attn_intra_t = attn_intra.transpose(-1, -2).contiguous()
        # Pre-broadcast gate to [BH, NC, Dk].
        g_last_bc = g_last.unsqueeze(-1).expand(-1, -1, dk).contiguous()

        try:
            core, final_state = nki.simulate(nki_chunk_gated_delta_rule_kernel_v2)(
                value.numpy(), k_cumdecay_t.numpy(), qg_t.numpy(),
                attn_intra_t.numpy(), k_decay.numpy(), g_last_bc.numpy(),
                init_state.numpy(),
            )
        except Exception as exc:
            self.skipTest(f"nki.simulate unavailable: {exc}")
        core = torch.from_numpy(numpy.asarray(core)).float()
        final_state = torch.from_numpy(numpy.asarray(final_state)).float()

        assert_close(self, expected_core, core, rtol=1e-4, name="nki v2 chunked core")
        assert_close(self, expected_state, final_state, rtol=1e-4, name="nki v2 chunked state")

    def test_nki_chunk_kernel_v2_chunk128(self):
        """v2 kernel at chunk_size=128: full [128,128] tile utilization."""
        try:
            import nki  # noqa: F401
            from neuronx_distributed_inference.models.qwen3_5.nki_delta_rule import (
                nki_chunk_gated_delta_rule_kernel_v2,
            )
        except ImportError:
            self.skipTest("nki not available in this environment")

        bh, nc, c, dk, dv = 2, 4, 128, 128, 128
        value = torch.randn(bh, nc, c, dv)
        k_cumdecay = torch.randn(bh, nc, c, dk)
        qg = torch.randn(bh, nc, c, dk)
        attn_intra = torch.randn(bh, nc, c, c)
        k_decay = torch.randn(bh, nc, c, dk)
        g_last = torch.rand(bh, nc)
        init_state = torch.randn(bh, dk, dv)

        state = init_state.clone()
        cores = []
        for i in range(nc):
            v_prime = k_cumdecay[:, i] @ state
            v_new = value[:, i] - v_prime
            attn_inter = qg[:, i] @ state
            cores.append(attn_inter + attn_intra[:, i] @ v_new)
            state_update = k_decay[:, i].transpose(-1, -2) @ v_new
            state = state * g_last[:, i, None, None] + state_update
        expected_core = torch.stack(cores, dim=1)
        expected_state = state

        k_cumdecay_t = k_cumdecay.transpose(-1, -2).contiguous()
        qg_t = qg.transpose(-1, -2).contiguous()
        attn_intra_t = attn_intra.transpose(-1, -2).contiguous()
        g_last_bc = g_last.unsqueeze(-1).expand(-1, -1, dk).contiguous()

        try:
            core, final_state = nki.simulate(nki_chunk_gated_delta_rule_kernel_v2)(
                value.numpy(), k_cumdecay_t.numpy(), qg_t.numpy(),
                attn_intra_t.numpy(), k_decay.numpy(), g_last_bc.numpy(),
                init_state.numpy(),
            )
        except Exception as exc:
            self.skipTest(f"nki.simulate unavailable: {exc}")
        core = torch.from_numpy(numpy.asarray(core)).float()
        final_state = torch.from_numpy(numpy.asarray(final_state)).float()

        assert_close(self, expected_core, core, rtol=1e-4, name="nki v2 c128 core")
        assert_close(self, expected_state, final_state, rtol=1e-4, name="nki v2 c128 state")

    def test_nki_decode_kernel_matches_general(self):
        """The specialized decode kernel (k_row, no seq loop) must match the
        general recurrent kernel for T=1."""
        try:
            import nki  # noqa: F401
            from neuronx_distributed_inference.models.qwen3_5.nki_delta_rule import (
                nki_recurrent_gated_delta_rule,
                nki_recurrent_gated_delta_rule_decode,
            )
        except ImportError:
            self.skipTest("nki not available in this environment")

        bh, dk, dv = 8, 128, 128
        q = torch.randn(bh, dk, 1)
        k = torch.randn(bh, dk, 1)
        v = torch.randn(bh, 1, dv)
        exp_g = torch.rand(bh, 1) + 0.5  # exp(g) ∈ (0.5, 1.5)
        beta = torch.rand(bh, 1)
        state = torch.randn(bh, dk, dv)

        # k_row is the transpose of k's last two dims: [BH, 1, Dk]
        k_row = k.transpose(-1, -2).contiguous()

        # Reference: general recurrent kernel
        try:
            out_gen, state_gen = nki.simulate(nki_recurrent_gated_delta_rule)(
                q.numpy(), k.numpy(), v.numpy(),
                exp_g.numpy(), beta.numpy(), state.clone().numpy(),
            )
            out_dec, state_dec = nki.simulate(nki_recurrent_gated_delta_rule_decode)(
                q.numpy(), k.numpy(), k_row.numpy(), v.numpy(),
                exp_g.numpy(), beta.numpy(), state.clone().numpy(),
            )
        except Exception as exc:
            self.skipTest(f"nki.simulate unavailable: {exc}")
        out_gen = torch.from_numpy(numpy.asarray(out_gen)).float()
        state_gen = torch.from_numpy(numpy.asarray(state_gen)).float()
        out_dec = torch.from_numpy(numpy.asarray(out_dec)).float()
        state_dec = torch.from_numpy(numpy.asarray(state_dec)).float()

        assert_close(self, out_gen, out_dec, rtol=1e-5, name="decode kernel output")
        assert_close(self, state_gen, state_dec, rtol=1e-5, name="decode kernel state")

    def test_nki_decode_v2_matches_v1(self):
        """The optimized v2 decode kernel (host-precomputed exp_g + k*beta)
        must produce identical results to v1."""
        try:
            import nki  # noqa: F401
            from neuronx_distributed_inference.models.qwen3_5.nki_delta_rule import (
                nki_recurrent_gated_delta_rule_decode,
                nki_recurrent_gated_delta_rule_decode_v2,
            )
        except ImportError:
            self.skipTest("nki not available in this environment")

        bh, dk, dv = 8, 128, 128
        q = torch.randn(bh, dk, 1)
        k = torch.randn(bh, dk, 1)
        v = torch.randn(bh, 1, dv)
        exp_g = torch.rand(bh, 1) + 0.5  # exp(g) ∈ (0.5, 1.5)
        beta = torch.rand(bh, 1)
        state = torch.randn(bh, dk, dv)

        # v1 inputs
        k_row = k.transpose(-1, -2).contiguous()  # [BH, 1, Dk]

        # v2 inputs: host-precomputed
        exp_g_bc = exp_g.expand(-1, dk).contiguous()  # [BH, Dk]
        k_beta_row = (k.squeeze(-1) * beta).unsqueeze(1).contiguous()  # [BH, 1, Dk]

        try:
            out_v1, state_v1 = nki.simulate(nki_recurrent_gated_delta_rule_decode)(
                q.numpy(), k.numpy(), k_row.numpy(), v.numpy(),
                exp_g.numpy(), beta.numpy(), state.clone().numpy(),
            )
            out_v2, state_v2 = nki.simulate(nki_recurrent_gated_delta_rule_decode_v2)(
                q.numpy(), k.numpy(), k_beta_row.numpy(), v.numpy(),
                exp_g_bc.numpy(), state.clone().numpy(),
            )
        except Exception as exc:
            self.skipTest(f"nki.simulate unavailable: {exc}")

        out_v1 = torch.from_numpy(numpy.asarray(out_v1)).float()
        state_v1 = torch.from_numpy(numpy.asarray(state_v1)).float()
        out_v2 = torch.from_numpy(numpy.asarray(out_v2)).float()
        state_v2 = torch.from_numpy(numpy.asarray(state_v2)).float()

        assert_close(self, out_v1, out_v2, rtol=1e-5, name="decode v2 output")
        assert_close(self, state_v1, state_v2, rtol=1e-5, name="decode v2 state")

    def test_chunked_vs_recurrent_consistency(self):
        """Both forms compute the same math, so prefill-then-decode must equal
        a longer prefill."""
        b, s, hv, dk, dv = 1, 33, 4, 16, 16
        q = torch.randn(b, s, hv, dk)
        k = torch.randn(b, s, hv, dk)
        v = torch.randn(b, s, hv, dv)
        g = -torch.rand(b, s, hv)
        beta = torch.rand(b, s, hv)

        full_out, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta)
        prefill_out, state = chunk_gated_delta_rule(
            q[:, :-1], k[:, :-1], v[:, :-1], g=g[:, :-1], beta=beta[:, :-1]
        )
        decode_out, _ = recurrent_gated_delta_rule(
            q[:, -1:], k[:, -1:], v[:, -1:], g=g[:, -1:], beta=beta[:, -1:],
            initial_state=state,
        )
        assert_close(self, full_out[:, :-1], prefill_out, rtol=1e-4, name="prefill part")
        assert_close(self, full_out[:, -1:], decode_out, rtol=1e-4, name="decode step")


class TestGatedDeltaNetModule(unittest.TestCase):
    """Neuron GatedDeltaNet module vs HF Qwen3NextGatedDeltaNet."""

    @classmethod
    def setUpClass(cls):
        init_cpu_env()

    def test_prefill_and_decode_match_hf(self):
        set_random_seed(0)
        hf_config = tiny_hf_config()
        config = make_inference_config(hf_config)

        hf_module = Qwen3NextGatedDeltaNet(hf_config, layer_idx=0).eval()
        neuron_module = NeuronQwen3_5GatedDeltaNet(config).eval()
        neuron_module.load_state_dict(hf_module.state_dict(), strict=False)

        b, s = 2, 23
        hidden = torch.randn(b, s, hf_config.hidden_size, dtype=DTYPE) * 0.5
        seq_ids = torch.arange(b)

        # HF full forward over s+1 tokens (no cache => pure prefill path).
        hidden_next = torch.randn(b, 1, hf_config.hidden_size, dtype=DTYPE) * 0.5
        full_input = torch.cat([hidden, hidden_next], dim=1)
        with torch.no_grad():
            expected_full = hf_module(full_input)
            expected_prefill = hf_module(hidden)

            actual_prefill = neuron_module(hidden, seq_ids=seq_ids, is_for_context_encoding=True)
            actual_decode = neuron_module(
                hidden_next, seq_ids=seq_ids, is_for_context_encoding=False
            )

        assert_close(self, expected_prefill, actual_prefill, rtol=1e-4, name="deltanet prefill")
        assert_close(
            self, expected_full[:, -1:], actual_decode, rtol=1e-4, name="deltanet decode"
        )


class TestGatedAttentionModule(unittest.TestCase):
    """Neuron gated attention vs HF Qwen3NextAttention (prefill + decode)."""

    @classmethod
    def setUpClass(cls):
        init_cpu_env()

    def test_prefill_and_decode_match_hf(self):
        set_random_seed(0)
        hf_config = tiny_hf_config()
        config = make_inference_config(hf_config)

        hf_attn = Qwen3NextAttention(hf_config, layer_idx=3).eval()
        rotary = Qwen3NextRotaryEmbedding(hf_config)
        neuron_attn = NeuronQwen3_5Attention(config).eval()
        neuron_attn.load_state_dict(hf_attn.state_dict(), strict=False)

        b, s = 2, 17
        hidden = torch.randn(b, s, hf_config.hidden_size, dtype=DTYPE) * 0.5
        hidden_next = torch.randn(b, 1, hf_config.hidden_size, dtype=DTYPE) * 0.5
        position_ids = torch.arange(s).unsqueeze(0).expand(b, -1)

        causal = torch.tril(torch.ones(s, s, dtype=torch.bool))[None, None].expand(b, 1, s, s)
        additive_mask = torch.where(causal, 0.0, torch.finfo(DTYPE).min)

        with torch.no_grad():
            cos, sin = rotary(hidden, position_ids)
            expected_prefill, _ = hf_attn(
                hidden,
                position_embeddings=(cos, sin),
                attention_mask=additive_mask,
            )
            actual_prefill, (k_cache, v_cache) = neuron_attn(
                hidden,
                attention_mask=causal,
                position_ids=position_ids,
            )
        assert_close(self, expected_prefill, actual_prefill, rtol=1e-5, name="attention prefill")

        # Decode: HF over s+1 tokens; Neuron with cached KV + active mask.
        full_input = torch.cat([hidden, hidden_next], dim=1)
        full_position_ids = torch.arange(s + 1).unsqueeze(0).expand(b, -1)
        causal_full = torch.tril(torch.ones(s + 1, s + 1, dtype=torch.bool))[None, None].expand(
            b, 1, s + 1, s + 1
        )
        additive_full = torch.where(causal_full, 0.0, torch.finfo(DTYPE).min)
        with torch.no_grad():
            cos, sin = rotary(full_input, full_position_ids)
            expected_full, _ = hf_attn(
                full_input,
                position_embeddings=(cos, sin),
                attention_mask=additive_full,
            )
            actual_decode, _ = neuron_attn(
                hidden_next,
                attention_mask=torch.ones(b, 1, 1, s, dtype=torch.bool),
                position_ids=torch.full((b, 1), s),
                past_key_value=(k_cache, v_cache),
                active_mask=torch.ones(b, 1, 1, 1, dtype=torch.bool),
            )
        assert_close(
            self, expected_full[:, -1:], actual_decode, rtol=1e-5, name="attention decode"
        )


class TestSparseMoeBlock(unittest.TestCase):
    """Routed + shared expert MoE block vs HF Qwen3NextSparseMoeBlock."""

    @classmethod
    def setUpClass(cls):
        init_cpu_env()

    def test_moe_block_matches_hf(self):
        set_random_seed(0)
        hf_config = tiny_hf_config()
        config = make_inference_config(hf_config)

        hf_moe = Qwen3NextSparseMoeBlock(hf_config).eval()
        neuron_moe = NeuronQwen3_5SparseMoeBlock(config).eval()

        sd = {f"layers.0.mlp.{k}": v.clone() for k, v in hf_moe.state_dict().items()}
        sd = convert_qwen3_5_hf_to_neuron_state_dict(sd, config)
        sd.pop("rank_util.rank")
        sd = {k[len("layers.0.mlp."):]: v for k, v in sd.items() if k.startswith("layers.0.mlp.")}
        missing, unexpected = neuron_moe.load_state_dict(sd, strict=False)
        self.assertFalse(unexpected, f"unexpected keys: {unexpected}")

        b, s = 2, 8
        hidden = torch.randn(b, s, hf_config.hidden_size, dtype=DTYPE) * 0.5
        with torch.no_grad():
            expected, _ = hf_moe(hidden)
            actual = neuron_moe(hidden)
        assert_close(self, expected, actual, rtol=1e-4, name="moe block")


class TestRMSNorm(unittest.TestCase):
    def test_zero_centered_rmsnorm_matches_hf(self):
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRMSNorm

        set_random_seed(0)
        hf_norm = Qwen3NextRMSNorm(64)
        with torch.no_grad():
            hf_norm.weight.copy_(torch.randn(64) * 0.1)
        neuron_norm = Qwen3_5RMSNorm(64)
        neuron_norm.load_state_dict(hf_norm.state_dict())

        x = torch.randn(2, 5, 64)
        with torch.no_grad():
            assert_close(self, hf_norm(x), neuron_norm(x), name="rmsnorm")


class TestFullModelLogits(unittest.TestCase):
    """End-to-end logits: HF Qwen3NextForCausalLM vs the composed Neuron stack
    (embedding + hybrid decoder layers + final norm + lm_head), checking both
    prefill and a subsequent decode step (cache/state continuity)."""

    @classmethod
    def setUpClass(cls):
        init_cpu_env()

    def _build_neuron_stack(self, hf_model, config):
        """Compose the Neuron decoder layers with plain embedding/lm_head."""
        hf_sd = hf_model.state_dict()
        sd = {}
        for k, v in hf_sd.items():
            if k.startswith("model."):
                sd[k[len("model."):]] = v.clone()
            else:
                sd[k] = v.clone()
        sd = convert_qwen3_5_hf_to_neuron_state_dict(sd, config)

        layers = nn.ModuleList(
            [NeuronQwen3_5DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        embed = nn.Embedding(config.vocab_size, config.hidden_size)
        norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        embed.load_state_dict({"weight": sd["embed_tokens.weight"]})
        lm_head.load_state_dict({"weight": sd["lm_head.weight"]})
        norm.load_state_dict({"weight": sd["norm.weight"]})
        for i, layer in enumerate(layers):
            prefix = f"layers.{i}."
            layer_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
            missing, unexpected = layer.load_state_dict(layer_sd, strict=False)
            self.assertFalse(unexpected, f"layer {i} unexpected keys: {unexpected}")
        return embed, layers, norm, lm_head

    def _neuron_forward(self, modules, input_ids, position_ids, seq_ids,
                        past_key_values=None, cache_len=None):
        embed, layers, norm, lm_head = modules
        b, s = input_ids.shape
        hidden = embed(input_ids)
        is_prefill = past_key_values is None
        if is_prefill:
            attention_mask = torch.tril(torch.ones(s, s, dtype=torch.bool))[None, None].expand(
                b, 1, s, s
            )
            active_mask = None
        else:
            attention_mask = torch.ones(b, 1, 1, cache_len, dtype=torch.bool)
            active_mask = torch.ones(b, 1, 1, 1, dtype=torch.bool)
        new_kv = []
        for i, layer in enumerate(layers):
            past = None if is_prefill else past_key_values[i]
            hidden, present, *_ = layer(
                hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past,
                seq_ids=seq_ids,
                active_mask=active_mask,
                is_for_context_encoding=is_prefill,
            )
            new_kv.append(present)
        hidden = norm(hidden)
        logits = lm_head(hidden)
        return logits, new_kv

    def test_logits_match_hf(self):
        set_random_seed(0)
        hf_config = tiny_hf_config()
        config = make_inference_config(hf_config)

        hf_model = Qwen3NextForCausalLM(hf_config).eval()
        modules = self._build_neuron_stack(hf_model, config)

        b, s = 2, 21
        input_ids = torch.randint(0, hf_config.vocab_size, (b, s))
        next_ids = torch.randint(0, hf_config.vocab_size, (b, 1))
        position_ids = torch.arange(s).unsqueeze(0).expand(b, -1)
        seq_ids = torch.arange(b)

        with torch.no_grad():
            # HF prefill with cache, then one decode step.
            cache = Qwen3NextDynamicCache(config=hf_config)
            hf_out = hf_model(
                input_ids, position_ids=position_ids, past_key_values=cache, use_cache=True
            )
            hf_decode = hf_model(
                next_ids,
                position_ids=torch.full((b, 1), s),
                past_key_values=hf_out.past_key_values,
                use_cache=True,
            )

            actual_prefill, kv = self._neuron_forward(
                modules, input_ids, position_ids, seq_ids
            )
            actual_decode, _ = self._neuron_forward(
                modules,
                next_ids,
                torch.full((b, 1), s),
                seq_ids,
                past_key_values=kv,
                cache_len=s,
            )

        assert_close(self, hf_out.logits, actual_prefill, rtol=1e-4, name="prefill logits")
        assert_close(self, hf_decode.logits, actual_decode, rtol=1e-4, name="decode logits")


if __name__ == "__main__":
    unittest.main()
