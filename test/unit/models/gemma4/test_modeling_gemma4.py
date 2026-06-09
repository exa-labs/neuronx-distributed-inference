from types import SimpleNamespace

import torch

from neuronx_distributed_inference.models.gemma4.modeling_gemma4 import (
    Gemma4InferenceConfig,
    NeuronGemma4ForCausalLM,
    get_updated_configs,
)


def _load_tiny_gemma4_config(config):
    config.attention_bias = False
    config.attention_k_eq_v = False
    config.final_logit_softcapping = 30.0
    config.global_head_dim = 64
    config.head_dim = 32
    config.hidden_activation = "gelu_pytorch_tanh"
    config.hidden_size = 128
    config.hidden_size_per_layer_input = 16
    config.intermediate_size = 512
    config.layer_types = ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]
    config.max_position_embeddings = 4096
    config.num_attention_heads = 4
    config.num_global_key_value_heads = None
    config.num_hidden_layers = 4
    config.num_key_value_heads = 1
    config.num_kv_shared_layers = 2
    config.rms_norm_eps = 1e-6
    config.rope_parameters = {
        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
        "full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.25, "rope_type": "proportional"},
    }
    config.sliding_window = 512
    config.tie_word_embeddings = True
    config.use_double_wide_mlp = True
    config.vocab_size = 1024
    config.vocab_size_per_layer_input = 1024


def test_gemma4_config_preserves_vllm_layer_types_and_head_dims():
    neuron_config = SimpleNamespace()

    config = Gemma4InferenceConfig(neuron_config=neuron_config, load_config=_load_tiny_gemma4_config)
    updated_configs = get_updated_configs(config)

    assert [layer.layer_type for layer in updated_configs] == [
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
    ]
    assert [layer.head_dim for layer in updated_configs] == [32, 64, 32, 64]
    assert [layer.sliding_window for layer in updated_configs] == [512, None, 512, None]
    assert [layer.intermediate_size for layer in updated_configs] == [512, 512, 1024, 1024]


def test_convert_hf_state_dict_copies_shared_kv_weights_and_renames_qk_norms():
    neuron_config = SimpleNamespace(
        tp_degree=1,
        local_ranks_size=1,
        vocab_parallel=False,
        fused_qkv=True,
    )
    config = SimpleNamespace(
        neuron_config=neuron_config,
        num_hidden_layers=4,
        num_kv_shared_layers=2,
        layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
    )
    state_dict = {
        "layers.0.self_attn.q_proj.weight": torch.ones(4, 4),
        "layers.0.self_attn.k_proj.weight": torch.ones(2, 4),
        "layers.0.self_attn.v_proj.weight": torch.full((2, 4), 2.0),
        "layers.0.self_attn.q_norm.weight": torch.ones(4),
        "layers.0.self_attn.k_norm.weight": torch.ones(4),
        "layers.1.self_attn.q_proj.weight": torch.full((4, 4), 3.0),
        "layers.1.self_attn.k_proj.weight": torch.full((2, 4), 4.0),
        "layers.1.self_attn.v_proj.weight": torch.full((2, 4), 5.0),
        "layers.1.self_attn.q_norm.weight": torch.full((4,), 3.0),
        "layers.1.self_attn.k_norm.weight": torch.full((4,), 4.0),
        "layers.2.self_attn.q_proj.weight": torch.full((4, 4), 6.0),
        "layers.2.self_attn.q_norm.weight": torch.full((4,), 6.0),
        "layers.3.self_attn.q_proj.weight": torch.full((4, 4), 7.0),
        "layers.3.self_attn.q_norm.weight": torch.full((4,), 7.0),
    }

    converted = NeuronGemma4ForCausalLM.convert_hf_to_neuron_state_dict(state_dict, config)

    assert torch.equal(converted["layers.2.self_attn.Wqkv.weight"], torch.cat([
        torch.full((4, 4), 6.0),
        torch.ones(2, 4),
        torch.full((2, 4), 2.0),
    ]))
    assert torch.equal(converted["layers.3.self_attn.Wqkv.weight"], torch.cat([
        torch.full((4, 4), 7.0),
        torch.full((2, 4), 4.0),
        torch.full((2, 4), 5.0),
    ]))
    assert "layers.3.self_attn.q_layernorm.weight" in converted
    assert "layers.3.self_attn.k_layernorm.weight" in converted
