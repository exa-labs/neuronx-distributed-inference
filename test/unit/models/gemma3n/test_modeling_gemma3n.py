import copy

import torch
from transformers.models.gemma3n.configuration_gemma3n import Gemma3nTextConfig
from transformers.models.gemma3n.modeling_gemma3n import (
    Gemma3nTextAltUp,
    Gemma3nTextLaurelBlock,
    Gemma3nTextMLP,
)

from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.models.gemma3n.modeling_gemma3n import (
    Gemma3nInferenceConfig,
    NeuronGemma3nForCausalLM,
    NeuronGemma3nTextAltUp,
    NeuronGemma3nTextLaurelBlock,
    NeuronGemma3nTextMLP,
)


def tiny_hf_config():
    return Gemma3nTextConfig(
        vocab_size=64,
        vocab_size_per_layer_input=64,
        hidden_size=16,
        hidden_size_per_layer_input=4,
        intermediate_size=[32, 24],
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=16,
        sliding_window=4,
        layer_types=["sliding_attention", "full_attention"],
        activation_sparsity_pattern=[0.0, 0.0],
        altup_num_inputs=2,
        laurel_rank=4,
        final_logit_softcapping=30.0,
    )


def tiny_neuron_config(hf_config=None):
    hf_config = hf_config or tiny_hf_config()

    def load_config(config):
        for key, value in hf_config.to_dict().items():
            setattr(config, key, value)

    return Gemma3nInferenceConfig(
        NeuronConfig(torch_dtype=torch.float32, on_cpu=True),
        load_config=load_config,
    )


def test_config_loads_gemma3n_text_attributes():
    config = tiny_neuron_config()

    assert config.hidden_size == 16
    assert config.hidden_activation == "gelu_pytorch_tanh"
    assert config.intermediate_size == [32, 24]
    assert config.layer_types == ["sliding_attention", "full_attention"]
    assert config.activation_sparsity_pattern == [0.0, 0.0]


def test_mlp_matches_transformers_cpu():
    torch.manual_seed(0)
    hf_config = tiny_hf_config()
    neuron_config = tiny_neuron_config(hf_config)
    hf_module = Gemma3nTextMLP(hf_config, layer_idx=1).eval()
    neuron_module = NeuronGemma3nTextMLP(neuron_config, layer_idx=1).eval()
    neuron_module.load_state_dict(copy.deepcopy(hf_module.state_dict()))
    hidden_states = torch.randn(2, 3, hf_config.hidden_size)

    with torch.no_grad():
        expected = hf_module(hidden_states)
        actual = neuron_module(hidden_states)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_laurel_matches_transformers_cpu():
    torch.manual_seed(0)
    hf_config = tiny_hf_config()
    neuron_config = tiny_neuron_config(hf_config)
    hf_module = Gemma3nTextLaurelBlock(hf_config).eval()
    neuron_module = NeuronGemma3nTextLaurelBlock(neuron_config).eval()
    neuron_module.load_state_dict(copy.deepcopy(hf_module.state_dict()))
    hidden_states = torch.randn(2, 3, hf_config.hidden_size)

    with torch.no_grad():
        expected = hf_module(hidden_states)
        actual = neuron_module(hidden_states)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_altup_matches_transformers_cpu():
    torch.manual_seed(0)
    hf_config = tiny_hf_config()
    neuron_config = tiny_neuron_config(hf_config)
    hf_module = Gemma3nTextAltUp(hf_config).eval()
    neuron_module = NeuronGemma3nTextAltUp(neuron_config).eval()
    neuron_module.load_state_dict(copy.deepcopy(hf_module.state_dict()))
    hidden_states = torch.randn(
        hf_config.altup_num_inputs,
        2,
        3,
        hf_config.hidden_size,
    )
    activated = torch.randn(2, 3, hf_config.hidden_size)

    with torch.no_grad():
        expected_predictions = hf_module.predict(hidden_states)
        actual_predictions = neuron_module.predict(hidden_states)
        expected_corrected = hf_module.correct(expected_predictions, activated)
        actual_corrected = neuron_module.correct(actual_predictions, activated)

    torch.testing.assert_close(actual_predictions, expected_predictions, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_corrected, expected_corrected, rtol=1e-5, atol=1e-5)


def test_state_dict_conversion_renames_gemma3n_attention_and_lm_head_keys():
    config = tiny_neuron_config()
    state_dict = {
        "model.norm.weight": torch.ones(config.hidden_size),
        "lm_head.weight": torch.randn(config.vocab_size, config.hidden_size),
    }
    for layer_idx in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer_idx}.self_attn"
        state_dict[f"{prefix}.q_norm.weight"] = torch.ones(config.head_dim)
        state_dict[f"{prefix}.k_norm.weight"] = torch.ones(config.head_dim)
        state_dict[f"{prefix}.q_proj.weight"] = torch.randn(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
        )
        state_dict[f"{prefix}.k_proj.weight"] = torch.randn(
            config.num_key_value_heads * config.head_dim,
            config.hidden_size,
        )
        state_dict[f"{prefix}.v_proj.weight"] = torch.randn(
            config.num_key_value_heads * config.head_dim,
            config.hidden_size,
        )

    converted = NeuronGemma3nForCausalLM.convert_hf_to_neuron_state_dict(state_dict, config)

    assert "norm.weight" in converted
    assert "lm_head.proj.weight" in converted
    for layer_idx in range(config.num_hidden_layers):
        assert f"layers.{layer_idx}.self_attn.q_layernorm.weight" in converted
        assert f"layers.{layer_idx}.self_attn.k_layernorm.weight" in converted
        assert f"layers.{layer_idx}.self_attn.q_norm.weight" not in converted
        assert f"layers.{layer_idx}.self_attn.k_norm.weight" not in converted
        assert f"layers.{layer_idx}.self_attn.rank_util.rank" in converted
