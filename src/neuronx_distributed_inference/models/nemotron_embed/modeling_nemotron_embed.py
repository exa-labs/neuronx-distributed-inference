# coding=utf-8
# Copyright 2025 Exa Labs and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""nvidia/Nemotron-3-Embed-8B-BF16 (Ministral3-based text embedding encoder) for NxD Inference.

Encoder-only single-forward model: bidirectional (non-causal) attention over the
padded sequence, masked mean pooling over valid tokens, fp32 L2 normalization.
No KV cache, no decode phase, no sampling.

Contract of the traced graph, per (batch_size, seq_len) bucket:
    (input_ids[int32 B,S], attention_mask[int32 B,S]) -> embeddings[fp32 B, hidden]

Notes vs the HF reference (transformers>=5.2 Ministral3Model):
  - RoPE is YaRN (factor 16, original_max_position_embeddings 16384). With
    mscale == mscale_all_dim the YaRN attention factor is exactly 1.0.
  - The llama-4 attention scale `1 + beta*log(1+floor(pos/16384))` equals 1.0
    for all positions < 16384, and this model is only compiled for seq_len
    <= 2048, so it is omitted from the graph.
"""

import math
from typing import List, Optional, Tuple

import torch
from neuronx_distributed.parallel_layers.layers import ParallelEmbedding
from neuronx_distributed.utils import cpu_mode
from torch import nn
from transformers.models.mistral.modeling_mistral import MistralRMSNorm

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.encoder_base import NeuronEncoderApplication
from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaMLP
from neuronx_distributed_inference.models.model_wrapper import EncoderModelInstance, ModelWrapper
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm

NEMOTRON_EMBED_MODEL_TAG = "nemotron_embed_encoder"


def get_rmsnorm_cls():
    # CustomRMSNorm does not work on CPU.
    return MistralRMSNorm if cpu_mode() else CustomRMSNorm


class NemotronEmbedYarnRotaryEmbedding(RotaryEmbedding):
    """YaRN rotary embedding matching transformers' `_compute_yarn_parameters`.

    The attention factor is 1.0 for this model (mscale == mscale_all_dim), so
    only the inverse frequencies differ from plain RoPE.
    """

    def __init__(self, dim, max_position_embeddings, base, rope_parameters: dict):
        self.rope_parameters = rope_parameters
        super().__init__(dim, max_position_embeddings=max_position_embeddings, base=base)

    def get_inv_freqs(self, device: Optional[torch.device] = None) -> torch.Tensor:
        p = self.rope_parameters
        factor = p["factor"]
        beta_fast = p.get("beta_fast") or 32
        beta_slow = p.get("beta_slow") or 1
        original_max = p["original_max_position_embeddings"]

        def find_correction_dim(num_rotations):
            return (self.dim * math.log(original_max / (num_rotations * 2 * math.pi))) / (
                2 * math.log(self.base)
            )

        low = max(math.floor(find_correction_dim(beta_fast)), 0)
        high = min(math.ceil(find_correction_dim(beta_slow)), self.dim - 1)
        if low == high:
            high += 0.001

        pos_freqs = self.base ** (
            torch.arange(0, self.dim, 2, dtype=torch.float, device=device) / self.dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (factor * pos_freqs)

        ramp = torch.clamp(
            (torch.arange(self.dim // 2, dtype=torch.float32, device=device) - low)
            / (high - low),
            0,
            1,
        )
        inv_freq_extrapolation_factor = 1 - ramp
        return (
            inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
            + inv_freq_extrapolation * inv_freq_extrapolation_factor
        )


class NemotronEmbedNeuronConfig(NeuronConfig):
    """Neuron config for the encoder; forces settings the encoder graph requires."""

    def __init__(self, **kwargs):
        # The NKI flash-attention kernel only supports causal or fully unmasked
        # attention; the encoder needs a padding-aware bidirectional mask, so
        # attention must go through the native (non-kernel) path.
        kwargs["attn_kernel_enabled"] = False
        kwargs["padding_side"] = "right"
        super().__init__(**kwargs)


class NemotronEmbedInferenceConfig(InferenceConfig):
    def add_derived_config(self):
        self.num_cores_per_group = 1

    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "vocab_size",
            "max_position_embeddings",
            "rope_theta",
            "rms_norm_eps",
            "hidden_act",
            "rope_parameters",
        ]

    @classmethod
    def get_neuron_config_cls(cls):
        return NemotronEmbedNeuronConfig


class NeuronNemotronEmbedAttention(NeuronAttentionBase):
    def __init__(self, config: InferenceConfig):
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        rotary_emb = NemotronEmbedYarnRotaryEmbedding(
            head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            rope_parameters=config.rope_parameters,
        )
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rotary_emb=rotary_emb,
        )


class NeuronNemotronEmbedLayer(nn.Module):
    """Transformer layer; module names match the HF checkpoint (layers.N.*)."""

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.self_attn = NeuronNemotronEmbedAttention(config)
        self.mlp = NeuronLlamaMLP(config)
        self.input_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )[0]
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]
        hidden_states = residual + hidden_states
        return hidden_states


class NeuronNemotronEmbedEncoder(nn.Module):
    """The traced module: embeddings -> layers -> norm -> masked mean pool -> L2 norm."""

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
        )
        self.layers = nn.ModuleList(
            [NeuronNemotronEmbedLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """input_ids/attention_mask: int32 [B, S] (right padded) -> fp32 [B, hidden]."""
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        position_ids = (
            torch.arange(seq_len, dtype=torch.int32, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
        )
        # Bidirectional padding mask: [B, 1, S(q), S(k)] bool. Every query may
        # attend to every valid (non-pad) key.
        mask_bool = attention_mask.to(torch.bool)
        attn_mask_4d = mask_bool[:, None, None, :].expand(
            batch_size, 1, seq_len, seq_len
        )

        for layer in self.layers:
            hidden_states = layer(hidden_states, attn_mask_4d, position_ids)
        hidden_states = self.norm(hidden_states)

        # Masked mean pooling + L2 normalization in fp32.
        hidden_states = hidden_states.to(torch.float32)
        mask_f = attention_mask.to(torch.float32).unsqueeze(-1)
        pooled = (hidden_states * mask_f).sum(dim=1) / mask_f.sum(dim=1)
        return pooled / torch.sqrt((pooled * pooled).sum(dim=1, keepdim=True))


class NemotronEmbedModelWrapper(ModelWrapper):
    """Wraps the encoder for NxD tracing with (batch_size, seq_len) buckets.

    Buckets come from `config.embed_buckets`, a list of [batch_size, seq_len]
    pairs. The caller (application) is responsible for padding inputs to an
    exact bucket shape; runtime bucket selection is by input shape.
    """

    def input_generator(self) -> List[Tuple[torch.Tensor, ...]]:
        inputs = []
        for batch_size, seq_len in self.config.embed_buckets:
            input_ids = torch.ones((batch_size, seq_len), dtype=torch.int32)
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.int32)
            inputs.append((input_ids, attention_mask))
        return inputs

    def get_model_instance(self):
        return EncoderModelInstance(model_cls=self.model_cls, config=self.config)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Forward called before load. Run load() first.")
        expected = [tuple(b) for b in self.config.embed_buckets]
        assert (
            tuple(input_ids.shape) in expected
        ), f"input shape {tuple(input_ids.shape)} must exactly match a bucket in {expected}"
        args = self.convert_int64_to_int32(input_ids, attention_mask)
        return self._forward(*args)


class NeuronNemotronEmbedModel(NeuronEncoderApplication):
    """Application: compile/load/run the Nemotron embed encoder on Neuron."""

    _model_cls = NeuronNemotronEmbedEncoder

    def get_model_wrapper_cls(self):
        return [[NeuronNemotronEmbedEncoder, NemotronEmbedModelWrapper]]

    def get_compiler_args(self) -> str:
        return (
            "--model-type=transformer --auto-cast=none -O2 "
            "--internal-hlo2tensorizer-options='--verify-hlo=true'"
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.models[0](input_ids, attention_mask)

    @classmethod
    def get_config_cls(cls):
        return NemotronEmbedInferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        tp_degree = config.neuron_config.tp_degree
        for i in range(config.num_hidden_layers):
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )
        return state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass
