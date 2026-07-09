# coding=utf-8
# Copyright 2026 Google Inc. HuggingFace Inc. team. All rights reserved.
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

"""Gemma 4 text model for NxD inference."""

import copy
from typing import List, Optional, Tuple, Type

import torch
from torch import nn

from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, ParallelEmbedding
from neuronx_distributed.utils import cpu_mode

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.gemma3.modeling_gemma3 import get_rmsnorm_cls as get_gemma3_rmsnorm_cls
from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaMLP
from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM, NeuronBaseModel
from neuronx_distributed_inference.models.model_wrapper import CONTEXT_ENCODING_MODEL_TAG, TOKEN_GENERATION_MODEL_TAG
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding, apply_rotary_pos_emb
from neuronx_distributed_inference.modules.kvcache.gemma4_kv_cache_manager import Gemma4KVCacheManager


# neuronx-cc rejects any single graph input larger than 4 GiB (NCC_EVRF023).
# The fused per-layer embedding table is vocab_size_per_layer_input x
# (num_hidden_layers * hidden_size_per_layer_input) -- ~4.7 GiB in bf16 for
# E2B -- so it is split column-wise into equal chunks that each stay under
# the limit on every tensor-parallel rank.
_MAX_WEIGHT_INPUT_BYTES = 4 * 2**30 - 2**20


def per_layer_embedding_split_sizes(config: InferenceConfig) -> List[int]:
    """Column split sizes for the per-layer embedding table such that each
    chunk's per-rank weight input stays under the compiler's 4 GiB limit."""
    total_cols = config.num_hidden_layers * config.hidden_size_per_layer_input
    element_size = torch.tensor([], dtype=config.neuron_config.torch_dtype).element_size()
    per_rank_rows = -(-config.vocab_size_per_layer_input // config.neuron_config.tp_degree)
    max_cols = max(1, _MAX_WEIGHT_INPUT_BYTES // (per_rank_rows * element_size))
    num_chunks = -(-total_cols // max_cols)
    base, remainder = divmod(total_cols, num_chunks)
    return [base + (1 if i < remainder else 0) for i in range(num_chunks)]


class NeuronGemma4RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.bfloat16))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = hidden_states.float()
        output = output * torch.rsqrt(output.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.with_scale:
            output = output * self.weight.float()
        return output.type_as(hidden_states)


class Gemma4LmHead(ColumnParallelLinear):
    def __init__(self, *args, final_logit_softcapping=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_logit_softcapping = final_logit_softcapping

    def forward(self, hidden_states):
        logits = super().forward(hidden_states)
        if self.final_logit_softcapping is None:
            return logits
        return torch.tanh(logits / self.final_logit_softcapping) * self.final_logit_softcapping


def get_rmsnorm_cls(with_scale: bool = True):
    if cpu_mode() and with_scale:
        return get_gemma3_rmsnorm_cls()
    return lambda hidden_size, eps: NeuronGemma4RMSNorm(hidden_size, eps=eps, with_scale=with_scale)


class Gemma4ProportionalRotaryEmbedding(RotaryEmbedding):
    """Proportional RoPE, matching HF gemma4 ``_compute_proportional_rope_parameters``.

    Unlike a plain partial-rotary slice, gemma4's "proportional" rope always
    produces an encoding spanning the *entire* head_dim: the first
    ``int(partial_rotary_factor * head_dim // 2)`` frequency pairs are the usual
    ``1 / base ** (2i / head_dim)`` (note the denominator is the full head_dim,
    not the rotated width), and the remaining pairs have zero frequency (cos=1,
    sin=0). Because cos/sin cover all ``head_dim`` dims, the standard NEOX
    ``rotate_half`` pairs dim ``i`` with dim ``i + head_dim/2`` exactly as HF.
    """

    def __init__(self, head_dim, partial_rotary_factor, max_position_embeddings, base):
        super().__init__(dim=head_dim, max_position_embeddings=max_position_embeddings, base=base)
        self.rope_angles = int(partial_rotary_factor * head_dim // 2)

    def get_inv_freqs(self, device: Optional[torch.device] = None) -> torch.Tensor:
        head_dim = self.dim
        idx = torch.arange(0, 2 * self.rope_angles, 2, dtype=torch.float, device=device)
        inv_freq_rotated = 1.0 / (self.base ** (idx / head_dim))
        nope_angles = head_dim // 2 - self.rope_angles
        if nope_angles > 0:
            return torch.cat(
                (inv_freq_rotated, torch.zeros(nope_angles, dtype=torch.float32, device=device)),
                dim=0,
            )
        return inv_freq_rotated


def get_updated_configs(config: "Gemma4InferenceConfig"):
    updated_configs = []
    for layer_idx, layer_type in enumerate(config.layer_types):
        updated_config = copy.deepcopy(config)
        updated_config.layer_idx = layer_idx
        updated_config.layer_type = layer_type
        updated_config.is_full_attention = layer_type == "full_attention"
        updated_config.sliding_window = config.sliding_window if layer_type == "sliding_attention" else None
        updated_config.head_dim = config.global_head_dim if updated_config.is_full_attention else config.local_head_dim
        updated_config.num_key_value_heads = (
            config.num_global_key_value_heads
            if updated_config.is_full_attention and config.num_global_key_value_heads is not None
            else config.local_num_key_value_heads
        )
        first_kv_shared_layer_idx = config.num_hidden_layers - config.num_kv_shared_layers
        updated_config.is_kv_shared_layer = config.num_kv_shared_layers > 0 and layer_idx >= first_kv_shared_layer_idx
        # The last non-shared layer of each type is the KV source for all shared
        # layers of that type (mirrors HF Gemma4TextAttention.store_full_length_kv).
        prev_layers = config.layer_types[:first_kv_shared_layer_idx]
        updated_config.store_full_length_kv = (
            not updated_config.is_kv_shared_layer
            and config.num_kv_shared_layers > 0
            and layer_idx == len(prev_layers) - 1 - prev_layers[::-1].index(layer_type)
        )
        updated_config.intermediate_size = config.intermediate_size * (
            2 if config.use_double_wide_mlp and updated_config.is_kv_shared_layer else 1
        )
        updated_configs.append(updated_config)
    return updated_configs


class Gemma4NeuronConfig(NeuronConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_cls = NeuronGemma4Attention


class Gemma4InferenceConfig(InferenceConfig):
    def __init__(self, neuron_config: NeuronConfig, fused_spec_config=None, load_config=None):
        self.attributes = [
            "attention_bias",
            "attention_k_eq_v",
            "final_logit_softcapping",
            "global_head_dim",
            "head_dim",
            "hidden_activation",
            "hidden_size",
            "hidden_size_per_layer_input",
            "intermediate_size",
            "layer_types",
            "max_position_embeddings",
            "num_attention_heads",
            "num_global_key_value_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "num_kv_shared_layers",
            "rms_norm_eps",
            "rope_parameters",
            "sliding_window",
            "tie_word_embeddings",
            "use_double_wide_mlp",
            "vocab_size",
            "vocab_size_per_layer_input",
        ]
        self.neuron_config = neuron_config
        self.fused_spec_config = fused_spec_config

        if load_config is not None:
            load_config(self)
        else:
            self.load_config()

        source_config = self.text_config if hasattr(self, "text_config") else self
        self.attention_bias = source_config.attention_bias if hasattr(source_config, "attention_bias") else False
        self.attention_k_eq_v = source_config.attention_k_eq_v if hasattr(source_config, "attention_k_eq_v") else False
        self.final_logit_softcapping = (
            source_config.final_logit_softcapping if hasattr(source_config, "final_logit_softcapping") else None
        )
        self.local_head_dim = source_config.head_dim if hasattr(source_config, "head_dim") else 256
        self.global_head_dim = source_config.global_head_dim if hasattr(source_config, "global_head_dim") else self.local_head_dim
        self.head_dim = self.local_head_dim
        self.hidden_activation = (
            source_config.hidden_activation if hasattr(source_config, "hidden_activation") else "gelu_pytorch_tanh"
        )
        self.hidden_act = self.hidden_activation
        self.hidden_size = source_config.hidden_size
        self.hidden_size_per_layer_input = (
            source_config.hidden_size_per_layer_input
            if hasattr(source_config, "hidden_size_per_layer_input")
            else 0
        )
        self.intermediate_size = source_config.intermediate_size
        self.layer_types = source_config.layer_types if hasattr(source_config, "layer_types") else None
        self.max_position_embeddings = source_config.max_position_embeddings
        self.num_attention_heads = source_config.num_attention_heads
        self.local_num_key_value_heads = source_config.num_key_value_heads
        self.num_global_key_value_heads = (
            source_config.num_global_key_value_heads if hasattr(source_config, "num_global_key_value_heads") else None
        )
        self.num_hidden_layers = source_config.num_hidden_layers
        self.num_key_value_heads = self.local_num_key_value_heads
        self.num_kv_shared_layers = (
            source_config.num_kv_shared_layers if hasattr(source_config, "num_kv_shared_layers") else 0
        )
        self.rms_norm_eps = source_config.rms_norm_eps
        self.rope_parameters = source_config.rope_parameters if hasattr(source_config, "rope_parameters") else None
        self.sliding_window = source_config.sliding_window
        self.tie_word_embeddings = (
            source_config.tie_word_embeddings if hasattr(source_config, "tie_word_embeddings") else True
        )
        self.use_double_wide_mlp = (
            source_config.use_double_wide_mlp if hasattr(source_config, "use_double_wide_mlp") else False
        )
        self.vocab_size = source_config.vocab_size
        self.vocab_size_per_layer_input = (
            source_config.vocab_size_per_layer_input
            if hasattr(source_config, "vocab_size_per_layer_input")
            else self.vocab_size
        )
        self.pad_token_id = source_config.pad_token_id if hasattr(source_config, "pad_token_id") else 0

        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % 5) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        if self.layer_types[-1] != "full_attention":
            self.layer_types[-1] = "full_attention"

        self.add_derived_config()
        self.validate_config()

    def add_derived_config(self):
        self.num_cores_per_group = 1

    def get_required_attributes(self) -> List[str]:
        return self.attributes

    @classmethod
    def get_neuron_config_cls(cls) -> Type[Gemma4NeuronConfig]:
        return Gemma4NeuronConfig


class NeuronGemma4Attention(NeuronAttentionBase):
    def __init__(self, config: Gemma4InferenceConfig):
        rope_parameters = config.rope_parameters[config.layer_type] if config.rope_parameters else None
        rope_theta = rope_parameters["rope_theta"] if rope_parameters and "rope_theta" in rope_parameters else 10000.0
        rope_type = rope_parameters.get("rope_type", "default") if rope_parameters else "default"

        if rope_type == "proportional":
            # gemma4 full-attention layers: proportional rope spans the full
            # head_dim (partial rotation encoded as trailing zero frequencies),
            # so the standard full-width rotate_half path applies.
            rotary_dim = config.head_dim
            rotary_emb = Gemma4ProportionalRotaryEmbedding(
                head_dim=config.head_dim,
                partial_rotary_factor=rope_parameters.get("partial_rotary_factor", 1.0),
                max_position_embeddings=config.max_position_embeddings,
                base=rope_theta,
            )
        else:
            rotary_dim = config.head_dim
            if rope_parameters and "partial_rotary_factor" in rope_parameters:
                rotary_dim = int(config.head_dim * rope_parameters["partial_rotary_factor"])
            rotary_emb = RotaryEmbedding(
                dim=rotary_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=rope_theta,
            )

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rotary_emb=rotary_emb,
            rms_norm_eps=config.rms_norm_eps,
            use_qk_norm=False,
            use_scaled_rope=None,
            sliding_window=config.sliding_window,
            softmax_scale=1.0,
            q_layernorm=get_rmsnorm_cls()(config.head_dim, eps=config.rms_norm_eps),
            k_layernorm=get_rmsnorm_cls()(config.head_dim, eps=config.rms_norm_eps),
        )
        self.v_layernorm = get_rmsnorm_cls(with_scale=False)(config.head_dim, eps=config.rms_norm_eps)
        self.rotary_dim = rotary_dim
        # KV-sharing metadata (Gemma 4 reuses the last non-shared same-type
        # layer's K/V for every shared layer). Set per forward by the decoder
        # layer via ``_shared_kv_states``; ``None`` disables sharing entirely.
        self.layer_type = getattr(config, "layer_type", None)
        self.is_kv_shared_layer = getattr(config, "is_kv_shared_layer", False)
        self.store_full_length_kv = getattr(config, "store_full_length_kv", False)
        self._shared_kv_states = None

    def apply_rotary_embedding(self, Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope):
        """Rotary embedding with partial-rotary support.

        Full-attention layers rotate only the first ``rotary_dim`` dims of each
        head (``partial_rotary_factor`` < 1); the remainder passes through
        unrotated. The cos/sin caches are computed per layer because layer
        types use different rotary dims and thetas.
        """
        if self.rotary_dim == self.head_dim:
            return super().apply_rotary_embedding(Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope)
        if cos_cache is None or sin_cache is None:
            cos_cache, sin_cache = self.rotary_emb(V, position_ids)
        q_rot, q_pass = Q[..., : self.rotary_dim], Q[..., self.rotary_dim :]
        k_rot, k_pass = K[..., : self.rotary_dim], K[..., self.rotary_dim :]
        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos_cache, sin_cache)
        Q = torch.cat((q_rot, q_pass), dim=-1)
        K = torch.cat((k_rot, k_pass), dim=-1)
        return Q, K, cos_cache, sin_cache

    def prep_qkv_tensors(self, *args, **kwargs):
        # cos/sin caches must not be shared across Gemma 4 layers: sliding and
        # full layers use different rotary dims and thetas.
        kwargs.pop("cos_cache", None)
        kwargs.pop("sin_cache", None)
        q, k, v, cos_cache, sin_cache, residual = super().prep_qkv_tensors(*args, **kwargs)
        v = self.v_layernorm(v)
        # True KV-sharing: a shared layer attends its own Q against the source
        # layer's post-rope K / post-norm V (never recomputing K/V from its own
        # hidden states), exactly as HF gemma4 does. The source layer stashes its
        # K/V here so every same-type shared layer downstream reuses them.
        shared = self._shared_kv_states
        if shared is not None:
            if self.is_kv_shared_layer:
                k, v = shared[self.layer_type]
            elif self.store_full_length_kv:
                shared[self.layer_type] = (k, v)
        return q, k, v, cos_cache, sin_cache, residual


class NeuronGemma4DecoderLayer(nn.Module):
    def __init__(self, config: Gemma4InferenceConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.attention_type = config.layer_type
        self.is_sliding_window_attention = config.sliding_window is not None

        self.self_attn = NeuronGemma4Attention(config)
        self.mlp = NeuronLlamaMLP(config)
        self.input_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)

        if config.hidden_size_per_layer_input and config.hidden_size_per_layer_input > 0:
            self.per_layer_input_gate = ColumnParallelLinear(
                config.hidden_size,
                config.hidden_size_per_layer_input,
                bias=False,
                gather_output=True,
                dtype=config.neuron_config.torch_dtype,
            )
            self.per_layer_projection = ColumnParallelLinear(
                config.hidden_size_per_layer_input,
                config.hidden_size,
                bias=False,
                gather_output=True,
                dtype=config.neuron_config.torch_dtype,
            )
            self.post_per_layer_input_norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.per_layer_input_gate = None
            self.per_layer_projection = None
            self.post_per_layer_input_norm = None

        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        local_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        adapter_ids=None,
        per_layer_input: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        per_layer_inputs = kwargs.pop("per_layer_inputs", None)
        if per_layer_input is None and per_layer_inputs is not None:
            per_layer_input = per_layer_inputs[:, :, self.layer_idx, :]
        # Hand the shared per-forward KV dict to this layer's attention so it can
        # publish (source layer) or consume (shared layer) the reused K/V.
        self.self_attn._shared_kv_states = kwargs.pop("shared_kv_states", None)
        mask = local_mask if self.is_sliding_window_attention and local_mask is not None else attention_mask

        residual = hidden_states
        hidden_states = self.input_layernorm(residual)
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            adapter_ids=adapter_ids,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = hidden_states + residual

        if per_layer_input is not None and self.per_layer_input_gate is not None:
            gate = self.per_layer_input_gate(hidden_states)
            gate = torch.nn.functional.gelu(gate, approximate="tanh")
            per_layer_contribution = self.per_layer_projection(gate * per_layer_input)
            per_layer_contribution = self.post_per_layer_input_norm(per_layer_contribution)
            hidden_states = hidden_states + per_layer_contribution

        hidden_states = hidden_states * self.layer_scalar
        # Never propagate cos/sin caches to the next layer: rotary dims/thetas
        # differ between sliding and full attention layers.
        return (hidden_states, present_key_value, None, None, None)


class NeuronGemma4TextModel(NeuronBaseModel):
    def setup_attr_for_model(self, config: Gemma4InferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
        self.head_dim = config.global_head_dim
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        self.vocab_size_per_layer_input = config.vocab_size_per_layer_input

        # Per-layer mixed attention: sliding_attention layers use a
        # sliding_window-long, local_head_dim-wide KV cache; full_attention
        # layers keep a max_length-long, global_head_dim-wide cache. Setting
        # sliding_window + has_mixed_attn makes the base model emit both the
        # global causal mask and the windowed local mask each step; the decoder
        # layers pick the right one per layer.
        self.sliding_window = config.sliding_window
        self.has_mixed_attn = True
        self.layer_is_sliding = [layer_type == "sliding_attention" for layer_type in config.layer_types]
        self.layer_head_dims = [
            config.local_head_dim if is_sliding else config.global_head_dim
            for is_sliding in self.layer_is_sliding
        ]

    def init_inference_optimization(self, config: Gemma4InferenceConfig):
        """Same as the base implementation, but with a per-layer KV cache manager.

        The generic ``KVCacheManager`` sizes every layer identically, which
        cannot represent Gemma 4's mix of sliding (window x 256) and full
        (max_length x 512) layer caches.
        """
        super().init_inference_optimization(config)
        self.kv_mgr = Gemma4KVCacheManager(
            config,
            num_kv_head=self.num_key_value_heads,
            layer_is_sliding=self.layer_is_sliding,
            layer_head_dims=self.layer_head_dims,
            sliding_window=config.sliding_window,
            global_rank=self.rank_util,
        )

    def init_model(self, config: Gemma4InferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        )
        if config.hidden_size_per_layer_input and config.hidden_size_per_layer_input > 0:
            self.embed_tokens_per_layer = nn.ModuleList(
                [
                    ParallelEmbedding(
                        config.vocab_size_per_layer_input,
                        split_size,
                        self.padding_idx,
                        dtype=config.neuron_config.torch_dtype,
                        shard_across_embedding=True,
                        sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
                    )
                    for split_size in per_layer_embedding_split_sizes(config)
                ]
            )
            self.per_layer_model_projection = ColumnParallelLinear(
                config.hidden_size,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
                bias=False,
                gather_output=True,
                dtype=config.neuron_config.torch_dtype,
            )
            self.per_layer_projection_norm = get_rmsnorm_cls()(config.hidden_size_per_layer_input, eps=config.rms_norm_eps)
            self.register_buffer("embed_scale_per_layer", torch.tensor(config.hidden_size_per_layer_input**0.5))
            self.register_buffer("per_layer_input_scale", torch.rsqrt(torch.tensor(2.0)))
            self.register_buffer("per_layer_projection_scale", torch.tensor(config.hidden_size**-0.5))
        else:
            self.embed_tokens_per_layer = None
            self.per_layer_model_projection = None
            self.per_layer_projection_norm = None

        self.lm_head = Gemma4LmHead(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            pad=True,
            gather_output=not self.on_device_sampling,
            dtype=config.neuron_config.torch_dtype,
            final_logit_softcapping=config.final_logit_softcapping,
        )

        updated_configs = get_updated_configs(config)
        self.layers = nn.ModuleList(
            [NeuronGemma4DecoderLayer(layer_config, idx) for idx, layer_config in enumerate(updated_configs)]
        )
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        self.register_buffer("normalizer", torch.tensor(config.hidden_size**0.5, dtype=config.neuron_config.torch_dtype))

    def get_per_layer_inputs(self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor):
        if self.embed_tokens_per_layer is None:
            return None
        per_layer_inputs_mask = torch.logical_and(input_ids >= 0, input_ids < self.vocab_size_per_layer_input)
        per_layer_input_ids = torch.where(per_layer_inputs_mask, input_ids, torch.zeros_like(input_ids))
        per_layer_embeds = (
            torch.cat(
                [chunk(per_layer_input_ids) for chunk in self.embed_tokens_per_layer],
                dim=-1,
            )
            * self.embed_scale_per_layer
        )
        per_layer_embeds = per_layer_embeds.reshape(
            *input_ids.shape,
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_model_projection(inputs_embeds) * self.per_layer_projection_scale
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)
        return (per_layer_projection + per_layer_embeds) * self.per_layer_input_scale

    def get_model_output(self, input_ids: torch.LongTensor = None, inputs_embeds=None, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = inputs_embeds * self.normalizer
        per_layer_inputs = None if input_ids is None else self.get_per_layer_inputs(input_ids, inputs_embeds)
        kwargs["inputs_embeds"] = inputs_embeds
        kwargs["per_layer_inputs"] = per_layer_inputs
        # Per-forward scratch dict for KV-sharing: source layers publish their
        # post-rope K/V here, shared layers read it. Flows to every decoder layer
        # through the base loop's **kwargs. Empty/no-op when num_kv_shared_layers=0.
        if self.config.num_kv_shared_layers > 0:
            kwargs["shared_kv_states"] = {}
        hidden_states = super().get_model_output(input_ids=input_ids, **kwargs)
        return hidden_states


class NeuronGemma4ForCausalLM(NeuronBaseForCausalLM):
    _model_cls = NeuronGemma4TextModel
    _STATE_DICT_MODEL_PREFIX = "language_model.model."

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        try:
            from transformers import Gemma4ForCausalLM

            return Gemma4ForCausalLM.from_pretrained(model_path, **kwargs)
        except ImportError:
            from transformers import Gemma4ForConditionalGeneration

            return Gemma4ForConditionalGeneration.from_pretrained(model_path, **kwargs)

    def enable_context_encoding(self):
        self.compile_tag = CONTEXT_ENCODING_MODEL_TAG
        super().enable_context_encoding()

    def enable_token_generation(self):
        self.compile_tag = TOKEN_GENERATION_MODEL_TAG
        super().enable_token_generation()

    def get_compiler_args(self):
        optimization_level = "-O1"
        compiler_args = f"--enable-saturate-infinity --enable-mixed-precision-accumulation --model-type transformer {optimization_level}"
        compiler_args += " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
        compiler_args += " --auto-cast=none"
        compiler_args += " --internal-enable-dge-levels vector_dynamic_offsets"
        compiler_args += " --internal-hlo2tensorizer-options='--verify-hlo=true'"
        return compiler_args

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        if any(k.startswith("model.language_model.") for k in state_dict):
            state_dict = {k.replace("model.language_model.", ""): v for k, v in state_dict.items()}
        elif "model.norm.weight" in state_dict:
            state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}

        per_layer_embedding_key = "embed_tokens_per_layer.weight"
        if per_layer_embedding_key in state_dict:
            fused_weight = state_dict.pop(per_layer_embedding_key)
            offset = 0
            for chunk_idx, split_size in enumerate(per_layer_embedding_split_sizes(config)):
                state_dict[f"embed_tokens_per_layer.{chunk_idx}.weight"] = (
                    fused_weight[:, offset : offset + split_size].detach().clone()
                )
                offset += split_size

        neuron_config = config.neuron_config
        if neuron_config.vocab_parallel:
            state_dict["embed_tokens.rank_util.rank"] = torch.arange(0, neuron_config.local_ranks_size)

        first_kv_shared_layer_idx = config.num_hidden_layers - config.num_kv_shared_layers
        kv_source_by_type = {}
        for layer_idx, layer_type in enumerate(config.layer_types):
            if layer_idx < first_kv_shared_layer_idx:
                kv_source_by_type[layer_type] = layer_idx
            elif layer_type in kv_source_by_type:
                source_idx = kv_source_by_type[layer_type]
                for proj_name in ("k_proj", "v_proj"):
                    for attr in ("weight", "bias"):
                        source_key = f"layers.{source_idx}.self_attn.{proj_name}.{attr}"
                        target_key = f"layers.{layer_idx}.self_attn.{proj_name}.{attr}"
                        if source_key in state_dict and target_key not in state_dict:
                            state_dict[target_key] = state_dict[source_key].detach().clone()
                for norm_name in ("k_norm",):
                    source_key = f"layers.{source_idx}.self_attn.{norm_name}.weight"
                    target_key = f"layers.{layer_idx}.self_attn.{norm_name}.weight"
                    if source_key in state_dict and target_key not in state_dict:
                        state_dict[target_key] = state_dict[source_key].detach().clone()

        for layer_idx in range(config.num_hidden_layers):
            state_dict[f"layers.{layer_idx}.self_attn.rank_util.rank"] = torch.arange(
                0, neuron_config.tp_degree, dtype=torch.int32
            )
            q_norm_key = f"layers.{layer_idx}.self_attn.q_norm.weight"
            if q_norm_key in state_dict:
                state_dict[f"layers.{layer_idx}.self_attn.q_layernorm.weight"] = state_dict.pop(q_norm_key)
            k_norm_key = f"layers.{layer_idx}.self_attn.k_norm.weight"
            if k_norm_key in state_dict:
                state_dict[f"layers.{layer_idx}.self_attn.k_layernorm.weight"] = state_dict.pop(k_norm_key)

            if config.neuron_config.fused_qkv:
                attr = "weight"
                q_key = f"layers.{layer_idx}.self_attn.q_proj.{attr}"
                k_key = f"layers.{layer_idx}.self_attn.k_proj.{attr}"
                v_key = f"layers.{layer_idx}.self_attn.v_proj.{attr}"
                if q_key in state_dict and k_key in state_dict and v_key in state_dict:
                    state_dict[f"layers.{layer_idx}.self_attn.Wqkv.{attr}"] = torch.cat(
                        [state_dict[q_key], state_dict[k_key], state_dict[v_key]]
                    )
                    del state_dict[q_key]
                    del state_dict[k_key]
                    del state_dict[v_key]

        state_dict["rank_util.rank"] = torch.arange(0, neuron_config.tp_degree, dtype=torch.int32)
        return state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        return Gemma4InferenceConfig
