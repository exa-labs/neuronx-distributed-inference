# coding=utf-8
"""PyTorch Gemma 3n text model for NXD inference."""

import copy
import math
import statistics
from typing import List, Type

import torch
from torch import nn
from transformers import Gemma3nForCausalLM
from transformers.activations import ACT2FN

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.models.model_wrapper import (
    CONTEXT_ENCODING_MODEL_TAG,
    TOKEN_GENERATION_MODEL_TAG,
)
from neuronx_distributed_inference.modules.attention.attention_base import (
    NeuronAttentionBase,
    QKNormPlacement,
)
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding


def _linear(in_features, out_features, bias=False, dtype=None, column=True, input_is_parallel=False):
    if parallel_state.model_parallel_is_initialized():
        if column:
            return ColumnParallelLinear(
                in_features,
                out_features,
                bias=bias,
                gather_output=False,
                dtype=dtype,
                pad=True,
                sequence_parallel_enabled=False,
            )
        return RowParallelLinear(
            in_features,
            out_features,
            bias=bias,
            input_is_parallel=input_is_parallel,
            dtype=dtype,
            pad=True,
            sequence_parallel_enabled=False,
        )
    return nn.Linear(in_features, out_features, bias=bias)


def _dense_linear(in_features, out_features, bias=False, dtype=None):
    return nn.Linear(in_features, out_features, bias=bias, dtype=dtype)


class NeuronGemma3nRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_buffer("weight", torch.tensor(1.0), persistent=False)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()) * self.weight.float()
        return output.type_as(x)


def get_rmsnorm_cls():
    return NeuronGemma3nRMSNorm


class Gemma3nNeuronConfig(NeuronConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_cls = NeuronGemma3nAttention


class Gemma3nInferenceConfig(InferenceConfig):
    attributes = [
        "hidden_size",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
        "head_dim",
        "pad_token_id",
        "vocab_size",
        "vocab_size_per_layer_input",
        "hidden_size_per_layer_input",
        "intermediate_size",
        "max_position_embeddings",
        "rope_theta",
        "rope_local_base_freq",
        "rms_norm_eps",
        "hidden_activation",
        "sliding_window",
        "layer_types",
        "final_logit_softcapping",
        "altup_active_idx",
        "altup_coef_clip",
        "altup_correct_scale",
        "altup_num_inputs",
        "num_kv_shared_layers",
        "laurel_rank",
        "activation_sparsity_pattern",
    ]

    def __init__(
        self,
        neuron_config: NeuronConfig,
        fused_spec_config=None,
        load_config=None,
        metadata=None,
        **kwargs,
    ):
        self.neuron_config = neuron_config
        self.fused_spec_config = fused_spec_config
        if load_config is not None:
            load_config(self)
        else:
            self.load_config()

        self.metadata = metadata
        for key, value in kwargs.items():
            setattr(self, key, value)

        text_config = getattr(self, "text_config", None)
        if text_config is not None:
            for attribute in self.attributes:
                if hasattr(text_config, attribute):
                    setattr(self, attribute, getattr(text_config, attribute))

        if not isinstance(self.intermediate_size, list):
            self.intermediate_size = [self.intermediate_size] * self.num_hidden_layers
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if (i + 1) % 5 == 0 else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]
        if self.activation_sparsity_pattern is None:
            num_sparse_layers = 10 if self.num_hidden_layers > 10 else 0
            self.activation_sparsity_pattern = [0.95] * num_sparse_layers + [
                0.0
            ] * (self.num_hidden_layers - num_sparse_layers)

        self.add_derived_config()
        self.validate_config()

    def add_derived_config(self):
        self.num_cores_per_group = 1
        self.hidden_act = self.hidden_activation

    def get_required_attributes(self) -> List[str]:
        return self.attributes

    @classmethod
    def get_neuron_config_cls(cls) -> Type[Gemma3nNeuronConfig]:
        return Gemma3nNeuronConfig


def get_updated_configs(config: Gemma3nInferenceConfig):
    updated_configs = []
    for layer_idx, layer_type in enumerate(config.layer_types):
        updated_config = copy.deepcopy(config)
        updated_config.layer_idx = layer_idx
        if layer_type != "sliding_attention":
            updated_config.sliding_window = None
        updated_configs.append(updated_config)
    return updated_configs


class NeuronGemma3nAttention(NeuronAttentionBase):
    def __init__(self, config: Gemma3nInferenceConfig):
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        rotary_emb = RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_local_base_freq if config.sliding_window is not None else config.rope_theta,
        )

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rotary_emb=rotary_emb,
            rms_norm_eps=config.rms_norm_eps,
            qk_norm_placement=QKNormPlacement.PRE_ROPE,
            q_layernorm=get_rmsnorm_cls()(hidden_size=head_dim, eps=config.rms_norm_eps),
            k_layernorm=get_rmsnorm_cls()(hidden_size=head_dim, eps=config.rms_norm_eps),
            sliding_window=config.sliding_window,
            softmax_scale=1.0,
        )
        self.v_layernorm = get_rmsnorm_cls()(hidden_size=head_dim, eps=config.rms_norm_eps, with_scale=False)

    def prep_qkv_tensors(self, *args, **kwargs):
        q, k, v, cos_cache, sin_cache, residual = super().prep_qkv_tensors(*args, **kwargs)
        v = self.v_layernorm(v)
        return q, k, v, cos_cache, sin_cache, residual


class NeuronGemma3nTextLaurelBlock(nn.Module):
    def __init__(self, config: Gemma3nInferenceConfig):
        super().__init__()
        dtype = config.neuron_config.torch_dtype
        self.linear_left = _linear(config.hidden_size, config.laurel_rank, dtype=dtype, column=True)
        self.linear_right = _linear(
            config.laurel_rank,
            config.hidden_size,
            dtype=dtype,
            column=False,
            input_is_parallel=parallel_state.model_parallel_is_initialized(),
        )
        self.post_laurel_norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states):
        laurel_hidden_states = self.linear_left(hidden_states)
        laurel_hidden_states = self.linear_right(laurel_hidden_states)
        return hidden_states + self.post_laurel_norm(laurel_hidden_states)


class NeuronGemma3nTextMLP(nn.Module):
    def __init__(self, config: Gemma3nInferenceConfig, layer_idx: int):
        super().__init__()
        dtype = config.neuron_config.torch_dtype
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size[layer_idx]
        self.gate_proj = _linear(self.hidden_size, self.intermediate_size, dtype=dtype, column=True)
        self.up_proj = _linear(self.hidden_size, self.intermediate_size, dtype=dtype, column=True)
        self.down_proj = _linear(
            self.intermediate_size,
            self.hidden_size,
            dtype=dtype,
            column=False,
            input_is_parallel=parallel_state.model_parallel_is_initialized(),
        )
        self.act_fn = ACT2FN[config.hidden_activation]
        self.activation_sparsity = config.activation_sparsity_pattern[layer_idx]
        self.std_multiplier = (
            statistics.NormalDist().inv_cdf(self.activation_sparsity)
            if self.activation_sparsity > 0.0
            else 0.0
        )

    def forward(self, hidden_states):
        gate_proj = self.gate_proj(hidden_states)
        if self.activation_sparsity > 0.0:
            gate_proj = self._gaussian_topk(gate_proj)
        return self.down_proj(self.act_fn(gate_proj) * self.up_proj(hidden_states))

    def _gaussian_topk(self, inputs):
        inputs_mean = torch.mean(inputs, dim=-1, keepdim=True)
        inputs_std = torch.std(inputs, dim=-1, keepdim=True, unbiased=False)
        cutoff_x = inputs_mean + inputs_std * self.std_multiplier
        return nn.functional.relu(inputs - cutoff_x)


class NeuronGemma3nTextAltUp(nn.Module):
    def __init__(self, config: Gemma3nInferenceConfig):
        super().__init__()
        self.config = config
        self.correct_output_scale = nn.Parameter(torch.zeros(config.hidden_size))
        self.correction_coefs = nn.Linear(config.altup_num_inputs, config.altup_num_inputs, bias=False)
        self.prediction_coefs = nn.Linear(
            config.altup_num_inputs,
            config.altup_num_inputs**2,
            bias=False,
        )
        self.modality_router = nn.Linear(config.hidden_size, config.altup_num_inputs, bias=False)
        self.router_norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        self.register_buffer("router_input_scale", torch.tensor(config.hidden_size**-1.0), persistent=False)

    def compute_router_modalities(self, hidden_states):
        router_inputs = self.router_norm(hidden_states) * self.router_input_scale
        return torch.tanh(self.modality_router(router_inputs).float()).type_as(hidden_states)

    def predict(self, hidden_states):
        modalities = self.compute_router_modalities(hidden_states[self.config.altup_active_idx])
        if self.training and self.config.altup_coef_clip is not None:
            self.prediction_coefs.weight.data.clamp_(
                -self.config.altup_coef_clip,
                self.config.altup_coef_clip,
            )
        all_coefs = (
            self.prediction_coefs(modalities)
            .reshape(*modalities.shape[:-1], self.config.altup_num_inputs, self.config.altup_num_inputs)
            .permute(0, 1, 3, 2)
        )
        predictions = torch.matmul(hidden_states.permute(1, 2, 3, 0), all_coefs)
        predictions = predictions.permute(3, 0, 1, 2)
        predictions += hidden_states
        return predictions.contiguous().type_as(hidden_states)

    def correct(self, predictions, activated):
        modalities = self.compute_router_modalities(activated)
        innovation = activated - predictions[self.config.altup_active_idx]
        innovation = innovation.repeat(self.config.altup_num_inputs, 1, 1, 1)
        if self.config.altup_coef_clip is not None:
            self.correction_coefs.weight.data.clamp_(
                -self.config.altup_coef_clip,
                self.config.altup_coef_clip,
            )
        all_coefs = self.correction_coefs(modalities) + 1.0
        all_coefs = all_coefs.permute(2, 0, 1).unsqueeze(-1)
        corrected = torch.mul(innovation, all_coefs)
        corrected += predictions
        return corrected.contiguous().type_as(activated)

    def scale_corrected_output(self, corrected):
        return (corrected.type_as(self.correct_output_scale) * self.correct_output_scale).type_as(corrected)


class NeuronGemma3nDecoderLayer(nn.Module):
    def __init__(self, config: Gemma3nInferenceConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx]
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronGemma3nAttention(config)
        self.mlp = NeuronGemma3nTextMLP(config, layer_idx=layer_idx)
        self.input_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        self.altup = NeuronGemma3nTextAltUp(config)
        self.laurel = NeuronGemma3nTextLaurelBlock(config)
        dtype = config.neuron_config.torch_dtype
        self.per_layer_input_gate = _dense_linear(
            config.hidden_size,
            config.hidden_size_per_layer_input,
            dtype=dtype,
        )
        self.per_layer_projection = _dense_linear(
            config.hidden_size_per_layer_input,
            config.hidden_size,
            dtype=dtype,
        )
        self.post_per_layer_input_norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(
        self,
        hidden_states,
        per_layer_input,
        attention_mask=None,
        local_mask=None,
        position_ids=None,
        past_key_value=None,
        adapter_ids=None,
        **kwargs,
    ):
        mask = local_mask if self.attention_type == "sliding_attention" and local_mask is not None else attention_mask
        predictions = self.altup.predict(hidden_states)
        active_prediction = predictions[self.config.altup_active_idx]
        active_prediction_normed = self.input_layernorm(active_prediction)
        laurel_output = self.laurel(active_prediction_normed)

        attn_output = self.self_attn(
            hidden_states=active_prediction_normed,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            adapter_ids=adapter_ids,
            **kwargs,
        )
        attn = self.post_attention_layernorm(attn_output.hidden_states)
        attn_gated = active_prediction + attn
        attn_laurel = (attn_gated + laurel_output) / math.sqrt(2)

        attn_norm = self.pre_feedforward_layernorm(attn_laurel)
        attn_ffw = self.mlp(attn_norm)
        attn_ffw_norm = self.post_feedforward_layernorm(attn_ffw)
        attn_ffw_laurel_gated = attn_laurel + attn_ffw_norm
        corrected_predictions = self.altup.correct(predictions, attn_ffw_laurel_gated)

        first_prediction = corrected_predictions[self.config.altup_active_idx].clone()
        if self.config.altup_correct_scale:
            first_prediction = self.altup.scale_corrected_output(first_prediction)
        first_prediction = self.per_layer_input_gate(first_prediction)
        first_prediction = self.act_fn(first_prediction)
        first_prediction = torch.multiply(first_prediction, per_layer_input)
        first_prediction = self.per_layer_projection(first_prediction)
        first_prediction = self.post_per_layer_input_norm(first_prediction)
        corrected_predictions[1:] += first_prediction

        return (
            corrected_predictions,
            attn_output.present_key_value,
            attn_output.cos_cache,
            attn_output.sin_cache,
            None,
        )


class NeuronGemma3nSoftcapLMHead(nn.Module):
    def __init__(self, config: Gemma3nInferenceConfig, gather_output: bool):
        super().__init__()
        self.final_logit_softcapping = config.final_logit_softcapping
        self.proj = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            pad=True,
            gather_output=gather_output,
            dtype=config.neuron_config.torch_dtype,
        )
        if hasattr(self.proj, "pad_size"):
            self.pad_size = self.proj.pad_size
        if hasattr(self.proj, "gather_output"):
            self.gather_output = self.proj.gather_output
        if hasattr(self.proj, "tensor_parallel_group"):
            self.tensor_parallel_group = self.proj.tensor_parallel_group

    def forward(self, hidden_states):
        logits = self.proj(hidden_states)
        if self.final_logit_softcapping is not None:
            logits = torch.tanh(logits / self.final_logit_softcapping) * self.final_logit_softcapping
        return logits


class NeuronGemma3nTextModel(NeuronBaseModel):
    def setup_attr_for_model(self, config: Gemma3nInferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
        self.sliding_window = config.sliding_window

    def init_model(self, config: Gemma3nInferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
            sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        )
        self.embed_tokens_per_layer = ParallelEmbedding(
            config.vocab_size_per_layer_input,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
            sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        )
        dtype = config.neuron_config.torch_dtype
        self.per_layer_model_projection = _dense_linear(
            config.hidden_size,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            dtype=dtype,
        )
        self.per_layer_projection_norm = get_rmsnorm_cls()(
            config.hidden_size_per_layer_input,
            eps=config.rms_norm_eps,
        )
        self.altup_projections = nn.ModuleList(
            [
                _dense_linear(config.hidden_size, config.hidden_size, dtype=dtype)
                for _ in range(1, config.altup_num_inputs)
            ]
        )
        self.altup_unembed_projections = nn.ModuleList(
            [
                _dense_linear(config.hidden_size, config.hidden_size, dtype=dtype)
                for _ in range(1, config.altup_num_inputs)
            ]
        )
        self.register_buffer("per_layer_projection_scale", torch.tensor(config.hidden_size**-0.5), persistent=False)
        self.register_buffer("per_layer_input_scale", torch.rsqrt(torch.tensor(2.0)), persistent=False)
        self.register_buffer("token_embedding_scale", torch.tensor(config.hidden_size**0.5), persistent=False)
        self.register_buffer(
            "per_layer_embedding_scale",
            torch.tensor(config.hidden_size_per_layer_input**0.5),
            persistent=False,
        )
        updated_configs = get_updated_configs(config)
        self.layers = nn.ModuleList(
            [NeuronGemma3nDecoderLayer(conf, idx) for idx, conf in enumerate(updated_configs)]
        )
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = NeuronGemma3nSoftcapLMHead(config, gather_output=not self.on_device_sampling)

    def get_per_layer_inputs(self, input_ids):
        embeddings = self.embed_tokens_per_layer(input_ids)
        embeddings = embeddings * self.per_layer_embedding_scale.to(embeddings.dtype)
        return embeddings.reshape(
            *input_ids.shape,
            self.config.num_hidden_layers,
            self.config.hidden_size_per_layer_input,
        )

    def project_per_layer_inputs(self, inputs_embeds, per_layer_inputs=None):
        per_layer_projection = self.per_layer_model_projection(inputs_embeds)
        per_layer_projection *= self.per_layer_projection_scale.to(
            dtype=inputs_embeds.dtype,
            device=per_layer_projection.device,
        )
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.config.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)
        if per_layer_inputs is None:
            return per_layer_projection
        if per_layer_projection.shape != per_layer_inputs.shape:
            per_layer_inputs = per_layer_inputs[..., : self.config.num_hidden_layers, :]
        return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale.to(
            dtype=inputs_embeds.dtype,
            device=per_layer_projection.device,
        )

    def get_model_output(
        self,
        input_ids=None,
        seq_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        active_mask=None,
        inputs_embeds=None,
        adapter_ids=None,
        update_cache=False,
        is_for_context_encoding=False,
        local_attn_mask=None,
        **kwargs,
    ):
        batch_size, seq_length = input_ids.shape[:2]
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.token_embedding_scale.to(
                self.config.neuron_config.torch_dtype
            )
            per_layer_inputs = self.get_per_layer_inputs(input_ids)
        else:
            per_layer_inputs = None
        per_layer_inputs = self.project_per_layer_inputs(inputs_embeds, per_layer_inputs)

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        target_magnitude = torch.mean(inputs_embeds**2, dim=-1, keepdim=True) ** 0.5
        epsilon_tensor = torch.tensor(1e-5, device=inputs_embeds.device)
        temp_hidden_states = [inputs_embeds]
        for projection in self.altup_projections:
            altup_proj = projection(inputs_embeds)
            current_hidden_state = altup_proj.to(dtype=inputs_embeds.dtype, device=target_magnitude.device)
            new_magnitude = torch.mean(current_hidden_state**2, dim=-1, keepdim=True)
            new_magnitude = torch.sqrt(torch.maximum(new_magnitude, epsilon_tensor))
            current_hidden_state = current_hidden_state * target_magnitude / new_magnitude
            temp_hidden_states.append(current_hidden_state)
        hidden_states = torch.stack(temp_hidden_states, dim=0)

        next_decoder_cache = ()
        rope_caches = {}
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            cos_cache, sin_cache = rope_caches.get(decoder_layer.attention_type, (None, None))
            layer_outputs = decoder_layer(
                hidden_states,
                per_layer_inputs[:, :, decoder_layer.layer_idx, :],
                attention_mask=attention_mask,
                local_mask=local_attn_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                active_mask=active_mask,
                adapter_ids=adapter_ids,
                seq_ids=seq_ids,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
                **kwargs,
            )
            hidden_states = layer_outputs[0]
            next_decoder_cache += (layer_outputs[1],)
            rope_caches[decoder_layer.attention_type] = layer_outputs[2:4]

        target_magnitude = torch.mean(hidden_states[0] ** 2, dim=-1, keepdim=True) ** 0.5
        temp_hidden_states = [hidden_states[0]]
        for idx, projection in enumerate(self.altup_unembed_projections, start=1):
            altup_unembed_proj = projection(hidden_states[idx])
            current_hidden_state = altup_unembed_proj.to(
                dtype=inputs_embeds.dtype,
                device=target_magnitude.device,
            )
            new_magnitude = torch.mean(current_hidden_state**2, dim=-1, keepdim=True)
            new_magnitude = torch.sqrt(torch.maximum(new_magnitude, epsilon_tensor))
            current_hidden_state = current_hidden_state * target_magnitude / new_magnitude
            temp_hidden_states.append(current_hidden_state)
        hidden_states = torch.stack(temp_hidden_states)
        hidden_states = torch.mean(hidden_states, dim=0)
        hidden_states = self.norm(hidden_states)

        if update_cache and self.kv_mgr is not None:
            next_decoder_cache = self.kv_mgr.update_cache(
                is_for_context_encoding=is_for_context_encoding,
                seq_ids=seq_ids,
                position_ids=position_ids,
                new_key_values=next_decoder_cache,
                seq_len=self.n_positions,
                **kwargs,
            )

        return hidden_states, next_decoder_cache


class NeuronGemma3nForCausalLM(NeuronBaseForCausalLM):
    _model_cls = NeuronGemma3nTextModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        return Gemma3nForCausalLM.from_pretrained(model_path, **kwargs)

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
        prefixes = ("model.language_model.model.", "language_model.model.", "model.")
        for prefix in prefixes:
            if any(key.startswith(prefix) for key in state_dict):
                state_dict = {key.removeprefix(prefix): value for key, value in state_dict.items()}
                break

        if "lm_head.weight" in state_dict:
            state_dict["lm_head.proj.weight"] = state_dict.pop("lm_head.weight")

        neuron_config = config.neuron_config
        if neuron_config.vocab_parallel:
            state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0,
                neuron_config.local_ranks_size,
            )
            state_dict["embed_tokens_per_layer.rank_util.rank"] = torch.arange(
                0,
                neuron_config.local_ranks_size,
            )

        tp_degree = neuron_config.tp_degree
        for i in range(config.num_hidden_layers):
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0,
                tp_degree,
                dtype=torch.int32,
            )
            state_dict[f"layers.{i}.self_attn.q_layernorm.weight"] = state_dict[
                f"layers.{i}.self_attn.q_norm.weight"
            ].detach().clone()
            state_dict[f"layers.{i}.self_attn.k_layernorm.weight"] = state_dict[
                f"layers.{i}.self_attn.k_norm.weight"
            ].detach().clone()
            del state_dict[f"layers.{i}.self_attn.q_norm.weight"]
            del state_dict[f"layers.{i}.self_attn.k_norm.weight"]

            if neuron_config.fused_qkv:
                attr = "weight"
                state_dict[f"layers.{i}.self_attn.Wqkv.{attr}"] = torch.cat(
                    [
                        state_dict[f"layers.{i}.self_attn.q_proj.{attr}"],
                        state_dict[f"layers.{i}.self_attn.k_proj.{attr}"],
                        state_dict[f"layers.{i}.self_attn.v_proj.{attr}"],
                    ]
                )
                del state_dict[f"layers.{i}.self_attn.q_proj.{attr}"]
                del state_dict[f"layers.{i}.self_attn.k_proj.{attr}"]
                del state_dict[f"layers.{i}.self_attn.v_proj.{attr}"]

        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        return state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        state_dict["lm_head.proj.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        return Gemma3nInferenceConfig
