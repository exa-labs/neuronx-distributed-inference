# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group and The HuggingFace Inc. team. All rights reserved.
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
"""
PyTorch Qwen3.5 / Qwen3-Next hybrid model for NXD inference.

Qwen3.5 uses the Qwen3-Next architecture: a hybrid stack mixing
Gated DeltaNet (linear attention with a constant-size recurrent state)
and gated full attention layers, plus a high-sparsity MoE with a
shared expert.

Layer types come from ``config.layer_types`` ("linear_attention" or
"full_attention"). DeltaNet layers keep their own per-sequence state
(conv state + recurrent state) in module-level buffers indexed by
``seq_ids``; the standard KV cache manager continues to manage the
full-attention layers (DeltaNet layers emit zero-valued KV entries to
keep the cache layout uniform).

Numerics follow the HuggingFace ``qwen3_next`` reference implementation
(``transformers.models.qwen3_next.modeling_qwen3_next``) so that logits
can be validated against the HF CPU model.
"""

import gc
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.utils import set_tensor_model_parallel_attributes
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)

from neuronx_distributed_inference.models.config import InferenceConfig, MoENeuronConfig
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.models.model_wrapper import (
    DecoderModelInstance,
    ModelWrapper,
)
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module


def _get_tp_degree():
    if parallel_state.model_parallel_is_initialized():
        return parallel_state.get_tensor_model_parallel_size()
    return 1


class Qwen3_5RMSNorm(nn.Module):
    """Zero-centered RMSNorm used by Qwen3-Next: output = norm(x) * (1 + weight)."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states * (1.0 + self.weight.float())
        return hidden_states.to(input_dtype)


class Qwen3_5RMSNormGated(nn.Module):
    """RMSNorm followed by SiLU gating, applied to the DeltaNet core output."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


class Qwen3_5NeuronConfig(MoENeuronConfig):
    pass


class Qwen3_5InferenceConfig(InferenceConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # MoE module (moe_v2) expects these standardized attribute names.
        self.num_local_experts = self.num_experts
        # The Qwen3-Next shared expert has its own sigmoid gate, so it is
        # implemented in the decoder layer rather than via moe_v2 SharedExperts.
        self.n_shared_experts = 0
        # Router must be FP32 softmax for accuracy parity with HF.
        self.neuron_config.router_config.dtype = torch.float32
        self.neuron_config.router_config.act_fn = "softmax"
        self.neuron_config.normalize_top_k_affinities = bool(
            getattr(self, "norm_topk_prob", False)
        )

    def add_derived_config(self):
        self.num_cores_per_group = 1

    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "max_position_embeddings",
            "rope_theta",
            "rms_norm_eps",
            "hidden_act",
            "layer_types",
            "linear_num_value_heads",
            "linear_num_key_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_conv_kernel_dim",
            "num_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "shared_expert_intermediate_size",
            "norm_topk_prob",
            "partial_rotary_factor",
        ]

    @classmethod
    def get_neuron_config_cls(cls):
        return Qwen3_5NeuronConfig


def _is_moe_layer(config, layer_idx: int) -> bool:
    mlp_only_layers = getattr(config, "mlp_only_layers", None) or []
    decoder_sparse_step = getattr(config, "decoder_sparse_step", 1)
    return (
        layer_idx not in mlp_only_layers
        and config.num_experts > 0
        and (layer_idx + 1) % decoder_sparse_step == 0
    )


# ---------------------------------------------------------------------------
# Gated DeltaNet (linear attention)
# ---------------------------------------------------------------------------


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: Optional[torch.Tensor] = None,
):
    """Chunked parallel form of the gated delta rule (prefill path).

    Mirrors ``torch_chunk_gated_delta_rule`` from the HF qwen3_next reference.
    Inputs are [B, S, H, D]-shaped (query/key/value) and [B, S, H] (g/beta).
    Returns (core_attn_out [B, S, H, Dv], final_state [B, H, Dk, Dv]).
    """
    initial_dtype = query.dtype
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    padded_length = sequence_length + pad_size

    total_sequence_length = query.shape[-2]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = l2norm(query, dim=-1) * scale
    key = l2norm(key, dim=-1)

    # reshape into chunks, flattening (batch, heads) into one dim: rank-5
    # tensors here exceed neuronx-cc stride/vectorizer limits (NCC_IBCG901/IMGN901)
    bh = batch_size * num_heads
    query, key, value = [
        x.reshape(bh, -1, chunk_size, x.shape[-1]) for x in (query, key, value)
    ]
    g = g.reshape(bh, -1, chunk_size)
    beta = beta.reshape(bh, -1, chunk_size)
    num_chunks = total_sequence_length // chunk_size

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()

    k_beta = key * beta.unsqueeze(-1)
    v_beta = value * beta.unsqueeze(-1)

    # multiplicative masks instead of masked_fill: high-rank masked_fill selects
    # hit a neuronx-cc codegen stride limit (NCC_IBCG901)
    keep_strict_lower = torch.tril(
        torch.ones(chunk_size, chunk_size, dtype=decay_mask.dtype, device=query.device),
        diagonal=-1,
    )
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask) * keep_strict_lower
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    if initial_state is None:
        last_recurrent_state = torch.zeros(
            bh, k_head_dim, v_head_dim, dtype=torch.float32, device=query.device
        )
    else:
        last_recurrent_state = initial_state.to(torch.float32).reshape(bh, k_head_dim, v_head_dim)

    keep_lower = torch.tril(
        torch.ones(chunk_size, chunk_size, dtype=decay_mask.dtype, device=query.device),
        diagonal=0,
    )

    chunk_outs = []
    for i in range(num_chunks):
        q_i, k_i, v_i = query[:, i], key[:, i], value[:, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, i]) * keep_lower
        v_prime = (k_cumdecay[:, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, i, :, None].exp()) @ last_recurrent_state
        chunk_outs.append(attn_inter + attn @ v_new)
        g_last = g[:, i, -1, None, None].exp()
        k_decay = k_i * (g[:, i, -1, None] - g[:, i]).exp()[..., None]
        state_update = last_recurrent_state * 0
        for c in range(chunk_size):
            state_update = state_update + k_decay[:, c, :, None] * v_new[:, c, None, :]
        last_recurrent_state = last_recurrent_state * g_last + state_update

    core_attn_out = torch.cat(chunk_outs, dim=-2)
    core_attn_out = core_attn_out.reshape(batch_size, num_heads, -1, v_head_dim)
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    last_recurrent_state = last_recurrent_state.reshape(
        batch_size, num_heads, k_head_dim, v_head_dim
    )
    return core_attn_out, last_recurrent_state


def recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
):
    """Sequential (token-by-token) form of the gated delta rule (decode path).

    Mirrors ``torch_recurrent_gated_delta_rule`` from the HF qwen3_next
    reference. Inputs are [B, S, H, D]-shaped; returns
    (core_attn_out [B, S, H, Dv], final_state [B, H, Dk, Dv]).
    """
    initial_dtype = query.dtype
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]

    scale = 1 / (query.shape[-1] ** 0.5)
    query = l2norm(query, dim=-1) * scale
    key = l2norm(key, dim=-1)

    core_attn_out = torch.zeros(
        batch_size, num_heads, sequence_length, v_head_dim,
        dtype=torch.float32, device=query.device,
    )
    last_recurrent_state = initial_state.to(torch.float32)

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class NeuronQwen3_5GatedDeltaNet(nn.Module):
    """Gated DeltaNet linear-attention mixer with per-sequence on-module state.

    State layout (per layer):
      - ``conv_state``      [max_batch, conv_dim, kernel_size - 1]
      - ``recurrent_state`` [max_batch, num_v_heads * head_k_dim * head_v_dim]
        (stored flat: high-rank state buffers trip neuronx-cc vectorizer)
    Both are indexed by ``seq_ids`` so continuous batching maps each vLLM
    sequence to a fixed state slot, analogous to the KV cache manager.

    Tensor-parallel sharding splits linear key heads across ranks (value
    heads follow, since num_v_heads is a multiple of num_k_heads).
    """

    def __init__(self, config: Qwen3_5InferenceConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.kernel_size = config.linear_conv_kernel_dim
        self.layer_norm_epsilon = config.rms_norm_eps

        tp_degree = _get_tp_degree()
        assert self.num_k_heads % tp_degree == 0, (
            f"linear_num_key_heads ({self.num_k_heads}) must be divisible by "
            f"tp_degree ({tp_degree})"
        )
        self.local_num_k_heads = self.num_k_heads // tp_degree
        self.local_num_v_heads = self.num_v_heads // tp_degree

        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.local_key_dim = self.key_dim // tp_degree
        self.local_value_dim = self.value_dim // tp_degree
        self.local_conv_dim = self.local_key_dim * 2 + self.local_value_dim

        dtype = config.neuron_config.torch_dtype

        # Projections. Shard along the head dimension; the per-key-head
        # interleaved layout of in_proj_qkvz/in_proj_ba is preserved by the
        # weight conversion (see convert_qwen3_5_hf_to_neuron_state_dict).
        self.in_proj_qkvz = ColumnParallelLinear(
            self.hidden_size,
            self.key_dim * 2 + self.value_dim * 2,
            bias=False,
            gather_output=False,
            dtype=dtype,
        )
        self.in_proj_ba = ColumnParallelLinear(
            self.hidden_size,
            self.num_v_heads * 2,
            bias=False,
            gather_output=False,
            dtype=dtype,
        )
        self.out_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=dtype,
        )

        # Depthwise causal conv over the local (q, k, v) channels.
        self.conv1d = nn.Conv1d(
            in_channels=self.local_conv_dim,
            out_channels=self.local_conv_dim,
            kernel_size=self.kernel_size,
            groups=self.local_conv_dim,
            padding=self.kernel_size - 1,
            bias=False,
        )

        self.dt_bias = nn.Parameter(torch.ones(self.local_num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.local_num_v_heads))

        # Mark plain per-head parameters for TP-0-dim checkpoint sharding;
        # the conv channels are reordered into per-rank slabs by
        # convert_qwen3_5_hf_to_neuron_state_dict.
        for param in (self.conv1d.weight, self.dt_bias, self.A_log):
            set_tensor_model_parallel_attributes(param, True, 0, 1, num_partitions=tp_degree)

        self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)

        max_batch = config.neuron_config.max_batch_size
        # nn.Parameter (not buffer) so torch_neuronx input/output aliasing can
        # match them by data_ptr against named_parameters() during trace.
        self.conv_state = nn.Parameter(
            torch.zeros(max_batch, self.local_conv_dim, self.kernel_size - 1, dtype=torch.float32),
            requires_grad=False,
        )
        self.recurrent_state = nn.Parameter(
            torch.zeros(
                max_batch, self.local_num_v_heads * self.head_k_dim * self.head_v_dim,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        # Updated full-buffer states for the current forward; returned as extra
        # traced outputs and aliased back onto conv_state / recurrent_state.
        self.next_conv_state = None
        self.next_recurrent_state = None

    def fix_query_key_value_ordering(self, mixed_qkvz, mixed_ba):
        """Split the interleaved qkvz / ba projections into per-head tensors."""
        batch, seq = mixed_qkvz.shape[0], mixed_qkvz.shape[1]
        nvk = self.local_num_v_heads // self.local_num_k_heads
        new_tensor_shape_qkvz = (
            batch, seq, self.local_num_k_heads,
            2 * self.head_k_dim + 2 * self.head_v_dim * nvk,
        )
        new_tensor_shape_ba = (batch, seq, self.local_num_k_heads, 2 * nvk)

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)
        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            nvk * self.head_v_dim,
            nvk * self.head_v_dim,
        ]
        split_arg_list_ba = [nvk, nvk]
        query, key, value, z = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=3)
        b, a = torch.split(mixed_ba, split_arg_list_ba, dim=3)

        value = value.reshape(batch, seq, self.local_num_v_heads, self.head_v_dim)
        z = z.reshape(batch, seq, self.local_num_v_heads, self.head_v_dim)
        b = b.reshape(batch, seq, self.local_num_v_heads)
        a = a.reshape(batch, seq, self.local_num_v_heads)
        return query, key, value, z, b, a

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_ids: Optional[torch.Tensor] = None,
        is_for_context_encoding: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        if seq_ids is not None:
            seq_ids = seq_ids.to(torch.long)

        projected_states_qkvz = self.in_proj_qkvz(hidden_states)
        projected_states_ba = self.in_proj_ba(hidden_states)
        query, key, value, z, b, a = self.fix_query_key_value_ordering(
            projected_states_qkvz, projected_states_ba
        )
        query, key, value = (x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value))

        mixed_qkv = torch.cat((query, key, value), dim=-1)
        mixed_qkv = mixed_qkv.transpose(1, 2)  # [B, conv_dim, S]

        if is_for_context_encoding:
            # Fresh sequence: zero conv state. Run the full causal conv and
            # save the trailing (kernel_size - 1) inputs for decode.
            conv_out = self.conv1d(mixed_qkv)[:, :, :seq_len]
            mixed_qkv_post_conv = F.silu(conv_out)
            new_conv_state = F.pad(
                mixed_qkv.float(), (self.kernel_size - 1 - seq_len, 0)
            )[:, :, -(self.kernel_size - 1):]
        else:
            # Decode: shift cached conv inputs, append the new token.
            prev_conv_state = self.conv_state[seq_ids].to(mixed_qkv.dtype)
            conv_input = torch.cat([prev_conv_state, mixed_qkv], dim=-1)
            weight = self.conv1d.weight.squeeze(1)  # [conv_dim, kernel]
            conv_out = (conv_input * weight.unsqueeze(0)).sum(dim=-1, keepdim=True)
            mixed_qkv_post_conv = F.silu(conv_out)
            new_conv_state = conv_input[:, :, 1:].float()

        query, key, value = torch.split(
            mixed_qkv_post_conv,
            [self.local_key_dim, self.local_key_dim, self.local_value_dim],
            dim=1,
        )
        query = query.transpose(1, 2).reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.transpose(1, 2).reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.transpose(1, 2).reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if self.local_num_v_heads // self.local_num_k_heads > 1:
            query = query.repeat_interleave(self.local_num_v_heads // self.local_num_k_heads, dim=2)
            key = key.repeat_interleave(self.local_num_v_heads // self.local_num_k_heads, dim=2)

        if is_for_context_encoding:
            core_attn_out, new_recurrent_state = chunk_gated_delta_rule(
                query, key, value, g=g, beta=beta, initial_state=None
            )
        else:
            initial_state = self.recurrent_state[seq_ids].reshape(
                batch_size, -1, self.head_k_dim, self.head_v_dim
            )
            core_attn_out, new_recurrent_state = recurrent_gated_delta_rule(
                query, key, value, g=g, beta=beta, initial_state=initial_state
            )

        if seq_ids is not None:
            self.next_conv_state = self.conv_state.index_copy(0, seq_ids, new_conv_state)
            self.next_recurrent_state = self.recurrent_state.index_copy(
                0, seq_ids, new_recurrent_state.reshape(batch_size, -1).float()
            )
            if hidden_states.device.type != "xla":
                # Eager (CPU) execution: persist state directly. Under XLA
                # trace, states flow out as aliased outputs instead.
                self.conv_state.copy_(self.next_conv_state)
                self.recurrent_state.copy_(self.next_recurrent_state)

        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        return self.out_proj(core_attn_out)


# ---------------------------------------------------------------------------
# Gated full attention
# ---------------------------------------------------------------------------


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_partial_rotary_pos_emb(q, k, cos, sin):
    """Apply RoPE to the first ``rotary_dim`` channels only (partial rotary)."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


class NeuronQwen3_5Attention(nn.Module):
    """Qwen3-Next gated attention: fused query+gate projection, per-head-dim
    zero-centered QK RMSNorm before partial RoPE, and a sigmoid output gate."""

    def __init__(self, config: Qwen3_5InferenceConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        self.rotary_dim = int(self.head_dim * partial_rotary_factor)
        self.rope_theta = config.rope_theta
        attention_bias = getattr(config, "attention_bias", False)

        tp_degree = _get_tp_degree()
        self.local_num_heads = self.num_heads // tp_degree
        self.local_num_kv_heads = max(self.num_kv_heads // tp_degree, 1)

        dtype = config.neuron_config.torch_dtype

        # q_proj emits query and output-gate channels, interleaved per head.
        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_heads * self.head_dim * 2,
            bias=attention_bias,
            gather_output=False,
            dtype=dtype,
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=attention_bias,
            gather_output=False,
            dtype=dtype,
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=attention_bias,
            gather_output=False,
            dtype=dtype,
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=dtype,
        )

        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rope_cos_sin(self, position_ids: torch.Tensor, dtype: torch.dtype):
        freqs = position_ids[:, :, None].float() * self.inv_freq[None, None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        active_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        q = q.view(batch_size, seq_len, self.local_num_heads, self.head_dim * 2)
        query_states, gate = torch.chunk(q, 2, dim=-1)
        gate = gate.reshape(batch_size, seq_len, -1)

        key_states = self.k_proj(hidden_states).view(
            batch_size, seq_len, self.local_num_kv_heads, self.head_dim
        )
        value_states = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.local_num_kv_heads, self.head_dim
        )

        query_states = self.q_norm(query_states).transpose(1, 2)
        key_states = self.k_norm(key_states).transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = self._rope_cos_sin(position_ids, hidden_states.dtype)
        query_states, key_states = apply_partial_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        present_key_value = (key_states, value_states)

        n_rep = self.local_num_heads // self.local_num_kv_heads

        def _expand_kv(k, v):
            if n_rep == 1:
                return k, v
            return (
                k.repeat_interleave(n_rep, dim=1),
                v.repeat_interleave(n_rep, dim=1),
            )

        if past_key_value is None:
            # Context encoding: causal attention over the input.
            k_full, v_full = _expand_kv(key_states, value_states)
            scores = query_states @ k_full.transpose(-1, -2) / (self.head_dim ** 0.5)
            scores = torch.where(
                attention_mask, scores, torch.finfo(scores.dtype).min
            )
            probs = F.softmax(scores.float(), dim=-1).to(scores.dtype)
            attn_output = probs @ v_full
        else:
            # Token generation: attend over the cache (masked by
            # attention_mask) plus the current token (masked by active_mask).
            k_cache, v_cache = past_key_value
            k_cache, v_cache = _expand_kv(k_cache, v_cache)
            k_new, v_new = _expand_kv(key_states, value_states)
            scores_prior = query_states @ k_cache.transpose(-1, -2) / (self.head_dim ** 0.5)
            scores_prior = torch.where(
                attention_mask, scores_prior, torch.finfo(scores_prior.dtype).min
            )
            scores_active = query_states @ k_new.transpose(-1, -2) / (self.head_dim ** 0.5)
            if active_mask is not None:
                scores_active = torch.where(
                    active_mask, scores_active, torch.finfo(scores_active.dtype).min
                )
            scores = torch.cat([scores_prior, scores_active], dim=-1)
            probs = F.softmax(scores.float(), dim=-1).to(scores.dtype)
            attn_output = probs @ torch.cat([v_cache, v_new], dim=2)

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)

        return attn_output, present_key_value


# ---------------------------------------------------------------------------
# MLP / MoE
# ---------------------------------------------------------------------------


class NeuronQwen3_5MLP(nn.Module):
    """Standard SwiGLU MLP on parallel layers (used for the shared expert
    and for dense layers when MoE is disabled)."""

    def __init__(self, config: Qwen3_5InferenceConfig, intermediate_size: int):
        super().__init__()
        dtype = config.neuron_config.torch_dtype
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size, intermediate_size, bias=False, gather_output=False, dtype=dtype
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_size, intermediate_size, bias=False, gather_output=False, dtype=dtype
        )
        self.down_proj = RowParallelLinear(
            intermediate_size, config.hidden_size, bias=False, input_is_parallel=True, dtype=dtype
        )

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class NeuronQwen3_5SparseMoeBlock(nn.Module):
    """Routed experts (via NxD MoE) plus the Qwen3-Next shared expert with
    its sigmoid gate."""

    def __init__(self, config: Qwen3_5InferenceConfig):
        super().__init__()
        import copy

        moe_config = copy.deepcopy(config)
        moe_config.intermediate_size = config.moe_intermediate_size
        self.moe = initialize_moe_module(config=moe_config)
        self.shared_expert = NeuronQwen3_5MLP(
            config, config.shared_expert_intermediate_size
        )
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states):
        routed_out = self.moe(hidden_states)[0]
        shared_out = self.shared_expert(hidden_states)
        shared_gate = torch.sigmoid(self.shared_expert_gate(hidden_states))
        return routed_out + shared_gate * shared_out


# ---------------------------------------------------------------------------
# Decoder layer / model / application head
# ---------------------------------------------------------------------------


class NeuronQwen3_5DecoderLayer(nn.Module):
    """Hybrid decoder layer: DeltaNet or gated attention mixer + MoE/dense MLP.

    DeltaNet layers manage their own recurrent state and return zero-valued
    KV tensors so the KV cache manager's per-layer layout stays uniform.
    """

    def __init__(self, config: Qwen3_5InferenceConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.is_linear_attention = self.layer_type == "linear_attention"

        self.linear_attn = None
        if self.is_linear_attention:
            self.linear_attn = NeuronQwen3_5GatedDeltaNet(config)
        else:
            self.self_attn = NeuronQwen3_5Attention(config)

        if _is_moe_layer(config, layer_idx):
            self.mlp = NeuronQwen3_5SparseMoeBlock(config)
            self.is_moe = True
        else:
            self.mlp = NeuronQwen3_5MLP(config, config.intermediate_size)
            self.is_moe = False

        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        tp_degree = _get_tp_degree()
        self._kv_shape_per_token = (
            max(config.num_key_value_heads // tp_degree, 1),
            head_dim,
        )

    def _dummy_kv(self, batch_size, seq_len, device, dtype):
        kv_heads, head_dim = self._kv_shape_per_token
        zeros = torch.zeros(batch_size, kv_heads, seq_len, head_dim, device=device, dtype=dtype)
        return (zeros, zeros)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        seq_ids: Optional[torch.Tensor] = None,
        active_mask: Optional[torch.Tensor] = None,
        is_for_context_encoding: Optional[bool] = None,
        **kwargs,
    ):
        if is_for_context_encoding is None:
            is_for_context_encoding = past_key_value is None

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.is_linear_attention:
            hidden_states = self.linear_attn(
                hidden_states,
                seq_ids=seq_ids,
                is_for_context_encoding=is_for_context_encoding,
            )
            present_key_value = self._dummy_kv(
                hidden_states.shape[0],
                hidden_states.shape[1],
                hidden_states.device,
                hidden_states.dtype,
            )
        else:
            hidden_states, present_key_value = self.self_attn(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                active_mask=active_mask,
                **kwargs,
            )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        hidden_states = residual + hidden_states

        return (hidden_states, present_key_value, None, None, None)


class NeuronQwen3_5Model(NeuronBaseModel):
    """Qwen3.5 / Qwen3-Next hybrid base model on NeuronBaseModel."""

    def setup_attr_for_model(self, config: Qwen3_5InferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: Qwen3_5InferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
        )
        self.layers = nn.ModuleList(
            [NeuronQwen3_5DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            pad=True,
            gather_output=not self.on_device_sampling,
            dtype=config.neuron_config.torch_dtype,
        )

    def linear_attn_modules(self):
        return [layer.linear_attn for layer in self.layers if layer.linear_attn is not None]

    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        for la in self.linear_attn_modules():
            if la.next_conv_state is not None:
                outputs = outputs + [la.next_conv_state, la.next_recurrent_state]
        return outputs


def convert_qwen3_5_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
    """Convert a HF qwen3_next / qwen3_5 checkpoint to the Neuron layout.

    Attention and DeltaNet parameter names are preserved 1:1; only the MoE
    experts are restacked into the NxD MoE layout (router + stacked
    gate_up/down projections).
    """
    neuron_config = config.neuron_config

    state_dict["rank_util.rank"] = torch.arange(
        0, neuron_config.tp_degree, dtype=torch.int32
    )

    # Reorder DeltaNet conv channels from the HF global [q | k | v] layout
    # into per-rank slabs [q_0 k_0 v_0 | q_1 k_1 v_1 | ...] so that a
    # contiguous 0-dim TP shard hands each rank its local (q, k, v) channels.
    tp = neuron_config.tp_degree
    key_dim = config.linear_num_key_heads * config.linear_key_head_dim
    value_dim = config.linear_num_value_heads * config.linear_value_head_dim
    for l in range(config.num_hidden_layers):  # noqa: E741
        key = f"layers.{l}.linear_attn.conv1d.weight"
        if key not in state_dict or tp == 1:
            continue
        w = state_dict[key]
        q, k, v = torch.split(w, [key_dim, key_dim, value_dim], dim=0)
        qs = torch.chunk(q, tp, dim=0)
        ks = torch.chunk(k, tp, dim=0)
        vs = torch.chunk(v, tp, dim=0)
        state_dict[key] = torch.cat(
            [t for r in range(tp) for t in (qs[r], ks[r], vs[r])], dim=0
        ).contiguous()

    for l in range(config.num_hidden_layers):  # noqa: E741
        if not _is_moe_layer(config, l):
            continue
        prefix = f"layers.{l}.mlp"
        if f"{prefix}.gate.weight" not in state_dict:
            continue

        state_dict[f"{prefix}.moe.router.linear_router.weight"] = (
            state_dict.pop(f"{prefix}.gate.weight").detach().clone()
        )

        intermediate_size, hidden_size = state_dict[f"{prefix}.experts.0.gate_proj.weight"].shape
        dtype = state_dict[f"{prefix}.experts.0.gate_proj.weight"].dtype
        num_experts = config.num_experts

        gate_up_proj = torch.empty(
            num_experts, hidden_size, 2 * intermediate_size, dtype=dtype
        )
        down_proj = torch.empty(num_experts, intermediate_size, hidden_size, dtype=dtype)
        for e in range(num_experts):
            gate_up_proj[e, :, :intermediate_size] = state_dict.pop(
                f"{prefix}.experts.{e}.gate_proj.weight"
            ).T
            gate_up_proj[e, :, intermediate_size:] = state_dict.pop(
                f"{prefix}.experts.{e}.up_proj.weight"
            ).T
            down_proj[e] = state_dict.pop(f"{prefix}.experts.{e}.down_proj.weight").T
        state_dict[f"{prefix}.moe.expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_proj
        state_dict[f"{prefix}.moe.expert_mlps.mlp_op.down_proj.weight"] = down_proj

        gc.collect()

    return state_dict


class Qwen3_5ModelInstance(DecoderModelInstance):
    """DecoderModelInstance that also aliases DeltaNet conv/recurrent state
    buffers onto the extra traced outputs appended after the KV cache."""

    def get(self, bucket_rank, **kwargs):
        module, aliases = super().get(bucket_rank, **kwargs)
        next_idx = max(aliases.values()) + 1 if aliases else 1
        for la in module.linear_attn_modules():
            aliases[la.conv_state] = next_idx
            aliases[la.recurrent_state] = next_idx + 1
            next_idx += 2
        return module, aliases


class Qwen3_5ModelWrapper(ModelWrapper):
    """ModelWrapper that aliases DeltaNet conv/recurrent state buffers."""

    def get_model_instance(self):
        return Qwen3_5ModelInstance(
            model_cls=self.model_cls,
            config=self.config,
            **self.model_init_kwargs,
        )


class NeuronQwen3_5ForCausalLM(NeuronBaseForCausalLM):
    """Application head for Qwen3.5 / Qwen3-Next checkpoints."""

    _model_cls = NeuronQwen3_5Model

    def get_model_wrapper_cls(self):
        return Qwen3_5ModelWrapper

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        return convert_qwen3_5_hf_to_neuron_state_dict(state_dict, config)

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        return Qwen3_5InferenceConfig
