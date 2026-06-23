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
import os
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

# NKI kernel for the delta-rule recurrence.  Always use NKI when available —
# the recurrent XLA graph triggers PGTiling (NCC_IPCC901) at batch>=8.
try:
    from neuronx_distributed_inference.models.qwen3_5.nki_delta_rule import (
        nki_chunk_gated_delta_rule_kernel,
        nki_recurrent_gated_delta_rule,
        nki_recurrent_gated_delta_rule_decode,
        nki_recurrent_gated_delta_rule_decode_v2,
        nki_within_chunk_state_update,
    )
    _NKI_AVAILABLE = True
except ImportError:
    _NKI_AVAILABLE = False
# Legacy env-var toggle (kept for backward compat; defaults to True now)
_USE_NKI_DELTA_RULE = (
    _NKI_AVAILABLE
    and os.environ.get("QWEN35_USE_NKI_DELTA_RULE", "1") != "0"
)

# Replicate the DeltaNet projections across tensor-parallel ranks instead of
# sharding them.  The hybrid token-generation graph fails neuronx-cc PGTiling
# (NCC_IPCC901) at decode batch>8 because the DeltaNet RowParallel out_proj
# all-reduce and the full-attention o_proj all-reduce land in the same DAG with
# incompatible axis groups.  A pure full-attention model (no DeltaNet) compiles
# at batch=14.  Running the DeltaNet layers replicated (no collectives) leaves
# the full-attention all-reduces as the only collectives in the decode DAG,
# matching the pure-attention structure that PGTiling can tile.
_DELTANET_REPLICATED = os.environ.get("QWEN35_DELTANET_REPLICATED") == "1"

# Shard only the *prefill* (context-encoding) DeltaNet recurrence across ranks
# while keeping the projections replicated.  Replication clears the decode
# PGTiling wall but unshards the recurrent scan, so prefill pays a ~4x compute
# tax (22s vs ~6s at tp=4).  With this flag the expensive recurrence runs on a
# per-rank head slice and the result is all-gathered before the (replicated)
# out_proj -- recovering the sharded prefill speed -- while the decode graph
# stays collective-free (full recurrence on every rank), so the wall remains
# cleared.  Only meaningful together with QWEN35_DELTANET_REPLICATED=1.
_DELTANET_SHARD_PREFILL = os.environ.get("QWEN35_DELTANET_SHARD_PREFILL") == "1"

# Use the chunk-parallel form of the gated delta rule for prefill instead of the
# token-sequential recurrence.  The recurrence is O(seq_len) sequential steps and
# dominates prefill latency (~6s sharded / ~22s replicated for 7500 tokens).  The
# chunked form does O(seq_len / chunk) sequential chunk-steps, each a batch of
# dense matmuls, collapsing prefill toward parallel-attention cost.  The pure
# torch chunked form previously tripped neuronx-cc PGTiling, but that was in the
# RowParallel (out_proj all-reduce) layout; under QWEN35_DELTANET_SHARD_PREFILL
# the DeltaNet compute is collective-free (only benign all-gathers remain), so it
# is worth re-evaluating.  "torch" runs chunk_gated_delta_rule (let neuronx-cc
# compile it); "nki" runs the hand-written chunked NKI kernel that bypasses the
# XLA trace entirely.
_DELTANET_CHUNK_PREFILL = os.environ.get("QWEN35_DELTANET_CHUNK_PREFILL", "")

# Fully shard the DeltaNet projections across ranks (in_proj ColumnParallel) in
# BOTH prefill and decode, but drive the output through an all-gather of the
# core attention output followed by a *replicated* out_proj -- instead of the
# RowParallelLinear out_proj whose all-reduce trips the decode PGTiling wall.
# This is the principled decode lever: in replicated/shard-prefill mode every
# rank stores and reads the full ~1B DeltaNet projection weights, which dominate
# the memory-bandwidth-bound decode at tp>=4; sharding in_proj cuts the per-rank
# DeltaNet weight/HBM traffic by tp.  The all-gather is the same collective the
# shard-prefill path already places in the CTE DAG alongside the full-attention
# all-reduce (which compiles clean at seqs=14), so extending it to the decode
# DAG -- replacing the incompatible second all-reduce -- is expected to tile.
# Implies non-replicated projections; set QWEN35_DELTANET_REPLICATED=0 with it.
_DELTANET_SHARD_DECODE = os.environ.get("QWEN35_DELTANET_SHARD_DECODE") == "1"

# Within-chunk DeltaNet *prefill* state update in chunk_gated_delta_rule.  The
# increment ``sum_c outer(k_decay[c], v_new[c]) == k_decay^T @ v_new`` is a
# batched transposed matmul.  The HF reference uses the matmul, but the bare
# matmul crashes neuronx-cc TensorEngine codegen (NCC_INLA001) on the flattened
# [bh, chunk, D] shape, so the default "loop" keeps the unrolled rank-1
# accumulation that always lowers.  That loop is ``chunk_size`` sequential steps
# per chunk, so its depth scales with sequence length and dominates single-chip
# (tp2) prefill latency -- the matmul/einsum forms collapse it to one dense op.
# All variants are bit-near-exact (CPU rel-err ~1e-7); this selector lets one
# image build A/B the lowerings on hardware.  Values: "loop" (default, safe),
# "matmul", "matmul_contig", "einsum", "bmm", and "nki".  The "nki" form emits
# the contraction as a hand-written NKI TensorEngine matmul
# (``nki_within_chunk_state_update``); unlike every XLA-lowered torch form (all
# of which crash neuronx-cc with NCC_INLA001) it compiles, collapsing the
# 64-step loop to one dense matmul per chunk while staying bit-near-exact.
_DELTANET_STATE_UPDATE = os.environ.get("QWEN35_DELTANET_STATE_UPDATE", "loop")

# Decode kernel selection.  "nki" (default) dispatches through the NKI recurrent
# kernel (nki_recurrent_gated_delta_rule) which is always safe at any batch size
# but carries per-invocation overhead (tensor reshapes + NKI trace).  "torch" uses
# the pure-torch single-step form (no explicit loop at seq_len=1) — this produces
# a simpler XLA graph with fused elementwise ops and eliminates the NKI wrapper
# overhead.  The torch path originally tripped PGTiling at batch>=8 because of the
# RowParallel out_proj all-reduce; with shard_decode (all-gather + replicated
# out_proj) or replicated DeltaNet, the DAG is collective-free and may tile.
# "nki" (default) = v1 kernel; "nki_v2" = optimized v2 (host-precomputed
# exp_g broadcast + k*beta fusion — fewer DMA + instructions per head);
# "torch" = pure-torch single-step (may crash neuronx-cc on big graphs).
_DELTANET_DECODE_KERNEL = os.environ.get("QWEN35_DELTANET_DECODE_KERNEL", "nki_v2")

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
    SPMDRank,
)
from neuronx_distributed.parallel_layers.mappings import _gather_along_dim

from neuronx_distributed_inference.models.config import InferenceConfig, MoENeuronConfig
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.models.model_wrapper import (
    CONTEXT_ENCODING_MODEL_TAG,
    TOKEN_GENERATION_MODEL_TAG,
    DecoderModelInstance,
    ModelWrapper,
)
from neuronx_distributed_inference.modules.attention.utils import manual_softmax
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


def _within_chunk_state_update(
    k_decay: torch.Tensor, v_new: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    """Within-chunk DeltaNet state increment ``sum_c outer(k_decay[:, c], v_new[:, c])``.

    Equivalent to the batched transposed matmul ``k_decay^T @ v_new`` producing
    ``[bh, Dk, Dv]``.  ``k_decay``/``v_new`` are the i-th chunk tensors shaped
    ``[bh, chunk_size, D]``.  The formulation is env-selectable
    (``QWEN35_DELTANET_STATE_UPDATE``): the bare matmul matches the HF reference
    and collapses the update to one dense op, but crashes neuronx-cc on some
    toolchains (NCC_INLA001); the "loop" fallback always lowers but serialises
    ``chunk_size`` rank-1 updates per chunk (the single-chip prefill bottleneck).
    All variants are bit-near-exact.
    """
    if _DELTANET_STATE_UPDATE == "nki" and _NKI_AVAILABLE:
        # Hand-written NKI TensorEngine matmul: the only form of the contraction
        # that compiles on neuronx-cc (all XLA-lowered torch forms hit
        # NCC_INLA001).  Collapses the default 64-step rank-1 loop into one dense
        # matmul per chunk, the principled single-chip (tp2) prefill win.  Inputs
        # are made contiguous so the NKI MLIR frontend can resolve their static
        # shapes during in-model tracing (the per-chunk slices ``k_i``/``v_new``
        # are non-contiguous views, which otherwise fail name resolution).
        return nki_within_chunk_state_update(k_decay.contiguous(), v_new.contiguous())
    if _DELTANET_STATE_UPDATE == "matmul":
        return k_decay.transpose(-1, -2) @ v_new
    if _DELTANET_STATE_UPDATE == "matmul_contig":
        return k_decay.transpose(-1, -2).contiguous() @ v_new.contiguous()
    if _DELTANET_STATE_UPDATE == "einsum":
        return torch.einsum("bck,bcv->bkv", k_decay, v_new)
    if _DELTANET_STATE_UPDATE == "bmm":
        return torch.bmm(k_decay.transpose(1, 2).contiguous(), v_new.contiguous())
    if _DELTANET_STATE_UPDATE == "outer_sum":
        # broadcast outer products then reduce over the chunk dim.  Unlike the
        # matmul/einsum/bmm forms (which all lower to the same transposed dot
        # that crashes neuronx-cc with NCC_INLA001), this emits a broadcast
        # multiply + reduce_sum -- a different HLO that may compile while
        # remaining bit-near-exact and collapsing the chunk loop to one op.
        return (k_decay.unsqueeze(-1) * v_new.unsqueeze(-2)).sum(dim=1)
    if _DELTANET_STATE_UPDATE in ("matmul_tiled", "outer_tiled"):
        # The matmul/einsum/bmm/outer_sum forms all crash neuronx-cc (NCC_INLA001
        # TPB_TENSOR2D) producing the full [bh, Dk, Dv] = [bh, 128, 128] state
        # tile by contracting the 64-chunk dim.  Split the Dv (free) output dim
        # so each contraction emits a narrower [bh, Dk, Dv/n] tile, which may
        # dodge the static-pattern assignment while still collapsing the 64-step
        # loop into a handful of dense ops.  Math is identical (column split of
        # the same product), so this stays bit-near-exact.
        n_tiles = 2
        dv = v_new.shape[-1]
        step = (dv + n_tiles - 1) // n_tiles
        tiles = []
        if _DELTANET_STATE_UPDATE == "matmul_tiled":
            k_t = k_decay.transpose(-1, -2)  # [bh, Dk, chunk]
            for start in range(0, dv, step):
                tiles.append(k_t @ v_new[..., start : start + step])
        else:
            k_e = k_decay.unsqueeze(-1)  # [bh, chunk, Dk, 1]
            for start in range(0, dv, step):
                v_e = v_new[..., start : start + step].unsqueeze(-2)
                tiles.append((k_e * v_e).sum(dim=1))
        return torch.cat(tiles, dim=-1)
    # default "loop": unrolled rank-1 accumulation -- always lowers cleanly.
    state_update = k_decay.new_zeros(
        k_decay.shape[0], k_decay.shape[-1], v_new.shape[-1]
    )
    for c in range(chunk_size):
        state_update = state_update + k_decay[:, c, :, None] * v_new[:, c, None, :]
    return state_update


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
        state_update = _within_chunk_state_update(k_decay, v_new, chunk_size)
        last_recurrent_state = last_recurrent_state * g_last + state_update

    core_attn_out = torch.cat(chunk_outs, dim=-2)
    core_attn_out = core_attn_out.reshape(batch_size, num_heads, -1, v_head_dim)
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    last_recurrent_state = last_recurrent_state.reshape(
        batch_size, num_heads, k_head_dim, v_head_dim
    )
    return core_attn_out, last_recurrent_state


def nki_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: Optional[torch.Tensor] = None,
):
    """Chunk-parallel gated delta rule with the inter-chunk recurrence in NKI.

    Same interface and result as ``chunk_gated_delta_rule`` (bit-near-exact, CPU
    rel-err ~2e-7), but the sequential chunk loop -- the prefill bottleneck on a
    single chip (tp2) -- runs as a hand-written NKI kernel instead of XLA-traced
    torch.  Torch precomputes every *state-independent* quantity (the UT
    transform, ``decay_mask``, ``value = attn @ v_beta``, ``k_cumdecay``, the
    intra-chunk attention ``attn_intra``, ``q*exp(g)``, ``k_decay``, the per-chunk
    gate ``exp(g_last)``) exactly as the reference does; the NKI kernel then runs
    the ~``num_chunks`` sequential steps, each five TensorEngine matmuls, keeping
    the [Dk, Dv] state resident in SBUF.  This collapses the reference's only
    neuronx-cc-compilable within-chunk form (the ``chunk_size``-step rank-1 loop,
    whose sequential depth scales with seq-len) to one dense matmul per chunk.

    The kernel contracts over the partition dim (``nc_matmul(out, A, B) =
    A^T @ B``), so the stationary operands of the three matmuls that contract a
    non-chunk axis (``k_cumdecay``, ``q*exp(g)``, ``attn_intra``) are
    pre-transposed here in torch; ``k_decay`` and ``value`` already contract the
    chunk axis and stay in natural layout.

    Inputs: [B, S, H, D] shaped (query/key/value), [B, S, H] (g/beta); returns
    (core_attn_out [B, S, H, Dv], final_state [B, H, Dk, Dv]).
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

    total_sequence_length = query.shape[-2]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = l2norm(query, dim=-1) * scale
    key = l2norm(key, dim=-1)

    bh = batch_size * num_heads
    query, key, value = [
        x.reshape(bh, -1, chunk_size, x.shape[-1]) for x in (query, key, value)
    ]
    g = g.reshape(bh, -1, chunk_size)
    beta = beta.reshape(bh, -1, chunk_size)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()

    k_beta = key * beta.unsqueeze(-1)
    v_beta = value * beta.unsqueeze(-1)

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

    keep_lower = torch.tril(
        torch.ones(chunk_size, chunk_size, dtype=decay_mask.dtype, device=query.device),
        diagonal=0,
    )

    # State-independent per-chunk quantities consumed by the NKI recurrence.
    attn_intra = (query @ key.transpose(-1, -2) * decay_mask) * keep_lower
    qg = query * g[..., None].exp()
    g_last = g[:, :, -1].exp()
    k_decay = key * (g[:, :, -1:] - g).exp()[..., None]

    if initial_state is None:
        init_state = torch.zeros(
            bh, k_head_dim, v_head_dim, dtype=torch.float32, device=query.device
        )
    else:
        init_state = initial_state.to(torch.float32).reshape(bh, k_head_dim, v_head_dim)

    # Pre-transpose the stationary operands whose contraction axis is not the
    # chunk dim so the NKI nc_matmul contracts over the partition (first) dim.
    k_cumdecay_t = k_cumdecay.transpose(-1, -2).contiguous()
    qg_t = qg.transpose(-1, -2).contiguous()
    attn_intra_t = attn_intra.transpose(-1, -2).contiguous()
    value = value.contiguous()
    k_decay = k_decay.contiguous()
    g_last = g_last.contiguous()

    core_flat, final_state_flat = nki_chunk_gated_delta_rule_kernel(
        value, k_cumdecay_t, qg_t, attn_intra_t, k_decay, g_last, init_state
    )

    core_attn_out = core_flat.reshape(batch_size, num_heads, total_sequence_length, v_head_dim)
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    final_state = final_state_flat.reshape(batch_size, num_heads, k_head_dim, v_head_dim)
    return core_attn_out, final_state


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


def nki_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
):
    """NKI-accelerated gated delta rule (decode and CTE).

    Same interface as recurrent_gated_delta_rule but dispatches to the
    custom NKI kernel that bypasses XLA trace and PGTiling. Handles the
    input layout transformation (transpose q/k to column-major) expected
    by the NKI kernel.

    Inputs: [B, S, H, D] shaped (same as recurrent_gated_delta_rule).

    For decode (S=1), uses a specialized layout path that avoids the
    transpose+contiguous chain — the seq-dim-1 tensors reshape directly
    to the NKI kernel layout without data copies.
    """
    initial_dtype = query.dtype
    batch_size = query.shape[0]
    sequence_length = query.shape[1]
    num_heads = query.shape[2]
    k_head_dim = query.shape[3]
    v_head_dim = value.shape[3]
    bh = batch_size * num_heads

    if sequence_length == 1:
        # ── Decode fast-path (seq_len=1) ──────────────────────────────────
        # Skip transpose+contiguous: [B,1,H,D] → squeeze → [B,H,D] → reshape
        # to [BH,D].  For S=1, the transpose [B,S,H,D]→[B,H,S,D] is a
        # stride-only change and the subsequent reshape to [BH,1,D] / [BH,D,1]
        # can be done without a copy.
        q_f32 = query.squeeze(1).to(torch.float32).reshape(bh, k_head_dim)
        k_f32 = key.squeeze(1).to(torch.float32).reshape(bh, k_head_dim)
        v_f32 = value.squeeze(1).to(torch.float32).reshape(bh, v_head_dim)

        scale = 1 / (k_head_dim ** 0.5)
        q_f32 = l2norm(q_f32, dim=-1) * scale
        k_f32 = l2norm(k_f32, dim=-1)

        # exp(g): [B,1,H,1] → squeeze → [B,H] → reshape → [BH]
        exp_g_flat = g.squeeze(1).squeeze(-1).to(torch.float32).exp().reshape(bh)
        # beta: [B,1,H,1] → squeeze → [B,H] → reshape → [BH]
        beta_flat = beta.squeeze(1).squeeze(-1).to(torch.float32).reshape(bh)

        q_col = q_f32.unsqueeze(-1).contiguous()  # [BH, Dk, 1]
        k_col = k_f32.unsqueeze(-1).contiguous()  # [BH, Dk, 1]
        state_flat = initial_state.reshape(bh, k_head_dim, v_head_dim).contiguous()

        if _DELTANET_DECODE_KERNEL == "nki_v2":
            # v2: host-precomputed exp_g broadcast + k*beta fusion
            # exp_g_bc: [BH, Dk] — same scalar repeated Dk times per head
            exp_g_bc = exp_g_flat.unsqueeze(-1).expand(-1, k_head_dim).contiguous()
            # k_beta_row: [BH, 1, Dk] — k * beta for outer product
            k_beta_row = (k_f32 * beta_flat.unsqueeze(-1)).unsqueeze(1).contiguous()
            v_row = v_f32.unsqueeze(1).contiguous()  # [BH, 1, Dv]

            out_flat, final_state_flat = nki_recurrent_gated_delta_rule_decode_v2(
                q_col, k_col, k_beta_row, v_row, exp_g_bc, state_flat
            )
        else:
            # v1: original kernel with in-kernel exp_g broadcast
            k_row = k_f32.unsqueeze(1).contiguous()   # [BH, 1, Dk]
            v_row = v_f32.unsqueeze(1).contiguous()   # [BH, 1, Dv]
            exp_g_2d = exp_g_flat.unsqueeze(-1).contiguous()  # [BH, 1]
            beta_2d = beta_flat.unsqueeze(-1).contiguous()    # [BH, 1]

            out_flat, final_state_flat = nki_recurrent_gated_delta_rule_decode(
                q_col, k_col, k_row, v_row, exp_g_2d, beta_2d, state_flat
            )

        # Output: [BH, 1, Dv] → [B, 1, H, Dv]
        core_attn_out = out_flat.reshape(
            batch_size, num_heads, 1, v_head_dim
        ).transpose(1, 2).contiguous().to(initial_dtype)
        final_state = final_state_flat.reshape(batch_size, num_heads, k_head_dim, v_head_dim)
        return core_attn_out, final_state

    # ── General path (seq_len > 1, used for CTE) ─────────────────────────
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    scale = 1 / (query.shape[-1] ** 0.5)
    query = l2norm(query, dim=-1) * scale
    key = l2norm(key, dim=-1)

    # Pre-compute exp(g) — the NKI kernel expects it pre-computed
    exp_g = g.exp().squeeze(-1)  # [B, H, T]

    # Reshape for NKI: merge batch and heads → BH
    # q, k: [B, H, T, D] → [BH, D, T] (column-major for [128,1] loads)
    q_col = query.reshape(bh, sequence_length, k_head_dim).transpose(1, 2).contiguous()
    k_col = key.reshape(bh, sequence_length, k_head_dim).transpose(1, 2).contiguous()
    # v: [B, H, T, D] → [BH, T, D] (row-major for [1,256] loads)
    v_row = value.reshape(bh, sequence_length, v_head_dim).contiguous()
    # exp_g: [B, H, T] → [BH, T]
    exp_g_flat = exp_g.reshape(bh, sequence_length).contiguous()
    # beta: [B, H, T, 1] → [BH, T]
    beta_flat = beta.reshape(bh, sequence_length, 1).squeeze(-1).contiguous()
    # state: [B, H, Dk, Dv] → [BH, Dk, Dv]
    state_flat = initial_state.reshape(bh, k_head_dim, v_head_dim).contiguous()

    # Call NKI kernel
    out_flat, final_state_flat = nki_recurrent_gated_delta_rule(
        q_col, k_col, v_row, exp_g_flat, beta_flat, state_flat
    )

    # Reshape outputs back: [BH, T, D] → [B, S, H, D]
    core_attn_out = out_flat.reshape(
        batch_size, num_heads, sequence_length, v_head_dim
    ).transpose(1, 2).contiguous().to(initial_dtype)
    final_state = final_state_flat.reshape(batch_size, num_heads, k_head_dim, v_head_dim)

    return core_attn_out, final_state


def _prefill_gated_delta_rule(query, key, value, g, beta, initial_state):
    """Dispatch the prefill (context-encoding) gated delta rule.

    Selection (env ``QWEN35_DELTANET_CHUNK_PREFILL``):
      ``"torch"`` -> ``chunk_gated_delta_rule`` (chunk-parallel, XLA-traced and
        compiled by neuronx-cc).
      ``"nki"``   -> ``nki_chunk_gated_delta_rule`` (chunk-parallel hand-written
        NKI kernel, bypasses the XLA trace / PGTiling).
    Otherwise the token-sequential recurrence is used: the NKI recurrent kernel
    when available, else the pure-torch reference.

    All implementations share the interface ``(query,key,value,g,beta,
    initial_state) -> (core_attn_out [B,S,H,Dv], final_state [B,H,Dk,Dv])``.
    """
    if _DELTANET_CHUNK_PREFILL == "torch":
        return chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta, initial_state=initial_state
        )
    if _DELTANET_CHUNK_PREFILL == "nki" and _NKI_AVAILABLE:
        return nki_chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta, initial_state=initial_state
        )
    if _USE_NKI_DELTA_RULE and _NKI_AVAILABLE:
        return nki_gated_delta_rule(
            query, key, value, g=g, beta=beta, initial_state=initial_state
        )
    return recurrent_gated_delta_rule(
        query, key, value, g=g, beta=beta, initial_state=initial_state
    )


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
        # When replicating the DeltaNet across ranks, the per-rank ("local")
        # head/channel counts equal the global counts: every rank holds the
        # full projections and runs the full recurrence with no collectives.
        # shard_decode fully shards in_proj (ColumnParallel) but routes the
        # output through an all-gather + replicated out_proj, so it is mutually
        # exclusive with the replicated layout (it implies non-replicated).
        self.shard_decode = _DELTANET_SHARD_DECODE
        self.replicated = _DELTANET_REPLICATED and not self.shard_decode
        shard_degree = 1 if self.replicated else tp_degree
        assert self.num_k_heads % shard_degree == 0, (
            f"linear_num_key_heads ({self.num_k_heads}) must be divisible by "
            f"shard_degree ({shard_degree})"
        )
        self.local_num_k_heads = self.num_k_heads // shard_degree
        self.local_num_v_heads = self.num_v_heads // shard_degree

        # Prefill-only recurrence sharding (requires replicated projections).
        # The per-rank value-head count for the context-encoding recurrence;
        # equals the global count unless prefill sharding is enabled.
        self.shard_prefill = _DELTANET_SHARD_PREFILL and self.replicated
        self.cte_shard_degree = tp_degree if self.shard_prefill else 1
        assert self.num_v_heads % self.cte_shard_degree == 0, (
            f"linear_num_value_heads ({self.num_v_heads}) must be divisible by "
            f"cte_shard_degree ({self.cte_shard_degree})"
        )
        self.cte_local_num_v_heads = self.num_v_heads // self.cte_shard_degree

        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.local_key_dim = self.key_dim // shard_degree
        self.local_value_dim = self.value_dim // shard_degree
        self.local_conv_dim = self.local_key_dim * 2 + self.local_value_dim

        dtype = config.neuron_config.torch_dtype

        # Projections. Sharded along the head dimension by default; the
        # per-key-head interleaved layout of in_proj_qkvz/in_proj_ba is
        # preserved by the weight conversion (see
        # convert_qwen3_5_hf_to_neuron_state_dict).  In replicated mode they are
        # plain nn.Linear so NxD's checkpoint sharder leaves them full-size on
        # every rank and they emit no tensor-parallel collectives.
        if self.replicated:
            self.in_proj_qkvz = nn.Linear(
                self.hidden_size, self.key_dim * 2 + self.value_dim * 2,
                bias=False, dtype=dtype,
            )
            self.in_proj_ba = nn.Linear(
                self.hidden_size, self.num_v_heads * 2, bias=False, dtype=dtype,
            )
            self.out_proj = nn.Linear(
                self.value_dim, self.hidden_size, bias=False, dtype=dtype,
            )
        else:
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
            if self.shard_decode:
                # in_proj is sharded (ColumnParallel) but the per-rank core
                # attention output is all-gathered to the full value_dim before
                # out_proj, so out_proj is a *replicated* nn.Linear over the full
                # value_dim (no RowParallel all-reduce -> no decode PGTiling wall).
                self.out_proj = nn.Linear(
                    self.value_dim, self.hidden_size, bias=False, dtype=dtype,
                )
            else:
                self.out_proj = RowParallelLinear(
                    self.value_dim,
                    self.hidden_size,
                    bias=False,
                    input_is_parallel=True,
                    dtype=dtype,
                )

        # Depthwise causal conv plus the per-head dt_bias / A_log gates.
        # NxD's checkpoint sharder only shards parameters owned by parallel
        # layer classes, so these plain parameters are kept full-size
        # (replicated on every rank) and sliced to the local channels/heads at
        # forward time via the SPMD rank. The conv channels are reordered into
        # per-rank [q | k | v] slabs by convert_qwen3_5_hf_to_neuron_state_dict
        # so the local slice is contiguous.
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.kernel_size,
            groups=self.conv_dim,
            padding=self.kernel_size - 1,
            bias=False,
        )

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))
        self.rank_util = SPMDRank(world_size=tp_degree)

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
        # NOTE: torch.split / torch.chunk miscompile on Trainium (neuronx-cc
        # produces wrong data for multi-output splits of a shared base in this
        # graph), and plain strided slices trip a PGTiling compiler assert at
        # tp=1; index_select with constant indices avoids both.
        dk, dv = self.head_k_dim, nvk * self.head_v_dim
        dev = mixed_qkvz.device
        idx = torch.arange(2 * dk + 2 * dv, device=dev)
        query = mixed_qkvz.index_select(-1, idx[:dk])
        key = mixed_qkvz.index_select(-1, idx[dk:2 * dk])
        value = mixed_qkvz.index_select(-1, idx[2 * dk:2 * dk + dv])
        z = mixed_qkvz.index_select(-1, idx[2 * dk + dv:2 * dk + 2 * dv])
        b = mixed_ba.index_select(-1, idx[:nvk])
        a = mixed_ba.index_select(-1, idx[nvk:2 * nvk])

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

        # Local slices of the replicated conv / per-head parameters
        # (conv channels are stored in per-rank slabs; see weight conversion).
        # In replicated mode the local dims equal the global dims, so every
        # rank owns the full conv/head ranges (slice from offset 0).
        if self.replicated:
            rank = torch.zeros((), dtype=torch.long, device=hidden_states.device)
        else:
            rank = self.rank_util.get_rank().to(torch.long)
        conv_idx = rank * self.local_conv_dim + torch.arange(
            self.local_conv_dim, device=hidden_states.device
        )
        head_idx = rank * self.local_num_v_heads + torch.arange(
            self.local_num_v_heads, device=hidden_states.device
        )
        conv_weight = self.conv1d.weight.index_select(0, conv_idx)
        dt_bias = self.dt_bias.index_select(0, head_idx)
        A_log = self.A_log.index_select(0, head_idx)

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
            conv_out = F.conv1d(
                mixed_qkv,
                conv_weight,
                groups=self.local_conv_dim,
                padding=self.kernel_size - 1,
            )[:, :, :seq_len]
            mixed_qkv_post_conv = F.silu(conv_out)
            new_conv_state = F.pad(
                mixed_qkv.float(), (self.kernel_size - 1 - seq_len, 0)
            )[:, :, -(self.kernel_size - 1):]
        else:
            # Decode: shift cached conv inputs, append the new token.
            prev_conv_state = self.conv_state[seq_ids].to(mixed_qkv.dtype)
            conv_input = torch.cat([prev_conv_state, mixed_qkv], dim=-1)
            weight = conv_weight.squeeze(1)  # [local_conv_dim, kernel]
            conv_out = (conv_input * weight.unsqueeze(0)).sum(dim=-1, keepdim=True)
            mixed_qkv_post_conv = F.silu(conv_out)
            new_conv_state = conv_input[:, :, 1:].float()

        kd, vd = self.local_key_dim, self.local_value_dim
        query = mixed_qkv_post_conv[:, :kd]
        key = mixed_qkv_post_conv[:, kd:2 * kd]
        value = mixed_qkv_post_conv[:, 2 * kd:2 * kd + vd]
        query = query.transpose(1, 2).reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.transpose(1, 2).reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.transpose(1, 2).reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)

        if self.local_num_v_heads // self.local_num_k_heads > 1:
            query = query.repeat_interleave(self.local_num_v_heads // self.local_num_k_heads, dim=2)
            key = key.repeat_interleave(self.local_num_v_heads // self.local_num_k_heads, dim=2)

        if is_for_context_encoding and self.shard_prefill:
            # CTE with prefill sharding: the recurrent scan is the dominant
            # prefill cost, so run it on a per-rank value-head slice and
            # all-gather the slices back to the full head range before the
            # (replicated) norm/out_proj.  The all-gather is the only
            # collective and lives solely in this context-encoding graph, so
            # the decode graph stays collective-free and the PGTiling wall
            # remains cleared.
            h = self.cte_local_num_v_heads
            prefill_rank = self.rank_util.get_rank().to(torch.long)
            hidx = prefill_rank * h + torch.arange(h, device=query.device)
            q_local = query.index_select(2, hidx)
            k_local = key.index_select(2, hidx)
            v_local = value.index_select(2, hidx)
            g_local = g.index_select(2, hidx)
            beta_local = beta.index_select(2, hidx)
            initial_state = torch.zeros(
                batch_size, h, self.head_k_dim, self.head_v_dim,
                dtype=torch.float32, device=query.device,
            )
            core_local, rstate_local = _prefill_gated_delta_rule(
                q_local, k_local, v_local, g=g_local, beta=beta_local,
                initial_state=initial_state,
            )
            # core_local: [B, S, h, head_v]; rstate_local: [B, h, head_k, head_v]
            core_attn_out = _gather_along_dim(core_local, partition_dim=2)
            new_recurrent_state = _gather_along_dim(rstate_local, partition_dim=1)
        elif is_for_context_encoding:
            # CTE (context encoding) for DeltaNet layers.  The implementation is
            # selected by _prefill_gated_delta_rule: chunk-parallel (torch or
            # NKI) when QWEN35_DELTANET_CHUNK_PREFILL is set, else the
            # token-sequential recurrence (NKI kernel / torch reference).
            initial_state = torch.zeros(
                batch_size, query.shape[2], self.head_k_dim, self.head_v_dim,
                dtype=torch.float32, device=query.device,
            )
            core_attn_out, new_recurrent_state = _prefill_gated_delta_rule(
                query, key, value, g=g, beta=beta, initial_state=initial_state
            )
        else:
            initial_state = self.recurrent_state[seq_ids].reshape(
                batch_size, -1, self.head_k_dim, self.head_v_dim
            )
            if _DELTANET_DECODE_KERNEL == "torch":
                core_attn_out, new_recurrent_state = recurrent_gated_delta_rule(
                    query, key, value, g=g, beta=beta, initial_state=initial_state
                )
            elif _USE_NKI_DELTA_RULE and _NKI_AVAILABLE:
                core_attn_out, new_recurrent_state = nki_gated_delta_rule(
                    query, key, value, g=g, beta=beta, initial_state=initial_state
                )
            else:
                raise RuntimeError(
                    f"NKI kernel required for TKG at batch>4 but not available! "
                    f"_USE_NKI={_USE_NKI_DELTA_RULE}, _AVAILABLE={_NKI_AVAILABLE}"
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
        if self.shard_decode:
            # core_attn_out is [B, S, local_num_v_heads, head_v]; gather the
            # per-rank value-head slices into the full head dimension so the
            # replicated out_proj sees the full value_dim.  This all-gather
            # replaces the RowParallel out_proj all-reduce in *both* the prefill
            # and decode graphs -- the same collective the shard-prefill path
            # already places in the CTE graph without tripping PGTiling.
            core_attn_out = _gather_along_dim(core_attn_out, partition_dim=2)
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

        # DIAGNOSTIC (temporary, env-gated): drop the Qwen3.5 sigmoid output gate
        # and its fused projection channels to test whether that structure (not
        # head_dim=256) is what trips neuronx-cc PGTiling at decode batch>8.
        # Unset by default so the HF-equivalence CPU unit test is unaffected.
        self.ablate_gate = os.environ.get("QWEN35_ABLATE_GATE") == "1"
        q_proj_mult = 1 if self.ablate_gate else 2

        # q_proj emits query and output-gate channels, interleaved per head.
        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_heads * self.head_dim * q_proj_mult,
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
        if self.ablate_gate:
            query_states = q.view(
                batch_size, seq_len, self.local_num_heads, self.head_dim
            )
            gate = None
        else:
            q = q.view(batch_size, seq_len, self.local_num_heads, self.head_dim * 2)
            query_states = q[..., : self.head_dim]
            gate = q[..., self.head_dim:]
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
            # Token generation: attend over the cache (masked by attention_mask)
            # plus the current token (masked by active_mask). Mirror
            # NeuronAttentionBase.compute_for_token_gen: split the softmax over
            # the prior (cached) and active (new) KV via manual_softmax instead
            # of concatenating the score tensors. The concat path materialises a
            # single (B, H, q, prior+active) DAG whose tensor-parallel axes
            # neuronx-cc's PGTiling pass cannot tile at batch > 8, which is the
            # AWS-supported attention's decode structure and the reason Qwen3
            # compiles at higher concurrency.
            k_cache, v_cache = past_key_value
            k_cache, v_cache = _expand_kv(k_cache, v_cache)
            k_new, v_new = _expand_kv(key_states, value_states)
            scores_prior = query_states @ k_cache.transpose(-1, -2) / (self.head_dim ** 0.5)
            scores_prior = torch.where(
                attention_mask, scores_prior, torch.finfo(scores_prior.dtype).min
            ).float()
            scores_active = query_states @ k_new.transpose(-1, -2) / (self.head_dim ** 0.5)
            if active_mask is not None:
                scores_active = torch.where(
                    active_mask, scores_active, torch.finfo(scores_active.dtype).min
                )
            scores_active = scores_active.float()
            softmax_prior, softmax_active = manual_softmax(
                scores_prior, scores_active, False
            )
            softmax_prior = softmax_prior.to(query_states.dtype)
            softmax_active = softmax_active.to(query_states.dtype)
            attn_output = softmax_prior @ v_cache + softmax_active @ v_new

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        if gate is not None:
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
    # into per-rank slabs [q_0 k_0 v_0 | q_1 k_1 v_1 | ...] so each rank's
    # local (q, k, v) channels form one contiguous slice of the replicated
    # weight, and provide a per-module SPMD rank tensor for the slicing.
    tp = neuron_config.tp_degree
    key_dim = config.linear_num_key_heads * config.linear_key_head_dim
    value_dim = config.linear_num_value_heads * config.linear_value_head_dim
    for l in range(config.num_hidden_layers):  # noqa: E741
        key = f"layers.{l}.linear_attn.conv1d.weight"
        if key not in state_dict:
            continue
        state_dict[f"layers.{l}.linear_attn.rank_util.rank"] = torch.arange(
            0, tp, dtype=torch.int32
        )
        # Replicated DeltaNet keeps the full HF conv layout on every rank (no
        # per-rank slabs), so skip the reorder exactly as for tp == 1.
        if tp == 1 or _DELTANET_REPLICATED:
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

    def enable_context_encoding(self, **model_init_kwargs):
        self.compile_tag = CONTEXT_ENCODING_MODEL_TAG
        super().enable_context_encoding(**model_init_kwargs)

    def enable_token_generation(self, **model_init_kwargs):
        self.compile_tag = TOKEN_GENERATION_MODEL_TAG
        # Must set cc_pipeline_tiling_factor=1 BEFORE super() traces the model.
        # The default model_wrapper __init__ does this at line 88, but only when
        # compiler_args is None.  With custom get_compiler_args(), that path is
        # skipped, so the attention layers would be traced with tiling=2 baked
        # into the XLA graph — causing PGTiling (NCC_IPCC901) at batch=14.
        self.neuron_config.cc_pipeline_tiling_factor = 1
        super().enable_token_generation(**model_init_kwargs)

    def get_compiler_args(self):
        is_tkg = getattr(self, "compile_tag", None) == TOKEN_GENERATION_MODEL_TAG
        # DeltaNet's recurrent scan DAG triggers PGTiling (NCC_IPCC901) at
        # batch=14.  Force aggressive modular-flow partitioning (mac-threshold=10)
        # so the graph is split into small modules where no single module has the
        # problematic multi-axis DAG configuration that PGTiling rejects.
        # NOTE: we include --verify-hlo=true here because model_wrapper skips its
        # own --internal-hlo2tensorizer-options when we already provide one.
        hlo2t = "--internal-hlo2tensorizer-options='--modular-flow-mac-threshold=10 --verify-hlo=true'"
        if is_tkg:
            return (
                "--auto-cast=none --model-type=transformer "
                f"{hlo2t} "
                f"--lnc={self.neuron_config.logical_nc_config} -O1"
            )
        # CTE: use ccop overlap with tiling-factor=2 for compute-communication
        # pipelining during long prefill passes.
        return (
            "--auto-cast=none --model-type=transformer "
            "--tensorizer-options='--enable-ccop-compute-overlap "
            "--cc-pipeline-tiling-factor=2 --vectorize-strided-dma ' "
            f"{hlo2t} "
            f"--lnc={self.neuron_config.logical_nc_config} -O1"
        )

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
