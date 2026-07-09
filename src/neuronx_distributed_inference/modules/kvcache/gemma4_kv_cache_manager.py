# Gemma 4 per-layer KV cache manager.
#
# Gemma 4 interleaves two attention flavours within a single network:
#   * "full_attention" layers  -> global causal attention, head_dim = global_head_dim (512 on E2B)
#   * "sliding_attention" layers -> windowed attention (window = sliding_window),
#                                   head_dim = head_dim (256 on E2B)
#
# The stock ``KVCacheManager`` sizes every layer with a single ``head_dim`` and a
# single cache length, so it cannot represent this mix: the full layers need an
# 8k-long, 512-wide cache while the sliding layers only need a ``sliding_window``
# long, 256-wide cache. Allocating the full cache for every layer is what pins
# decode to HBM bandwidth (each token-gen step re-reads the entire 8k cache on
# all 35 layers).
#
# This manager sizes ``k``/``v`` per layer using the real ``layer_types`` list
# (so sliding layers shrink to ``sliding_window`` x ``head_dim``) and wraps the
# write position modulo the window only on sliding layers, exactly mirroring the
# proven windowed path in the base manager. Everything else (fetch, slice,
# scatter, quant, tiling) is inherited unchanged.
#
# True KV-sharing (Milestone B). Gemma 4 reuses the K/V of the last non-shared
# layer of each type for every ``num_kv_shared_layers`` trailing layer (HF's
# ``store_full_length_kv`` semantics). Those shared layers must not own a KV
# cache at all: they attend the *source* layer's cache (past tokens) plus the
# source layer's freshly computed K/V (the modeling threads the latter through
# ``_shared_kv_states``). This manager therefore allocates physical buffers only
# for the non-shared "owner" layers and aliases every shared layer's reads to its
# source buffer, skipping its writes. On E2B (35 layers, 20 shared) that removes
# the 4 shared full-attention caches (~1 GiB each at b64) and 16 shared sliding
# caches, freeing ~57% of the KV footprint and lifting the batch ceiling.

from typing import List

import torch
from torch import nn

from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.modules.kvcache.kv_cache_manager import (
    KV_CACHE_PAD_FOR_SEQ_IDS_MASKING,
    KVCacheManager,
    get_kv_shapes,
    tile_cache,
)


class Gemma4KVCacheManager(KVCacheManager):
    """KV cache manager with per-layer head_dim and per-layer cache length.

    Args:
        config: inference config.
        num_kv_head: number of KV heads (1 on Gemma 4 E2B, i.e. MQA).
        layer_is_sliding: per-layer flag, ``True`` for sliding_attention layers.
        layer_head_dims: per-layer attention head dimension.
        sliding_window: window size used by the sliding_attention layers.
        global_rank: SPMD rank helper, forwarded to the base manager.
    """

    def __init__(
        self,
        config: InferenceConfig,
        num_kv_head: int,
        layer_is_sliding: List[bool],
        layer_head_dims: List[int],
        sliding_window: int,
        global_rank=None,
        **kwargs,
    ):
        # Stash per-layer metadata before ``super().__init__`` runs, because it
        # invokes ``_init_kv_shape`` (overridden below) during construction.
        self.layer_is_sliding = list(layer_is_sliding)
        self.layer_head_dims = list(layer_head_dims)
        self._gemma4_sliding_window = sliding_window
        self._build_kv_sharing_map(config)
        super().__init__(
            config,
            num_kv_head=num_kv_head,
            global_rank=global_rank,
            sliding_window=sliding_window,
            layer_to_cache_size_mapping=self._build_cache_size_mapping(config),
            **kwargs,
        )
        # Milestone B: drop the physical buffers for the trailing shared layers.
        # ``super().__init__`` allocated one (k, v) pair per layer; the owner
        # layers are exactly the first ``_first_shared_layer_idx`` layers (HF
        # marks the *trailing* ``num_kv_shared_layers`` as shared), so their
        # parameters are the leading ``2 * _first_shared_layer_idx`` entries.
        if self._kv_sharing_enabled and hasattr(self, "past_key_values"):
            keep = 2 * self._first_shared_layer_idx
            self.past_key_values = nn.ParameterList(list(self.past_key_values)[:keep])

    def _build_kv_sharing_map(self, config: InferenceConfig) -> None:
        """Map each layer to the owner layer whose KV cache it physically uses.

        Non-shared ("owner") layers map to themselves. Each trailing shared layer
        maps to the last non-shared layer of the *same* ``layer_type`` -- exactly
        the layer the modeling marks ``store_full_length_kv`` and publishes its
        post-rope K/V from, so the aliased past cache and the threaded current K/V
        come from one and the same source layer (HF-faithful).
        """
        num_layers = config.num_hidden_layers
        num_shared = getattr(config, "num_kv_shared_layers", 0) or 0
        self._kv_sharing_enabled = num_shared > 0
        self._first_shared_layer_idx = num_layers - num_shared
        layer_types = list(config.layer_types)
        prev_layers = layer_types[: self._first_shared_layer_idx]
        self._layer_owner = []
        for layer_idx in range(num_layers):
            if not self._kv_sharing_enabled or layer_idx < self._first_shared_layer_idx:
                self._layer_owner.append(layer_idx)
                continue
            layer_type = layer_types[layer_idx]
            source_idx = len(prev_layers) - 1 - prev_layers[::-1].index(layer_type)
            self._layer_owner.append(source_idx)

    def _build_cache_size_mapping(self, config: InferenceConfig) -> List[int]:
        """Cache length per layer: window for sliding layers, max_length for full."""
        max_len = config.neuron_config.max_length
        return [
            self._gemma4_sliding_window if is_sliding else max_len
            for is_sliding in self.layer_is_sliding
        ]

    def _init_kv_shape(self, config: InferenceConfig, layer_to_cache_size_mapping=None):
        """Build per-layer (batch, kv_head, cache_len, head_dim) K/V shapes.

        Unlike the base implementation this varies ``head_dim`` per layer, so the
        512-wide full-attention caches and 256-wide sliding caches coexist.
        """
        assert layer_to_cache_size_mapping is not None, "Gemma4KVCacheManager requires a cache size mapping"
        max_batch_size = (
            config.neuron_config.kv_cache_batch_size + config.neuron_config.kv_cache_padding_size
        )
        num_kv_heads_per_rank = self._get_num_kv_heads_per_rank(config)

        self.padded_layer_ids = []
        self.k_shapes = []
        self.v_shapes = []
        for idx, cache_len in enumerate(layer_to_cache_size_mapping):
            if self.neuron_config.apply_seq_ids_mask:
                cache_len += KV_CACHE_PAD_FOR_SEQ_IDS_MASKING
                self.padded_layer_ids.append(idx)
            head_dim = self.layer_head_dims[idx]
            k_shape, v_shape = get_kv_shapes(
                cache_len,
                max_batch_size,
                num_kv_heads_per_rank,
                head_dim,
                self.k_cache_transposed,
                self.is_kv_cache_tiled,
            )
            self.k_shapes.append(k_shape)
            self.v_shapes.append(v_shape)

    def _get_index_to_update_new_position(
        self, seq_ids, scatter_index, position_ids, full_k, transposed: bool, layer_idx: int
    ):
        """Ring-buffer the write index modulo the window, but only for sliding layers.

        Full-attention layers keep their absolute position (no wrap), matching
        their full-length cache; sliding layers wrap into their window exactly
        like the base single-window path (``% (sliding_window - 1)``).
        """
        if self.layer_is_sliding[layer_idx]:
            position_ids = position_ids % (self._gemma4_sliding_window - 1)
        index = scatter_index if self.is_medusa else position_ids
        view_shape = (-1, 1, index.shape[-1], 1) if not transposed else (-1, 1, 1, index.shape[-1])
        return index.view(*view_shape).expand_as(full_k)

    def _fetch_cache(self, idx: int, kvcache_buffer=None):
        """Fetch the buffer a layer reads/writes, aliasing shared layers to source.

        Owner layers map to themselves; shared layers map to their source layer's
        physical buffer, so a shared layer attends the source's accumulated cache.
        """
        return super()._fetch_cache(self._layer_owner[idx], kvcache_buffer)

    def get_cache(
        self, seq_len: int, skip_slice=False, kvcache_buffer=None, seq_ids=None, windowed_context_encoding_window_idx=-1, **kwargs
    ):
        """Return per-layer (K, V) for *all* layers, not just the owned buffers.

        The base implementation iterates ``len(self.past_key_values) // 2`` layers,
        which after Milestone B is only the owner count. The decoder loop indexes
        the result by absolute layer id, so we iterate all ``num_hidden_layers``
        and let ``_fetch_cache`` alias shared layers to their source buffer.
        """
        past_key_values = []
        for idx in range(self.config.num_hidden_layers):
            k_cache, v_cache = self.get_kv_by_layer_id(
                idx=idx,
                skip_slice=skip_slice,
                seq_len=seq_len,
                kvcache_buffer=kvcache_buffer,
                seq_ids=seq_ids,
                windowed_context_encoding_window_idx=windowed_context_encoding_window_idx,
                **kwargs,
            )
            past_key_values.append([k_cache, v_cache])
        return past_key_values

    def update_cache(
        self,
        is_for_context_encoding: bool,
        seq_ids,
        position_ids,
        new_key_values,
        seq_len: int,
        scatter_index=None,
        kv_active_mask=None,
        kvcache_buffer=None,
        windowed_context_encoding_window_idx: int = -1,
        **kwargs,
    ):
        """Write only the owner layers' caches; shared layers reuse their source.

        ``new_key_values`` still has one entry per layer (the decoder loop appends
        for every layer). Shared layers carry the source layer's reused K/V, which
        the source layer already writes, so we skip them -- and the returned list
        then matches the compacted ``past_key_values`` state (owner buffers only).
        """
        updated_kv_cache = []
        for idx, kv_per_layer in enumerate(new_key_values):
            if self._kv_sharing_enabled and idx >= self._first_shared_layer_idx:
                continue
            k_cache, v_cache = self.update_kv_by_layer_id(
                idx=idx,
                is_for_context_encoding=is_for_context_encoding,
                seq_ids=seq_ids,
                position_ids=position_ids,
                kv_per_layer=kv_per_layer,
                seq_len=seq_len,
                scatter_index=scatter_index,
                kv_active_mask=kv_active_mask,
                kvcache_buffer=kvcache_buffer,
                windowed_context_encoding_window_idx=windowed_context_encoding_window_idx,
                **kwargs,
            )
            if self.is_kv_cache_tiled:
                k_cache = tile_cache(k_cache, self.k_cache_transposed)
                v_cache = tile_cache(v_cache, False)
            updated_kv_cache.append(k_cache)
            updated_kv_cache.append(v_cache)
        return updated_kv_cache
