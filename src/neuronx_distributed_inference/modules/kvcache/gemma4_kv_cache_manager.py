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

from typing import List

import torch

from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.modules.kvcache.kv_cache_manager import (
    KV_CACHE_PAD_FOR_SEQ_IDS_MASKING,
    KVCacheManager,
    get_kv_shapes,
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
        super().__init__(
            config,
            num_kv_head=num_kv_head,
            global_rank=global_rank,
            sliding_window=sliding_window,
            layer_to_cache_size_mapping=self._build_cache_size_mapping(config),
            **kwargs,
        )

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
