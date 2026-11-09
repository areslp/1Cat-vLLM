# SPDX-License-Identifier: Apache-2.0
"""PPU reference sparse MLA backend.

The upstream sparse MLA kernels target Hopper/Blackwell (FLASHMLA_SPARSE,
FLASHINFER_MLA_SPARSE) or exact SM70 devices (GLM5_SM70_SPARSE).  PPU exposes a
CUDA 8.0-compatible device while having no HPU/FlashMLA sparse kernel.  This
backend preserves the model's top-k token selection and MLA math using plain
PyTorch operations.  It is intentionally a correctness-first reference path,
not a performance kernel.
"""

import os
from typing import TYPE_CHECKING, ClassVar

import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.ppu_sparse_attn_ref import sparse_mla_attention_ref
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionLayer, SparseMLAAttentionImpl
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    FlashMLASparseBackend,
    FlashMLASparseMetadata,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer

logger = init_logger(__name__)


def ppu_sparse_ref_enabled() -> bool:
    return os.environ.get("VLLM_PPU_SPARSE_REF", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class PPUSparseRefBackend(FlashMLASparseBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_name() -> str:
        return "PPU_SPARSE_REF"

    @staticmethod
    def get_impl_cls() -> type["PPUSparseRefImpl"]:
        return PPUSparseRefImpl

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [576]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        del capability
        return ppu_sparse_ref_enabled()

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        del dtype, use_mla, has_sink, use_sparse, device_capability
        if head_size != 576:
            return f"PPU sparse reference requires MLA head_size=576, got {head_size}"
        if kv_cache_dtype not in (None, "auto", "float16", "bfloat16"):
            return "PPU sparse reference only supports BF16/FP16 KV cache"
        if block_size is not None and block_size % 64 != 0:
            return "PPU sparse reference requires block_size divisible by 64"
        config = get_current_vllm_config_or_none()
        if config is not None and config.model_config is not None:
            text_config = config.model_config.hf_text_config
            if getattr(text_config, "kv_lora_rank", None) != 512:
                return "PPU sparse reference requires kv_lora_rank=512"
            if getattr(text_config, "qk_rope_head_dim", None) != 64:
                return "PPU sparse reference requires qk_rope_head_dim=64"
        return None


class PPUSparseRefImpl(SparseMLAAttentionImpl[FlashMLASparseMetadata]):
    """Correctness-first sparse MLA attention over selected cache slots."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        del (
            alibi_slopes,
            sliding_window,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
        )
        if num_kv_heads != 1:
            raise NotImplementedError("PPU sparse reference requires one KV head")
        if head_size != 576:
            raise NotImplementedError("PPU sparse reference requires head_size=576")
        if mla_args.get("kv_lora_rank") != 512:
            raise NotImplementedError("PPU sparse reference requires kv_lora_rank=512")
        if mla_args.get("qk_rope_head_dim") != 64:
            raise NotImplementedError(
                "PPU sparse reference requires qk_rope_head_dim=64"
            )
        if indexer is None or indexer.topk_indices_buffer is None:
            raise ValueError("PPU sparse reference requires an indexer top-k buffer")

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_lora_rank = 512
        self.qk_rope_head_dim = 64
        self.topk_indices_buffer = indexer.topk_indices_buffer
        self.topk_width = self.topk_indices_buffer.shape[1]
        # Bound the gathered-K workspace.  This is a reference path; small
        # chunks keep peak memory predictable at long context.
        self.token_chunk = 8
        logger.info_once(
            "PPU sparse reference MLA enabled: topk=%d, head=576, tokens/chunk=%d",
            self.topk_width,
            self.token_chunk,
        )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del layer
        if isinstance(q, tuple):
            q_nope, q_pe = q
            q = torch.cat([q_nope, q_pe], dim=-1)
        if q.shape[-1] != self.head_size:
            raise ValueError(
                f"PPU sparse reference expected q width {self.head_size}, "
                f"got {q.shape[-1]}"
            )

        num_tokens = q.shape[0]
        if num_tokens == 0:
            return q.new_empty((0, self.num_heads, self.kv_lora_rank)), None

        topk_indices = self.topk_indices_buffer[:num_tokens]
        global_slots = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[:num_tokens],
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
        )

        cache_flat = kv_c_and_k_pe_cache.reshape(-1, self.head_size)
        output = sparse_mla_attention_ref(
            q,
            cache_flat,
            global_slots,
            self.scale,
            self.kv_lora_rank,
            self.token_chunk,
        )
        return output, None
