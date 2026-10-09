# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MTP shared metadata policy captured at model-state initialization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from vllm import envs
from vllm.platforms import current_platform
from vllm.v1.attention.backends import sm70_mtp_shortconv_meta as _sc_meta
from vllm.v1.attention.backends.gdn_attn import onecat_mtp_gdn_group_meta_enabled
from vllm.v1.attention.backends.sm70_mtp_shortconv_meta import (
    ShortConvGroupDescriptor as ShortConvGroupDescriptor,
)
from vllm.v1.worker.gpu.spec_decode import uses_dflash_selector_engine

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.attn_utils import AttentionGroup
    from vllm.v1.worker.gpu.input_batch import InputBatch


def capture_flags(config, device, cache_mode) -> tuple[bool, bool]:
    spec = config.speculative_config
    mtp_on_device = (
        spec is not None
        and spec.method == "mtp"
        and device.type == "cuda"
        and current_platform.is_device_capability(70)
    )
    grouped_gdn = onecat_mtp_gdn_group_meta_enabled() and mtp_on_device
    grouped_shortconv = bool(
        _sc_meta.enabled()
        and mtp_on_device
        and not uses_dflash_selector_engine(config)
        and cache_mode in ("none", "align")
    )
    return grouped_gdn, grouped_shortconv


def prepare_shortconv_metadata(
    self: Any,
    input_batch: InputBatch,
    block_tables: tuple[torch.Tensor, ...],
    attn_groups: list[list[AttentionGroup]],
    query_start_loc_cpu: torch.Tensor,
    num_decode_draft_tokens_cpu: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    num_reqs: int,
    num_tokens: int,
) -> dict[int, Any] | None:
    """ONECAT_MTP_SHORTCONV_GROUP_META for one FULL-graph batch: every PLE
    short-conv group's graph metadata in one launch, or None with a counted
    reason, which keeps the per-group path."""
    reason = _sc_meta.pure_spec_block_reason(
        num_decode_draft_tokens_cpu, query_start_loc_cpu, num_tokens
    )
    prepared_result = None
    if reason is None:
        num_spec_decodes = int((num_decode_draft_tokens_cpu >= 0).sum().item())
        builders = self._get_ple_shortconv_builders(attn_groups)
        if not builders:
            reason = "no_groups"
        else:
            prepared_result = _sc_meta.prepare_ple_shortconv_group_metadata(
                builders_by_group=builders,
                block_tables=block_tables,
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=input_batch.query_start_loc,
                num_spec_decodes=num_spec_decodes,
                num_spec_decode_tokens=int(
                    query_start_loc_cpu[num_spec_decodes].item()
                ),
                num_reqs=num_reqs,
                descriptor=self._ple_shortconv_descriptor,
                state_start_indices=self._mamba_state_idx_gpu
                if self._align_mode
                else None,
                req_index_mapping=input_batch.idx_mapping if self._align_mode else None,
            )
            if prepared_result is None:
                reason = "prepare"
    if reason is not None or prepared_result is None:
        _sc_meta.note_route(f"fallback:{reason}")
        return None
    prepared, self._ple_shortconv_descriptor = prepared_result
    _sc_meta.note_route("fused")
    return prepared


def common_gdn_enabled(config, device, grouped: bool) -> bool:
    spec = config.speculative_config
    return bool(
        (envs.VLLM_SM70_MTP4_SHARED_GDN_METADATA or grouped)
        and spec is not None
        and spec.method == "mtp"
        and device.type == "cuda"
        and current_platform.is_device_capability(70)
    )


def fused_gdn_enabled(common: bool, grouped: bool, cache_mode: str) -> bool:
    return bool(
        common
        and (envs.VLLM_SM70_MTP4_FUSED_GDN_METADATA or grouped)
        and cache_mode in ("none", "align")
    )
