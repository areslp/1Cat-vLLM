# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admission for the single-request E4M3 FP32 small-Q route."""

import os

import torch


def load_grouped_e4m3_fp32():
    try:
        from flash_attn_v100 import (
            flash_attn_grouped_e4m3_fp32_available,
            flash_attn_grouped_e4m3_fp32_paged,
        )
    except ImportError:
        return None
    if not flash_attn_grouped_e4m3_fp32_available():
        return None
    from vllm.v1.attention.ops.sm70_e4m3_long import wrap_long_attention

    return wrap_long_attention(flash_attn_grouped_e4m3_fp32_paged)


def grouped_e4m3_fp32_allowed(
    instance, query, k, v, table, lengths, metadata, *, out, partition_size_hint
):
    parent_table = getattr(metadata, "block_table", None)
    parent_seq = getattr(metadata, "seq_lens", None)
    if not (
        getattr(instance, "flash_attn_grouped_e4m3_fp32_paged", None) is not None
        and instance.kv_cache_dtype == "fp8_e4m3"
        and instance.use_smallq_decode_xqa
        and partition_size_hint is None
        and not os.environ.get("VLLM_FLASH_V100_DECODE_PARTITION_SIZE")
        and getattr(metadata, "causal", True)
        and instance._flash_v100_window_size(causal=True) == (-1, -1)
        and query.ndim == 3
        and 2 <= query.shape[0] <= 8
        and query.shape[1:] == (6, 256)
        and query.dtype == torch.float16
        and query.is_contiguous()
        and out.shape == query.shape
        and out.dtype == query.dtype
        and out.device == query.device
        and out.is_contiguous()
        and k.ndim == 4
        and k.shape[1] in (800, 848, 1616, 1648, 1728, 3296, 3456)
        and k.shape[2:] == (1, 256)
        and v.shape == k.shape
        and k.dtype == torch.uint8
        and v.dtype == torch.uint8
        and parent_seq is not None
        and parent_seq.shape == (1,)
        and parent_table is not None
        and parent_table.ndim == 2
        and parent_table.shape[0] == 1
        and 0 < parent_table.shape[1] * k.shape[1] <= 266240
        and lengths.shape == (query.shape[0],)
        and table.shape == (query.shape[0], parent_table.shape[1])
    ):
        return False
    # The parent metadata proves that every real query belongs to one request.
    # Device row lengths, including graph padding, define the visible KV prefix.
    return all(
        t.device == query.device and t.dtype == torch.int32 and t.is_contiguous()
        for t in (table, lengths, parent_table, parent_seq)
    ) and all(
        t.device == query.device
        and t.stride(-1) == 1
        and t.data_ptr() % 16 == 0
        and all(s % 8 == 0 for s in t.stride()[:3])
        for t in (k, v)
    )


MULTI_KV_HEAD_ENV = "VLLM_FLASH_V100_GROUPED_VERIFY_MULTI_KV_HEAD"
GROUP_SIZE = 6


def multi_kv_head_enabled() -> bool:
    return bool(int(os.environ.get(MULTI_KV_HEAD_ENV, "0") or 0))


def grouped_e4m3_fp32_kv_head_views(query, k, v, out):
    """Per-KV-head (q, k, v, out) views for a rank holding several KV heads.

    The native entry serves one KV head with its six query heads. A TP2 rank of
    a 24/4-head model holds two KV heads (q ``[rows, 12, 256]``, KV
    ``[pages, page, 2, 256]``); slicing head ``h`` gives the exact single-head
    layout: the query slice is copied contiguous, the KV slice is a strided
    view (token stride ``2 * 256``, head offset ``256`` bytes) that the entry
    addresses through its runtime strides. Returns ``None`` unless the shapes
    describe ``6 * num_kv_heads`` query heads over ``num_kv_heads >= 2``.
    """
    if not (
        query.ndim == 3
        and k.ndim == 4
        and v.shape == k.shape
        and k.shape[2] >= 2
        and query.shape[1] == GROUP_SIZE * k.shape[2]
        and query.shape[2] == 256
        and k.shape[3] == 256
        and out.shape == query.shape
    ):
        return None
    views = []
    for head in range(k.shape[2]):
        heads = slice(head * GROUP_SIZE, (head + 1) * GROUP_SIZE)
        q_h = query[:, heads, :].contiguous()
        views.append(
            (
                q_h,
                k[:, :, head : head + 1, :],
                v[:, :, head : head + 1, :],
                torch.empty_like(q_h),
            )
        )
    return views


def run_grouped_e4m3_fp32_per_kv_head(
    op,
    instance,
    query,
    k,
    v,
    table,
    lengths,
    metadata,
    *,
    out,
    softmax_scale,
    k_scale,
    v_scale,
    partition_size_hint,
) -> bool:
    """Run the single-request E4M3 FP32 grouped route once per KV head.

    Every head view must pass ``grouped_e4m3_fp32_allowed`` before any call
    is issued, so a rejected shape falls back with ``out`` untouched.
    Returns ``True`` when ``out`` holds the result.
    """
    if op is None or not multi_kv_head_enabled():
        return False
    views = grouped_e4m3_fp32_kv_head_views(query, k, v, out)
    if views is None:
        return False
    for q_h, k_h, v_h, out_h in views:
        if not grouped_e4m3_fp32_allowed(
            instance,
            q_h,
            k_h,
            v_h,
            table,
            lengths,
            metadata,
            out=out_h,
            partition_size_hint=partition_size_hint,
        ):
            return False
    for head, (q_h, k_h, v_h, out_h) in enumerate(views):
        # The gate validates the per-row table; the entry takes the request's
        # single block-table row and the per-row lengths.
        op(
            q_h,
            k_h,
            v_h,
            metadata.block_table,
            lengths,
            out=out_h,
            softmax_scale=softmax_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        out[:, head * GROUP_SIZE : (head + 1) * GROUP_SIZE, :].copy_(out_h)
    return True
