# SPDX-License-Identifier: Apache-2.0
"""Correctness-first PPU implementation of the DSA sparse indexer.

The production indexer depends on DeepGEMM FP8 MQA-logits kernels.  PPU has no
DeepGEMM, so this module computes the same indexer logits with PyTorch while
preserving the checkpoint's FP8 K-cache quantization, per-head weighting,
causal range, and top-k selection.
"""

import os

import torch


def ppu_sparse_ref_enabled() -> bool:
    return os.environ.get("VLLM_PPU_SPARSE_REF", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _cache_geometry(
    kv_cache: torch.Tensor, quant_block_size: int | None = None
) -> tuple[int, int, int, int]:
    """Return (block_size, cache_stride, head_dim, page_bytes).

    The on-device kernel treats each cache page as one flat byte range:
    ``block_size * head_dim`` contiguous FP8 key bytes followed by one FP32
    scale per token.  The 3D ``[num_blocks, block_size, cache_stride]`` shape is
    only a storage description; slicing it as ``cache[block, offset, :head_dim]``
    would use ``cache_stride`` as the row pitch and therefore intermix the key
    and scale regions.  All access below goes through flat page offsets.
    """
    if kv_cache.ndim != 3 or kv_cache.dtype != torch.uint8:
        raise ValueError("Indexer cache must be a 3D uint8 tensor")
    if not kv_cache.is_contiguous():
        raise ValueError("PPU indexer reference requires a contiguous cache")
    block_size = int(kv_cache.shape[1])
    cache_stride = int(kv_cache.shape[2])
    # The reference cache stores one scale per key vector (4 bytes per token).
    head_dim = cache_stride - 4
    if head_dim <= 0:
        raise ValueError(f"Unexpected indexer cache stride {cache_stride}")
    if quant_block_size is not None and quant_block_size != head_dim:
        raise NotImplementedError(
            "PPU indexer reference requires one scale per key, got "
            f"quant_block_size={quant_block_size}, head_dim={head_dim}"
        )
    if int(kv_cache.stride(0)) != block_size * cache_stride:
        raise ValueError("Indexer cache pages must be densely packed")
    return block_size, cache_stride, head_dim, block_size * cache_stride


def _cache_flat(kv_cache: torch.Tensor) -> torch.Tensor:
    return kv_cache.view(-1)


def _insert_indexer_k(
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    quant_block_size: int,
) -> None:
    """Write BF16 indexer K into the uint8 FP8/scale cache.

    The layout matches ``indexer_k_quant_and_cache_kernel``: each page stores
    ``block_size * head_dim`` contiguous FP8 bytes followed by one FP32 scale
    per token.  K rows are not interleaved with the scale tail.
    """
    if k is None or k.numel() == 0:
        return
    num_tokens = min(k.shape[0], slot_mapping.shape[0])
    if num_tokens == 0:
        return
    k = k[:num_tokens].float()
    slots = slot_mapping[:num_tokens].to(torch.int64)
    valid = slots >= 0
    if not valid.any():
        return
    k = k[valid]
    slots = slots[valid]

    block_size, _cache_stride, head_dim, page_bytes = _cache_geometry(
        kv_cache, quant_block_size
    )
    if head_dim != k.shape[1]:
        raise ValueError(f"Indexer cache head_dim={head_dim}, k width={k.shape[1]}")

    amax = k.abs().amax(dim=-1).clamp_min(1e-4)
    scale = amax / 448.0
    # The original kernel uses UE8M0 scales for this model.
    scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    q = torch.clamp(k / scale[:, None], -448.0, 448.0).to(torch.float8_e4m3fn)

    block = (slots // block_size).to(torch.int64)
    offset = (slots % block_size).to(torch.int64)
    q_bytes = q.view(torch.uint8)

    flat = _cache_flat(kv_cache)
    page_base = block * page_bytes
    k_cols = torch.arange(head_dim, device=kv_cache.device, dtype=torch.int64)
    k_idx = page_base[:, None] + offset[:, None] * head_dim + k_cols[None, :]
    flat[k_idx.reshape(-1)] = q_bytes.reshape(-1)

    # Scales occupy the page tail, one FP32 per token.
    scale_base = page_base + block_size * head_dim + offset * 4
    cols = torch.arange(4, device=kv_cache.device, dtype=torch.int64)
    scale_bytes = scale.to(torch.float32).contiguous().view(torch.uint8).reshape(-1, 4)
    flat[(scale_base[:, None] + cols[None, :]).reshape(-1)] = scale_bytes.reshape(-1)


def _dequant_page_keys(
    kv_cache: torch.Tensor,
    block_ids: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """Return dequantized keys [seq_len, head_dim] for one request."""
    block_size, _cache_stride, head_dim, page_bytes = _cache_geometry(kv_cache)
    if seq_len <= 0:
        return kv_cache.new_empty((0, head_dim), dtype=torch.float32)
    num_blocks = (seq_len + block_size - 1) // block_size
    blocks = block_ids[:num_blocks].to(torch.int64)
    if (blocks < 0).any():
        raise ValueError("Negative block id in an indexer block table")

    flat = _cache_flat(kv_cache)
    page_base = blocks * page_bytes
    offsets = torch.arange(block_size, device=kv_cache.device, dtype=torch.int64)
    cols = torch.arange(head_dim, device=kv_cache.device, dtype=torch.int64)
    k_idx = (
        page_base[:, None, None]
        + offsets[None, :, None] * head_dim
        + cols[None, None, :]
    )
    values = flat[k_idx.reshape(-1)].view(torch.float8_e4m3fn)
    values = values.reshape(-1, head_dim)[:seq_len].float()

    scale_cols = torch.arange(4, device=kv_cache.device, dtype=torch.int64)
    scale_idx = (
        page_base[:, None, None]
        + block_size * head_dim
        + offsets[None, :, None] * 4
        + scale_cols[None, None, :]
    )
    scales = flat[scale_idx.reshape(-1)].view(torch.float32).reshape(-1)[:seq_len]
    return values * scales[:, None]


def _topk_from_logits(
    logits: torch.Tensor,
    topk: int,
    out: torch.Tensor,
) -> None:
    width = min(topk, logits.shape[1])
    values, indices = torch.topk(logits, k=width, dim=1)
    indices = indices.to(torch.int32)
    indices = torch.where(torch.isfinite(values), indices, -1)
    out[:, :width] = indices


def ppu_reference_sparse_indexer(
    hidden_states: torch.Tensor,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    k: torch.Tensor | None,
    weights: torch.Tensor,
    quant_block_size: int,
    topk_tokens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
    attn_metadata_narrowed,
) -> torch.Tensor:
    del hidden_states
    if not isinstance(q_quant, torch.Tensor):
        raise NotImplementedError("PPU indexer reference expects FP8 q, not FP4 q")
    slot_mapping = attn_metadata_narrowed.slot_mapping
    num_tokens = slot_mapping.shape[0]
    if k is not None:
        k = k[:num_tokens]
    if not skip_k_cache_insert:
        _insert_indexer_k(k, kv_cache, slot_mapping, quant_block_size)

    topk_indices_buffer[: q_quant.shape[0]] = -1
    q_all = q_quant.float()
    w_all = weights.float()

    if attn_metadata_narrowed.num_prefills > 0:
        prefill = attn_metadata_narrowed.prefill
        assert prefill is not None
        for chunk in prefill.chunks:
            seq_lens = torch.diff(chunk.cu_seq_lens).tolist()
            key_parts = [
                _dequant_page_keys(
                    kv_cache,
                    chunk.block_table[i],
                    int(seq_lens[i]),
                )
                for i in range(chunk.num_reqs)
            ]
            keys = torch.cat(key_parts, dim=0)
            ks = chunk.cu_seqlen_ks.to(torch.int64)
            ke = chunk.cu_seqlen_ke.to(torch.int64)
            q = q_all[chunk.token_start : chunk.token_end]
            w = w_all[chunk.token_start : chunk.token_end]
            out = topk_indices_buffer[chunk.token_start : chunk.token_end]
            total_seq = keys.shape[0]
            positions = torch.arange(total_seq, device=keys.device)

            for start in range(0, q.shape[0], 32):
                end = min(start + 32, q.shape[0])
                scores = torch.einsum("qhd,kd->qhk", q[start:end], keys)
                logits = (scores.relu() * w[start:end, :, None]).sum(dim=1)
                in_range = (positions[None, :] >= ks[start:end, None]) & (
                    positions[None, :] < ke[start:end, None]
                )
                logits = logits.masked_fill(~in_range, -torch.inf)
                values, indices = torch.topk(logits, min(topk_tokens, total_seq), dim=1)
                indices = indices - ks[start:end, None]
                indices = torch.where(torch.isfinite(values), indices, -1).to(
                    torch.int32
                )
                out[start:end, : indices.shape[1]] = indices

    if attn_metadata_narrowed.num_decodes > 0:
        decode = attn_metadata_narrowed.decode
        assert decode is not None
        num_decode_tokens = attn_metadata_narrowed.num_decode_tokens
        seq_lens = decode.seq_lens
        if seq_lens.ndim == 2:
            seq_lens = seq_lens[:, -1]
        seq_lens = seq_lens[:num_decode_tokens].tolist()
        q = q_all[:num_decode_tokens]
        w = w_all[:num_decode_tokens]
        out = topk_indices_buffer[:num_decode_tokens]
        for i, seq_len in enumerate(seq_lens):
            if seq_len <= 0:
                continue
            keys = _dequant_page_keys(
                kv_cache,
                decode.block_table[i],
                int(seq_len),
            )
            scores = torch.einsum("hd,kd->hk", q[i], keys)
            logits = (scores.relu() * w[i, :, None]).sum(dim=0).unsqueeze(0)
            _topk_from_logits(
                logits,
                topk_tokens,
                out[i : i + 1],
            )

    return topk_indices_buffer
