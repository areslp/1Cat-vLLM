# SPDX-License-Identifier: Apache-2.0
"""Pure-PyTorch sparse MLA attention reference for PPU.

Given a per-token list of selected cache slots, this gathers the selected
576-wide MLA cache rows, computes the QK scores, softmaxes over the selected
tokens, and multiplies by the 512-wide latent V.  Invalid slots are masked out.
"""

import torch


def sparse_mla_attention_ref(
    q: torch.Tensor,
    cache_flat: torch.Tensor,
    global_slots: torch.Tensor,
    scale: float,
    kv_lora_rank: int,
    token_chunk: int = 8,
) -> torch.Tensor:
    """q [T, H, 576], cache_flat [slots, 576], global_slots [T, K]."""
    if q.ndim != 3 or cache_flat.ndim != 2 or global_slots.ndim != 2:
        raise ValueError("Expected q/cache/global_slots to be 2D/3D tensors")
    if q.shape[0] != global_slots.shape[0]:
        raise ValueError("q and global_slots token counts differ")
    if cache_flat.shape[1] != q.shape[2]:
        raise ValueError("cache row width must equal q head width")
    if q.shape[2] != kv_lora_rank + 64:
        raise ValueError("PPU sparse MLA expects 512 latent + 64 RoPE")

    num_tokens = q.shape[0]
    output = q.new_empty((num_tokens, q.shape[1], kv_lora_rank))
    if num_tokens == 0:
        return output

    for start in range(0, num_tokens, token_chunk):
        end = min(start + token_chunk, num_tokens)
        q_chunk = q[start:end]
        slots_chunk = global_slots[start:end]
        valid = slots_chunk >= 0
        safe = slots_chunk.clamp_min(0)
        gathered = cache_flat.index_select(0, safe.reshape(-1)).view(
            *slots_chunk.shape, cache_flat.shape[1]
        )
        if not valid.all():
            gathered = gathered * valid.unsqueeze(-1).to(gathered.dtype)

        scores = torch.einsum("thd,tkd->thk", q_chunk, gathered) * scale
        scores = scores.float().masked_fill(~valid[:, None, :], -torch.inf)
        probs = torch.softmax(scores, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        v = gathered[..., :kv_lora_rank]
        output[start:end] = torch.einsum("thk,tkd->thd", probs, v.float()).to(q.dtype)
    return output
