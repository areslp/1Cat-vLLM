# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse Mamba state-snapshot retention (``prefix_cache_retention_interval``).

In ``mamba_cache_mode="align"`` every Mamba group snapshots its recurrent
state at each block boundary and frees the superseded snapshot into the free
queue. Retaining every snapshot (dense, the default) makes one long prefill
cycle the whole free queue and evict other requests' cached attention blocks
(1CatAI/1Cat-vLLM#490). With a retention interval only reachable snapshots
keep a hash; the rest are reused before any other free block.
"""

import pytest

from tests.v1.core.test_prefix_caching import (
    _make_hybrid_kv_cache_config,
    make_request,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    get_group_id,
    init_none_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.core.single_type_kv_cache_manager import MambaManager

BLOCK_SIZE = 16


@pytest.fixture(autouse=True)
def _none_hash():
    init_none_hash(sha256)


def _free_order(reuse_unhashed_first: bool) -> list[int]:
    pool = BlockPool(
        num_gpu_blocks=6,
        enable_caching=True,
        hash_block_size=BLOCK_SIZE,
        reuse_unhashed_first=reuse_unhashed_first,
    )
    b1, b2, b3 = pool.get_new_blocks(3)
    b2.block_hash = make_block_hash_with_group_id(BlockHash(b"hash-b2"), 0)
    pool.free_blocks([b1, b2, b3])
    if reuse_unhashed_first:
        # Unhashed frees are held back until the next step starts.
        assert [b.block_id for b in pool.free_block_queue.get_all_free_blocks()] == [
            4,
            5,
            2,
        ]
        pool.flush_pending_front()
    return [b.block_id for b in pool.free_block_queue.get_all_free_blocks()]


def test_free_blocks_default_order_unchanged():
    # Default: every freed block is appended in the given order.
    assert _free_order(reuse_unhashed_first=False) == [4, 5, 1, 2, 3]


def test_free_blocks_reuse_unhashed_first():
    # Unhashed blocks go to the front (in order), hashed ones to the back.
    assert _free_order(reuse_unhashed_first=True) == [1, 3, 4, 5, 2]


def test_retention_block_mask_dense_when_unset():
    assert (
        MambaManager.retention_block_mask(
            start_block=0,
            end_block=6,
            block_size=BLOCK_SIZE,
            alignment_tokens=BLOCK_SIZE,
            retention_interval=None,
            replay_boundaries=(95,),
        )
        is None
    )


def test_retention_block_mask_keeps_only_replay_boundary():
    # 6 full blocks; a 103-token prompt is reachable at 102 (resend) and
    # 103 (extension), both aligned down to token 96 = block index 5.
    mask = MambaManager.retention_block_mask(
        start_block=0,
        end_block=6,
        block_size=BLOCK_SIZE,
        alignment_tokens=BLOCK_SIZE,
        retention_interval=0,
        replay_boundaries=(102, 103),
    )
    assert mask == [False, False, False, False, False, True]


def test_retention_block_mask_block_aligned_prompt_keeps_both():
    # A 96-token prompt: resend hits at most 95 tokens (block index 4),
    # an extension matches block index 5. Both must stay.
    mask = MambaManager.retention_block_mask(
        start_block=0,
        end_block=6,
        block_size=BLOCK_SIZE,
        alignment_tokens=BLOCK_SIZE,
        retention_interval=0,
        replay_boundaries=(95, 96),
    )
    assert mask == [False, False, False, False, True, True]


def test_retention_block_mask_periodic_segments():
    # 32-token interval = 2 blocks per segment: keep the last block of each
    # segment (indices 1, 3, 5) plus the replay boundary at index 5.
    mask = MambaManager.retention_block_mask(
        start_block=0,
        end_block=6,
        block_size=BLOCK_SIZE,
        alignment_tokens=BLOCK_SIZE,
        retention_interval=32,
        replay_boundaries=(102, 103),
    )
    assert mask == [False, True, False, True, False, True]
    # Chunked caching: the window [2, 6) sees the same segment tails.
    mask = MambaManager.retention_block_mask(
        start_block=2,
        end_block=6,
        block_size=BLOCK_SIZE,
        alignment_tokens=BLOCK_SIZE,
        retention_interval=32,
        replay_boundaries=(102, 103),
    )
    assert mask == [False, True, False, True]


def test_retention_block_mask_interval_at_block_size_is_dense():
    assert (
        MambaManager.retention_block_mask(
            start_block=0,
            end_block=6,
            block_size=BLOCK_SIZE,
            alignment_tokens=BLOCK_SIZE,
            retention_interval=BLOCK_SIZE,
            replay_boundaries=(102,),
        )
        is None
    )


# ---------------------------------------------------------------------------
# End to end through KVCacheManager with a full-attention + Mamba(align) model
# ---------------------------------------------------------------------------

MAMBA_GROUP_ID = 1


def _make_manager(retention_interval: int | None, num_blocks: int = 40):
    kv_cache_config = _make_hybrid_kv_cache_config(
        BLOCK_SIZE, num_blocks, ["full", "mamba_align"]
    )
    kv_cache_config.prefix_cache_retention_interval = retention_interval
    return KVCacheManager(
        kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=BLOCK_SIZE,
    )


def _prefill(manager: KVCacheManager, req, chunk: int = BLOCK_SIZE) -> int:
    """Chunked prefill ending every chunk on a block boundary (align mode).
    Returns the prefix-cache hit length."""
    computed_blocks, num_hit = manager.get_computed_blocks(req)
    pos = num_hit
    first = True
    while pos < req.num_tokens:
        num_new = min(chunk, req.num_tokens - pos)
        manager.new_step_starts()
        blocks = manager.allocate_slots(
            req,
            num_new,
            num_hit if first else 0,
            computed_blocks if first else None,
        )
        assert blocks is not None
        pos += num_new
        req.num_computed_tokens = pos
        first = False
    return num_hit


def _num_cached_mamba_hashes(manager: KVCacheManager) -> int:
    cache = manager.block_pool.cached_block_hash_to_block._cache
    return sum(1 for key in cache if get_group_id(key) == MAMBA_GROUP_ID)


def _record_pops(manager: KVCacheManager) -> list[int]:
    popped: list[int] = []
    original = manager.block_pool.get_new_blocks

    def recording_get_new_blocks(num_blocks: int):
        blocks = original(num_blocks)
        popped.extend(block.block_id for block in blocks)
        return blocks

    manager.block_pool.get_new_blocks = recording_get_new_blocks  # type: ignore[method-assign]
    return popped


# 6 full blocks (96 tokens) + 7 partial tokens.
_PROMPT = [i for i in range(6) for _ in range(BLOCK_SIZE)] + [6] * 7
# Shares the first 3 blocks (48 tokens) with _PROMPT, then diverges.
_SIBLING = _PROMPT[: 3 * BLOCK_SIZE] + [7] * 20


@pytest.mark.parametrize(
    ("retention_interval", "expected_mamba_hashes", "expected_sibling_hit"),
    [
        # Dense: every boundary snapshot stays cached, a sibling sharing 3
        # blocks resumes at 48 tokens.
        (None, 6, 48),
        # Only the prompt-end boundary survives: the sibling cannot resume.
        (0, 1, 0),
        # One snapshot per 32 tokens (block indices 1, 3, 5): the sibling
        # resumes at the nearest retained boundary at or before 48 -> 32.
        (32, 3, 32),
    ],
)
def test_mamba_align_retention_end_to_end(
    retention_interval, expected_mamba_hashes, expected_sibling_hit
):
    manager = _make_manager(retention_interval)

    req0 = make_request("0", _PROMPT, BLOCK_SIZE, sha256)
    assert _prefill(manager, req0) == 0
    assert _num_cached_mamba_hashes(manager) == expected_mamba_hashes
    manager.free(req0)

    # An identical resend hits every full block (the last token is recomputed).
    req1 = make_request("1", _PROMPT, BLOCK_SIZE, sha256)
    _, num_hit = manager.get_computed_blocks(req1)
    assert num_hit == 6 * BLOCK_SIZE

    # Partial-prefix sharing needs a snapshot at (or before) the shared end.
    req2 = make_request("2", _SIBLING, BLOCK_SIZE, sha256)
    _, num_hit = manager.get_computed_blocks(req2)
    assert num_hit == expected_sibling_hit


def test_mamba_align_retention_recycles_superseded_snapshots():
    """With retention on, superseded snapshots are reused by the next
    boundary instead of consuming fresh blocks from the free queue."""
    distinct: dict[int | None, int] = {}
    for retention_interval in (None, 0):
        manager = _make_manager(retention_interval)
        popped = _record_pops(manager)
        req = make_request("0", _PROMPT, BLOCK_SIZE, sha256)
        _prefill(manager, req)
        distinct[retention_interval] = len(set(popped))
        manager.free(req)
    # Dense: 7 attention blocks + one fresh Mamba block per boundary.
    # Retention 0: the same 7 attention blocks, Mamba cycles a few blocks.
    assert distinct[0] < distinct[None]
    assert distinct[0] <= 7 + 3
