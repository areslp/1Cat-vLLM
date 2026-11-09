# SPDX-License-Identifier: Apache-2.0
"""CPU regression tests for the PPU sparse indexer reference path.

These tests deliberately build their expectation from the raw BF16 keys:
quantize them independently to FP8/E8M0, compute causal per-head ReLU-weighted
logits, and then take top-k.  They never use the implementation's own cache
readback as the oracle.
"""

import importlib.util
import pathlib
from types import SimpleNamespace

import torch

ROOT = pathlib.Path(__file__).resolve().parents[3]


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


idx = _load_module(
    "ppu_sparse_indexer_ref_under_test",
    "vllm/model_executor/layers/ppu_sparse_indexer_ref.py",
)
attn = _load_module(
    "ppu_sparse_attn_ref_under_test",
    "vllm/model_executor/layers/ppu_sparse_attn_ref.py",
)

HEAD_DIM = 128
BLOCK_SIZE = 64
TOPK = 2048


def _quantize_dequantize(raw_k: torch.Tensor) -> torch.Tensor:
    """Independent FP8/E8M0 round-trip straight from raw BF16 keys."""
    amax = raw_k.float().abs().amax(dim=-1).clamp_min(1e-4)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    quant = torch.clamp(raw_k.float() / scale[:, None], -448.0, 448.0).to(
        torch.float8_e4m3fn
    )
    return quant.float() * scale[:, None]


def _reference_logits(
    q: torch.Tensor,
    weights: torch.Tensor,
    raw_keys: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
) -> torch.Tensor:
    keys = _quantize_dequantize(raw_keys)
    scores = torch.einsum("qhd,kd->qhk", q.float(), keys)
    logits = (scores.relu() * weights[:, :, None]).sum(dim=1)
    positions = torch.arange(keys.shape[0])
    mask = (positions[None, :] >= ks[:, None]) & (positions[None, :] < ke[:, None])
    return logits.masked_fill(~mask, -torch.inf)


def _reference_topk(
    q: torch.Tensor,
    weights: torch.Tensor,
    raw_keys: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    logits = _reference_logits(q, weights, raw_keys, ks, ke)
    width = min(topk, logits.shape[1])
    values, indices = torch.topk(logits, width, dim=1)
    indices = torch.where(
        torch.isfinite(values), indices - ks[:, None], torch.full_like(indices, -1)
    )
    return indices.to(torch.int32)


def _expected_dq(raw_k: torch.Tensor) -> torch.Tensor:
    return _quantize_dequantize(raw_k)


def _assert_same_selection(actual: torch.Tensor, expected: torch.Tensor):
    # BLAS batching may reorder ties at the FP32 ulp level, so compare the
    # selected sets rather than the order returned by topk.
    assert actual.shape == expected.shape
    assert torch.equal(actual.sort(dim=1).values, expected.sort(dim=1).values)


def _slot_mapping(block_table: torch.Tensor, seq_len: int) -> torch.Tensor:
    positions = torch.arange(seq_len, dtype=torch.int64)
    blocks = block_table[positions // BLOCK_SIZE].to(torch.int64)
    return blocks * BLOCK_SIZE + positions % BLOCK_SIZE


def _write_keys(raw_keys: torch.Tensor, block_table: torch.Tensor, cache):
    slots = _slot_mapping(block_table, raw_keys.shape[0])
    idx._insert_indexer_k(raw_keys, cache, slots, HEAD_DIM)


def _stack_block_tables(block_tables):
    width = max(len(table) for table in block_tables)
    padded = [
        torch.cat([table, torch.zeros(width - len(table), dtype=table.dtype)])
        for table in block_tables
    ]
    return torch.stack(padded)


def _logit_scale(raw_keys: torch.Tensor) -> torch.Tensor:
    amax = raw_keys.float().abs().amax(dim=-1).clamp_min(1e-4)
    return torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))


def test_physical_page_layout_and_roundtrip():
    torch.manual_seed(609)
    num_tokens = 2 * BLOCK_SIZE
    cache = torch.zeros((2, BLOCK_SIZE, HEAD_DIM + 4), dtype=torch.uint8)
    raw = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16)
    raw *= torch.exp(torch.rand(num_tokens, 1) * 4.0)
    slots = torch.arange(num_tokens, dtype=torch.int64)
    idx._insert_indexer_k(raw, cache, slots, HEAD_DIM)

    page_bytes = BLOCK_SIZE * (HEAD_DIM + 4)
    flat = cache.reshape(-1)
    scale = _logit_scale(raw)
    quant = torch.clamp(raw.float() / scale[:, None], -448.0, 448.0).to(
        torch.float8_e4m3fn
    )
    quant_bytes = quant.view(torch.uint8).reshape(num_tokens, HEAD_DIM)
    scale_bytes = scale.view(torch.uint8).reshape(num_tokens, 4)

    for page in range(2):
        page_base = page * page_bytes
        page_keys = flat[page_base : page_base + BLOCK_SIZE * HEAD_DIM].reshape(
            BLOCK_SIZE, HEAD_DIM
        )
        page_scales = flat[
            page_base + BLOCK_SIZE * HEAD_DIM : page_base + page_bytes
        ].reshape(BLOCK_SIZE, 4)
        rows = slice(page * BLOCK_SIZE, (page + 1) * BLOCK_SIZE)
        assert torch.equal(page_keys, quant_bytes[rows])
        assert torch.equal(page_scales, scale_bytes[rows])

    restored = idx._dequant_page_keys(cache, torch.tensor([0, 1]), num_tokens)
    expected = _expected_dq(raw)
    assert torch.equal(restored, expected)
    assert torch.isfinite(restored).all()

    # The previous 3D slice wrote K with a 132-byte row pitch.  Verify that
    # rendering is observably different at the page tail, i.e. this regression
    # fails for the old layout instead of merely passing both implementations.
    old_keys = torch.zeros_like(flat)
    for page in range(2):
        for off in range(BLOCK_SIZE):
            dst = page * page_bytes + off * (HEAD_DIM + 4)
            src = page * BLOCK_SIZE + off
            old_keys[dst : dst + HEAD_DIM] = quant_bytes[src]
    assert not torch.equal(old_keys, flat)


def test_incremental_write_invalid_slots_and_cross_page():
    torch.manual_seed(610)
    seq_len = 2 * BLOCK_SIZE + 7
    block_table = torch.tensor([5, 1, 3], dtype=torch.int64)
    cache = torch.zeros((6, BLOCK_SIZE, HEAD_DIM + 4), dtype=torch.uint8)
    raw = torch.randn(seq_len, HEAD_DIM, dtype=torch.bfloat16) * 3
    slots = _slot_mapping(block_table, seq_len)

    idx._insert_indexer_k(raw[:BLOCK_SIZE], cache, slots[:BLOCK_SIZE], HEAD_DIM)
    before = cache.clone()

    second_slots = slots[BLOCK_SIZE:].clone()
    second_slots[:] = -1
    idx._insert_indexer_k(raw[BLOCK_SIZE:], cache, second_slots, HEAD_DIM)
    assert torch.equal(cache, before)

    second_slots = slots[BLOCK_SIZE:].clone()
    second_slots[0] = -1
    second_slots[3] = -1
    idx._insert_indexer_k(raw[BLOCK_SIZE:], cache, second_slots, HEAD_DIM)
    for token in (BLOCK_SIZE, BLOCK_SIZE + 3):
        block = int(slots[token]) // BLOCK_SIZE
        offset = int(slots[token]) % BLOCK_SIZE
        page = cache[block].reshape(-1)
        assert page[offset * HEAD_DIM : (offset + 1) * HEAD_DIM].sum() == 0
        assert (
            page[
                BLOCK_SIZE * HEAD_DIM + offset * 4 : BLOCK_SIZE * HEAD_DIM
                + (offset + 1) * 4
            ].sum()
            == 0
        )

    idx._insert_indexer_k(raw[BLOCK_SIZE:], cache, slots[BLOCK_SIZE:], HEAD_DIM)
    restored = idx._dequant_page_keys(cache, block_table, seq_len)
    assert torch.equal(restored, _expected_dq(raw))

    # A full page followed by a partial page must not bleed into the next page.
    page_bytes = BLOCK_SIZE * (HEAD_DIM + 4)
    flat = cache.reshape(-1)
    assert flat[5 * page_bytes : 6 * page_bytes - BLOCK_SIZE * 4].sum() != 0
    assert flat[1 * page_bytes : 2 * page_bytes - BLOCK_SIZE * 4].sum() != 0
    assert flat[:page_bytes].sum() == 0


def _make_cache(num_blocks: int):
    return torch.zeros((num_blocks, BLOCK_SIZE, HEAD_DIM + 4), dtype=torch.uint8)


def _build_chunks(
    block_tables,
    keys,
    q_lens,
    chunk_splits,
    token_offsets,
):
    """Build the prefill chunk metadata used by the reference indexer."""
    seq_lens = [k.shape[0] for k in keys]
    chunks = []
    for row, split in enumerate(chunk_splits):
        query_rows = []
        seq_starts = [0]
        for seq_len in seq_lens:
            seq_starts.append(seq_starts[-1] + seq_len)
        ks_parts = []
        ke_parts = []
        for req, q_len in enumerate(q_lens):
            lo, hi = split[req]
            ks_parts.append(torch.full((hi - lo,), seq_starts[req], dtype=torch.int32))
            prefix = seq_lens[req] - q_len
            ke_parts.append(
                torch.arange(prefix + lo + 1, prefix + hi + 1, dtype=torch.int32)
                + seq_starts[req]
            )
            query_rows.append((req, lo, hi))
        num_rows = sum(hi - lo for _, lo, hi in query_rows)
        chunks.append(
            SimpleNamespace(
                cu_seq_lens=torch.tensor(seq_starts, dtype=torch.int32),
                cu_seqlen_ks=torch.cat(ks_parts),
                cu_seqlen_ke=torch.cat(ke_parts),
                block_table=_stack_block_tables(
                    [block_tables[req] for req, _, _ in query_rows]
                ),
                token_start=token_offsets[row],
                token_end=token_offsets[row] + num_rows,
                num_reqs=len(q_lens),
            )
        )
    return chunks


def test_long_prefill_beyond_index_topk_multi_request():
    torch.manual_seed(611)
    seq_lens = [2304, 2112]
    q_lens = [6, 4]
    # Deliberately non-sequential pages, including page boundaries.
    generator = torch.Generator().manual_seed(611)
    block_tables = [
        torch.randperm(36, generator=generator),
        torch.randperm(33, generator=generator) + 36,
    ]
    cache = _make_cache(69)
    keys = [
        torch.randn(seq_len, HEAD_DIM, dtype=torch.bfloat16) * 2.5
        for seq_len in seq_lens
    ]
    slot_parts = []
    for block_table, raw in zip(block_tables, keys):
        _write_keys(raw, block_table, cache)
        slot_parts.append(_slot_mapping(block_table, raw.shape[0]))
    slot_mapping = torch.cat(slot_parts)

    q_len = sum(q_lens)
    num_heads = 4
    q = (torch.randn(q_len, num_heads, HEAD_DIM) * 2).to(torch.float8_e4m3fn)
    weights = torch.randn(q_len, num_heads)
    topk_buffer = torch.full((q_len, TOPK), -1, dtype=torch.int32)

    chunk_splits = [((0, 6), (0, 4))]
    token_offsets = [0]
    chunks = _build_chunks(block_tables, keys, q_lens, chunk_splits, token_offsets)
    metadata = SimpleNamespace(
        slot_mapping=slot_mapping,
        num_prefills=1,
        num_prefill_tokens=q_len,
        prefill=SimpleNamespace(chunks=chunks),
        num_decodes=0,
        num_decode_tokens=0,
        decode=None,
    )

    idx.ppu_reference_sparse_indexer(
        None,
        cache,
        q,
        None,
        weights,
        HEAD_DIM,
        TOPK,
        topk_buffer,
        True,
        metadata,
    )

    for req, (lo, hi) in enumerate(((0, 6), (6, 10))):
        expected = _reference_topk(
            q[lo:hi],
            weights[lo:hi],
            keys[req],
            torch.zeros(hi - lo, dtype=torch.int64),
            torch.arange(seq_lens[req] - (hi - lo) + 1, seq_lens[req] + 1),
            TOPK,
        )
        actual = topk_buffer[lo:hi]
        _assert_same_selection(actual, expected)
        # >2048 valid keys must actually drop positions, not just return all.
        assert int(actual[0].max()) < seq_lens[req]
        assert len(set(actual[0].tolist())) == TOPK
        # Every selected slot must be strictly before the causal boundary.
        for row in range(hi - lo):
            causal_end = seq_lens[req] - (hi - lo) + row + 1
            valid = actual[row][actual[row] >= 0]
            assert int(valid.max()) < causal_end


def test_chunked_prefill_single_request_query_slices():
    torch.manual_seed(614)
    seq_len = 2304
    q_len = 8
    block_table = torch.randperm(36, generator=torch.Generator().manual_seed(614))
    cache = _make_cache(36)
    raw = torch.randn(seq_len, HEAD_DIM, dtype=torch.bfloat16) * 2.5
    _write_keys(raw, block_table, cache)
    q = (torch.randn(q_len, 4, HEAD_DIM) * 2).to(torch.float8_e4m3fn)
    weights = torch.randn(q_len, 4)
    topk_buffer = torch.full((q_len, TOPK), -1, dtype=torch.int32)
    chunks = _build_chunks(
        [block_table],
        [raw],
        [q_len],
        [((0, 4),), ((4, 8),)],
        [0, 4],
    )
    metadata = SimpleNamespace(
        slot_mapping=_slot_mapping(block_table, seq_len),
        num_prefills=1,
        num_prefill_tokens=q_len,
        prefill=SimpleNamespace(chunks=chunks),
        num_decodes=0,
        num_decode_tokens=0,
        decode=None,
    )
    idx.ppu_reference_sparse_indexer(
        None,
        cache,
        q,
        None,
        weights,
        HEAD_DIM,
        TOPK,
        topk_buffer,
        True,
        metadata,
    )

    expected = _reference_topk(
        q,
        weights,
        raw,
        torch.zeros(q_len, dtype=torch.int64),
        torch.arange(seq_len - q_len + 1, seq_len + 1),
        TOPK,
    )
    _assert_same_selection(topk_buffer, expected)
    for row in range(q_len):
        causal_end = seq_len - q_len + row + 1
        valid = topk_buffer[row][topk_buffer[row] >= 0]
        assert int(valid.max()) < causal_end
    # The second query slice must still see the full prefix.
    assert int(topk_buffer[4:].max()) > seq_len // 2


def test_decode_multiple_requests_long_and_short():
    torch.manual_seed(612)
    seq_lens = [2304, 70]
    block_tables = [
        torch.arange(35, -1, -1, dtype=torch.int64),
        torch.tensor([80, 77], dtype=torch.int64),
    ]
    cache = _make_cache(82)
    keys = [
        torch.randn(seq_len, HEAD_DIM, dtype=torch.bfloat16) * 2 for seq_len in seq_lens
    ]
    for block_table, raw in zip(block_tables, keys):
        _write_keys(raw, block_table, cache)

    num_heads = 4
    q = (torch.randn(2, num_heads, HEAD_DIM) * 2).to(torch.float8_e4m3fn)
    weights = torch.randn(2, num_heads)
    topk_buffer = torch.full((2, TOPK), -1, dtype=torch.int32)
    decode = SimpleNamespace(
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        block_table=_stack_block_tables(block_tables),
    )
    metadata = SimpleNamespace(
        slot_mapping=torch.zeros(2, dtype=torch.int64),
        num_prefills=0,
        num_prefill_tokens=0,
        prefill=None,
        num_decodes=2,
        num_decode_tokens=2,
        decode=decode,
    )
    idx.ppu_reference_sparse_indexer(
        None,
        cache,
        q,
        None,
        weights,
        HEAD_DIM,
        TOPK,
        topk_buffer,
        True,
        metadata,
    )

    for req, seq_len in enumerate(seq_lens):
        expected = _reference_topk(
            q[req : req + 1],
            weights[req : req + 1],
            keys[req],
            torch.zeros(1, dtype=torch.int64),
            torch.tensor([seq_len]),
            TOPK,
        )
        actual = topk_buffer[req : req + 1]
        _assert_same_selection(actual[:, : expected.shape[1]], expected)
        assert (actual[:, expected.shape[1] :] == -1).all()
        if seq_len < TOPK:
            assert set(actual[0, :seq_len].tolist()) == set(range(seq_len))
            assert (actual[0, seq_len:] == -1).all()


def test_sparse_mla_latent_and_rope_semantics():
    torch.manual_seed(613)
    tokens, heads, slots, topk = 5, 3, 17, 9
    q = torch.randn(tokens, heads, 576, dtype=torch.bfloat16)
    cache = torch.randn(slots, 576, dtype=torch.bfloat16)
    global_slots = torch.randint(0, slots, (tokens, topk))
    global_slots[0, :] = -1
    out = attn.sparse_mla_attention_ref(
        q, cache, global_slots, scale=0.0625, kv_lora_rank=512, token_chunk=2
    )
    assert out.shape == (tokens, heads, 512)
    assert torch.count_nonzero(out[0]) == 0

    for token in range(1, tokens):
        valid = global_slots[token] >= 0
        selected = cache.index_select(0, global_slots[token][valid])
        scores = torch.einsum("hd,kd->hk", q[token], selected).float() * 0.0625
        probs = torch.softmax(scores, dim=-1)
        expected = torch.einsum("hk,kd->hd", probs, selected[:, :512].float())
        assert (out[token].float() - expected).abs().max().item() < 0.02

    bad_shape = torch.randn(tokens, heads, 512, dtype=torch.bfloat16)
    try:
        attn.sparse_mla_attention_ref(
            bad_shape, cache[:, :512], global_slots, 0.0625, 512
        )
    except ValueError:
        pass
    else:
        raise AssertionError("512-wide MLA attention must be rejected")


def _run_all():
    tests = [
        test_physical_page_layout_and_roundtrip,
        test_incremental_write_invalid_slots_and_cross_page,
        test_long_prefill_beyond_index_topk_multi_request,
        test_chunked_prefill_single_request_query_slices,
        test_decode_multiple_requests_long_and_short,
        test_sparse_mla_latent_and_rope_semantics,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    _run_all()
