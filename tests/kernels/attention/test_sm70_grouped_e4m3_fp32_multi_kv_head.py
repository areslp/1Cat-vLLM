# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-KV-head E4M3 FP32 grouped verify for TP ranks holding several KV heads.

The native entry serves one KV head and its six query heads. A TP2 rank of a
24/4-head model holds two KV heads, so the opt-in route slices the query and
the paged KV per head and runs the entry twice. Each head's result must equal
the entry called on a contiguous single-head copy, and the FP64 oracle bound
of the single-head tests must hold.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.ops import sm70_e4m3_grouped as ops


def _native():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    module = pytest.importorskip("flash_attn_v100")
    if not module.flash_attn_grouped_e4m3_fp32_available():
        pytest.skip("rebuild native E4M3 FP32 entry")
    return module.flash_attn_grouped_e4m3_fp32_paged


class _Instance:
    def __init__(self, op):
        self.flash_attn_grouped_e4m3_fp32_paged = op
        self.kv_cache_dtype = "fp8_e4m3"
        self.use_smallq_decode_xqa = True

    def _flash_v100_window_size(self, causal=True):
        return (-1, -1)


def _case(rows, page, length, num_kv_heads, seed):
    torch.manual_seed(seed)
    pages = (length + page - 1) // page
    capacity = pages * page
    q = torch.randn(
        (rows, ops.GROUP_SIZE * num_kv_heads, 256), device="cuda", dtype=torch.float16
    )
    raw = torch.randn(
        (2, capacity, num_kv_heads, 256), device="cuda", dtype=torch.float16
    )
    encoded = raw.to(torch.float8_e4m3fn).view(torch.uint8)
    backing = torch.empty(
        (pages, 2, page, num_kv_heads, 256), device="cuda", dtype=torch.uint8
    )
    k, v = backing.unbind(1)
    order = torch.randperm(pages, device="cuda")
    k[order] = encoded[0].reshape_as(k)
    v[order] = encoded[1].reshape_as(v)
    parent_table = order.int()[None].contiguous()
    parent_seq = torch.tensor([length], device="cuda", dtype=torch.int32)
    lengths = torch.arange(
        length - rows + 1, length + 1, device="cuda", dtype=torch.int32
    )
    table = parent_table.expand(rows, -1).contiguous()
    metadata = SimpleNamespace(
        block_table=parent_table, seq_lens=parent_seq, causal=True
    )
    return q, k, v, encoded, table, lengths, metadata


def _reference_single_head(op, q, k, v, table, lengths, head, ks, vs):
    heads = slice(head * ops.GROUP_SIZE, (head + 1) * ops.GROUP_SIZE)
    q_h = q[:, heads, :].contiguous()
    k_h = k[:, :, head : head + 1, :].contiguous()
    v_h = v[:, :, head : head + 1, :].contiguous()
    out_h = torch.empty_like(q_h)
    op(
        q_h,
        k_h,
        v_h,
        table[:1],
        lengths,
        out=out_h,
        softmax_scale=0.0625,
        k_scale=ks,
        v_scale=vs,
    )
    return out_h


@pytest.mark.parametrize(
    "rows,page,length",
    [
        (5, 1616, 8197),
        (8, 1648, 8197),
        (5, 1616, 65536),
        (8, 1648, 131072),
        (2, 848, 1700),
    ],
)
def test_per_kv_head_matches_contiguous_single_head_and_oracle(
    monkeypatch, rows, page, length
):
    op = _native()
    monkeypatch.setenv(ops.MULTI_KV_HEAD_ENV, "1")
    num_kv_heads = 2
    q, k, v, encoded, table, lengths, metadata = _case(
        rows, page, length, num_kv_heads, 20260914
    )
    ks, vs = 0.5, 1.25
    out = torch.empty_like(q)
    ran = ops.run_grouped_e4m3_fp32_per_kv_head(
        op,
        _Instance(op),
        q,
        k,
        v,
        table,
        lengths,
        metadata,
        out=out,
        softmax_scale=0.0625,
        k_scale=ks,
        v_scale=vs,
        partition_size_hint=None,
    )
    assert ran
    for head in range(num_kv_heads):
        expected = _reference_single_head(op, q, k, v, table, lengths, head, ks, vs)
        heads = slice(head * ops.GROUP_SIZE, (head + 1) * ops.GROUP_SIZE)
        assert torch.equal(out[:, heads, :], expected)
        # FP64 oracle on the same quantized KV, as in the single-head tests.
        rk = encoded[0, :length, head].view(torch.float8_e4m3fn).double() * ks
        rv = encoded[1, :length, head].view(torch.float8_e4m3fn).double() * vs
        score = q[:, heads, :].transpose(0, 1).double() @ rk.T * 0.0625
        mask = torch.arange(length, device="cuda")[None] >= lengths[:, None]
        score.masked_fill_(mask[None], -torch.inf)
        oracle = (score.softmax(-1) @ rv).transpose(0, 1)
        relative_l2 = (out[:, heads, :].double() - oracle).norm() / oracle.norm()
        assert float(relative_l2) < 0.001


def test_per_kv_head_is_graph_replayable(monkeypatch):
    op = _native()
    monkeypatch.setenv(ops.MULTI_KV_HEAD_ENV, "1")
    rows, page, length = 5, 1616, 4000
    q, k, v, _, table, lengths, metadata = _case(rows, page, length, 2, 20260915)
    out = torch.empty_like(q)

    def call():
        assert ops.run_grouped_e4m3_fp32_per_kv_head(
            op,
            _Instance(op),
            q,
            k,
            v,
            table,
            lengths,
            metadata,
            out=out,
            softmax_scale=0.0625,
            k_scale=1.0,
            v_scale=1.0,
            partition_size_hint=None,
        )

    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for _ in range(2):
        q.normal_()
        expected = torch.cat(
            [
                _reference_single_head(op, q, k, v, table, lengths, head, 1.0, 1.0)
                for head in range(2)
            ],
            dim=1,
        )
        graph.replay()
        assert torch.equal(out, expected)


def test_route_is_opt_in_and_shape_gated(monkeypatch):
    op = _native()
    rows, page, length = 5, 1616, 4000
    q, k, v, _, table, lengths, metadata = _case(rows, page, length, 2, 20260916)
    out = torch.full_like(q, 7.0)
    kwargs = dict(
        out=out,
        softmax_scale=0.0625,
        k_scale=1.0,
        v_scale=1.0,
        partition_size_hint=None,
    )
    monkeypatch.delenv(ops.MULTI_KV_HEAD_ENV, raising=False)
    assert not ops.run_grouped_e4m3_fp32_per_kv_head(
        op, _Instance(op), q, k, v, table, lengths, metadata, **kwargs
    )
    monkeypatch.setenv(ops.MULTI_KV_HEAD_ENV, "1")
    # Single KV head is the native entry's own shape, not this route.
    assert (
        ops.grouped_e4m3_fp32_kv_head_views(
            q[:, :6, :], k[:, :, :1, :], v[:, :, :1, :], out[:, :6, :]
        )
        is None
    )
    # Two requests in the batch are rejected by the single-request gate.
    two = SimpleNamespace(
        block_table=torch.cat([metadata.block_table] * 2),
        seq_lens=torch.cat([metadata.seq_lens] * 2),
        causal=True,
    )
    assert not ops.run_grouped_e4m3_fp32_per_kv_head(
        op, _Instance(op), q, k, v, table, lengths, two, **kwargs
    )
    assert bool((out == 7.0).all())


# ---------------------------------------------------------------------------
# Legacy E5M2 one-pass verifier (DFlash2 q8) on per-KV-head strided views.
# ---------------------------------------------------------------------------


def _native_e5m2():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    module = pytest.importorskip("flash_attn_v100")
    return module.flash_attn_grouped_verify_paged


@pytest.mark.parametrize("page,length", [(1648, 8197), (1648, 65536), (3296, 131072)])
def test_e5m2_one_pass_verifier_per_kv_head_matches_contiguous(page, length):
    op = _native_e5m2()
    torch.manual_seed(20260917)
    rows, num_kv_heads = 8, 2
    pages = (length + page - 1) // page
    capacity = pages * page
    q = torch.randn(
        (rows, ops.GROUP_SIZE * num_kv_heads, 256), device="cuda", dtype=torch.float16
    )
    raw = torch.randn(
        (2, capacity, num_kv_heads, 256), device="cuda", dtype=torch.float16
    )
    encoded = raw.to(torch.float8_e5m2).view(torch.uint8)
    backing = torch.empty(
        (pages, 2, page, num_kv_heads, 256), device="cuda", dtype=torch.uint8
    )
    k, v = backing.unbind(1)
    order = torch.randperm(pages, device="cuda")
    k[order] = encoded[0].reshape_as(k)
    v[order] = encoded[1].reshape_as(v)
    table = order.int()[None].contiguous()
    seq_lens = torch.tensor([length], device="cuda", dtype=torch.int32)
    views = ops.grouped_e4m3_fp32_kv_head_views(q, k, v, torch.empty_like(q))
    assert views is not None and len(views) == num_kv_heads
    for head, (q_h, k_h, v_h, out_h) in enumerate(views):
        op(
            q_h,
            k_h,
            v_h,
            table,
            seq_lens,
            softmax_scale=0.0625,
            out=out_h,
            kv_cache_dtype="fp8_e5m2",
            k_scale=0.5,
            v_scale=1.25,
            one_pass=True,
        )
        expected = torch.empty_like(q_h)
        op(
            q_h,
            k_h.contiguous(),
            v_h.contiguous(),
            table,
            seq_lens,
            softmax_scale=0.0625,
            out=expected,
            kv_cache_dtype="fp8_e5m2",
            k_scale=0.5,
            v_scale=1.25,
            one_pass=True,
        )
        assert torch.equal(out_h, expected)
        # FP64 oracle on the same quantized KV (causal rows ending at length).
        rk = encoded[0, :length, head].view(torch.float8_e5m2).double() * 0.5
        rv = encoded[1, :length, head].view(torch.float8_e5m2).double() * 1.25
        lengths = torch.arange(length - rows + 1, length + 1, device="cuda")
        score = q_h.transpose(0, 1).double() @ rk.T * 0.0625
        mask = torch.arange(length, device="cuda")[None] >= lengths[:, None]
        score.masked_fill_(mask[None], -torch.inf)
        oracle = (score.softmax(-1) @ rv).transpose(0, 1)
        relative_l2 = (out_h.double() - oracle).norm() / oracle.norm()
        assert float(relative_l2) < 0.01
