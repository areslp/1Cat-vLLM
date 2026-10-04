# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical checks for E7's isolated ambiguous-row reference resolver.

These exercise the resolver bundled with E7. They intentionally do not change
or call the shared upstream top-k/top-p implementation.
"""

import pytest
import torch

from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
from vllm.v1.worker.gpu.sample.sm70_e7 import packet_topk

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

V = 32768
KC = 256


def _logits(batch_size: int, k: int, extra_ties: int, temperature: float, seed: int):
    """Build half-precision-grid logits with optional ties at the k cutoff."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    logits = (torch.randn(batch_size, V, generator=generator) * 3).half().float()
    logits[:, :50] += 12
    logits = (logits.half().float() / temperature).cuda()
    if extra_ties:
        kth = logits.sort(dim=-1, descending=True).values[:, k - 1].clone()
        for row in range(batch_size):
            ids = (logits[row] < kth[row]).nonzero().flatten()[:extra_ties]
            logits[row, ids] = kth[row]
    return logits


def _resolve(logits, k, p, reference_rows=None, initial=None):
    values, indices = packet_topk._sorted_candidates(logits, KC)
    if reference_rows is None:
        reference_rows = torch.ones(
            logits.shape[0], dtype=torch.bool, device=logits.device
        )
    if initial is None:
        initial = torch.full_like(logits, 123.0)
    return packet_topk._resolve_reference_rows_nosync(
        initial, reference_rows, values, indices, k, p
    )


def _bits_equal(a, b):
    return torch.equal(
        a.contiguous().view(torch.int32), b.contiguous().view(torch.int32)
    )


@pytest.mark.parametrize("extra_ties", [0, 8, 40])
@pytest.mark.parametrize("top_p", [None, 0.95, 0.5])
@pytest.mark.parametrize("k_value", [1, 20, 64])
def test_resolved_rows_match_pytorch_reference(k_value, top_p, extra_ties):
    logits = _logits(5, k_value, extra_ties, 0.6, 1000 + k_value + extra_ties)
    k = torch.full((5,), k_value, dtype=torch.int32, device="cuda")
    p = None if top_p is None else torch.full((5,), top_p, device="cuda")

    actual = _resolve(logits, k, p)
    expected = apply_top_k_top_p_pytorch(logits.clone(), k, p)

    assert _bits_equal(actual, expected)


@pytest.mark.parametrize("num_flagged", [1, 3])
def test_single_and_multiple_flagged_rows(num_flagged):
    batch_size = 5
    logits = _logits(batch_size, 20, 8, 0.6, 77)
    flags = torch.zeros(batch_size, dtype=torch.bool, device="cuda")
    rows = torch.arange(1, 1 + num_flagged, device="cuda")
    flags[rows] = True
    k = torch.full((batch_size,), 20, dtype=torch.int32, device="cuda")
    p = torch.full((batch_size,), 0.95, device="cuda")
    initial = torch.full_like(logits, 123.0)
    untouched = initial.clone()

    actual = _resolve(logits, k, p, flags, initial)
    expected = apply_top_k_top_p_pytorch(logits.clone(), k, p)

    assert _bits_equal(actual[flags], expected[flags])
    assert torch.equal(actual[~flags], untouched[~flags])


@pytest.mark.parametrize("case", ["all_equal", "ties_after_head"])
def test_unresolved_cutoff_ties_leave_kernel_output_untouched(case):
    batch_size = 2
    logits = torch.full((batch_size, V), -20.0, device="cuda")
    if case == "all_equal":
        logits[:] = 1.0
    else:
        logits[:, :10] = torch.arange(20, 10, -1, device="cuda")
        logits[:, 10:310] = 3.0
    k = torch.full((batch_size,), 20, dtype=torch.int32, device="cuda")
    p = torch.full((batch_size,), 0.95, device="cuda")
    flags = torch.ones(batch_size, dtype=torch.bool, device="cuda")
    initial = torch.randn_like(logits)
    untouched = initial.clone()
    before = packet_topk.nosync_unresolved_rows("cuda")

    actual = _resolve(logits, k, p, flags, initial)

    assert _bits_equal(actual, untouched)
    assert packet_topk.nosync_unresolved_rows("cuda") - before == batch_size


def test_resolver_does_not_host_synchronize():
    logits = _logits(5, 20, 8, 0.6, 1)
    k = torch.full((5,), 20, dtype=torch.int32, device="cuda")
    p = torch.full((5,), 0.95, device="cuda")
    packet_topk._sorted_candidates(logits, KC)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        _resolve(logits, k, p)
    finally:
        torch.cuda.set_sync_debug_mode("default")


@pytest.mark.parametrize("case", ["k_tie", "p_tie", "unique"])
@pytest.mark.parametrize("top_p", [1.0, 0.95, 0.6])
def test_tied_cutoff_patterns_match_reference(case, top_p):
    logits = torch.full((3, V), -20.0, device="cuda")
    if case == "k_tie":
        logits[:, :24] = 1.0
        logits[:, :18] = 2.0
    elif case == "p_tie":
        logits[:, :20] = 1.0
        logits[:, :2] = 2.0
    else:
        logits[:, :32] = torch.arange(32, 0, -1, device="cuda") / 8
    k = torch.full((3,), 20, dtype=torch.int32, device="cuda")
    p = torch.full((3,), top_p, device="cuda")

    actual = _resolve(logits, k, p)
    expected = apply_top_k_top_p_pytorch(logits.clone(), k, p)

    assert _bits_equal(actual, expected)


def test_signed_zero_cutoff_ties_match_reference():
    batch_size = 3
    logits = torch.full((batch_size, V), -30.0, device="cuda")
    logits[:, :15] = torch.linspace(0.9, 0.1, 15, device="cuda")
    logits[:, 15:40:2] = -0.0
    logits[:, 16:40:2] = 0.0
    logits[1] = torch.randn(V, device="cuda")
    k = torch.full((batch_size,), 20, dtype=torch.int32, device="cuda")

    for top_p in (
        None,
        torch.full((batch_size,), 0.6, device="cuda"),
        torch.full((batch_size,), 0.8, device="cuda"),
    ):
        actual = _resolve(logits, k, top_p)
        expected = apply_top_k_top_p_pytorch(logits.clone(), k, top_p)
        assert _bits_equal(actual, expected)
