# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only regression for the conservative E7 serving envelope."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.sm70_e7 import runtime as e


def params():
    return SamplingParams(temperature=0.6, top_k=20, top_p=0.95)


def batch():
    return SimpleNamespace(
        num_reqs=8,
        num_tokens=40,
        num_draft_tokens=32,
        logits_indices=torch.arange(40),
        is_prefilling_np=np.zeros(8, dtype=bool),
        num_draft_tokens_per_req=np.full(8, 4),
        cu_num_logits_np=np.arange(0, 41, 5),
    )


def test_supported_default():
    assert e.parameter_reason(params()) is None
    assert e.batch_reason(batch()) is None


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 9])
def test_other_counts_never_enter_e7(n):
    b = batch()
    b.num_reqs = n
    assert e.batch_reason(b) == "request_count"


@pytest.mark.parametrize(
    "key,value",
    [
        ("logprobs", 0),
        ("prompt_logprobs", 0),
        ("min_tokens", 1),
        ("repetition_penalty", 1.1),
        ("presence_penalty", 1.0),
        ("frequency_penalty", 1.0),
        ("min_p", 0.1),
        ("logit_bias", {1: 1}),
        ("allowed_token_ids", [1]),
        ("bad_words", ["test"]),
        ("logprob_token_ids", [1]),
        ("thinking_token_budget", 5),
        ("extra_args", {}),
        ("temperature", 0),
        ("top_k", 0),
        ("top_k", 257),
    ],
)
def test_feature_fallback(key, value):
    p = params()
    setattr(p, key, value)
    assert e.parameter_reason(p) is not None


@pytest.mark.parametrize("k", [1, 20, 255, 256])
def test_supported_topk(k):
    p = params()
    p.top_k = k
    assert e.parameter_reason(p) is None


def test_prefill_and_partial_mapping():
    b = batch()
    b.is_prefilling_np[-1] = True
    assert e.batch_reason(b) == "prefill"
    b = batch()
    b.num_draft_tokens_per_req[-1] = 3
    assert e.batch_reason(b) == "draft_shape"
    b = batch()
    b.cu_num_logits_np[-1] = 39
    assert e.batch_reason(b) == "logits_mapping"


def test_slot_reuse(monkeypatch):
    monkeypatch.setattr(e, "ENABLED", True)
    monkeypatch.setattr(e, "_STATS_DIR", "")
    sampler = SimpleNamespace()
    p = params()
    e.record_request(sampler, 2, p)
    assert sampler._e7_reasons[2] is None
    p.logprobs = 0
    e.record_request(sampler, 2, p)
    assert sampler._e7_reasons[2] == "feature:logprobs"


def test_request_modality_guard(monkeypatch):
    monkeypatch.setattr(e, "ENABLED", True)
    monkeypatch.setattr(e, "_STATS_DIR", "")
    sampler = SimpleNamespace()
    e.record_request(sampler, 1, params(), unsupported="multimodal_unvalidated")
    assert sampler._e7_reasons[1] == "multimodal_unvalidated"
    e.record_request(sampler, 1, params())
    assert sampler._e7_reasons[1] is None


def test_dynamic_fallback_reuses_original_local_logits():
    local = torch.arange(24, dtype=torch.float16).view(3, 8)
    batch_marker = object()
    draft_marker = object()
    called = []

    class Logits:
        def _gather_logits(self, x):
            assert x is local
            called.append("gather")
            return x

    def original_rejection(raw, batch, draft):
        assert torch.equal(raw, local)
        assert batch is batch_marker and draft is draft_marker
        called.append("original_rejection")
        return "baseline-output"

    runner = SimpleNamespace(
        rejection_sampler=original_rejection,
        speculator=SimpleNamespace(draft_logits=draft_marker),
    )
    assert (
        e._baseline_from_local(runner, batch_marker, Logits(), local)
        == "baseline-output"
    )
    assert called == ["gather", "original_rejection"]
