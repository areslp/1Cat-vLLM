# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of portable collection; no live inference endpoint is used."""

import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "docs/sm70-main5dc-integration"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase_delta_sums_engines_and_keeps_missing_distinct_from_zero():
    replay = load("replay")
    before = replay.metric_values(
        'vllm:request_prefill_time_seconds_sum{engine="0"} 10\n'
        'vllm:request_prefill_time_seconds_count{engine="0"} 2\n'
    )
    after = replay.metric_values(
        'vllm:request_prefill_time_seconds_sum{engine="0"} 13\n'
        'vllm:request_prefill_time_seconds_count{engine="0"} 3\n'
        'vllm:request_prefill_time_seconds_sum{engine="1"} 5\n'
        'vllm:request_prefill_time_seconds_count{engine="1"} 1\n'
    )
    _, phases = replay.metric_delta(before, after)
    assert phases["request_prefill_time_seconds"] == {
        "count": 2,
        "sum_s": 8,
        "mean_s": 4,
    }
    assert phases["request_queue_time_seconds"]["mean_s"] is None


@pytest.mark.parametrize("after", [{"vllm:x_total": 1}, {}])
def test_counter_reset_or_disappearance_is_not_a_measurement(after):
    with pytest.raises(ValueError, match="reset/disappeared"):
        load("replay").metric_delta({"vllm:x_total": 2}, after)


@pytest.mark.parametrize("observed,completion", [(1, 2), (2, 2), (1, 1)])
def test_cli_collects_phases_and_flags_extra_requests_or_short_budget(
    monkeypatch, tmp_path, observed, completion
):
    replay = load("replay")
    tokens = [1, 2, 3]
    digest = hashlib.sha256(b"[1,2,3]").hexdigest()
    data = {
        "prompts": [
            {
                "id": "synthetic",
                "group": "cohort",
                "messages": [],
                "max_output_tokens": 2,
                "expected_input_tokens": 3,
                "input_token_sha256": digest,
            }
        ]
    }
    source = tmp_path / "observations.json"
    source.write_text(json.dumps(data))
    output = tmp_path / "run.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "replay",
            "--base-url",
            "http://synthetic.invalid",
            "--group",
            "cohort",
            "--observations",
            str(source),
            "--out",
            str(output),
            "--label",
            "U-timed",
        ],
    )
    scrapes: list[int] = []

    def urlopen(request, timeout):
        if isinstance(request, str):
            assert request.endswith("/metrics")
            n = observed if scrapes else 0
            scrapes.append(n)
            text = "vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n"
            for phase in replay.PHASES:
                text += f"vllm:{phase}_count {n}\nvllm:{phase}_sum {n * 0.1}\n"
            text += f"vllm:prompt_tokens_cached_total {n}\n"
            return io.BytesIO(text.encode())
        if request.full_url.endswith("/tokenize"):
            return io.BytesIO(json.dumps({"tokens": tokens, "count": 3}).encode())
        body = json.loads(request.data)
        assert body["max_tokens"] == 2 and body["seed"] == 1234
        packet = {
            "choices": [
                {
                    "delta": {"content": "ok"},
                    "finish_reason": "length" if completion == 2 else "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": completion},
        }
        return io.BytesIO(
            ("data: " + json.dumps(packet) + "\n\ndata: [DONE]\n").encode()
        )

    monkeypatch.setattr(replay.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(replay.time, "sleep", lambda _: None)
    if observed == 1 and completion == 2:
        replay.main()
    else:
        with pytest.raises(SystemExit, match="unqualified"):
            replay.main()
    result = json.loads(output.read_text())
    assert result["qualified_metric_counts"] == (observed == 1)
    assert result["fixed_output_budget_completed"] == (completion == 2)
    assert result["server_metric_delta"]["vllm:prompt_tokens_cached_total"] == observed
    assert output.with_suffix(".metrics-before.txt").is_file()
    assert output.with_suffix(".metrics-after.txt").is_file()


def test_cache_snapshot_detects_same_size_content_change(tmp_path):
    module = load("cache_snapshot")
    data = tmp_path / "kernel.py"
    data.write_text("before")
    first = module.snapshot(tmp_path)
    data.write_text("after!")
    second = module.snapshot(tmp_path)
    assert first["kernel.py"]["bytes"] == second["kernel.py"]["bytes"]
    assert first["kernel.py"]["sha256"] != second["kernel.py"]["sha256"]
