# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay one published synthetic cohort against an ordinary OpenAI endpoint."""

import argparse
import concurrent.futures
import hashlib
import json
import threading
import time
import urllib.request
from pathlib import Path

from prometheus_client.parser import text_string_to_metric_families

PHASES = (
    "request_queue_time_seconds",
    "request_prefill_time_seconds",
    "request_decode_time_seconds",
    "time_to_first_token_seconds",
    "e2e_request_latency_seconds",
)


def metrics_snapshot(base, output):
    with urllib.request.urlopen(base + "/metrics", timeout=15) as response:
        raw = response.read().decode()
    output.write_text(raw)
    return metric_values(raw)


def metric_values(raw):
    values = {}
    for family in text_string_to_metric_families(raw):
        for sample in family.samples:
            name = sample.name
            if name.startswith("vllm:") and (
                name.endswith(("_sum", "_count", "_total"))
                or name in ("vllm:num_requests_running", "vllm:num_requests_waiting")
            ):
                values[name] = values.get(name, 0) + sample.value
    return values


def metric_delta(before, after):
    delta = {
        name: value - before.get(name, 0)
        for name, value in after.items()
        if name.endswith(("_sum", "_count", "_total"))
    }
    if any(value < 0 for value in delta.values()) or before.keys() - after.keys():
        raise ValueError("Metrics reset/disappeared; retain snapshots and repeat")
    phases = {}
    for phase in PHASES:
        count = delta.get("vllm:" + phase + "_count")
        total = delta.get("vllm:" + phase + "_sum")
        phases[phase] = {
            "count": count,
            "sum_s": total,
            "mean_s": total / count if count and total is not None else None,
        }
    return delta, phases


def post(url, body):
    payload = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
    request = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(request, timeout=240), hashlib.sha256(
        payload
    ).hexdigest()


def validate(base, case):
    messages = case["messages"]
    template = {"enable_thinking": False}
    response, _ = post(
        base + "/tokenize",
        {"messages": messages, "model": "flash-next", "chat_template_kwargs": template},
    )
    with response:
        tokens = json.load(response)
    digest = hashlib.sha256(
        json.dumps(tokens["tokens"], separators=(",", ":")).encode()
    ).hexdigest()
    assert tokens["count"] == case["expected_input_tokens"]
    assert digest == case["input_token_sha256"], "different tokenizer/template"


def run(base, case, barrier):
    messages = case["messages"]
    template = {"enable_thinking": False}
    body = {
        "model": "flash-next",
        "messages": messages,
        "temperature": 0,
        "top_p": 1,
        "repetition_penalty": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "max_tokens": case["max_output_tokens"],
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": template,
        "seed": 1234,
    }
    barrier.wait()
    started = time.perf_counter()
    response, request_sha = post(base + "/v1/chat/completions", body)
    text, reasoning, usage, done, finish = "", "", None, False, None
    first, last = None, None
    with response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                done = True
                break
            row = json.loads(data)
            assert "error" not in row, row
            usage = row.get("usage") or usage
            for choice in row.get("choices", []):
                chunk = choice.get("delta") or {}
                content = chunk.get("content") or ""
                thought = chunk.get("reasoning_content") or ""
                if content or thought:
                    last = time.perf_counter() - started
                    if first is None:
                        first = last
                text += content
                reasoning += thought
                finish = choice.get("finish_reason") or finish
    assert done and usage and finish
    assert usage["prompt_tokens"] == case["expected_input_tokens"]
    return {
        "id": case["id"],
        "request_sha256": request_sha,
        "input_token_sha256": case["input_token_sha256"],
        "text": text,
        "reasoning": reasoning,
        "usage": usage,
        "finish_reason": finish,
        "wall_s": time.perf_counter() - started,
        "ttft_s": first,
        "last_text_chunk_s": last,
        "input_expected": case["expected_input_tokens"],
        "requested_output": case["max_output_tokens"],
        "fixed_output_budget_completed": (
            usage["completion_tokens"] == case["max_output_tokens"]
            and finish == "length"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--label", required=True, help="Arm and warm/timed phase")
    parser.add_argument("--reset-prefix-cache", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.observations.read_text())
    cases = [
        case
        for case in payload["prompts"]
        if case.get("group", case["id"]) == args.group or case["id"] == args.group
    ]
    assert cases, args.group
    base = args.base_url.rstrip("/")
    for case in cases:
        validate(base, case)
    if args.reset_prefix_cache:
        request = urllib.request.Request(
            base + "/reset_prefix_cache", data=b"", method="POST"
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            assert response.status == 200
    args.out.parent.mkdir(parents=True, exist_ok=True)
    before = metrics_snapshot(base, args.out.with_suffix(".metrics-before.txt"))
    for name in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
        assert before[name] == 0, "Use an idle dedicated comparison endpoint"
    barrier = threading.Barrier(len(cases) + 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cases)) as pool:
        futures = [pool.submit(run, base, case, barrier) for case in cases]
        started = time.perf_counter()
        barrier.wait(timeout=60)
        rows = [future.result() for future in futures]
        wall = time.perf_counter() - started
    # Match the historical collector: allow the emitter one event-loop tick.
    time.sleep(0.5)
    after = metrics_snapshot(base, args.out.with_suffix(".metrics-after.txt"))
    delta, phases = metric_delta(before, after)
    total = sum(row["usage"]["completion_tokens"] for row in rows)
    counts_match = all(phases[p]["count"] == len(cases) for p in PHASES)
    budget = all(row["fixed_output_budget_completed"] for row in rows)
    result = {
        "group": args.group,
        "label": args.label,
        "concurrency": len(cases),
        "requests": rows,
        "wall_s": wall,
        "actual_output_tokens": total,
        "group_output_tok_s": total / wall,
        "successful": True,
        "fixed_output_budget_completed": budget,
        "server_metric_delta": delta,
        "server_phases": phases,
        "external_traffic_contamination": not counts_match,
        "qualified_metric_counts": counts_match,
        "prefix_reset_performed": args.reset_prefix_cache,
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps({key: value for key, value in result.items() if key != "requests"})
    )
    if not counts_match or not budget:
        raise SystemExit("Saved unqualified group: metric counts or budget mismatch")


if __name__ == "__main__":
    main()
