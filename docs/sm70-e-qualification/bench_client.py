# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: BLE001
"""Paired synthetic HTTP/SSE client. Parent enforces bounded group time."""

import concurrent.futures
import hashlib
import http.client
import json
import threading
import time
import urllib.parse
from pathlib import Path

import regex as re


def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def get(url, path):
    u = urllib.parse.urlsplit(url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=10)
    try:
        c.request("GET", path)
        r = c.getresponse()
        body = r.read()
        if r.status != 200:
            raise RuntimeError("GET status " + str(r.status))
        return body.decode()
    finally:
        c.close()


def metrics(text):
    data = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.match(r"([^\s{]+)(?:\{[^}]*\})?\s+([-+\d.eE]+)", line)
        if match:
            key = match[1]
            if key.endswith(("_sum", "_count", "_total")) or key in (
                "vllm:num_requests_running",
                "vllm:num_requests_waiting",
            ):
                data[key] = data.get(key, 0) + float(match[2])
    return data


def request(url, case, barrier, out):
    u = urllib.parse.urlsplit(url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=900)
    body = {
        "model": "flash-next",
        "messages": case["messages"],
        "temperature": 0,
        "top_p": 1,
        "repetition_penalty": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "max_tokens": case["max_output_tokens"],
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
        "seed": 1234,
    }
    payload = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
    result = {
        "id": case["id"],
        "input_expected": case["expected_input_tokens"],
        "input_token_sha256": case["input_token_sha256"],
        "request_sha256": hashlib.sha256(payload).hexdigest(),
        "requested_output": case["max_output_tokens"],
        "chunks": [],
        "text": "",
        "reasoning": "",
        "finish_reason": None,
        "usage": None,
        "timings": None,
        "status": "pending",
    }
    barrier.wait()
    start = time.perf_counter()
    result["start_epoch"] = time.time()
    first = None
    last = None
    usage = None
    try:
        c.request(
            "POST",
            "/v1/chat/completions",
            body=payload,
            headers={"Content-Type": "application/json", "X-Request-Id": case["id"]},
        )
        r = c.getresponse()
        result["http_status"] = r.status
        result["headers_s"] = time.perf_counter() - start
        if r.status != 200:
            raise RuntimeError(
                "HTTP " + str(r.status) + ": " + r.read(8192).decode("utf-8", "replace")
            )
        completed = False
        while True:
            line = r.readline()
            if not line:
                break
            elapsed = time.perf_counter() - start
            if not line.startswith(b"data:"):
                continue
            raw = line[5:].strip()
            if raw == b"[DONE]":
                completed = True
                break
            item = json.loads(raw)
            result["chunks"].append({"at_s": elapsed, "data": item})
            if "error" in item:
                raise RuntimeError(str(item["error"]))
            if item.get("usage"):
                usage = item["usage"]
            if item.get("timings"):
                result["timings"] = item["timings"]
            for choice in item.get("choices", []):
                delta = choice.get("delta") or {}
                text = delta.get("content") or ""
                reasoning = delta.get("reasoning_content") or ""
                if text or reasoning:
                    if first is None:
                        first = elapsed
                    last = elapsed
                    result["text"] += text
                    result["reasoning"] += reasoning
                if choice.get("finish_reason"):
                    result["finish_reason"] = choice["finish_reason"]
        result["usage"] = usage
        result["wall_s"] = time.perf_counter() - start
        result["ttft_s"] = first
        result["last_text_chunk_s"] = last
        result["stream_decode_wall_s"] = (
            None if first is None or last is None else max(0, last - first)
        )
        if not completed or not usage or not result["finish_reason"]:
            raise RuntimeError("incomplete stream/usage/finish")
        if usage["prompt_tokens"] != case["expected_input_tokens"]:
            raise RuntimeError("actual input token count mismatch")
        if usage["completion_tokens"] > case["max_output_tokens"]:
            raise RuntimeError("output budget exceeded")
        result["fixed_output_budget_completed"] = (
            usage["completion_tokens"] == case["max_output_tokens"]
            and result["finish_reason"] == "length"
        )
        result["early_EOS"] = not result["fixed_output_budget_completed"]
        result["status"] = "completed"
        n = usage["completion_tokens"]
        duration = result["stream_decode_wall_s"]
        result["stream_observed_decode_tok_s"] = (
            None if not duration else max(0, n - 1) / duration
        )
        result["chunk_gaps_s"] = [
            b["at_s"] - a["at_s"]
            for a, b in zip(result["chunks"], result["chunks"][1:])
            if b["at_s"] >= a["at_s"]
        ]
        result["chunk_gap_semantics"] = (
            "HTTP SSE chunk gaps, not per-token ITL; "
            "MTP can deliver several tokens together"
        )
    except Exception as e:
        result.update(
            status="failed",
            error=str(e),
            wall_s=time.perf_counter() - start,
            ttft_s=first,
        )
    finally:
        c.close()
        atomic(out, result)
    return result


def group(root, arm, url, key, case_files, resume=False):
    out = root / "evidence" / arm / key
    if resume and (out / "GROUP_RESULT.json").exists():
        data = json.loads((out / "GROUP_RESULT.json").read_text())
        cases = [json.loads(Path(x).read_text()) for x in case_files]
        if not data["successful"] or [
            x["input_token_sha256"] for x in data["requests"]
        ] != [x["input_token_sha256"] for x in cases]:
            raise RuntimeError("existing group not valid paired evidence")
        print(json.dumps({"resumed_existing_fresh_group": key, "arm": arm}), flush=True)
        return data
    out.mkdir(parents=True, exist_ok=False)
    cases = [json.loads(Path(x).read_text()) for x in case_files]
    before_text = get(url, "/metrics")
    before = metrics(before_text)
    (out / "metrics-before.txt").write_text(before_text)
    if arm == "C" and any(
        before.get(x, 0) != 0
        for x in ("vllm:num_requests_running", "vllm:num_requests_waiting")
    ):
        raise RuntimeError("production requests already active")
    barrier = threading.Barrier(len(cases) + 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cases)) as pool:
        fs = [
            pool.submit(request, url, case, barrier, out / (case["id"] + ".json"))
            for case in cases
        ]
        started = time.perf_counter()
        start_epoch = time.time()
        barrier.wait()
        results = [f.result() for f in fs]
        wall = time.perf_counter() - started
    # Existing metric emitter can trail the final SSE by an event loop tick.
    time.sleep(0.5)
    after_text = get(url, "/metrics")
    after = metrics(after_text)
    (out / "metrics-after.txt").write_text(after_text)
    delta = {
        k: after[k] - before.get(k, 0)
        for k in after
        if k.endswith(("_sum", "_count", "_total"))
    }
    phases = {}
    for phase in (
        "request_queue_time_seconds",
        "request_prefill_time_seconds",
        "request_decode_time_seconds",
        "time_to_first_token_seconds",
        "e2e_request_latency_seconds",
    ):
        count = delta.get("vllm:" + phase + "_count")
        total = delta.get("vllm:" + phase + "_sum")
        phases[phase] = {
            "count": count,
            "sum_s": total,
            "mean_s": total / count if total is not None and count else None,
            "scope": (
                "aggregate of server request phase wall clocks; "
                "not GPU kernel compute time"
            ),
        }
    completed = [x for x in results if x["status"] == "completed"]
    totalout = sum(x["usage"]["completion_tokens"] for x in completed)
    data = {
        "group": key,
        "arm": arm,
        "concurrency": len(cases),
        "start_epoch": start_epoch,
        "wall_s": wall,
        "actual_output_tokens": totalout,
        "group_output_tok_s": totalout / wall,
        "group_input_tok_s": sum(x["input_expected"] for x in completed) / wall,
        "requests": results,
        "server_metric_delta": delta,
        "server_phases": phases,
        "successful": len(completed) == len(cases),
        "fixed_output_budget_completed": all(
            x.get("fixed_output_budget_completed") for x in results
        ),
        "external_traffic_contamination": None,
    }
    if arm == "C":
        observed = phases["request_prefill_time_seconds"]["count"]
        data["external_traffic_contamination"] = observed != len(cases)
        if data["external_traffic_contamination"]:
            data["qualified"] = False
    else:
        data["external_traffic_contamination"] = False
    atomic(out / "GROUP_RESULT.json", data)
    print(
        json.dumps(
            {
                k: data[k]
                for k in (
                    "group",
                    "arm",
                    "wall_s",
                    "actual_output_tokens",
                    "group_output_tok_s",
                    "successful",
                    "fixed_output_budget_completed",
                )
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if not data["successful"]:
        raise RuntimeError("request group failed; stop expansion: " + key)
    return data


def gold_matches(text, case):
    text = text.strip()
    expect = case["expected"]
    if case["match"] == "integer":
        values = re.findall(r"(?<!\d)-?\d+(?!\d)", text)
        return len(values) == 1 and int(values[0]) == expect
    if case["match"] == "exact_text":
        return text == expect
    try:
        return json.loads(text) == expect
    except ValueError:
        return False


def parallel_probe(root, arm, url, entries, resume=False):
    ids = [
        "q_ascii_copy_0",
        "q_utf8_copy_0",
        "q_arithmetic_regression_0",
        "q_parentheses_0",
        "q_sum_0",
        "q_remainder_0",
        "q_json_fields_0",
        "q_json_boolean_0",
    ]
    if arm == "S4long":
        ids = ids[:2]
    files = [next(x["file"] for x in entries if x["id"] == key) for key in ids]
    key = "parallel_gold_" + str(len(files))
    if (root / "evidence" / arm / key / "GROUP_RESULT.json").exists():
        return
    data = group(root, arm, url, key, files, resume)
    cases = [json.loads(Path(p).read_text()) for p in files]
    checks = [
        gold_matches(r["text"], c["gold"]) for r, c in zip(data["requests"], cases)
    ]
    copies = {
        c["gold"]["expected"]: i
        for i, c in enumerate(cases)
        if c["gold"]["match"] == "exact_text"
    }
    cross = [
        i
        for i, r in enumerate(data["requests"])
        if r["text"].strip() in copies and copies[r["text"].strip()] != i
    ]
    atomic(
        root / "evidence" / arm / key / "PARALLEL_ROUTING_RESULT.json",
        {
            "concurrency": len(files),
            "strict_gold_checks": checks,
            "unique_copy_cross_talk": cross,
            "unique_copy_checks": checks[:2],
            "performance_sample": False,
        },
    )
    if cross:
        raise RuntimeError("response cross-talk in unique copied text")
