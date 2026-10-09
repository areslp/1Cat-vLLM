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
    text, usage, done, finish = "", None, False, None
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
                text += (choice.get("delta") or {}).get("content") or ""
                finish = choice.get("finish_reason") or finish
    assert done and usage and finish
    assert usage["prompt_tokens"] == case["expected_input_tokens"]
    return {
        "id": case["id"],
        "request_sha256": request_sha,
        "input_token_sha256": case["input_token_sha256"],
        "text": text,
        "usage": usage,
        "finish_reason": finish,
        "wall_s": time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reset-prefix-cache", action="store_true")
    args = parser.parse_args()
    payload = json.loads(Path(__file__).with_name("observations.json").read_text())
    cases = [
        case
        for case in payload["prompts"]
        if case.get("group", case["id"]) == args.group
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
    barrier = threading.Barrier(len(cases) + 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cases)) as pool:
        futures = [pool.submit(run, base, case, barrier) for case in cases]
        started = time.perf_counter()
        barrier.wait(timeout=60)
        rows = [future.result() for future in futures]
        wall = time.perf_counter() - started
    result = {
        "group": args.group,
        "concurrency": len(cases),
        "requests": rows,
        "wall_s": wall,
        "output_tokens": sum(row["usage"]["completion_tokens"] for row in rows),
        "prefix_reset_performed": args.reset_prefix_cache,
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps({key: value for key, value in result.items() if key != "requests"})
    )


if __name__ == "__main__":
    main()
