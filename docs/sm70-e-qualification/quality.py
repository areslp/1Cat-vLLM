# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: BLE001
"""One frozen synthetic cohort; retain streams and separate server phases."""

import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

import bench_client as bench

W = Path(__file__).resolve().parent
URL = "http://127.0.0.1:18200"


def protocol_features(arm):
    root = W / "evidence" / arm / "protocol_features"
    root.mkdir()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "echo_marker",
                "description": "Echo the supplied synthetic marker.",
                "parameters": {
                    "type": "object",
                    "properties": {"marker": {"type": "string"}},
                    "required": ["marker"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    tests = [
        (
            "tool_auto",
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Call echo_marker with marker SYNTHETIC_OK_742. "
                            "Do not answer in prose."
                        ),
                    }
                ],
                "tools": tools,
                "tool_choice": "auto",
            },
        ),
        (
            "json_constraint",
            {
                "messages": [
                    {
                        "role": "user",
                        "content": 'Return exactly the JSON object {"ok":true,"n":7}.',
                    }
                ],
                "response_format": {"type": "json_object"},
            },
        ),
    ]
    for name, additions in tests:
        body = {
            "model": "flash-next",
            "temperature": 0,
            "top_p": 1,
            "repetition_penalty": 1,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "max_tokens": 128,
            "stream": False,
            "seed": 1234,
            "chat_template_kwargs": {"enable_thinking": False},
            **additions,
        }
        payload = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
        token_body = {
            k: body[k]
            for k in ("model", "messages", "chat_template_kwargs", "tools")
            if k in body
        }
        result = {
            "name": name,
            "body": body,
            "request_sha256": hashlib.sha256(payload).hexdigest(),
            "status": "pending",
            "gold_passed": False,
        }
        started = time.monotonic()
        try:
            request = urllib.request.Request(
                URL + "/tokenize",
                data=json.dumps(token_body).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                tokens = json.load(response)
            result["input_tokens"] = tokens["count"]
            result["input_token_sha256"] = hashlib.sha256(
                json.dumps(tokens["tokens"], separators=(",", ":")).encode()
            ).hexdigest()
            request = urllib.request.Request(
                URL + "/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                answer = json.load(response)
            result.update(status="completed", response=answer)
            msg = answer["choices"][0]["message"]
            if name == "tool_auto":
                calls = msg.get("tool_calls") or []
                result["gold_passed"] = (
                    len(calls) == 1
                    and calls[0]["function"]["name"] == "echo_marker"
                    and json.loads(calls[0]["function"]["arguments"])
                    == {"marker": "SYNTHETIC_OK_742"}
                )
            else:
                result["gold_passed"] = json.loads(msg["content"].strip()) == {
                    "ok": True,
                    "n": 7,
                }
            result["actual_prompt_matches_tokenize"] = (
                answer["usage"]["prompt_tokens"] == tokens["count"]
            )
        except Exception as exc:
            result.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
        result["wall_s"] = time.monotonic() - started
        bench.atomic(root / (name + ".json"), result)


def main():
    global URL
    if len(sys.argv) != 4:
        raise SystemExit("usage: quality.py ARM GROUP BASE_URL")
    arm, key, URL = sys.argv[1:4]
    cases = json.loads((W / "quality-observations.json").read_text())
    prompt_dir = W / "runtime/prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for case in cases:
        path = prompt_dir / (case["id"] + ".json")
        path.write_text(json.dumps(case, ensure_ascii=False, indent=2) + "\n")
        entries.append({"id": case["id"], "file": str(path)})
    if key == "parallel_gold_8":
        bench.parallel_probe(W, arm, URL, entries)
        return
    if key == "protocol_features":
        protocol_features(arm)
        return
    files, cases = [], []
    for entry in entries:
        case = json.loads(Path(entry["file"]).read_text())
        if case.get("group") == key or case["id"] == key:
            files.append(entry["file"])
            cases.append(case)
    assert files, key
    for file, case in zip(files, cases):
        # Validate the actual serving tokenizer in each arm, not only saved hashes.
        payload = {k: case[k] for k in ("messages",)}
        payload.update(
            model="flash-next", chat_template_kwargs={"enable_thinking": False}
        )
        request = urllib.request.Request(
            URL + "/tokenize",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            tokens = json.load(response)
        digest = hashlib.sha256(
            json.dumps(tokens["tokens"], separators=(",", ":")).encode()
        ).hexdigest()
        assert (
            tokens["count"] == case["expected_input_tokens"]
            and digest == case["input_token_sha256"]
        ), "live tokenizer drift: " + case["id"]
    data = bench.group(W, arm, URL, key, files)
    if any("gold" in c for c in cases):
        checks = [
            bench.gold_matches(r["text"], c["gold"])
            for r, c in zip(data["requests"], cases)
        ]
    elif key.startswith("retrieval_"):
        checks = []
        for r, c in zip(data["requests"], cases):
            try:
                checks.append(json.loads(r["text"].strip()) == c["expected_json"])
            except ValueError:
                checks.append(False)
    else:
        checks = None
    if checks is not None:
        bench.atomic(
            W / "evidence" / arm / key / "GOLD_RESULT.json",
            {"checks": checks, "passed": all(checks)},
        )
        assert all(checks), "new correctness failure: " + key
    if key == "protocol_features":
        protocol_features(arm)
    # Gold failure is retained for paired relative assessment. It never becomes
    # qualified performance simply because the reference also has model errors.


if __name__ == "__main__":
    main()
