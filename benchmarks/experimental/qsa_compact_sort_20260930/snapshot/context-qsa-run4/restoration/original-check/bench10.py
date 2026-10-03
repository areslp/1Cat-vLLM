"""STEP-10 timing harness (STEP-22 copy): frozen token IDs, greedy, ignore_eos, unique cache_salt.

Each request records client-side SSE chunk timestamps and tokens per chunk, so the
per-engine-step time and the tokens committed per step (MTP acceptance) can be
separated. Isolated requests also record server counter deltas.
"""
from pathlib import Path
import concurrent.futures
import datetime
import hashlib
import json
import time
import uuid

import requests

D = Path(__file__).resolve().parent
BASE = "http://127.0.0.1:8200"
KEYS = (
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:prefix_cache_hits_total",
    "vllm:request_generation_tokens_count",
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)


def tokens(name, start, n):
    ids = json.loads((D / name).read_text())["tokens"][start : start + n]
    if len(ids) != n:
        raise ValueError("insufficient tokens")
    return ids


def metrics():
    r = requests.get(BASE + "/metrics", timeout=10)
    r.raise_for_status()
    return r.text


def totals(text):
    out = {k: 0.0 for k in KEYS}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            key = line.split("{", 1)[0].split(" ", 1)[0]
            if key in out:
                out[key] += float(line.rsplit(" ", 1)[1])
    return out


def req(tag, label, ids, out):
    body = {
        "request_id": f"step41a-{tag}-{label}-{uuid.uuid4().hex}",
        "model": "flash-next",
        "prompt": ids,
        "max_tokens": out,
        "temperature": 0,
        "top_p": 1,
        "top_k": -1,
        "ignore_eos": True,
        "seed": 1729,
        "cache_salt": uuid.uuid4().hex,
        "return_token_ids": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    save(D / f"{tag}-{label}.request.json", body)
    start = time.perf_counter()
    wall = time.time()
    chunks, token_ids, usage, error = [], [], None, None
    try:
        with requests.post(BASE + "/v1/completions", json=body, stream=True, timeout=(10, 300)) as r:
            r.raise_for_status()
            for line in r.iter_lines(chunk_size=4096):
                if not line.startswith(b"data:"):
                    continue
                raw = line[5:].strip()
                if raw == b"[DONE]":
                    break
                obj = json.loads(raw)
                if "error" in obj:
                    raise RuntimeError(str(obj["error"]))
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices", []):
                    ts = ch.get("token_ids") or []
                    if ts:
                        chunks.append([time.perf_counter() - start, len(ts)])
                        token_ids.extend(ts)
    except Exception as e:  # recorded, validated by the caller
        error = str(e)
    elapsed = time.perf_counter() - start
    ok = (
        error is None
        and usage is not None
        and usage.get("completion_tokens") == out
        and usage.get("prompt_tokens") == len(ids)
        and len(token_ids) == out
    )
    return {
        "label": label,
        "ok": ok,
        "error": error,
        "start_wall": wall,
        "elapsed": elapsed,
        "ttft": chunks[0][0] if chunks else None,
        "chunks": chunks,
        "output_token_ids": token_ids,
        "prompt_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "usage": usage,
    }


def save(path, obj):
    if path.exists():
        raise RuntimeError(f"refusing overwrite {path}")
    path.write_text(json.dumps(obj, indent=1))


def isolated(tag, label, ids, out):
    before = metrics()
    b = totals(before)
    row = req(tag, label, ids, out)
    after = metrics()
    a = totals(after)
    delta = {k: a[k] - b[k] for k in KEYS}
    checks = {
        "request_ok": row["ok"],
        "input_counter_ok": delta[KEYS[0]] == len(ids),
        "output_counter_ok": delta[KEYS[1]] == out,
        "no_prefix_hits": delta[KEYS[2]] == 0,
        "request_counter_ok": delta[KEYS[3]] == 1,
    }
    rec = {"tag": tag, "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), **row, "metrics_delta": delta, "checks": checks}
    save(D / f"{tag}-{label}.json", rec)
    print(json.dumps({"tag": tag, "label": label, "elapsed": row["elapsed"], "ttft": row["ttft"], "chunks": len(row["chunks"]), "delta": delta, "checks": checks}), flush=True)
    if not all(checks.values()):
        raise RuntimeError(f"validation failed {tag}-{label}")
    return rec


def group(tag, label, prompts, out):
    before = metrics()
    b = totals(before)
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        rows = list(pool.map(lambda i: req(tag, f"{label}-r{i}", prompts[i], out), range(len(prompts))))
    wall = time.perf_counter() - t0
    after = metrics()
    a = totals(after)
    delta = {k: a[k] - b[k] for k in KEYS}
    n = len(prompts)
    checks = {
        "requests_ok": all(r["ok"] for r in rows),
        "input_counter_ok": delta[KEYS[0]] == sum(len(p) for p in prompts),
        "output_counter_ok": delta[KEYS[1]] == out * n,
        "no_prefix_hits": delta[KEYS[2]] == 0,
        "request_counter_ok": delta[KEYS[3]] == n,
    }
    rec = {"tag": tag, "label": label, "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "wall": wall, "rows": rows, "metrics_delta": delta, "checks": checks}
    save(D / f"{tag}-{label}.json", rec)
    print(json.dumps({"tag": tag, "label": label, "wall": wall, "tps": out * n / wall, "delta": delta, "checks": checks}), flush=True)
    if not all(checks.values()):
        raise RuntimeError(f"validation failed {tag}-{label}")
    return rec
