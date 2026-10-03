#!/usr/bin/env python3
"""STEP-38 fixed-work performance suite for one arm. Usage: perf38.py <arm>

256 output tokens per request, ignore_eos, unique cache salts, 512-token prompts from
input-8192.json at offsets 512*i. G = greedy; N = T0.6 / top-p 0.95 / top-k 20 (seed 1729).
  c1: G and N at offsets 0/1024/2560/4096 (isolated requests with server counter checks)
  c2, c4, c8: G x2 groups and N x2 groups
  STEP-01 short suite (bench.py cases, sampling, c1..c8)
Per request: median SSE step interval (engine step time) and tokens per step (acceptance).
Guard: every 10 s, any GPU free (total - used) < 150 MiB or /health failing -> exit 3.
"""
import concurrent.futures
import datetime
import json
import os
import statistics as st
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import requests

import bench as bp
import bench10 as b

D = Path(__file__).resolve().parent
BASE = "http://127.0.0.1:8200"
MIN_FREE_MIB = 150
G = {"temperature": 0, "top_p": 1, "top_k": -1}
N = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
OFFS = (0, 1024, 2560, 4096)
SHORT = [(512, 128, 1, 1), (512, 256, 1, 3), (512, 256, 2, 6), (512, 256, 4, 8), (512, 256, 8, 16)]


def note(t):
    print(datetime.datetime.now(datetime.timezone.utc).isoformat(), t, flush=True)


def tok(off):
    return b.tokens("input-8192.json", off, 512)


def req(tag, label, ids, params, seed):
    body = {"request_id": f"step41a-{tag}-{label}-{uuid.uuid4().hex}", "model": "flash-next", "prompt": ids,
            "max_tokens": 256, "ignore_eos": True, "cache_salt": uuid.uuid4().hex, "return_token_ids": True,
            "stream": True, "stream_options": {"include_usage": True}, **params}
    if seed:
        body["seed"] = 1729
    b.save(D / f"{tag}-{label}.request.json", body)
    t0, chunks, toks, usage, err = time.perf_counter(), [], [], None, None
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
                        chunks.append([time.perf_counter() - t0, len(ts)])
                        toks.extend(ts)
    except Exception as e:
        err = str(e)
    ok = err is None and usage is not None and usage.get("completion_tokens") == 256 and len(toks) == 256
    t = [x[0] for x in chunks]
    iv = [1000 * (y - x) for x, y in zip(t[1:], t[2:])]
    return {"label": label, "ok": ok, "error": err, "elapsed": time.perf_counter() - t0, "ttft": t[0] if t else None,
            "chunks": chunks, "output_token_ids": toks, "usage": usage, "params": params, "seed": seed,
            "step_ms": st.median(iv) if iv else None,
            "tok_per_step": (256 - chunks[0][1]) / (len(chunks) - 1) if len(chunks) > 1 else None}


def iso(tag, label, ids, params, seed=False):
    before = b.totals(b.metrics())
    row = req(tag, label, ids, params, seed)
    after = b.totals(b.metrics())
    d = {k: after[k] - before[k] for k in b.KEYS}
    row["metrics_delta"] = d
    row["checks"] = {"ok": row["ok"], "input": d[b.KEYS[0]] == len(ids), "output": d[b.KEYS[1]] == 256,
                     "no_prefix_hits": d[b.KEYS[2]] == 0, "one_request": d[b.KEYS[3]] == 1}
    b.save(D / f"{tag}-{label}.json", row)
    note(f"{tag}-{label} step {row['step_ms'] and round(row['step_ms'], 3)} tok/step {row['tok_per_step'] and round(row['tok_per_step'], 3)} checks {all(row['checks'].values())}")
    if not all(row["checks"].values()):
        raise RuntimeError(f"validation failed {tag}-{label}")
    return row


def grp(tag, label, prompts, params, seed=False):
    before = b.totals(b.metrics())
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        rows = list(pool.map(lambda i: req(tag, f"{label}-r{i}", prompts[i], params, seed), range(len(prompts))))
    wall = time.perf_counter() - t0
    after = b.totals(b.metrics())
    d = {k: after[k] - before[k] for k in b.KEYS}
    n = len(prompts)
    checks = {"ok": all(r["ok"] for r in rows), "input": d[b.KEYS[0]] == sum(len(p) for p in prompts),
              "output": d[b.KEYS[1]] == 256 * n, "no_prefix_hits": d[b.KEYS[2]] == 0, "requests": d[b.KEYS[3]] == n}
    rec = {"tag": tag, "label": label, "params": params, "wall": wall, "rows": rows, "metrics_delta": d, "checks": checks}
    b.save(D / f"{tag}-{label}.json", rec)
    note(f"{tag}-{label} step {round(st.median(r['step_ms'] for r in rows), 3)} tok/s {round(256 * n / wall, 1)} checks {all(checks.values())}")
    if not all(checks.values()):
        raise RuntimeError(f"validation failed {tag}-{label}")
    return rec


def gpu_free():
    q = ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"]
    rows = subprocess.run(q, check=True, capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
    return [int(t) - int(u) for u, t in (r.split(", ") for r in rows)]


def healthy():
    try:
        return requests.get(BASE + "/health", timeout=5).status_code == 200
    except Exception:
        return False


def monitor():
    while True:
        free, ok = gpu_free(), healthy()
        with (D / "guard38.jsonl").open("a") as f:
            f.write(json.dumps({"t": time.time(), "free": free, "healthy": ok}) + "\n")
        if min(free) < MIN_FREE_MIB or not ok:
            (D / "guard-tripped.json").write_text(json.dumps({"t": time.time(), "free": free, "healthy": ok}))
            note(f"GUARD TRIPPED free={free} healthy={ok}")
            os._exit(3)
        time.sleep(10)


def main():
    arm = sys.argv[1]
    threading.Thread(target=monitor, daemon=True).start()
    raw = requests.get(BASE + "/metrics", timeout=10).text
    if sum(float(l.rsplit(" ", 1)[1]) for l in raw.splitlines() if l.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{"))):
        raise SystemExit("service not idle")
    t = f"{arm}-P"
    for off in OFFS:
        iso(t, f"G1-o{off}", tok(off), G)
    for off in OFFS:
        iso(t, f"N1-o{off}", tok(off), N, seed=True)
    for c in (2, 4, 8):
        for g in range(2):
            prompts = [tok((512 * (i + c * g)) % 7680) for i in range(c)]
            grp(t, f"G{c}-{g}", prompts, G)
        for g in range(2):
            prompts = [tok((512 * (i + c * g)) % 7680) for i in range(c)]
            grp(t, f"N{c}-{g}", prompts, N, seed=True)
    for j, (n, out, cc, r) in enumerate(SHORT):
        bp.run_case(f"{t}-short_{j}_p{n}_o{out}_c{cc}", n, out, cc, r, "sample")
    note("PERF DONE")


if __name__ == "__main__":
    main()
