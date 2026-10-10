# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Repeat saved synthetic cohorts with phase/cache accounting outside timing."""

import concurrent.futures
import json
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from pathlib import Path

import replay

ROOT = Path(__file__).resolve().parent


def cohort(plan, group, repetition):
    cases = [
        x
        for x in json.loads((ROOT / "observations.json").read_text())["prompts"]
        if x.get("group", x["id"]) == group or x["id"] == group
    ]
    assert cases, group
    if plan.get("cache_control") == "request_salt":
        phase = plan["label"].split("-", 1)[1]
        salt_repetition = 0 if plan.get("reuse_prefix", False) else repetition
        cases = [
            dict(
                case,
                cache_salt=f"pr903-7ab-v1:{phase}:{salt_repetition}:{group}:{case['id']}",
            )
            for case in cases
        ]
    base = plan["base_url"].rstrip("/")
    out = ROOT / "results" / plan["label"] / f"{repetition:02d}-{group}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        replay.validate(base, case)
    preflight = replay.metrics_snapshot(base, out.with_suffix(".idle.txt"))
    assert (
        preflight["vllm:num_requests_running"]
        == preflight["vllm:num_requests_waiting"]
        == 0
    )
    if plan.get("reset_prefix_cache", True):
        request = urllib.request.Request(
            base + "/reset_prefix_cache", data=b"", method="POST"
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            assert response.status == 200
    before = replay.metrics_snapshot(base, out.with_suffix(".metrics-before.txt"))
    barrier = threading.Barrier(len(cases) + 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cases)) as pool:
        futures = [pool.submit(replay.run, base, case, barrier) for case in cases]
        start = time.perf_counter()
        barrier.wait(timeout=60)
        rows = [f.result() for f in futures]
        wall = time.perf_counter() - start
    time.sleep(0.5)
    after = replay.metrics_snapshot(base, out.with_suffix(".metrics-after.txt"))
    delta, phases = replay.metric_delta(before, after)
    cached = delta.get("vllm:prompt_tokens_cached_total")
    computed = delta.get("vllm:request_prefill_kv_computed_tokens_sum")
    expected = sum(x["expected_input_tokens"] for x in cases)
    require_cached = plan.get("reuse_prefix", False) and repetition > 0
    cache_qualified = (
        (
            cached is not None
            and cached > 0
            and computed is not None
            and cached + computed == expected
        )
        if require_cached
        else (
            not plan.get("require_uncached", True)
            or (cached == 0 and computed == expected)
        )
    )
    count_ok = all(phases[p]["count"] == len(cases) for p in replay.PHASES)
    for row in rows:
        row["request_started_offset_s"] = (
            row.pop("request_start_perf_counter_s") - start
        )
    data = dict(
        label=plan["label"],
        commit=plan["runtime_commit"],
        group=group,
        repetition=repetition,
        requests=rows,
        concurrency=len(cases),
        wall_s=wall,
        actual_output_tokens=sum(x["usage"]["completion_tokens"] for x in rows),
        fixed_output_budget_completed=all(
            x["fixed_output_budget_completed"] for x in rows
        ),
        server_metric_delta=delta,
        server_phases=phases,
        qualified_metric_counts=count_ok,
        cached_tokens=cached,
        computed_kv_tokens=computed,
        expected_prompt_tokens=expected,
        cache_work_qualified=cache_qualified,
        require_cached=require_cached,
        priming_cohort=plan.get("reuse_prefix", False) and repetition == 0,
        cache_control=plan.get("cache_control", "reset_prefix_cache"),
        prefix_reset_performed=plan.get("reset_prefix_cache", True),
    )
    data["qualified"] = (
        count_ok
        and data["fixed_output_budget_completed"]
        and data["cache_work_qualified"]
    )
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: data[k]
                for k in (
                    "label",
                    "group",
                    "repetition",
                    "qualified",
                    "wall_s",
                    "cached_tokens",
                    "computed_kv_tokens",
                )
            }
        ),
        flush=True,
    )
    return {
        "path": str(out.relative_to(ROOT)),
        "qualified": data["qualified"],
        "wall_s": wall,
    }


def main():
    plan = json.loads(Path(sys.argv[1]).read_text())
    commits = json.loads((ROOT / "runtime-commits.json").read_text())
    commits["D"] = "c64c4824e830301b3f0fec2da47a202e49ede20e"
    plan["runtime_commit"] = commits[plan["label"].split("-")[0]]
    out = ROOT / "results" / plan["label"] / "SUITE.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    assert not out.exists(), "Preserve the existing attempt before repeating this label"
    result = {
        "plan": plan,
        "started_epoch": time.time(),
        "groups": [],
        "status": "running",
    }
    gpu_command = [
        "nvidia-smi",
        "--query-gpu=index,name,temperature.gpu,clocks.sm,clocks.mem,power.limit",
        "--format=csv",
    ]
    result["gpu_before"] = subprocess.check_output(gpu_command, text=True)
    try:
        for repetition in range(plan.get("repetitions", 3)):
            groups = (
                plan["groups"]
                if repetition % 2 == 0
                else list(reversed(plan["groups"]))
            )
            for group in groups:
                result["groups"].append(cohort(plan, group, repetition))
                out.write_text(json.dumps(result, indent=2) + "\n")
        result["status"] = "completed"
    except BaseException:
        result.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        result["finished_epoch"] = time.time()
        result["gpu_after"] = subprocess.check_output(gpu_command, text=True)
        out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
