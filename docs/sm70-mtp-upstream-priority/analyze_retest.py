# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recompute all frozen retest pairs. No CUDA, logs or hardware access."""

import argparse
import json
from pathlib import Path
from statistics import median


def load(path):
    return json.loads(path.read_text())


def per_group(old, fixed):
    errors = []
    for arm, data in (("old", old), ("fixed", fixed)):
        if not data.get("successful"):
            errors.append(arm + " request failure")
        if len(data["requests"]) != data["concurrency"]:
            errors.append(arm + " wrong concurrency")
        if (
            sum(
                r["usage"]["completion_tokens"]
                for r in data["requests"]
                if r.get("usage")
            )
            != data["actual_output_tokens"]
        ):
            errors.append(arm + " wrong aggregate output count")
        if (
            abs(
                data["group_output_tok_s"]
                - data["actual_output_tokens"] / data["wall_s"]
            )
            > 1e-8
        ):
            errors.append(arm + " wrong throughput")
        for r in data["requests"]:
            if r.get("status") != "completed" or not r.get("usage"):
                errors.append(arm + " incomplete request: " + r["id"])
                continue
            if r["usage"]["prompt_tokens"] != r["input_expected"]:
                errors.append(arm + " input count: " + r["id"])
    if len(old["requests"]) != len(fixed["requests"]):
        errors.append("pair concurrency differs")
    for a, b in zip(old["requests"], fixed["requests"]):
        for key in (
            "id",
            "input_token_sha256",
            "input_expected",
            "request_sha256",
            "requested_output",
        ):
            if a[key] != b[key]:
                errors.append("pair differs: " + key)
    fixed_budget = not errors and all(
        r.get("fixed_output_budget_completed")
        and r["usage"]["completion_tokens"] == r["requested_output"]
        and r["finish_reason"] == "length"
        for d in (old, fixed)
        for r in d["requests"]
    )
    row = {
        "group": old["group"],
        "concurrency": old["concurrency"],
        "validation_errors": errors,
        "fixed_budget_pair": fixed_budget,
        "old_output_tokens": old["actual_output_tokens"],
        "fixed_output_tokens": fixed["actual_output_tokens"],
        "old_wall_s": old["wall_s"],
        "fixed_wall_s": fixed["wall_s"],
        "throughput_ratio_fixed_over_old": old["wall_s"] / fixed["wall_s"]
        if fixed_budget
        else None,
        "same_text_requests": sum(
            a["text"] == b["text"] for a, b in zip(old["requests"], fixed["requests"])
        ),
        "request_count": len(old["requests"]),
        "phases": {},
        "mtp": {},
    }
    for phase in (
        "request_prefill_time_seconds",
        "request_decode_time_seconds",
        "request_queue_time_seconds",
    ):
        row["phases"][phase] = [
            d["server_phases"][phase]["mean_s"] for d in (old, fixed)
        ]
    for name in ("num_drafts", "num_draft_tokens", "num_accepted_tokens"):
        key = "vllm:spec_decode_" + name + "_total"
        row["mtp"][name] = [d["server_metric_delta"].get(key) for d in (old, fixed)]
    return row


def summary(rows):
    qualified = [r for r in rows if r["fixed_budget_pair"]]
    values = [r["throughput_ratio_fixed_over_old"] for r in qualified]
    return {
        "groups": len(rows),
        "qualified_fixed_budget_pairs": len(qualified),
        "ratio_median": median(values) if values else None,
        "ratio_range": [min(values), max(values)] if values else None,
        "old_wall_sum_s": sum(r["old_wall_s"] for r in qualified),
        "fixed_wall_sum_s": sum(r["fixed_wall_s"] for r in qualified),
        "same_text_requests": sum(r["same_text_requests"] for r in rows),
        "total_requests_per_arm": sum(r["request_count"] for r in rows),
        "phase_mean_medians": {
            p: [median([r["phases"][p][i] for r in qualified]) for i in range(2)]
            for p in (
                "request_prefill_time_seconds",
                "request_decode_time_seconds",
                "request_queue_time_seconds",
            )
        }
        if qualified
        else {},
        "mtp_totals": {
            k: [sum(r["mtp"][k][i] or 0 for r in qualified) for i in range(2)]
            for k in ("num_drafts", "num_draft_tokens", "num_accepted_tokens")
        },
    }


def analyze(root):
    plan = load(root / "MATRIX.json")["groups"]
    rows, missing = [], []
    quality = {}
    for key in plan:
        paths = [
            root / "evidence" / a / key / "GROUP_RESULT.json"
            for a in ("main", "patched")
        ]
        present = [p.exists() for p in paths]
        if not all(present):
            missing.append(
                {"group": key, "old_present": present[0], "fixed_present": present[1]}
            )
            continue
        a, b = map(load, paths)
        row = per_group(a, b)
        rows.append(row)
        if key.startswith("q_"):
            quality[key] = [
                load(root / "evidence" / arm / key / "GOLD_RESULT.json")
                if (root / "evidence" / arm / key / "GOLD_RESULT.json").exists()
                else None
                for arm in ("main", "patched")
            ]
    perf = [r for r in rows if not r["group"].startswith("q_")]
    return {
        "complete_group_pairs": len(rows),
        "planned_groups": len(plan),
        "missing_pairs": missing,
        "validation_error_count": sum(len(r["validation_errors"]) for r in rows),
        "quality": quality,
        "original_c1": summary(
            [r for r in perf if r["group"].startswith("main_1024_1_")]
        ),
        "all_c1_128": summary(
            [r for r in perf if r["group"].startswith(("main_1024_1_", "ablation_c1_"))]
        ),
        "c4_128": summary([r for r in perf if r["group"].startswith("main_1024_4_")]),
        "c8_128": summary([r for r in perf if r["group"].startswith("main_1024_8_")]),
        "c1_1024_output": summary(
            [r for r in perf if r["group"].startswith("decode_")]
        ),
        "rows": rows,
        "limitations": [
            "sequential arms in one finite window, not interleaved model trials",
            "pure upstream main not rerun",
            "new source only changes M1 tile; no broad requalification",
            (
                "ablation prompts repeat earlier concurrency prompts "
                "and can use prefix cache"
            ),
            "component graph timings are not HTTP or whole-draft timing",
            (
                "small synthetic cohort does not establish "
                "comprehensive quality equivalence"
            ),
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    assert not result["validation_error_count"], "paired evidence validation failure"


if __name__ == "__main__":
    main()
