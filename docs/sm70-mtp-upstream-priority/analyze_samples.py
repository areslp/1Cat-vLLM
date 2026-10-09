# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recompute every single-request phase/work change from frozen observations."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import median


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def common_prefix(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def analyze(root):
    rows, all_requests, duplicate_texts = [], [], defaultdict(list)
    for directory in sorted((root / "evidence/main").iterdir()):
        group = directory.name
        old = json.loads((directory / "GROUP_RESULT.json").read_text())
        fixed = json.loads(
            (root / "evidence/patched" / group / "GROUP_RESULT.json").read_text()
        )
        for old_request, fixed_request in zip(old["requests"], fixed["requests"]):
            assert (
                old_request["input_token_sha256"] == fixed_request["input_token_sha256"]
            )
            assert old_request["request_sha256"] == fixed_request["request_sha256"]
            row = {
                "group": group,
                "id": old_request["id"],
                "input_sha256": old_request["input_token_sha256"],
                "output_sha256": [sha(r["text"]) for r in (old_request, fixed_request)],
                "same_text": old_request["text"] == fixed_request["text"],
                "common_prefix_chars": common_prefix(
                    old_request["text"], fixed_request["text"]
                ),
            }
            all_requests.append(row)
            for arm, request in (("old", old_request), ("fixed", fixed_request)):
                duplicate_texts[(arm, sha(request["text"]))].append(request["id"])
        if not group.startswith(("ablation_c1_", "main_1024_1_")):
            continue
        row = all_requests[-1].copy()
        assert old["concurrency"] == fixed["concurrency"] == 1
        metrics = [d["server_metric_delta"] for d in (old, fixed)]
        phases = [d["server_phases"] for d in (old, fixed)]
        counters = lambda key, metrics=metrics: [
            m.get("vllm:" + key, 0) for m in metrics
        ]
        row["wall_s"] = [d["wall_s"] for d in (old, fixed)]
        row["ratio_fixed_over_old"] = old["wall_s"] / fixed["wall_s"]
        for phase in ("prefill", "decode", "queue"):
            row[phase + "_s"] = [
                p[f"request_{phase}_time_seconds"]["mean_s"] for p in phases
            ]
        for name in ("num_drafts", "num_draft_tokens", "num_accepted_tokens"):
            row[name] = counters("spec_decode_" + name + "_total")
        for name in (
            "prefix_cache_hits_total",
            "num_preemptions_total",
            "prompt_tokens_cached_total",
        ):
            row[name] = counters(name)
        row["external_traffic_contamination"] = [
            d.get("external_traffic_contamination") for d in (old, fixed)
        ]
        n0, n1 = row["num_drafts"]
        d0, d1 = row["decode_s"]
        c0, c1 = d0 / n0, d1 / n1
        # Symmetric exact decomposition: work plus cost, no arbitrary baseline arm.
        row["per_draft_s"] = [c0, c1]
        row["per_draft_ratio_fixed_over_old"] = c1 / c0
        row["draft_count_delta"] = n1 - n0
        row["decode_delta_s"] = d1 - d0
        row["decode_delta_work_s"] = (n1 - n0) * (c0 + c1) / 2
        row["decode_delta_cost_s"] = (c1 - c0) * (n0 + n1) / 2
        row["wall_delta_non_decode_s"] = fixed["wall_s"] - old["wall_s"] - (d1 - d0)
        assert (
            abs(
                row["decode_delta_s"]
                - row["decode_delta_work_s"]
                - row["decode_delta_cost_s"]
            )
            < 1e-12
        )
        rows.append(row)
    slow = [r for r in rows if r["ratio_fixed_over_old"] < 1]
    genuine = [r for r in slow if r["draft_count_delta"] > 0]
    inputs = defaultdict(list)
    for r in rows:
        inputs[r["input_sha256"]].append(r["id"])
    repeat_inputs = defaultdict(list)
    for r in all_requests:
        repeat_inputs[r["input_sha256"]].append(r["id"])
    unique_rows = []
    for input_hash, ids in inputs.items():
        cases = [r for r in rows if r["input_sha256"] == input_hash]
        old_wall, fixed_wall = [
            sum(r["wall_s"][i] for r in cases) / len(cases) for i in (0, 1)
        ]
        unique_rows.append(
            {
                "input_sha256": input_hash,
                "ids": ids,
                "ratio_fixed_over_old": old_wall / fixed_wall,
                "draft_count_delta": cases[0]["draft_count_delta"],
                "same_old_text_across_repeats": len(
                    {r["output_sha256"][0] for r in cases}
                )
                == 1,
                "same_fixed_text_across_repeats": len(
                    {r["output_sha256"][1] for r in cases}
                )
                == 1,
                "same_round_counts_across_repeats": len(
                    {tuple(r["num_drafts"]) for r in cases}
                )
                == 1,
            }
        )
    return {
        "single_request_pairs": len(rows),
        "slower_count": len(slow),
        "slower_with_more_drafts": len(genuine),
        "slower_with_equal_drafts": len(
            [r for r in slow if r["draft_count_delta"] == 0]
        ),
        "slower_with_fewer_drafts": len(
            [r for r in slow if r["draft_count_delta"] < 0]
        ),
        "slow_more_work_draft_delta_sum": sum(r["draft_count_delta"] for r in genuine),
        "slow_more_work_wall_delta_s": sum(
            r["wall_s"][1] - r["wall_s"][0] for r in genuine
        ),
        "slow_more_work_decode_delta_s": sum(r["decode_delta_s"] for r in genuine),
        "slow_more_work_decode_work_delta_s": sum(
            r["decode_delta_work_s"] for r in genuine
        ),
        "slow_more_work_decode_cost_delta_s": sum(
            r["decode_delta_cost_s"] for r in genuine
        ),
        "all_per_draft_ratio_median": median(
            r["per_draft_ratio_fixed_over_old"] for r in rows
        ),
        "slower_per_draft_ratio_range": [
            min(r["per_draft_ratio_fixed_over_old"] for r in slow),
            max(r["per_draft_ratio_fixed_over_old"] for r in slow),
        ],
        "single_request_distinct_inputs": len(inputs),
        "repeated_single_request_inputs": [v for v in inputs.values() if len(v) > 1],
        "repeated_inputs_across_concurrency": [
            v for v in repeat_inputs.values() if len(v) > 1
        ],
        "duplicate_output_groups": [
            {"arm": arm, "sha256": h, "ids": v}
            for (arm, h), v in duplicate_texts.items()
            if len(v) > 1
        ],
        "same_text_count": sum(r["same_text"] for r in rows),
        "all_prefix_cache_hits_zero": all(
            r["prefix_cache_hits_total"] == [0, 0] for r in rows
        ),
        "all_preemptions_zero": all(r["num_preemptions_total"] == [0, 0] for r in rows),
        "all_cached_prompt_tokens_zero": all(
            r["prompt_tokens_cached_total"] == [0, 0] for r in rows
        ),
        "deduplicated_case_count": len(unique_rows),
        "deduplicated_ratio_median": median(
            r["ratio_fixed_over_old"] for r in unique_rows
        ),
        "deduplicated_slower_cases": sum(
            r["ratio_fixed_over_old"] < 1 for r in unique_rows
        ),
        "deduplicated_slower_with_more_drafts": sum(
            r["ratio_fixed_over_old"] < 1 and r["draft_count_delta"] > 0
            for r in unique_rows
        ),
        "deduplicated_rows": unique_rows,
        "rows_worst_first": sorted(rows, key=lambda r: r["ratio_fixed_over_old"]),
        "all_request_text_divergences": all_requests,
        "qualification": [
            "Exact phase/work accounting, not isolated GPU kernel timing",
            "HTTP text chunks are neither token IDs nor per-token timings",
            "Single sequential old/fixed trial cannot establish repeatability",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                not in (
                    "rows_worst_first",
                    "all_request_text_divergences",
                    "duplicate_output_groups",
                    "repeated_inputs_across_concurrency",
                )
            },
            indent=2,
        )
    )
