# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline descriptive cohort/phase/tail/MTP analysis; no hardware required."""

import argparse
import json
import math
import statistics
from pathlib import Path

import review_evidence


def quantile_nearest(values, q):
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * q) - 1)] if values else None


def describe(values):
    return {
        "n": len(values),
        "median": statistics.median(values) if values else None,
        "p95_nearest_rank": quantile_nearest(values, 0.95),
        "max": max(values) if values else None,
        "min": min(values) if values else None,
        "interpretation": "descriptive sample quantiles, not production p95 estimates",
    }


def analyze(root):
    reviewed = review_evidence.review(root)
    if reviewed["wire_errors"]:
        raise ValueError("invalid wire/count evidence")
    groups = {
        arm: {
            p.parent.name: json.loads(p.read_text())
            for p in (root / "observations" / arm).glob("*/GROUP_RESULT.json")
        }
        for arm in ("main", "patched")
    }
    paired = set(groups["main"]) & set(groups["patched"])
    performance = [r for r in reviewed["pairs"] if "wall_speedup" in r]
    rows = []
    total = {
        a: {
            "requests": 0,
            "output_tokens": 0,
            "drafts": 0,
            "draft_tokens": 0,
            "accepted_tokens": 0,
        }
        for a in groups
    }
    for r in performance:
        key = r["group"]
        row = {
            "group": key,
            "throughput_ratio_main_over_patched": r["wall_speedup"],
            "arms": {},
        }
        for arm in groups:
            g = groups[arm][key]
            m = g["server_metric_delta"]
            drafts = m.get("vllm:spec_decode_num_drafts_total", 0)
            dt = m.get("vllm:spec_decode_num_draft_tokens_total", 0)
            ac = m.get("vllm:spec_decode_num_accepted_tokens_total", 0)
            row["arms"][arm] = {
                "group_wall_s": g["wall_s"],
                "group_output_tok_s": g["group_output_tok_s"],
                "request_wall_s": describe([x["wall_s"] for x in g["requests"]]),
                "TTFT_s": describe(
                    [x["ttft_s"] for x in g["requests"] if x.get("ttft_s") is not None]
                ),
                "server_phase_means_s": {
                    k: v["mean_s"] for k, v in g["server_phases"].items()
                },
                "MTP": {
                    "drafts": drafts,
                    "draft_tokens": dt,
                    "accepted_tokens": ac,
                    "acceptance_fraction": ac / dt if dt else None,
                    "mean_acceptance_length": 1 + ac / drafts if drafts else None,
                },
            }
            for k, v in [
                ("requests", len(g["requests"])),
                ("output_tokens", g["actual_output_tokens"]),
                ("drafts", drafts),
                ("draft_tokens", dt),
                ("accepted_tokens", ac),
            ]:
                total[arm][k] += v
        rows.append(row)
    for a, t in total.items():
        t["acceptance_fraction"] = (
            t["accepted_tokens"] / t["draft_tokens"] if t["draft_tokens"] else None
        )
        t["mean_acceptance_length"] = (
            1 + t["accepted_tokens"] / t["drafts"] if t["drafts"] else None
        )
    return {
        "same_window": False,
        "both_complete_groups": sorted(paired),
        "main_only_complete_groups": sorted(set(groups["main"]) - paired),
        "candidate_only_complete_groups": sorted(set(groups["patched"]) - paired),
        "neither_complete_groups": sorted(
            set(json.loads((root / "MATRIX.json").read_text())["groups"])
            - set(groups["main"])
            - set(groups["patched"])
        ),
        "performance_groups": len(performance),
        "fixed_budget_aggregate": total,
        "rows": rows,
        "cohort_summaries": reviewed["summaries"],
        "limits": [
            "budget timeout excluded from gains",
            "phase clocks include host/runtime/JIT effects; not GPU kernel time",
            "tail quantiles descriptive small samples only",
            "SSE gaps not per-token ITL",
            (
                "MTP rates differ with generation chains; no token-parity "
                "or per-family attribution claim"
            ),
            "last memory snapshots are not peak profiling",
        ],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("evidence", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = analyze(a.evidence)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "both_complete": len(result["both_complete_groups"]),
                "main_only": len(result["main_only_complete_groups"]),
                "neither": len(result["neither_complete_groups"]),
                "performance_groups": result["performance_groups"],
                "fixed_budget_aggregate": result["fixed_budget_aggregate"],
            }
        )
    )


if __name__ == "__main__":
    main()
