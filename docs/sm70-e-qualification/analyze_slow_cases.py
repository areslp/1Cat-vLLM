# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explain C1 work changes and prefill batch formation without hiding samples."""

import json
import math
import statistics
from pathlib import Path

import analyze_performance as analysis

ROOT = Path(__file__).resolve().parent
arms = {}
for arm in ("D", "U", "E"):
    groups = {}
    for phase in ("a", "b"):
        for path in sorted(
            (ROOT / "results" / f"{arm}-timed-{phase}").glob("[0-9]*.json")
        ):
            row = json.loads(path.read_text())
            if row["qualified"]:
                groups.setdefault(row["group"], []).append(row)
    arms[arm] = groups


def median(rows, key):
    return statistics.median(analysis.values(row)[key] for row in rows)


comparisons = {}
for baseline in ("D", "U"):
    rows = []
    names = arms[baseline].keys() & arms["E"].keys()
    for group in sorted(names):
        if not (
            group.startswith("regression_c1_")
            or group in ("main_1024_1_0", "main_1024_1_1")
        ):
            continue
        a, b = arms[baseline][group], arms["E"][group]
        metrics = {
            key: {
                "baseline": median(a, key),
                "candidate": median(b, key),
                "relative_change_pct": 100 * (median(b, key) / median(a, key) - 1),
            }
            for key in (
                "wall_s",
                "prefill_s",
                "server_ttft_s",
                "decode_s",
                "draft_rounds",
                "decode_s_per_draft_round",
            )
        }
        rows.append(
            {
                "group": group,
                "repeats": [len(a), len(b)],
                "metrics": metrics,
                "output_signature_sets_equal": analysis.output_signatures(a)
                == analysis.output_signatures(b),
            }
        )
    comparisons["E_vs_" + baseline] = {
        "C1_cases": len(rows),
        "slower_any_amount": sum(
            row["metrics"]["wall_s"]["relative_change_pct"] > 0 for row in rows
        ),
        "slower_over_3pct": [
            row["group"]
            for row in rows
            if row["metrics"]["wall_s"]["relative_change_pct"] > 3
        ],
        "faster_over_3pct": [
            row["group"]
            for row in rows
            if row["metrics"]["wall_s"]["relative_change_pct"] < -3
        ],
        "equal_weight_geometric_mean_wall_change_pct": 100
        * (
            math.exp(
                statistics.mean(
                    math.log(
                        row["metrics"]["wall_s"]["candidate"]
                        / row["metrics"]["wall_s"]["baseline"]
                    )
                    for row in rows
                )
            )
            - 1
        )
        if rows
        else None,
        "median_decode_per_draft_round_change_pct": statistics.median(
            row["metrics"]["decode_s_per_draft_round"]["relative_change_pct"]
            for row in rows
        )
        if rows
        else None,
        "cases": rows,
    }
strata = {}
for arm, groups in arms.items():
    strata[arm] = {}
    for group, rows in groups.items():
        if not group.startswith("prefill_main_1024_"):
            continue
        batches = {}
        for row in rows:
            count = int(row["server_metric_delta"]["vllm:iteration_tokens_total_count"])
            batches.setdefault(str(count), []).append(row)
        strata[arm][group] = {
            count: {
                "samples": len(cohorts),
                "wall_median_s": median(cohorts, "wall_s"),
                "server_TTFT_median_s": median(cohorts, "server_ttft_s"),
                "prefill_median_s": median(cohorts, "prefill_s"),
                "queue_median_s": median(cohorts, "queue_s"),
            }
            for count, cohorts in sorted(batches.items())
        }
result = {
    "comparisons": comparisons,
    "prefill_batch_strata": strata,
    "limitations": [
        "All qualified samples remain in the primary comparison.",
        (
            "C1 decode duration per draft round is a serving-work proxy, "
            "not a GPU-kernel timer."
        ),
        (
            "Batch strata explain observed request grouping; "
            "they are not randomized independent experiments."
        ),
        (
            "E versus D can change generated output and MTP work as main evolves. "
            "Quality is assessed separately."
        ),
    ],
}
(ROOT / "SLOW_CASE_ANALYSIS.json").write_text(json.dumps(result, indent=2) + "\n")
for name, data in comparisons.items():
    print(name, {key: value for key, value in data.items() if key != "cases"})
    for row in data["cases"]:
        if row["metrics"]["wall_s"]["relative_change_pct"] > 3:
            print(
                row["group"],
                {
                    key: round(row["metrics"][key]["relative_change_pct"], 2)
                    for key in ("wall_s", "draft_rounds", "decode_s_per_draft_round")
                },
            )
