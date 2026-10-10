# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Report the fixed-compiler comparison separately from earlier tuning runs."""

import json
import math
import statistics
from pathlib import Path

import analyze_performance as analysis
import analyze_quality as quality

ROOT = Path(__file__).resolve().parent
LABELS = {
    "U_fixed": ["U-fixed-timed", "U-fixed-cached"],
    "E_fixed": ["E-fixed-timed", "E-fixed-cached"],
    "D_reference": ["D-timed-a"],
    "U_previous": ["U-timed-a", "U-timed-b", "U-cached-a", "U-cached-b"],
}


def collect():
    arms, inventory = {}, {}
    for arm, labels in LABELS.items():
        groups, commits, rejected, priming = {}, set(), [], []
        for label in labels:
            for path in sorted((ROOT / "results" / label).glob("[0-9]*.json")):
                row = json.loads(path.read_text())
                commits.add(row["commit"])
                if not row["qualified"]:
                    rejected.append(str(path.relative_to(ROOT)))
                    continue
                if row.get("priming_cohort"):
                    priming.append(str(path.relative_to(ROOT)))
                    continue
                group = ("cached/" if row.get("require_cached") else "") + row["group"]
                groups.setdefault(group, []).append(row)
        arms[arm] = groups
        inventory[arm] = dict(
            commits=sorted(commits),
            rejected=rejected,
            priming=priming,
            groups=len(groups),
            cohorts=sum(map(len, groups.values())),
        )
    return arms, inventory


def compare(arms, candidate, baseline):
    rows = {}
    for group in sorted(arms[candidate].keys() & arms[baseline].keys()):
        left, right = arms[baseline][group], arms[candidate][group]
        rows[group] = {
            "repeats": [len(left), len(right)],
            "output_signature_sets_equal": analysis.output_signatures(left)
            == analysis.output_signatures(right),
            "baseline_output_signatures": analysis.output_signatures(left),
            "candidate_output_signatures": analysis.output_signatures(right),
            "metrics": {
                key: analysis.comparison(left, right, key)
                for key in analysis.values(left[0])
            },
        }
    c1 = {
        name: row
        for name, row in rows.items()
        if name.startswith("regression_c1_")
        or name in ("main_1024_1_0", "main_1024_1_1")
    }
    changes = [row["metrics"]["wall_s"]["relative_change_pct"] for row in c1.values()]
    summary = {
        "historical_C1_cases": len(c1),
        "slower_any_amount": sum(change > 0 for change in changes),
        "slower_over_3pct": [
            name
            for name, row in c1.items()
            if row["metrics"]["wall_s"]["relative_change_pct"] > 3
        ],
        "faster_over_3pct": [
            name
            for name, row in c1.items()
            if row["metrics"]["wall_s"]["relative_change_pct"] < -3
        ],
        "equal_weight_geomean_wall_change_pct": 100
        * (
            math.exp(statistics.mean(math.log1p(change / 100) for change in changes))
            - 1
        )
        if changes
        else None,
        "same_output_signature_sets": sum(
            row["output_signature_sets_equal"] for row in c1.values()
        ),
    }
    return dict(summary=summary, groups=rows)


def prefill_strata(arms):
    """Keep every cohort, while exposing different scheduler iteration counts."""
    result = {}
    for arm, groups in arms.items():
        result[arm] = {}
        for name, rows in groups.items():
            if not name.startswith("prefill_main_1024_"):
                continue
            strata = {}
            for row in rows:
                count = str(
                    int(row["server_metric_delta"]["vllm:iteration_tokens_total_count"])
                )
                strata.setdefault(count, []).append(row)
            result[arm][name] = {
                count: dict(
                    cohorts=len(cohorts),
                    client_max_ttft_median_s=statistics.median(
                        max(request["ttft_s"] for request in row["requests"])
                        for row in cohorts
                    ),
                    request_start_spread_median_s=statistics.median(
                        max(
                            request["request_started_offset_s"]
                            for request in row["requests"]
                        )
                        - min(
                            request["request_started_offset_s"]
                            for request in row["requests"]
                        )
                        for row in cohorts
                    ),
                    medians={
                        metric: statistics.median(
                            analysis.values(row)[metric] for row in cohorts
                        )
                        for metric in (
                            "wall_s",
                            "prefill_s",
                            "queue_s",
                            "server_ttft_s",
                        )
                    },
                )
                for count, cohorts in sorted(strata.items())
            }
    return result


def main():
    arms, inventory = collect()
    comparisons = {
        candidate + "_vs_" + baseline: compare(arms, candidate, baseline)
        for candidate, baseline in [
            ("E_fixed", "U_fixed"),
            ("E_fixed", "D_reference"),
            ("U_fixed", "U_previous"),
        ]
    }
    qa = {arm: quality.arm_result(arm) for arm in ("D", "U-fixed", "E-fixed")}
    qc = {}
    for base in ("D", "U-fixed"):
        a, b = qa[base]["cases"], qa["E-fixed"]["cases"]
        qc["E-fixed_vs_" + base] = dict(
            new_failures=[
                name
                for name in sorted(a.keys() & b.keys())
                if a[name]["passed"] and not b[name]["passed"]
            ],
            improved=[
                name
                for name in sorted(a.keys() & b.keys())
                if not a[name]["passed"] and b[name]["passed"]
            ],
            missing=sorted(a.keys() - b.keys()),
        )
    result = dict(
        inventory=inventory,
        comparisons=comparisons,
        quality=qa,
        prefill_batch_strata=prefill_strata(arms),
        quality_comparisons=qc,
        limitations=[
            (
                "The primary patch comparison is E_fixed versus U_fixed, "
                "three repeats per cold cohort."
            ),
            (
                "D_reference uses the previous production compiler configuration; "
                "it is a deployment observation, not patch-only attribution."
            ),
            (
                "Earlier automatic-tuning runs are retained separately and "
                "are not pooled into the primary comparison."
            ),
            (
                "One process startup per fixed arm does not establish "
                "restart reproducibility."
            ),
            (
                "Decode time per draft round is a serving-work proxy, "
                "not an isolated kernel timer."
            ),
            "Prefill-only C1/C4/C8 and cold/cached 32K/64K are reported separately.",
        ],
    )
    (ROOT / "FIXED_PERFORMANCE_ANALYSIS.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print("INVENTORY", inventory)
    for name, comparison in comparisons.items():
        print(name, comparison["summary"])
        for group, row in comparison["groups"].items():
            print(
                group,
                {
                    key: round(
                        row["metrics"][key].get("relative_change_pct", float("nan")), 2
                    )
                    for key in (
                        "wall_s",
                        "prefill_s",
                        "server_ttft_s",
                        "draft_rounds",
                        "decode_s_per_draft_round",
                    )
                },
            )
    print("QUALITY", qc)
    for arm, data in qa.items():
        gold = {
            name: row for name, row in data["cases"].items() if name.startswith("q_")
        }
        print(
            arm,
            sum(row["passed"] for row in gold.values()),
            "/",
            len(gold),
            "protocol",
            data["protocol"],
            "media",
            data["multimodal"],
        )


if __name__ == "__main__":
    main()
