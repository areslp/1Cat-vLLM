# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Separate phase/work accounting from the PR and deployment comparisons."""

import hashlib
import json
import math
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PHASES = {
    "prefill_s": "request_prefill_time_seconds",
    "server_ttft_s": "time_to_first_token_seconds",
    "decode_s": "request_decode_time_seconds",
    "queue_s": "request_queue_time_seconds",
    "server_e2e_s": "e2e_request_latency_seconds",
}


def values(row):
    data = {"wall_s": row["wall_s"]}
    data.update({k: row["server_phases"][v]["mean_s"] for k, v in PHASES.items()})
    data["tokens_per_s"] = row["actual_output_tokens"] / row["wall_s"]
    delta = row["server_metric_delta"]
    drafts = delta.get("vllm:spec_decode_num_drafts_total", 0)
    proposed = delta.get("vllm:spec_decode_num_draft_tokens_total", 0)
    accepted = delta.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    data["draft_rounds"] = drafts
    data["accepted_draft_tokens"] = accepted
    data["draft_acceptance_fraction"] = accepted / proposed if proposed else None
    # Serving-phase duration divided by work: not an isolated kernel timing.
    decode = row["server_phases"]["request_decode_time_seconds"]["mean_s"]
    data["decode_s_per_draft_round"] = (
        decode / drafts if drafts and row["concurrency"] == 1 else None
    )
    return data


def output_signatures(rows):
    outputs = {}
    for row in rows:
        for request in row["requests"]:
            payload = json.dumps(
                [request["text"], request["reasoning"]], ensure_ascii=False
            )
            signature = hashlib.sha256(payload.encode()).hexdigest()
            outputs.setdefault(request["id"], set()).add(signature)
    return {name: sorted(signatures) for name, signatures in outputs.items()}


def percentile(values, p):
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def comparison(left, right, metric):
    a, b = [values(x)[metric] for x in left], [values(x)[metric] for x in right]
    a, b = [x for x in a if x is not None], [x for x in b if x is not None]
    if not a or not b or statistics.median(a) == 0:
        return {"baseline": a, "candidate": b, "relative_change_unavailable": True}
    ratio = statistics.median(b) / statistics.median(a) - 1
    rng = random.Random(1234)
    boots = []
    for _ in range(4000):
        ma = statistics.median(rng.choices(a, k=len(a)))
        mb = statistics.median(rng.choices(b, k=len(b)))
        if ma:
            boots.append(100 * (mb / ma - 1))
    ci = [percentile(boots, 0.025), percentile(boots, 0.975)]
    return {
        "baseline_median": statistics.median(a),
        "candidate_median": statistics.median(b),
        "relative_change_pct": 100 * ratio,
        "bootstrap_95pct_interval_pct": ci,
        "baseline_samples": a,
        "candidate_samples": b,
        "interpretation": (
            "descriptive small-sample bootstrap; repeated cohorts, "
            "not independent model draws"
        ),
    }


def main():
    arms = {}
    excluded = []
    observed_commits = {}
    for arm in ("D", "U", "E"):
        groups = {}
        observed_commits[arm] = set()
        for phase in ("a", "b"):
            paths = [
                path
                for kind in ("timed", "cached")
                for path in (ROOT / "results" / f"{arm}-{kind}-{phase}").glob(
                    "[0-9]*.json"
                )
            ]
            for path in sorted(paths):
                data = json.loads(path.read_text())
                observed_commits[arm].add(data["commit"])
                if data.get("priming_cohort"):
                    continue
                if not data["qualified"]:
                    excluded.append(
                        {
                            "file": str(path.relative_to(ROOT)),
                            "counts": data["qualified_metric_counts"],
                            "cache": data["cache_work_qualified"],
                            "budget": data["fixed_output_budget_completed"],
                        }
                    )
                    continue
                group = ("cached/" if data.get("require_cached") else "") + data[
                    "group"
                ]
                groups.setdefault(group, []).append(data)
        arms[arm] = groups
    result = {
        "source_commits": {
            arm: next(iter(commits)) if len(commits) == 1 else sorted(commits)
            for arm, commits in observed_commits.items()
        },
        "current_runtime_manifest": json.loads(
            (ROOT / "runtime-commits.json").read_text()
        ),
        "excluded_unqualified_groups": excluded,
        "comparisons": {},
    }
    for baseline in ("U", "D"):
        rows = {}
        for group in sorted(set(arms[baseline]) & set(arms["E"])):
            a, b = arms[baseline][group], arms["E"][group]
            output_a, output_b = output_signatures(a), output_signatures(b)
            rows[group] = {
                "baseline_repeats": len(a),
                "candidate_repeats": len(b),
                "expected_prompt_tokens": a[0]["expected_prompt_tokens"],
                "baseline_output_signatures": output_a,
                "candidate_output_signatures": output_b,
                "output_signature_sets_equal": output_a == output_b,
                "metrics": {
                    metric: comparison(a, b, metric) for metric in values(a[0])
                },
            }
        result["comparisons"]["E_vs_" + baseline] = rows
    (ROOT / "PERFORMANCE_ANALYSIS.json").write_text(json.dumps(result, indent=2) + "\n")
    for comparison_name, rows in result["comparisons"].items():
        print(comparison_name)
        for group, row in rows.items():
            print(
                group,
                "n=",
                row["baseline_repeats"],
                row["candidate_repeats"],
                {
                    name: round(
                        row["metrics"][name].get("relative_change_pct", float("nan")), 2
                    )
                    for name in ("wall_s", "prefill_s", "server_ttft_s", "decode_s")
                },
            )
    print("excluded", len(excluded))


if __name__ == "__main__":
    main()
