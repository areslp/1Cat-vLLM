# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only review of frozen synthetic wire identity, gold and paired samples.

Usage: python review_evidence.py public-evidence --output REVIEW.json
No tokenizer, model, GPU or network is required. All missing/failed/unequal-work
samples remain visible. A relative timing result does not certify model quality.
"""

import argparse
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def expected_body(case):
    return {
        "model": "flash-next",
        "messages": case["messages"],
        "temperature": 0,
        "top_p": 1,
        "repetition_penalty": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "max_tokens": case["max_output_tokens"],
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
        "seed": 1234,
    }


def integer_runs(text):
    values = []
    i = 0
    while i < len(text):
        start = i
        if text[i] == "-" and i + 1 < len(text) and text[i + 1].isdecimal():
            i += 1
        if text[i].isdecimal():
            while i < len(text) and text[i].isdecimal():
                i += 1
            values.append(text[start:i])
        else:
            i += 1
    return values


def performance_dimensions(group):
    parts = group.split("_")
    if (
        len(parts) == 4
        and parts[0] in ("main", "decode")
        and all(x.isdecimal() for x in parts[1:])
    ):
        return int(parts[1]), int(parts[2])
    return None


def gold_matches(text, gold):
    text = text.strip()
    if gold["match"] == "exact_text":
        return text == gold["expected"]
    if gold["match"] == "integer":
        values = integer_runs(text)
        return len(values) == 1 and int(values[0]) == gold["expected"]
    try:
        return json.loads(text) == gold["expected"]
    except ValueError:
        return False


def review(root):
    with gzip.open(
        root / "synthetic-fixtures.jsonl.gz", "rt", encoding="utf8"
    ) as stream:
        fixtures = {case["id"]: case for case in map(json.loads, stream)}
    matrix = json.loads((root / "MATRIX.json").read_text())["groups"]
    arms, errors, gold, integrity = {}, [], {}, {}
    for arm in ("main", "patched"):
        arms[arm], gold[arm] = {}, []
        for path in sorted((root / "observations" / arm).glob("*/GROUP_RESULT.json")):
            group = json.loads(path.read_text())
            arms[arm][group["group"]] = group
            group_failures = []
            output_sum = sum(
                r.get("usage", {}).get("completion_tokens", 0)
                for r in group["requests"]
            )
            computed_full = all(
                r["status"] == "completed"
                and r.get("usage", {}).get("completion_tokens") == r["requested_output"]
                for r in group["requests"]
            )
            if group["concurrency"] != len(group["requests"]):
                group_failures.append("concurrency/count mismatch")
            if group["actual_output_tokens"] != output_sum:
                group_failures.append("group output/count mismatch")
            if group["fixed_output_budget_completed"] != computed_full:
                group_failures.append("false fixed-budget flag")
            if group["wall_s"] <= 0 or not math.isfinite(group["wall_s"]):
                group_failures.append("invalid wall duration")
            elif not math.isclose(
                group["group_output_tok_s"], output_sum / group["wall_s"], rel_tol=1e-8
            ):
                group_failures.append("group throughput mismatch")
            integrity[(arm, group["group"])] = list(group_failures)
            if group_failures:
                errors.append(
                    {"arm": arm, "group": group["group"], "errors": group_failures}
                )
            for request in group["requests"]:
                case = fixtures[request["id"]]
                check = {"arm": arm, "group": group["group"], "request": request["id"]}
                failures = []
                if request["request_sha256"] != digest(expected_body(case)):
                    failures.append("wire hash mismatch")
                if request["input_token_sha256"] != case["input_token_sha256"]:
                    failures.append("input hash mismatch")
                if request["input_expected"] != case["expected_input_tokens"]:
                    failures.append("input count mismatch")
                if request["requested_output"] != case["max_output_tokens"]:
                    failures.append("output budget mismatch")
                if request["status"] != "completed":
                    failures.append("request failed or incomplete")
                elif request["usage"]["completion_tokens"] > case["max_output_tokens"]:
                    failures.append("output exceeds requested budget")
                elif request["usage"]["prompt_tokens"] != case["expected_input_tokens"]:
                    failures.append("actual token count mismatch")
                if failures:
                    errors.append({**check, "errors": failures})
                    integrity[(arm, group["group"])].extend(failures)
                if "gold" in case:
                    gold[arm].append(
                        {**check, "passed": gold_matches(request["text"], case["gold"])}
                    )
                elif "expected_json" in case:
                    try:
                        passed = (
                            json.loads(request["text"].strip()) == case["expected_json"]
                        )
                    except ValueError:
                        passed = False
                    gold[arm].append({**check, "passed": passed})
    pairs = []
    for key in matrix:
        rows = {arm: groups.get(key) for arm, groups in arms.items()}
        row = {
            "group": key,
            "measured_arms": [a for a, v in rows.items() if v is not None],
            "errors": [],
        }
        if any(value is None for value in rows.values()):
            row["errors"].append("missing arm")
        else:
            main, patched = rows["main"], rows["patched"]
            identities = lambda group: [
                (
                    r["id"],
                    r["request_sha256"],
                    r["input_token_sha256"],
                    r["input_expected"],
                    r["requested_output"],
                )
                for r in group["requests"]
            ]
            if identities(main) != identities(patched):
                row["errors"].append("different requests or budgets")
            if main["concurrency"] != patched["concurrency"]:
                row["errors"].append("different concurrency")
            for arm, group in rows.items():
                if integrity.get((arm, key)):
                    row["errors"].append(arm + " invalid evidence integrity")
                if not group["successful"]:
                    row["errors"].append(arm + " unsuccessful")
                if not group["fixed_output_budget_completed"]:
                    row["errors"].append(arm + " early EOS or unequal output work")
                if any(r["status"] != "completed" for r in group["requests"]):
                    row["errors"].append(arm + " incomplete stream")
            row["concurrency"] = main["concurrency"]
            row["outputs_equal"] = [
                a["text"] == b["text"]
                for a, b in zip(main["requests"], patched["requests"])
            ]
            row["arms"] = {
                arm: {
                    "wall_s": group["wall_s"],
                    "group_output_tok_s": group["group_output_tok_s"],
                    "actual_output_tokens": group["actual_output_tokens"],
                    "server_phases": group["server_phases"],
                    "MTP": {
                        k: v
                        for k, v in group["server_metric_delta"].items()
                        if "spec_decode" in k
                    },
                    "cache_and_resource_counters": {
                        k: v
                        for k, v in group["server_metric_delta"].items()
                        if any(t in k for t in ("cache", "preempt", "queue"))
                    },
                }
                for arm, group in rows.items()
            }
            if not row["errors"] and key.startswith(("main_", "decode_")):
                row["wall_speedup"] = main["wall_s"] / patched["wall_s"]
                row["phase_mean_speedups"] = {}
                for phase in (
                    "request_prefill_time_seconds",
                    "request_decode_time_seconds",
                    "request_queue_time_seconds",
                ):
                    a, b = (
                        main["server_phases"][phase]["mean_s"],
                        patched["server_phases"][phase]["mean_s"],
                    )
                    row["phase_mean_speedups"][phase] = (
                        a / b if a is not None and b else None
                    )
                row["interpretation"] = (
                    "Server phase wall clocks, not GPU kernel time. "
                    "Cache/JIT/queue effects retained."
                )
        pairs.append(row)
    summaries = []
    for length, concurrency in sorted(
        {
            dims
            for row in pairs
            if (dims := performance_dimensions(row["group"])) is not None
        }
    ):
        for family in ("main", "decode"):
            subset = [
                r
                for r in pairs
                if r["group"].startswith(f"{family}_{length}_{concurrency}_")
            ]
            if not subset:
                continue
            ratios = [r["wall_speedup"] for r in subset if "wall_speedup" in r]
            summaries.append(
                {
                    "family": family,
                    "input_tokens": length,
                    "concurrency": concurrency,
                    "retained_cohorts": len(subset),
                    "valid_paired_fixed_budget_cohorts": len(ratios),
                    "median_wall_speedup": statistics.median(ratios)
                    if ratios
                    else None,
                    "min_wall_speedup": min(ratios) if ratios else None,
                    "max_wall_speedup": max(ratios) if ratios else None,
                }
            )
    return {
        "wire_errors": errors,
        "gold": gold,
        "planned_groups": len(matrix),
        "pairs": pairs,
        "summaries": summaries,
        "all_planned_groups_paired": all(len(r["measured_arms"]) == 2 for r in pairs),
        "quality_note": (
            "Strict gold failures and early EOS are retained; "
            "timings alone do not certify quality."
        ),
        "phase_note": (
            "SSE chunk gaps are not per-token ITL; "
            "MTP can return multiple tokens in one chunk."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = review(args.evidence)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "wire_errors": len(report["wire_errors"]),
                "planned_groups": report["planned_groups"],
                "all_planned_groups_paired": report["all_planned_groups_paired"],
            }
        )
    )
    if report["wire_errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
