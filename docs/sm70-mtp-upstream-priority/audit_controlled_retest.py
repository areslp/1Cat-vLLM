# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit a whole-graph/config control as data, without Torch or devices."""

import argparse
import json
from pathlib import Path


def signature(row):
    return {key: row[key] for key in ("source_ast_sha256", "kernel_sha256", "config")}


def analyze(data):
    expected = list(range(data["expected_ranks"]))
    launch_comparison = []
    errors = []
    arms = data["arms"]
    for arm in arms:
        ranks = [row["rank"] for row in arm["autotuner_runs"]]
        if sorted(ranks) != expected:
            errors.append("missing or repeated rank in autotuner observations")
    for rank in expected:
        rows = [
            [row for row in arm["autotuner_runs"] if row["rank"] == rank]
            for arm in arms
        ]
        if any(len(values) != 1 for values in rows):
            continue
        events = [values[0]["records"] for values in rows]
        if any(
            len(values) != data["expected_unique_runs_per_rank"] for values in events
        ):
            errors.append(f"incomplete rank {rank} observation")
        comparisons = [
            signature(a) == signature(b) for a, b in zip(*events, strict=False)
        ]
        launch_comparison.append(
            {
                "rank": rank,
                "counts": [len(values) for values in events],
                "equal_ordered_signatures": sum(comparisons),
                "all_signatures_equal": (
                    len(events[0]) == len(events[1]) and all(comparisons)
                ),
            }
        )
    graphs = data["graph_pairs"]
    keys = [(row["rank"], row["graph"], row["subgraph"]) for row in graphs]
    declared_keys = {tuple(row) for row in data["expected_graph_keys"]}
    if (
        len(keys) != data["expected_graph_pairs"]
        or len(set(keys)) != len(keys)
        or set(keys) != declared_keys
    ):
        errors.append("missing or repeated generated graph")
    equal_runtime = sum(
        row["ast_hashes"][0][0] == row["ast_hashes"][1][0] for row in graphs
    )
    equal_benchmark = sum(
        row["ast_hashes"][0][1] == row["ast_hashes"][1][1] for row in graphs
    )
    aligned = (
        not errors
        and len(launch_comparison) == data["expected_ranks"]
        and all(row["all_signatures_equal"] for row in launch_comparison)
        and equal_runtime == equal_benchmark == len(graphs)
    )
    complete = all(
        arm["healthy"]
        and arm["finished"]
        and arm["gold_passed"]
        and not arm["startup_failure"]
        and arm["timed_groups"] == data["expected_timed_groups_per_arm"]
        for arm in arms
    )
    admitted = aligned and complete and data["request_pair_validation_passed"]
    return {
        "schema": 1,
        "observed_run_comparison_by_rank": launch_comparison,
        "generated_graph_pairs": len(graphs),
        "equal_runtime_ast_pairs": equal_runtime,
        "equal_benchmark_ast_pairs": equal_benchmark,
        "complete_config_and_generated_source_alignment": aligned,
        "completed_both_arms": complete,
        "performance_pair_admitted": admitted,
        "usable_performance_pairs": data["expected_performance_groups_per_arm"]
        if admitted
        else 0,
        "throughput_ratio": None,
        "validation_errors": errors,
        "limits": [
            "Autotuner.run observations may include compilation benchmarks",
            "Generated runtime references are audited separately",
            "No direct CUDA graph replay trace is claimed",
            "An incomplete driver with no recorded failures is not a pass",
            "A startup failure does not isolate its cause",
            "Wire/input/output validation is a separate required admission gate",
            "This audit computes admission, not a throughput estimate",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(json.loads(args.observations.read_text()))
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
