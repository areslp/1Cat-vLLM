# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recompute paired endpoint observations without a model or GPU."""

import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path(__file__).with_name("observations.json")
    )
    parser.add_argument(
        "--labels",
        nargs=2,
        default=["upstream-timed", "E-timed"],
        help="Two recorded labels; default is the aligned-choice run",
    )
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    groups = {}
    for row in data["observations"]:
        if row["label"] in args.labels:
            groups.setdefault(row["group"], {})[row["label"]] = row
    rows = []
    for key, pair in sorted(groups.items()):
        if any(label not in pair for label in args.labels):
            continue
        a, b = (pair[label] for label in args.labels)
        matched = len(a["requests"]) == len(b["requests"]) and all(
            all(
                x[field] == y[field]
                for field in (
                    "request_sha256",
                    "input_token_sha256",
                    "input_expected",
                    "requested_output",
                )
            )
            for x, y in zip(a["requests"], b["requests"])
        )
        qualified = (
            matched
            and a["successful"]
            and b["successful"]
            and a["fixed_output_budget_completed"]
            and b["fixed_output_budget_completed"]
            and a["actual_output_tokens"] == b["actual_output_tokens"]
            and all(
                r["server_phases"]["request_prefill_time_seconds"]["count"]
                == len(r["requests"])
                for r in (a, b)
            )
        )
        row = {
            "group": key,
            "qualified_request_pair": qualified,
            "wall_ratio_E_over_upstream": b["wall_s"] / a["wall_s"],
            "output_text_identical": all(
                (x["text"], x.get("reasoning", ""))
                == (y["text"], y.get("reasoning", ""))
                for x, y in zip(a["requests"], b["requests"])
            ),
            "draft_rounds": [
                r["server_metric_delta"].get("vllm:spec_decode_num_drafts_total")
                for r in (a, b)
            ],
            "cached_prefill_tokens_upstream_E": [
                r["server_metric_delta"].get("vllm:prompt_tokens_cached_total", 0)
                for r in (a, b)
            ],
        }
        work = []
        for observation in (a, b):
            metrics = observation["server_metric_delta"]
            drafts = metrics["vllm:spec_decode_num_drafts_total"]
            draft_tokens = metrics["vllm:spec_decode_num_draft_tokens_total"]
            accepted = metrics["vllm:spec_decode_num_accepted_tokens_total"]
            decode = observation["server_phases"]["request_decode_time_seconds"][
                "sum_s"
            ]
            work.append(
                {
                    "drafts": drafts,
                    "accepted_fraction": accepted / draft_tokens,
                    "accepted_tokens_per_draft": accepted / drafts,
                    "request_decode_sum_s": decode,
                    "request_decode_sum_per_draft_s": decode / drafts,
                    "engine_iterations_including_prefill": metrics.get(
                        "vllm:iteration_tokens_total_count"
                    ),
                }
            )
        row["MTP_work_upstream_E"] = work
        row["cached_prefill_work_matches"] = all(
            a["server_metric_delta"].get(key, 0) == b["server_metric_delta"].get(key, 0)
            for key in (
                "vllm:prompt_tokens_cached_total",
                "vllm:request_prefill_kv_computed_tokens_sum",
            )
        )
        n0, n1 = (r["drafts"] for r in work)
        c0, c1 = (r["request_decode_sum_per_draft_s"] for r in work)
        row["request_decode_delta_work_s"] = (n1 - n0) * (c0 + c1) / 2
        row["request_decode_delta_cost_s"] = (c1 - c0) * (n0 + n1) / 2
        row["request_cost_ratio_E_over_upstream"] = c1 / c0
        row["request_cost_scope"] = (
            "Summed request wall clocks per summed request draft. Concurrent "
            "requests overlap; includes scheduling, CPU and collectives. "
            "Neither isolated kernel cost nor GPU batch-step cost."
        )
        rows.append(row)
        print(json.dumps(row))
    ratios = [
        r["wall_ratio_E_over_upstream"] for r in rows if r["qualified_request_pair"]
    ]
    cache = data.get("kernel_choice_review")
    active = data.get("active_role_choice_review")
    if all(label.endswith("-unaligned") for label in args.labels):
        previous = data["previous_unaligned_pair"]
        cache = previous["cache"]
        active = {
            "parameter_differences": previous["active_choice_differences"],
            "qualified_kernel_control": False,
        }
    print(
        json.dumps(
            {
                "pairs": len(rows),
                "qualified_request_pairs": len(ratios),
                "median_wall_ratio_descriptive_only": statistics.median(ratios)
                if ratios
                else None,
                "faster": sum(r < 1 for r in ratios),
                "slower": sum(r > 1 for r in ratios),
                "same_output_text_pairs": sum(r["output_text_identical"] for r in rows),
                "labels": args.labels,
                "kernel_choice_review": cache,
                "active_role_choice_review": active,
            }
        )
    )


if __name__ == "__main__":
    main()
