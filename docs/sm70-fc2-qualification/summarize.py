# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recompute every paired fixed-budget cohort using public records only."""

import json
import statistics
from pathlib import Path


def main():
    data = json.loads(Path(__file__).with_name("observations.json").read_text())
    groups = {}
    for row in data["observations"]:
        if row["label"] in ("main-timed", "patched-timed"):
            groups.setdefault(row["group"], {})[row["label"]] = row
    ratios = []
    for key, pair in sorted(groups.items()):
        a, b = pair["main-timed"], pair["patched-timed"]
        assert a["successful"] and b["successful"]
        assert a["fixed_output_budget_completed"] and b["fixed_output_budget_completed"]
        assert len(a["requests"]) == len(b["requests"])
        for x, y in zip(a["requests"], b["requests"]):
            for field in ("request_sha256", "input_token_sha256"):
                assert x[field] == y[field]
            assert x["usage"]["completion_tokens"] == y["usage"]["completion_tokens"]
        ratio = b["wall_s"] / a["wall_s"]
        ratios.append(ratio)
        print(
            f"{key}: main={a['wall_s']:.6f}s candidate={b['wall_s']:.6f}s "
            f"ratio={ratio:.9f}"
        )
    assert len(ratios) == data["paired_summary"]["expected_timed_cohorts"] == 12
    print(
        json.dumps(
            {
                "pairs": len(ratios),
                "faster": sum(ratio < 1 for ratio in ratios),
                "slower": sum(ratio > 1 for ratio in ratios),
                "median_wall_ratio_descriptive_only": statistics.median(ratios),
                "selected_configuration_difference_count": data[
                    "selected_configuration_difference_count"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
