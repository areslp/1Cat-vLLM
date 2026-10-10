# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare synthetic correctness cases individually, preserving baseline failures."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def signature(row):
    return hashlib.sha256(
        json.dumps([row.get("text"), row.get("reasoning")], ensure_ascii=False).encode()
    ).hexdigest()


def arm_result(arm):
    root = ROOT / "evidence" / arm
    cases = {}
    for path in sorted(root.glob("*/GOLD_RESULT.json")):
        group = path.parent.name
        group_result = json.loads((path.parent / "GROUP_RESULT.json").read_text())
        cases[group] = {
            "passed": json.loads(path.read_text())["passed"],
            "checks": json.loads(path.read_text())["checks"],
            "outputs": {
                row["id"]: {
                    "signature": signature(row),
                    "text": row.get("text"),
                    "input_token_sha256": row.get("input_token_sha256"),
                    "request_sha256": row.get("request_sha256"),
                }
                for row in group_result["requests"]
            },
        }
    protocol = {}
    for path in sorted((root / "protocol_features").glob("*.json")):
        data = json.loads(path.read_text())
        protocol[path.stem] = {
            k: data.get(k)
            for k in (
                "gold_passed",
                "status",
                "actual_prompt_matches_tokenize",
                "request_sha256",
            )
        }
    parallel = root / "parallel_gold_8" / "PARALLEL_ROUTING_RESULT.json"
    multimodal = {}
    for name in ("image", "video"):
        path = ROOT / f"{arm}-{name}.json"
        if path.exists():
            data = json.loads(path.read_text())
            multimodal[name] = {
                k: data.get(k)
                for k in ("gold_passed", "completed", "text", "image_sha256")
            }
    return dict(
        cases=cases,
        protocol=protocol,
        parallel=json.loads(parallel.read_text()) if parallel.exists() else None,
        multimodal=multimodal,
    )


def main():
    arms = {arm: arm_result(arm) for arm in ("D", "U", "E")}
    comparisons = {}
    for baseline in ("D", "U"):
        left, right = arms[baseline]["cases"], arms["E"]["cases"]
        common = sorted(left.keys() & right.keys())
        comparisons["E_vs_" + baseline] = {
            "common_cases": common,
            "new_failures": [
                name
                for name in common
                if left[name]["passed"] and not right[name]["passed"]
            ],
            "improved_cases": [
                name
                for name in common
                if not left[name]["passed"] and right[name]["passed"]
            ],
            "different_outputs": [
                name
                for name in common
                if left[name]["outputs"] != right[name]["outputs"]
            ],
            "missing_candidate_cases": sorted(left.keys() - right.keys()),
        }
    result = dict(arms=arms, comparisons=comparisons)
    (ROOT / "QUALITY_ANALYSIS_PRIVATE.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    for arm, data in arms.items():
        quality = {
            name: row for name, row in data["cases"].items() if name.startswith("q_")
        }
        print(
            arm,
            sum(row["passed"] for row in quality.values()),
            "/",
            len(quality),
            "failures",
            [name for name, row in quality.items() if not row["passed"]],
        )
    print(json.dumps(comparisons, ensure_ascii=False))


if __name__ == "__main__":
    main()
