# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Write recorded choices to a stopped, task-local Torch 2.10 role cache."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument(
        "--choices",
        type=Path,
        default=Path(__file__).with_name("canonical-kernel-choices.json"),
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    recorded = json.loads(args.choices.read_text())
    counts = {}
    for role, root in [("backbone", args.backbone), ("draft", args.draft)]:
        helpers = {path.name: path for path in root.rglob("*.py")}
        for row in recorded[role]:
            helper = helpers[row["helper"]]
            assert (
                hashlib.sha256(helper.read_bytes()).hexdigest() == row["helper_sha256"]
            ), "Different generated helper; do not force this choice"
            destination = helper.parent / row["key"]
            if destination.exists():
                previous = json.loads(destination.read_text())
                assert (
                    previous["configs_hash"] == row["configuration"]["configs_hash"]
                ), "Different candidate set"
            if args.apply:
                destination.write_text(json.dumps(row["configuration"]) + "\n")
        counts[role] = len(recorded[role])
    print(json.dumps({"applied": args.apply, "choices": counts}))


if __name__ == "__main__":
    main()
