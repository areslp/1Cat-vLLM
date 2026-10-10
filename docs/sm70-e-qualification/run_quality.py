# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run every synthetic correctness group and retain failures for paired review."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--arm", choices=("D", "U", "E", "U-fixed", "E-fixed"), required=True
)
parser.add_argument("--url", default="http://127.0.0.1:18200")
args = parser.parse_args()
root = Path(__file__).resolve().parent
(root / "evidence" / args.arm).mkdir(parents=True, exist_ok=True)
plan = json.loads((root / "quality-plan.json").read_text())
outcomes = {}
for group in plan["quality_groups"] + plan["additional_groups"]:
    result = subprocess.run(
        [sys.executable, str(root / "quality.py"), args.arm, group, args.url]
    )
    outcomes[group] = result.returncode
for kind in ("image", "video"):
    command = [
        sys.executable,
        str(root / "multimodal_normal_smoke.py"),
        "--url",
        args.url,
        "--out",
        str(root / (args.arm + "-" + kind + ".json")),
    ]
    if kind == "video":
        command += ["--video", str(root / "synthetic-red-1s.mp4")]
    outcomes[kind] = subprocess.run(command).returncode
(root / (args.arm + "-quality-exit-codes.json")).write_text(
    json.dumps(outcomes, indent=2) + "\n"
)
print(json.dumps(outcomes, indent=2))
