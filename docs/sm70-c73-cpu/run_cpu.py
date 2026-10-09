# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the refreshed owner-contract scope on the real CPU platform."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junitxml", type=Path)
    args = parser.parse_args()
    evidence = Path(__file__).resolve().parent
    root = evidence.parents[1]
    scope = json.loads((evidence / "scope.json").read_text())
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    paths = [evidence / "cpu_bootstrap", root, root / "flash-attention-v100"]
    inherited = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(map(str, paths))
    if inherited:
        env["PYTHONPATH"] += os.pathsep + inherited
    command = [sys.executable, "-B", "-m", "pytest", "--noconftest", "-q"]
    if args.junitxml is not None:
        command.append(f"--junitxml={args.junitxml}")
    command.extend(scope["tests"])
    return subprocess.run(command, cwd=root, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
