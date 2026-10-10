# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Start one comparison arm using a normally built source checkout."""

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("U", "E"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = json.loads((root / "serving-configuration.json").read_text())
    source, model = args.source.resolve(), args.model_dir.resolve()
    cache = args.cache_root.resolve() / args.arm
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("ONECAT_", "VLLM_", "TORCHINDUCTOR_", "TRITON_"))
    }
    env.update(config["main" if args.arm == "U" else "patched"])
    env.update(
        PYTHONPATH=os.pathsep.join(
            (
                str(source),
                str(source / "vllm/third_party"),
                str(source / "flash-attention-v100"),
                env.get("PYTHONPATH", ""),
            )
        ),
        VLLM_CACHE_ROOT=str(cache / "vllm"),
        TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
        TRITON_CACHE_DIR=str(cache / "triton"),
        TORCH_EXTENSIONS_DIR=str(cache / "cpp"),
    )
    command = [sys.executable] + [
        str(model) if value == "${MODEL_DIR}" else value
        for value in config["server_arguments"]
    ]
    os.chdir(source)
    os.execve(sys.executable, command, env)


if __name__ == "__main__":
    main()
