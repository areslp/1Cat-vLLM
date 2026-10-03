#!/usr/bin/env python3
"""Build the isolated qsa_planner58 PyTorch CUDA extension."""

import os
from pathlib import Path
import json
import sys

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise SystemExit("CPU-only build requires CUDA_VISIBLE_DEVICES=''")
os.environ["CUDA_HOME"] = "/home/l/work/qwen38-bench/cuda-home"
os.environ["MAX_JOBS"] = "2"
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0"

import torch  # noqa: E402
from torch.utils.cpp_extension import load  # noqa: E402


source_dir = Path(__file__).resolve().parent
build_dir_value = os.environ.get("BUILD_DIR")
if not build_dir_value:
    raise SystemExit("Set BUILD_DIR to a new, separate extension build directory")
build_dir = Path(build_dir_value).expanduser().resolve()
if build_dir == source_dir or source_dir in build_dir.parents:
    raise SystemExit("BUILD_DIR must be outside the source directory")
if build_dir.exists():
    raise SystemExit("BUILD_DIR must be a new directory; retain prior failures")
if torch.__version__ != "2.10.0+cu128" or sys.version_info[:2] != (3, 12):
    raise SystemExit("Unexpected declared Python/Torch build runtime")
build_dir.mkdir(parents=True, exist_ok=False)
print(json.dumps({"scope": "CPU_ONLY_BUILD_NO_GPU_SMOKE",
                  "Python": sys.version, "Torch": torch.__version__,
                  "CUDA_HOME": os.environ["CUDA_HOME"],
                  "MAX_JOBS": os.environ["MAX_JOBS"],
                  "TORCH_CUDA_ARCH_LIST": os.environ["TORCH_CUDA_ARCH_LIST"],
                  "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
                  "build_directory": str(build_dir)}), flush=True)

module = load(
    name="qsa_planner58",
    sources=[str(source_dir / "bindings.cpp"), str(source_dir / "planner58.cu")],
    build_directory=str(build_dir),
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "-Xptxas=-v"],
    with_cuda=True,
    verbose=True,
)
print(f"built {module.__name__} with torch {torch.__version__}")
print(f"CANDIDATE_PATH={module.__file__}")
