#!/usr/bin/env python3
"""Build the isolated qsa_planner54 PyTorch CUDA extension."""

import os
from pathlib import Path

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
build_dir.mkdir(parents=True, exist_ok=True)

module = load(
    name="qsa_planner54",
    sources=[str(source_dir / "bindings.cpp"), str(source_dir / "planner54.cu")],
    build_directory=str(build_dir),
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    with_cuda=True,
    verbose=True,
)
print(f"built {module.__name__} with torch {torch.__version__}")
print(f"CANDIDATE_PATH={module.__file__}")
