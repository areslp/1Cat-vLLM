# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real frozen draft weights: old Triton, upstream Triton, upstream native.

Diagnostic graph timings are separate from HTTP performance requests.
"""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

import torch

from benchmarks.kernels.benchmark_sm70_moe_packed_w13 import graph, latency
from benchmarks.kernels.benchmark_sm70_mtp_moe_fp16 import checkpoint_weight
from vllm.model_executor.layers.fused_moe import fused_moe as moe
from vllm.triton_utils import tl

OUTPUT = None


def atomic(data):
    tmp = OUTPUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(OUTPUT)


def measure(weight, m, down):
    n, k = weight.shape[-2:]
    x = torch.randn(m * 10 if down else m, k, device="cuda", dtype=torch.float16)
    ids = torch.empty(m * 10, device="cuda", dtype=torch.int32).random_(0, 512)
    weights = torch.softmax(torch.randn(m, 10, device="cuda"), -1)
    padded = torch.tensor([m * 20], device="cuda", dtype=torch.int32)
    outputs = [x.new_empty(m, 10, n) for _ in range(3)]
    upstream = dict(
        BLOCK_SIZE_M=2,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=64,
        GROUP_SIZE_M=1,
        SPLIT_K=1,
        num_warps=4,
        num_stages=3,
    )
    old = upstream | {"BLOCK_SIZE_N": 64, "num_warps": 2, "num_stages": 4}
    selected = moe.get_default_config(m, 512, 160, 2560, 10, None)
    assert all(selected[key] == value for key, value in upstream.items()), selected
    # Actual fixed dispatcher, including its native admission guard.
    original = torch.ops._C.sm70_mtp_moe_fp16_out
    calls = []

    def record(*args):
        calls.append(True)
        return original(*args)

    torch.ops._C.sm70_mtp_moe_fp16_out = record
    try:
        moe.dispatch_fused_moe_kernel(
            x,
            weight,
            outputs[2],
            None,
            None,
            None,
            weights,
            None,
            ids,
            padded,
            down,
            1 if down else 10,
            selected,
            tl.float16,
            False,
            False,
            False,
            False,
            False,
        )
    finally:
        torch.ops._C.sm70_mtp_moe_fp16_out = original
    assert len(calls) == 1, "fixed real dispatcher did not call upstream native"

    def triton(arm, config):
        moe.invoke_fused_moe_triton_kernel(
            x,
            weight,
            outputs[arm],
            None,
            None,
            weights,
            None,
            ids,
            padded,
            down,
            1 if down else 10,
            config,
            tl.float16,
            False,
            False,
            False,
            False,
            False,
        )

    functions = [
        lambda: triton(0, old),
        lambda: triton(1, upstream),
        lambda: original(outputs[2], x, weight, ids, weights, padded, down),
    ]
    graphs = []
    try:
        for function in functions:
            graphs.append(graph(function, unroll=8))
        comparisons = []
        for scale in (0.0, 0.001, 0.03, 0.1, 1.0, 3.0):
            x.normal_(0, scale)
            ids.random_(0, 512)
            ids[0] = -1
            weights.copy_(torch.softmax(torch.randn_like(weights), -1))
            for output in outputs:
                output.fill_(float("nan"))
            for capture in graphs:
                capture.replay()
            torch.accelerator.synchronize()
            native_diff = int(
                (outputs[1].view(torch.int16) != outputs[2].view(torch.int16)).sum()
            )
            old_diff = int(
                (outputs[0].view(torch.int16) != outputs[1].view(torch.int16)).sum()
            )
            assert native_diff == 0, (m, down, scale, native_diff)
            comparisons.append(
                {
                    "input_scale": scale,
                    "native_vs_upstream_bit_mismatches": native_diff,
                    "old_vs_upstream_bit_mismatches": old_diff,
                }
            )
        ids.random_(0, 512)
        samples = [[], [], []]
        for trial in range(7):
            order = (0, 1, 2) if trial % 2 == 0 else (2, 1, 0)
            for arm in order:
                samples[arm].append(latency(graphs[arm], repeats=20, unroll=8))
        return {
            "m": m,
            "down": down,
            "fixed_actual_dispatch_native_calls": len(calls),
            "comparisons": comparisons,
            "samples_us": samples,
            "median_us": list(map(median, samples)),
            "timing_order": "alternating 0,1,2 / 2,1,0",
            "old_tile_is_production_regression_only_for_m1": m == 1,
        }
    finally:
        for capture in graphs:
            capture.reset()
        torch.accelerator.synchronize()


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    OUTPUT = args.out
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(20261008)
    assert torch.cuda.get_device_capability() == (7, 0)
    source = Path(moe.__file__).parents[3]
    native = source / "_C.abi3.so"
    report = {
        "passed": False,
        "complete": False,
        "rows": [],
        "weights_unmodified": True,
        "activations_and_routes": "synthetic",
        "scope": "real checkpoint MTP expert projection slices, all four TP ranks",
        "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
    }
    atomic(report)
    for rank in range(4):
        for down in (False, True):
            weight = checkpoint_weight(args.model, rank, down)
            for m in (1, 5):
                row = {"rank": rank, **measure(weight, m, down)}
                report["rows"].append(row)
                atomic(report)
                print(json.dumps(row), flush=True)
            del weight
            torch.accelerator.empty_cache()
    report.update(passed=True, complete=True)
    atomic(report)


if __name__ == "__main__":
    main()
