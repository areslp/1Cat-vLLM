# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synthetic H2 scratch and TP4 top1 changed-input CUDA-graph contracts."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

W = Path(os.environ.get("SM70_REPRO_OUTPUT", "."))


def finish_tp_group(graph, dist):
    """Release NCCL graph user objects before collective teardown."""
    graph.reset()
    dist.barrier()
    dist.destroy_process_group()


def h2():
    from vllm.model_executor.layers import sm70_fuse47 as op

    results = []
    layer = SimpleNamespace(
        expert_map=None, local_num_experts=192, global_num_experts=192
    )
    for rows in (1, 5, 8, 40, 128):
        slots, hidden, k = rows * 8, 2560, 8
        gen = torch.Generator(device="cuda").manual_seed(3300 + rows)
        x = torch.randn(rows, hidden, generator=gen, device="cuda", dtype=torch.float16)
        ids = torch.randint(
            0, 192, (rows, k), generator=gen, device="cuda", dtype=torch.int32
        )
        ids[:, 0] = 17

        def ints(n, dtype=torch.int32):
            return torch.full((n,), -999, dtype=dtype, device="cuda")

        buffers = {
            "token_expert_indices": torch.arange(
                slots, device="cuda", dtype=torch.int32
            ),
            "output": torch.full_like(x, 99),
            "permuted_input": torch.empty(
                slots, hidden, device="cuda", dtype=torch.float16
            ),
            "expert_offsets64": ints(193, torch.int64),
            "expert_offsets": ints(193),
            "inv_permuted_idx": ints(slots),
            "permuted_idx": ints(slots),
            "permuted_experts_id": ints(slots),
            "sorted_row_idx": ints(slots),
            "topk_ids": ints(slots),
        }
        assert op.h2_supported(layer, x, ids, buffers)

        def call(x=x, ids=ids, buffers=buffers):
            op.moe_permute_fused(layer, x, ids, buffers)

        def check(x=x, ids=ids, buffers=buffers, k=k):
            flat = ids.flatten()
            order = torch.argsort(flat, stable=True)
            expected_offset = (
                flat[:, None] < torch.arange(193, device="cuda")[None]
            ).sum(0)
            assert torch.equal(buffers["output"], torch.zeros_like(x))
            assert torch.equal(buffers["topk_ids"], flat)
            assert torch.equal(buffers["permuted_idx"].long(), order)
            assert torch.equal(buffers["sorted_row_idx"].long(), order)
            assert torch.equal(buffers["inv_permuted_idx"].long(), torch.argsort(order))
            assert torch.equal(buffers["permuted_experts_id"], flat[order])
            assert torch.equal(buffers["expert_offsets64"], expected_offset)
            assert torch.equal(buffers["expert_offsets"].long(), expected_offset)
            assert torch.equal(buffers["permuted_input"], x[order // k])

        call()
        check()
        for _ in range(3):
            call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        x.add_(0.25)
        ids.copy_((ids + 23) % 192)
        graph.replay()
        torch.accelerator.synchronize()
        check()
        results.append(
            {
                "rows": rows,
                "slots": slots,
                "scratch_and_row_map_exact": True,
                "changed_input_graph_replay": True,
            }
        )
    (W / "H2_COMPONENT_RESULT.json").write_text(json.dumps(results, indent=2) + "\n")


def tp_worker(rank):
    import torch.distributed as dist

    from vllm.model_executor.layers import sm70_draft47 as op

    torch.accelerator.set_device_index(rank)
    dist.init_process_group("nccl", rank=rank, world_size=4)
    valid, width, rows = 62077, 62080, 8
    logits = torch.empty(rows, width, device="cuda", dtype=torch.float16)
    packets = torch.empty(4 * rows, 2, device="cuda", dtype=torch.float32)
    full = torch.empty(4 * rows, width, device="cuda", dtype=torch.float16)

    def fill(case):
        torch.manual_seed(777 + rank)
        logits.normal_()
        if case == "ties":
            logits[:, 7] = float("inf")
            logits[:, 513] = float("inf")
        elif case == "nans":
            logits[:, 7] = float("nan")
            logits[:, 515] = float("nan")
        elif case == "zero":
            logits.zero_()
        elif case == "minus_inf":
            logits.fill_(-float("inf"))
        # Mask shard padding exactly as the production logits processor does.
        logits[:, valid:] = -float("inf")

    def call():
        pair = op.local_top1_pair(logits, rank * width)
        dist.all_gather_into_tensor(packets, pair)
        return op.global_top1(packets, 4, rows)

    def check(actual):
        dist.all_gather_into_tensor(full, logits)
        expected = (
            full.view(4, rows, width)
            .permute(1, 0, 2)
            .reshape(rows, 4 * width)
            .argmax(-1)
        )
        assert torch.equal(actual, expected), (rank, actual, expected)

    fill("finite")
    for _ in range(3):
        check(call())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = call()
    results = []
    for case in ("finite", "ties", "nans", "zero", "minus_inf"):
        fill(case)
        graph.replay()
        torch.accelerator.synchronize()
        check(actual)
        results.append(
            {
                "case": case,
                "exact_global_ids": True,
                "NCCL_TP4_packet_transport": True,
                "changed_input_graph_replay": True,
            }
        )
    if rank == 0:
        (W / "TP4_COMPONENT_RESULT.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
    # NCCL graph communicators retain graph user objects. Release the captured
    # graph before destroying its process group, after synchronized replays.
    finish_tp_group(graph, dist)
    del graph


if __name__ == "__main__":
    W.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    if sys.argv[1] == "h2":
        h2()
    else:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ.setdefault("MASTER_PORT", "18344")
        torch.multiprocessing.spawn(tp_worker, nprocs=4, join=True)
