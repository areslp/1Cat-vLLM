# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preserve the exact first-block pivot and temperature arithmetic for E7."""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import apply_temperature


@triton.jit
def _pivot(PREFIX, STRIDE, K, PCT, OUT, V: tl.constexpr, BLOCK: tl.constexpr = 8192):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < V
    x = tl.load(PREFIX + row * STRIDE + offs, mask, other=-float("inf"))
    finite = (x > -float("inf")) & mask
    nf = tl.sum(finite)
    xf = tl.where(finite, x, 0.0)
    avg = tl.where(nf > 0, tl.sum(xf) / nf, 0.0)
    sq = tl.where(nf > 0, tl.sum(xf * xf) / nf, 0.0)
    std = tl.sqrt(tl.maximum(sq - avg * avg, 0.0))
    k = tl.load(K + row)
    percentile = tl.cast(k / V * 200, tl.uint32)
    percentile = tl.minimum(percentile, 199)
    sigma = tl.load(PCT + percentile)
    sigma = sigma + tl.abs(sigma) * -0.15
    pivot = avg + std * sigma
    tl.store(OUT + row, pivot)


def temperature(raw, temp):
    x = raw.float()
    apply_temperature(x, torch.arange(raw.shape[0], device=raw.device), temp)
    return x
