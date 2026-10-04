# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qrita-compatible outlier packet v3; correctness and speed are separate gates.
The original first-block pivot is computed before sharding; all outliers are
stably compacted and joined in the original Qrita BUFFER order. Same ternary
search, exp/sum/div/log and reference flags. No approximate boundary epsilon.
"""

import torch

from vllm.utils.platform_utils import num_compute_units
from vllm.v1.sample.ops import topk_topp_triton as tt
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch as reference

from . import e7_compat as old
from . import packet_ops as po
from . import packet_topk as pt


def canonical(values, ids):
    bits = values.contiguous().view(torch.int32).to(torch.int64)
    bits = torch.where(bits == -(2**31), torch.zeros_like(bits), bits)
    order = torch.where(bits < 0, (~bits) & 0xFFFFFFFF, bits | 0x80000000)
    perm = torch.sort((order - 0x80000000) * 0x100000000 + ids, dim=-1).indices
    return values.gather(1, perm), ids.gather(1, perm)


CAP = po.CAP
TABLES = {}


def tables(dev):
    if dev not in TABLES:
        TABLES[dev] = (
            torch.tensor(tt._PERCENTILE_TO_STD_TABLE, device=dev),
            torch.tensor(tt._NORMAL_CDF_TO_SIGMA_TABLE, device=dev),
        )
    return TABLES[dev]


def pivot(x, k, V, top_p_enabled):
    B = x.shape[0]
    out = torch.empty(B, device=x.device)
    kw = {}
    if tt._use_sm70_topk_topp_8_warps(x.device, B, V, True, top_p_enabled):
        kw["num_warps"] = 8
    old._pivot[(B,)](x, x.stride(0), k, tables(x.device)[0], out, V, **kw)
    return out


def compact_candidates(values, ids):
    _, B, C = values.shape
    v = values.transpose(0, 1).reshape(B, 4 * C)
    i = ids.transpose(0, 1).reshape(B, 4 * C)
    vv, j = torch.topk(v, 256, dim=1, sorted=False)
    ii = i.gather(1, j).long()
    return canonical(vv, ii)


def finish(values, ids, meta, piv, k, p, V, mode, precomputed=None, bad_out=None):
    _, B, C = values.shape
    assert num_compute_units(values.device.index) >= B
    x, buf = po.join(values, ids, meta, V)
    count = (
        precomputed[0]
        if precomputed is not None
        else meta[:, :, 0].sum(0).to(torch.int32)
    )
    flags = torch.empty(B, device=x.device, dtype=torch.bool)
    dbg = torch.empty((B, 18), device=x.device)
    if mode == "nosync":
        va, ia = po.candidates(values, ids, x)
    else:
        original = x.clone()
    kw = {}
    if tt._use_sm70_topk_topp_8_warps(x.device, B, V, True, p is not None):
        kw["num_warps"] = 8
    pct, cdf = tables(x.device)
    pt._topk_topp_kernel[(B,)](
        x,
        x.stride(0),
        buf,
        pct,
        cdf,
        k,
        p if p is not None else x,
        B,
        VOCAB_SIZE=V,
        MASK_VALUE=-float("inf"),
        BLOCK_SIZE=8192,
        BLOCK_SIZE_TRUNC=4096,
        TOPK_ENABLED=True,
        TOPP_ENABLED=p is not None,
        REFERENCE_ROWS=flags,
        DEBUG=dbg,
        PRE_PIVOT=piv,
        PRE_COUNT=count,
        **kw,
    )
    if mode == "nosync":
        x = pt._resolve_reference_rows_nosync(x, flags, va, ia, k, p)
    elif bool(flags.any()):
        x[flags] = reference(
            original[flags], k[flags], p[flags] if p is not None else None
        )
    safe, _ = po.certificate(
        dbg,
        piv,
        count,
        precomputed[1] if precomputed is not None else meta[:, :, 1].amax(0),
        bad_out,
    )
    return x, safe, dbg


def process(raw, temp, k, p, mode="sync"):
    B, V = raw.shape
    x = old.temperature(raw, temp)

    def fallback(reason):
        return tt.apply_top_k_top_p_triton(x.clone(), k, p), {
            "route": "original_fallback",
            "reason": reason,
        }

    if (
        V < 32768
        or num_compute_units(x.device.index) < B
        or raw.dtype != torch.float16
        or bool(((temp <= 0) | ~torch.isfinite(temp) | (k < 1) | (k > 256)).any())
    ):
        return fallback("metadata")
    if p is not None and bool((~torch.isfinite(p) | (p <= 0) | (p > 1)).any()):
        return fallback("probability")
    pv = pivot(x, k, V, p is not None)
    packets = [
        po.pack(x[:, r * V // 4 : (r + 1) * V // 4], pv, r * V // 4) for r in range(4)
    ]
    vals = torch.stack([a[0] for a in packets])
    ids = torch.stack([a[1] for a in packets])
    meta = torch.stack([a[2] for a in packets])
    counts = meta[:, :, 0]
    good = (
        (counts <= CAP).all(0)
        & (counts.sum(0) > k)
        & torch.isfinite(pv)
        & (meta[:, :, 2].sum(0) == 0)
        & (meta[:, :, 3].amax(0) != 0)
    )
    if not bool(good.all()):
        return fallback("capacity_or_numerics")
    out, safe, dbg = finish(vals, ids, meta, pv, k, p, V, mode)
    if not bool(safe.all()):
        return fallback("certificate")
    # The pre-mask joined buffer can be regenerated for independent L1 checks.
    sparse, buffer = po.join(vals, ids, meta, V)
    return out, {
        "route": "compressed_qrita",
        "outliers": counts.sum(0).cpu().tolist(),
        "certificate": True,
        "pivot": pv,
        "debug": dbg,
        "sparse": sparse,
        "buffer": buffer,
    }


# Testing helpers compatible with verify_extended
temperature = old.temperature
