# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stable GPU compaction preserving Qrita BUFFER order; isolated prototype."""

import torch

from vllm.triton_utils import tl, triton

CAP = 2048
TILE = 2048


@triton.jit
def _counts(
    X,
    PIV,
    CNT,
    OMIT,
    BAD,
    MAXV,
    W: tl.constexpr,
    STR: tl.constexpr,
    LO: tl.constexpr,
    NB: tl.constexpr,
    T: tl.constexpr,
):
    row = tl.program_id(0)
    b = tl.program_id(1)
    off = b * T + tl.arange(0, T)
    m = off < W
    x = tl.load(X + row * STR + off, m, other=-float("inf"))
    pv = tl.load(PIV + row)
    take = (x > pv) & m
    tl.store(CNT + row * NB + b, tl.sum(take.to(tl.int32)))
    omitted = tl.where(m & ~take & ((LO + off) >= 8192), x, -float("inf"))
    tl.store(OMIT + row * NB + b, tl.max(omitted))
    tl.store(MAXV + row * NB + b, tl.max(x))
    tl.store(
        BAD + row * NB + b, tl.sum((m & ((x != x) | (x == float("inf")))).to(tl.int32))
    )


@triton.jit
def _offsets(CNT, OMIT, BAD, MAXV, OFF, META, NB: tl.constexpr, P: tl.constexpr):
    r = tl.program_id(0)
    i = tl.arange(0, P)
    n = tl.load(CNT + r * NB + i, i < NB, other=0)
    s = tl.cumsum(n)
    tl.store(OFF + r * NB + i, s - n, i < NB)
    tl.store(META + r * 4, tl.sum(n).to(tl.float32))
    tl.store(
        META + r * 4 + 1,
        tl.max(tl.load(OMIT + r * NB + i, i < NB, other=-float("inf"))),
    )
    tl.store(
        META + r * 4 + 2,
        tl.sum(tl.load(BAD + r * NB + i, i < NB, other=0)).to(tl.float32),
    )
    tl.store(
        META + r * 4 + 3,
        tl.max(tl.load(MAXV + r * NB + i, i < NB, other=-float("inf"))),
    )


@triton.jit
def _init(V, IDS, N: tl.constexpr, T: tl.constexpr):
    ix = tl.program_id(0) * T + tl.arange(0, T)
    tl.store(V + ix, -float("inf"), ix < N)
    tl.store(IDS + ix, 1073741824, ix < N)


@triton.jit
def _pack(
    X,
    PIV,
    OFF,
    VAL,
    IDS,
    W: tl.constexpr,
    STR: tl.constexpr,
    LO: tl.constexpr,
    NB: tl.constexpr,
    C: tl.constexpr,
    T: tl.constexpr,
):
    row = tl.program_id(0)
    b = tl.program_id(1)
    off = b * T + tl.arange(0, T)
    m = off < W
    x = tl.load(X + row * STR + off, m, other=-float("inf"))
    pv = tl.load(PIV + row)
    take = (x > pv) & m
    pos = tl.cumsum(take.to(tl.int32)) - 1 + tl.load(OFF + row * NB + b)
    live = take & (pos < C)
    tl.store(VAL + row * C + pos, x, live)
    tl.store(IDS + row * C + pos, LO + off, live)


@triton.jit
def _fill(X, N: tl.constexpr, T: tl.constexpr):
    ix = tl.program_id(0) * T + tl.arange(0, T)
    tl.store(X + ix, -float("inf"), ix < N)


@triton.jit
def _join(
    VALUES,
    IDS,
    META,
    X,
    BUF,
    B: tl.constexpr,
    V: tl.constexpr,
    C: tl.constexpr,
    T: tl.constexpr,
):
    row = tl.program_id(0)
    ix = tl.program_id(1) * T + tl.arange(0, T)
    rank = ix // C
    loc = ix % C
    live = ix < 4 * C
    count = tl.load(META + rank * B * 4 + row * 4, live, other=0).to(tl.int32)
    live = live & (loc < count)
    vals = tl.load(VALUES + rank * B * C + row * C + loc, live, other=-float("inf"))
    ids = tl.load(IDS + rank * B * C + row * C + loc, live, other=V)
    offset = tl.zeros((T,), tl.int32)
    for j in tl.static_range(3):
        offset += tl.where(
            rank > j, tl.load(META + j * B * 4 + row * 4).to(tl.int32), 0
        )
    tl.store(BUF + row * V + offset + loc, vals, live)
    tl.store(X + row * V + ids, vals, live & (ids < V))


def pack(x, piv, lo):
    B, W = x.shape
    nb = triton.cdiv(W, TILE)
    cnt = torch.empty((B, nb), dtype=torch.int32, device=x.device)
    om = torch.empty((B, nb), device=x.device)
    bad = torch.empty_like(cnt)
    mv = torch.empty_like(om)
    off = torch.empty_like(cnt)
    meta = torch.empty((B, 4), device=x.device)
    _counts[(B, nb)](x, piv, cnt, om, bad, mv, W, x.stride(0), lo, nb, TILE)
    _offsets[(B,)](cnt, om, bad, mv, off, meta, nb, triton.next_power_of_2(nb))
    vals = torch.empty((B, CAP), device=x.device)
    ids = torch.empty((B, CAP), dtype=torch.int32, device=x.device)
    _init[(triton.cdiv(B * CAP, 256),)](vals, ids, B * CAP, 256)
    _pack[(B, nb)](x, piv, off, vals, ids, W, x.stride(0), lo, nb, CAP, TILE)
    return vals, ids, meta


def join(values, ids, meta, V):
    _, B, C = values.shape
    x = torch.empty((B, V), device=values.device)
    buf = torch.empty_like(x)
    _fill[(triton.cdiv(B * V, 1024),)](x, B * V, 1024)
    _join[(B, triton.cdiv(4 * C, 256))](values, ids, meta, x, buf, B, V, C, 256)
    return x, buf


@triton.jit
def _candidate_keys(
    VALUES,
    IDS,
    X,
    KEYS,
    B: tl.constexpr,
    V: tl.constexpr,
    C: tl.constexpr,
    T: tl.constexpr,
):
    row = tl.program_id(0)
    ix = tl.program_id(1) * T + tl.arange(0, T)
    N: tl.constexpr = 4 * C + 256
    m = ix < N
    packet = ix < 4 * C
    rank = ix // C
    loc = ix % C
    vv = tl.load(VALUES + rank * B * C + row * C + loc, m & packet, other=-float("inf"))
    ids = tl.load(IDS + rank * B * C + row * C + loc, m & packet, other=V)
    prefix = ix - 4 * C
    pv = tl.load(X + row * V + prefix, m & ~packet, other=-float("inf"))
    val = tl.where(packet, vv, pv)
    id = tl.where(packet, ids, prefix)
    # Remove duplicates with the 256 explicit valid vocabulary IDs. The latter
    # guarantee >=256 valid slots even when there are few outliers. Invalid packet
    # padding receives INT64_MIN and can never be selected over a valid -inf slot.
    valid = m & tl.where(packet, (ids >= 256) & (ids < V), True)
    bits = val.to(tl.uint32, bitcast=True)
    bits = tl.where(bits == 2147483648, 0, bits)
    order = tl.where((bits & 2147483648) != 0, ~bits, bits | 2147483648).to(tl.uint32)
    key = (order.to(tl.int64) - 2147483648) * 4294967296 + id.to(tl.int64)
    key = tl.where(valid, key, tl.full((T,), -9223372036854775808, tl.int64))
    tl.store(KEYS + row * N + ix, key, m)


@triton.jit
def _candidate_fetch(
    SELECT,
    VALUES,
    IDS,
    X,
    OUTV,
    OUTI,
    B: tl.constexpr,
    V: tl.constexpr,
    C: tl.constexpr,
):
    row = tl.program_id(0)
    j = tl.arange(0, 256)
    ix = tl.load(SELECT + row * 256 + 255 - j)
    packet = ix < 4 * C
    rank = ix // C
    loc = ix % C
    vv = tl.load(VALUES + rank * B * C + row * C + loc, packet, other=-float("inf"))
    id = tl.load(IDS + rank * B * C + row * C + loc, packet, other=V)
    pref = ix - 4 * C
    pv = tl.load(X + row * V + pref, ~packet, other=-float("inf"))
    tl.store(OUTV + row * 256 + j, tl.where(packet, vv, pv))
    tl.store(OUTI + row * 256 + j, tl.where(packet, id, pref).to(tl.int64))


def candidates(values, ids, x):
    _, B, C = values.shape
    V = x.shape[1]
    N = 4 * C + 256
    keys = torch.empty((B, N), device=x.device, dtype=torch.int64)
    _candidate_keys[(B, triton.cdiv(N, 256))](values, ids, x, keys, B, V, C, 256)
    sel = torch.topk(keys, 256, dim=1, largest=True, sorted=True).indices
    v = torch.empty((B, 256), device=x.device)
    i = torch.empty((B, 256), device=x.device, dtype=torch.int64)
    _candidate_fetch[(B,)](sel, values, ids, x, v, i, B, V, C)
    return v, i


@triton.jit
def _mark(
    META,
    TEMP,
    K,
    P,
    B: tl.constexpr,
    T: tl.constexpr,
    HAS_P: tl.constexpr,
    EXTERNAL_BAD: tl.constexpr,
):
    i = tl.arange(0, T)
    m = i < B
    t = tl.load(TEMP + i, m, other=1.0)
    k = tl.load(K + i, m, other=1)
    bad = (t <= 0) | (t != t) | (t == float("inf")) | (k < 1) | (k > 256)
    if HAS_P:
        p = tl.load(P + i, m, other=1.0)
        bad = bad | (p <= 0) | (p > 1) | (p != p)
    if EXTERNAL_BAD:
        bad = tl.full((T,), True, tl.int1)
    prev = tl.load(META + i * 4 + 2, m, other=0.0)
    tl.store(META + i * 4 + 2, prev + bad.to(tl.float32), m)


def mark(meta, temp, k, p, external=False):
    B = temp.numel()
    _mark[(1,)](
        meta,
        temp,
        k,
        p if p is not None else temp,
        B,
        triton.next_power_of_2(B),
        p is not None,
        external,
    )


@triton.jit
def _gate(
    META, PIV, K, TOTAL, OMIT, BAD, B: tl.constexpr, T: tl.constexpr, C: tl.constexpr
):
    i = tl.arange(0, T)
    m = i < B
    n = tl.zeros((T,), tl.int32)
    om = tl.full((T,), -float("inf"), tl.float32)
    mx = tl.full((T,), -float("inf"), tl.float32)
    bad = tl.full((T,), False, tl.int1)
    for r in tl.static_range(4):
        c = tl.load(META + r * B * 4 + i * 4, m, other=0).to(tl.int32)
        b = tl.load(META + r * B * 4 + i * 4 + 2, m, other=0)
        n += c
        bad = bad | (c > C) | (b > 0)
        om = tl.maximum(
            om, tl.load(META + r * B * 4 + i * 4 + 1, m, other=-float("inf"))
        )
        mx = tl.maximum(
            mx, tl.load(META + r * B * 4 + i * 4 + 3, m, other=-float("inf"))
        )
    pv = tl.load(PIV + i, m, other=0.0)
    k = tl.load(K + i, m, other=0)
    bad = bad | (n <= k) | ~((pv > -float("inf")) & (pv < float("inf"))) | (mx == 0)
    tl.store(TOTAL + i, n, m)
    tl.store(OMIT + i, om, m)
    tl.store(BAD, tl.sum((bad & m).to(tl.int32)))


def gate(meta, piv, k):
    B = k.numel()
    total = torch.empty(B, device=k.device, dtype=torch.int32)
    om = torch.empty(B, device=k.device)
    bad = torch.empty((), device=k.device, dtype=torch.int32)
    _gate[(1,)](meta, piv, k, total, om, bad, B, triton.next_power_of_2(B), CAP)
    return bad, total, om


@triton.jit
def _certificate(DBG, PIV, TOTAL, OMIT, SAFE, BAD, B: tl.constexpr, T: tl.constexpr):
    i = tl.arange(0, T)
    m = i < B
    fp = tl.load(DBG + i * 18, m, other=0.0)
    mx = tl.load(DBG + i * 18 + 1, m, other=1.0)
    pv = tl.load(DBG + i * 18 + 8, m, other=0.0)
    n = tl.load(DBG + i * 18 + 9, m, other=0.0)
    s = tl.load(DBG + i * 18 + 17, m, other=0.0)
    expected = tl.load(PIV + i, m, other=0.0)
    total = tl.load(TOTAL + i, m, other=0)
    om = tl.load(OMIT + i, m, other=0.0)
    good = (
        (s == 0)
        & (n == total)
        & (pv == expected)
        & (om <= pv)
        & (fp > pv)
        & (fp < float("inf"))
        & (fp > -float("inf"))
        & (mx != 0)
    )
    tl.store(SAFE + i, good, m)
    tl.store(BAD, tl.sum((~good & m).to(tl.int32)))


def certificate(dbg, piv, total, om, bad_out=None):
    B = total.numel()
    safe = torch.empty(B, device=total.device, dtype=torch.bool)
    bad = (
        bad_out
        if bad_out is not None
        else torch.empty((), device=total.device, dtype=torch.int32)
    )
    _certificate[(1,)](dbg, piv, total, om, safe, bad, B, triton.next_power_of_2(B))
    return safe, bad
