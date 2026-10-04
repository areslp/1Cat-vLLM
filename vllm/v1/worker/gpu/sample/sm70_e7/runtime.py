# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in E7 for exactly eight pure MTP4 verify requests.

All numerical kernels are the STEP50 V4 admission snapshot. This adapter owns
only eligibility, TP packet transport, original rejection invocation and optional
shadow observations. Single requests never enter its CUDA/collective path.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from vllm.distributed import get_tp_group
from vllm.model_executor.layers.sm70_fuse47 import _model_lm_head
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.spec_decode import rejection_sampler_utils as rs

from . import e7_fast as fast
from . import packet_ops as po
from .e7_compat import temperature

MODE = os.getenv("ONECAT_E7_MODE", "off")
if MODE not in ("off", "shadow", "on"):
    raise ValueError("ONECAT_E7_MODE must be off, shadow, or on")
ENABLED = MODE != "off"
_STATS_DIR = os.getenv("ONECAT_E7_STATS_DIR", "")
_COUNTS: Counter = Counter()
_THREAD_STARTED = False
_IDENTITY: dict = {}

# Schema-aware guards: SamplingParams normalizes absent bad_words to [].
# Custom per-request logits_processors is not a field in the pinned MRv2 schema;
# any dynamically attached processors are nevertheless rejected below.
_SCALAR_NEUTRAL = {
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repetition_penalty": 1.0,
    "min_tokens": 0,
}
_NULL_FIELDS = (
    "logprobs",
    "prompt_logprobs",
    "logit_bias",
    "allowed_token_ids",
    "structured_outputs",
    "thinking_token_budget",
    "extra_args",
    "logprob_token_ids",
)
_MISSING = object()


def parameter_reason(params) -> str | None:
    for name, neutral in _SCALAR_NEUTRAL.items():
        if getattr(params, name, _MISSING) != neutral:
            return "feature:" + name
    for name in _NULL_FIELDS:
        if getattr(params, name, _MISSING) is not None:
            return "feature:" + name
    for name in ("bad_words", "_bad_words_token_ids"):
        val = getattr(params, name, _MISSING)
        if val is _MISSING or val:
            return "feature:" + name
    if getattr(params, "logits_processors", None):
        return "feature:logits_processors"
    t = getattr(params, "temperature", None)
    k = getattr(params, "top_k", None)
    p = getattr(params, "top_p", None)
    if not isinstance(t, (int, float)) or not math.isfinite(t) or t <= 0:
        return "greedy_or_temperature"
    if not isinstance(k, int) or not 1 <= k <= 256:
        return "top_k"
    if not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 < p <= 1:
        return "top_p"
    return None


def _stats_loop() -> None:
    dest = Path(_STATS_DIR) / f"e7-{os.getpid()}.json"
    while True:
        try:
            obj = dict(_IDENTITY)
            obj.update(
                pid=os.getpid(), time=time.time(), mode=MODE, counters=dict(_COUNTS)
            )
            tmp = dest.with_suffix(".tmp")
            tmp.write_text(json.dumps(obj, sort_keys=True))
            tmp.replace(dest)
        except OSError:
            # Telemetry never affects model output.
            pass
        time.sleep(1)


def _start_stats() -> None:
    global _THREAD_STARTED
    if not _STATS_DIR or _THREAD_STARTED:
        return
    _THREAD_STARTED = True
    pkg = Path(__file__).parent
    _IDENTITY["package_hashes"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(pkg.glob("*.py"))
    }
    _IDENTITY["module_path"] = str(pkg)
    try:
        Path(_STATS_DIR).mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    threading.Thread(target=_stats_loop, name="e7-aggregate", daemon=True).start()


def record_request(sampler, req_idx: int, params, *, unsupported=None) -> None:
    if not ENABLED:
        return
    _start_stats()
    if not hasattr(sampler, "_e7_reasons"):
        sampler._e7_reasons = {}
    why = unsupported or parameter_reason(params)
    # Slots are overwritten on every add, including streaming re-adds.
    sampler._e7_reasons[req_idx] = why
    _COUNTS["requests_seen"] += 1
    _COUNTS["requests_supported" if why is None else "request_" + why] += 1


def batch_reason(batch) -> str | None:
    """CPU-only and independent of model tensor values."""
    if batch.num_reqs != 8:
        return "request_count"
    if batch.num_tokens != 40 or batch.num_draft_tokens != 32:
        return "not_pure_mtp4"
    if batch.logits_indices.numel() != 40:
        return "logits_shape"
    if np.any(batch.is_prefilling_np):
        return "prefill"
    if batch.num_draft_tokens_per_req is None or not np.all(
        batch.num_draft_tokens_per_req == 4
    ):
        return "draft_shape"
    if not np.all(np.diff(batch.cu_num_logits_np) == 5):
        return "logits_mapping"
    return None


def _allg(t: torch.Tensor, tp) -> torch.Tensor:
    t = t.contiguous()
    out = torch.empty((4 * t.shape[0], *t.shape[1:]), dtype=t.dtype, device=t.device)
    dist.all_gather_into_tensor(out, t, group=tp.device_group)
    return out.reshape(4, *t.shape)


def _uniform_bad(bad: bool, device, tp) -> bool:
    value = torch.tensor(int(bad), dtype=torch.int32, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.MAX, group=tp.device_group)
    return bool(value)


def _equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(
        a.contiguous().view(torch.int32), b.contiguous().view(torch.int32)
    )


def _valid_equal(a, b) -> bool:
    x, n = a.sampled_token_ids, a.num_sampled
    y, m = b.sampled_token_ids, b.num_sampled
    if not torch.equal(n, m):
        return False
    mask = torch.arange(x.shape[1], device=x.device)[None, :] < n[:, None]
    return torch.equal(x[mask], y[mask])


class _RejectionObserver:
    """Shadow-only observation, never installed during on-mode inference."""

    def __init__(self, kernel):
        self.kernel, self.state = kernel, None

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def call(*args, **kwargs):
            ret = launch(*args, **kwargs)
            self.state = tuple(args[i].clone() for i in (2, 3, 4))
            return ret

        return call


def _observe(fn):
    original = rs._rejection_kernel
    observer = _RejectionObserver(original)
    try:
        rs._rejection_kernel = observer
        output = fn()
    finally:
        rs._rejection_kernel = original
    if observer.state is None:
        raise RuntimeError("E7 shadow rejection state unavailable")
    return output, observer.state


def _sample_processed(runner, batch, processed) -> SamplerOutput:
    rej = runner.rejection_sampler
    sampler = rej.sampler
    sampled, n = rs.rejection_sample(
        processed,
        runner.speculator.draft_logits,
        batch.input_ids[batch.logits_indices],
        batch.cu_num_logits,
        batch.positions[batch.logits_indices],
        batch.idx_mapping,
        batch.expanded_idx_mapping,
        batch.expanded_local_pos,
        sampler.sampling_states.temperature.gpu,
        sampler.sampling_states.seeds.gpu,
        rej.num_speculative_steps,
        rej.synthetic_conditional_rates,
        use_fp64=sampler.use_fp64_gumbel,
    )
    return SamplerOutput(
        sampled_token_ids=sampled, num_sampled=n, logprobs_tensors=None, num_nans=None
    )


def _baseline_from_local(runner, batch, lp, local):
    """Reuse the identical LM-head output if a packet/certificate rejects E7.

    The caller has already guarded scale=1, soft_cap=None, unpadded original
    vocabulary and no grammar. This is the original compute_logits gather and
    original RejectionSampler, without evaluating the LM-head GEMM twice.
    """
    raw = lp._gather_logits(local)[..., :248320]
    return runner.rejection_sampler(raw, batch, runner.speculator.draft_logits)


def try_sample(runner, hidden: torch.Tensor, batch, grammar_output):
    """Return None for the unmodified model_runner baseline path."""
    _COUNTS["hook_steps"] += 1
    why = batch_reason(batch)
    if why:
        _COUNTS["fallback_" + why] += 1
        return None
    if torch.cuda.is_current_stream_capturing():
        _COUNTS["fallback_capture"] += 1
        return None
    tp = get_tp_group()
    if (
        tp.world_size != 4
        or runner.use_pp
        or not current_platform.is_device_capability(70)
    ):
        _COUNTS["fallback_static_layout"] += 1
        return None
    _IDENTITY.update(rank=tp.rank_in_group, ranks=list(tp.ranks))
    pair = _model_lm_head(runner.model)
    if pair is None:
        raise RuntimeError("E7 enabled on an unsupported model")
    lp, head = pair
    shard = head.shard_indices
    if not (
        lp.org_vocab_size == 248320
        and lp.soft_cap is None
        and lp.scale == 1.0
        and not lp.logits_as_input
        and shard.org_vocab_start_index == tp.rank_in_group * 62080
        and head.num_embeddings_per_partition == 62080
        and shard.num_org_vocab_padding == 0
    ):
        _COUNTS["fallback_vocab_layout"] += 1
        return None
    sampler = runner.sampler
    indices = batch.idx_mapping_np
    reasons = getattr(sampler, "_e7_reasons", {})
    feature_bad = (
        grammar_output is not None
        or batch.has_structured_output_reqs
        or getattr(runner, "lora_config", None) is not None
        or sampler.compute_nans
        or runner.rejection_sampler.num_speculative_steps != 4
        or runner.rejection_sampler.synthetic_conditional_rates is not None
        or runner.speculative_config.method != "mtp"
        or any(i not in reasons or reasons[i] is not None for i in indices)
        or np.any(sampler.logit_bias_state.use_logit_bias[indices])
        or np.any(sampler.penalties_state.use_penalty[indices])
        or np.any(sampler.bad_words_state.num_bad_words.np[indices])
        or sampler.logprob_token_ids_state.max_num_token_ids(indices) != 0
        or sampler.sampling_states.max_num_logprobs(indices) != -1
    )
    # Once the common scheduler envelope is met, rank-local request guards are
    # resolved collectively before any conditional sequence of collectives.
    if _uniform_bad(bool(feature_bad), hidden.device, tp):
        _COUNTS["fallback_features"] += 1
        return None
    _COUNTS["eligible_steps"] += 1
    local = head.quant_method.apply(head, hidden, bias=None)
    if _uniform_bad(
        local.dtype != torch.float16 or local.shape != (40, 62080), hidden.device, tp
    ):
        _COUNTS["fallback_dtype"] += 1
        return None
    mapping = batch.expanded_idx_mapping
    state = sampler.sampling_states
    temp = state.temperature.gpu[mapping]
    k = state.top_k.gpu[mapping]
    p = state.top_p.gpu[mapping] if np.any(state.top_p.np[indices] != 1.0) else None
    x = temperature(local, temp)
    pivot = (
        fast.pivot(x, k, 248320, p is not None)
        if tp.rank_in_group == 0
        else torch.empty(40, device=x.device)
    )
    dist.broadcast(pivot, src=tp.ranks[0], group=tp.device_group)
    values, ids, meta = po.pack(x, pivot, shard.org_vocab_start_index)
    po.mark(meta, temp, k, p)
    metas = _allg(meta, tp)
    bad, total, omitted = po.gate(metas, pivot, k)
    if bool(bad):
        _COUNTS["fallback_structure"] += 1
        if MODE == "shadow":
            # Shadow-only aggregate diagnostics; never log logits or prompts.
            _COUNTS["structure_capacity"] += int(bool((metas[:, :, 0] > po.CAP).any()))
            _COUNTS["structure_low_count"] += int(bool((total <= k).any()))
            _COUNTS["structure_nonfinite"] += int(bool((metas[:, :, 2] > 0).any()))
            _COUNTS["structure_zero_max"] += int(
                bool((metas[:, :, 3].amax(0) == 0).any())
            )
            _COUNTS["structure_bad_pivot"] += int(bool((~torch.isfinite(pivot)).any()))
        _COUNTS["baseline_local_reused"] += 1
        return _baseline_from_local(runner, batch, lp, local)
    gathered_v = _allg(values, tp)
    gathered_i = _allg(ids, tp)
    processed, safe, debug = fast.finish(
        gathered_v,
        gathered_i,
        metas,
        pivot,
        k,
        p,
        248320,
        "nosync",
        (total, omitted),
        bad,
    )
    dist.all_reduce(bad, op=dist.ReduceOp.MAX, group=tp.device_group)
    if bool(bad):
        _COUNTS["fallback_certificate"] += 1
        _COUNTS["baseline_local_reused"] += 1
        return _baseline_from_local(runner, batch, lp, local)
    _COUNTS["compressed_steps"] += 1
    if MODE == "shadow":
        # Gather the EXACT same LM-head result used by the compressed candidate.
        # Baseline output is the sole output returned to the model state update.
        raw = lp._gather_logits(local)[..., :248320]
        ref_processed = sampler.apply_sampling_params(
            raw,
            mapping,
            indices,
            batch.positions[batch.logits_indices],
            batch.input_ids[batch.logits_indices],
            batch.expanded_local_pos,
        )
        processed_ok = _equal(ref_processed, processed)
        candidate, cs = _observe(lambda: _sample_processed(runner, batch, processed))
        reference, bs = _observe(
            lambda: runner.rejection_sampler(raw, batch, runner.speculator.draft_logits)
        )
        state_ok = all(_equal(a, b) for a, b in zip(cs, bs))
        token_ok = _valid_equal(reference, candidate)
        _COUNTS["shadow_checked_steps"] += 1
        _COUNTS["shadow_checked_rows"] += 40
        _COUNTS["shadow_processed_fail"] += int(not processed_ok)
        _COUNTS["shadow_state_fail"] += int(not state_ok)
        _COUNTS["shadow_token_fail"] += int(not token_ok)
        if _uniform_bad(
            not (processed_ok and state_ok and token_ok), hidden.device, tp
        ):
            _COUNTS["shadow_global_fail"] += 1
        return reference
    return _sample_processed(runner, batch, processed)
