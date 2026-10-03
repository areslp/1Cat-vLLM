"""Opt-in torch-profiler stage annotations for the reviewed flash-next vLLM.

This is loaded as a vLLM general plugin from the maintenance-only FAV100_SHIM
path.  It never imports or shadows ``sitecustomize`` and does nothing unless
``HOST_INSIGHT_STAGE_MARKERS=1`` is set by the profiling launcher.

The outer annotations are scheduler classifications, not per-request claims:
``prefill`` is a context-only scheduler iteration, ``decode`` is a
generation-only iteration (including target speculative verification), and
``mixed`` contains both.  ``mtp`` is nested only around the actual MTP draft
proposal method.  Labels deliberately contain no request or prompt data.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import inspect
import logging
import os
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any


_OPT_IN_ENV = "HOST_INSIGHT_STAGE_MARKERS"
_PLUGIN_NAME = "host_insight_stage_markers"
_PATCHED_ATTR = "__host_insight_stage_markers_v2__"
_LOGGER = logging.getLogger(__name__)
_REVIEWED_SOURCE_HASHES = {
    "execute_model": "004a7e025e1eb1943f7c6bc89dcfe86d2b3b079bb5a1a5110f50d8833ee1ea0c",
    "eagle_propose": "7d4dd5634dc9c03b0a148566d2dcb47c2772f687d2e9df05ccd5a440c294432f",
    "worker_init_device": "72378ad5ffe3dafaddd5e8886ebeb42404d929a2a2efb61698251c55ad41c452",
    "compute_iteration_details": (
        "53cf7f27f46f56137f9f51f73e82f4158f342982b62463cadd260864854e3676"
    ),
}


def _enabled() -> bool:
    return os.environ.get(_OPT_IN_ENV) == "1"


def _source_hash(function: Callable[..., Any]) -> str:
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError) as error:
        raise RuntimeError(
            f"{_PLUGIN_NAME}: cannot inspect reviewed vLLM method "
            f"{function!r}"
        ) from error
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _require_reviewed_source(name: str, function: Callable[..., Any]) -> None:
    actual = _source_hash(function)
    expected = _REVIEWED_SOURCE_HASHES[name]
    if actual != expected:
        raise RuntimeError(
            f"{_PLUGIN_NAME}: unsupported vLLM {name} source "
            f"(expected {expected}, got {actual}); refusing stage markers"
        )


def _load_reviewed_targets() -> tuple[type[Any], type[Any], type[Any], Callable[..., Any]]:
    try:
        from vllm.v1.utils import compute_iteration_details
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
        from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator
        from vllm.v1.worker.gpu_worker import Worker
    except ImportError as error:
        raise RuntimeError(
            f"{_PLUGIN_NAME}: reviewed vLLM V2 stage-marker targets are unavailable"
        ) from error

    execute_model = getattr(GPUModelRunner, "execute_model", None)
    eagle_propose = getattr(EagleSpeculator, "propose", None)
    worker_init_device = getattr(Worker, "init_device", None)
    if not callable(execute_model) or not callable(eagle_propose) or not callable(worker_init_device):
        raise RuntimeError(
            f"{_PLUGIN_NAME}: reviewed V2 runner, MTP proposer, or worker hook is unavailable"
        )
    if not callable(compute_iteration_details):
        raise RuntimeError(
            f"{_PLUGIN_NAME}: reviewed scheduler classification helper is unavailable"
        )
    _require_reviewed_source("execute_model", execute_model)
    _require_reviewed_source("eagle_propose", eagle_propose)
    _require_reviewed_source("worker_init_device", worker_init_device)
    _require_reviewed_source("compute_iteration_details", compute_iteration_details)
    _require_scheduler_contract(compute_iteration_details)
    return GPUModelRunner, EagleSpeculator, Worker, compute_iteration_details


def _phase(compute_details: Callable[[Any], Any], scheduler_output: Any) -> str:
    """Classify only the scheduler's public context/generation accounting."""
    try:
        details = compute_details(scheduler_output)
    except Exception:
        return "host_insight::phase::unknown"
    context_requests = getattr(details, "num_ctx_requests", None)
    generation_requests = getattr(details, "num_generation_requests", None)
    if not isinstance(context_requests, int) or not isinstance(generation_requests, int):
        return "host_insight::phase::unknown"
    if context_requests > 0 and generation_requests > 0:
        return "host_insight::phase::mixed"
    if context_requests > 0:
        return "host_insight::phase::prefill"
    if generation_requests > 0:
        return "host_insight::phase::decode"
    return "host_insight::phase::unknown"


def _require_scheduler_contract(compute_details: Callable[[Any], Any]) -> None:
    """Exercise the installed helper and its actual return schema before startup."""
    for phases, expected in [((True,), "prefill"), ((False,), "decode"), ((True, False), "mixed")]:
        requests = {str(index): context for index, context in enumerate(phases)}
        scheduler = SimpleNamespace(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(is_context_phase=requests.__getitem__),
            num_scheduled_tokens={request: 1 for request in requests},
        )
        if _phase(compute_details, scheduler) != "host_insight::phase::" + expected:
            raise RuntimeError(
                f"{_PLUGIN_NAME}: incompatible scheduler classification contract ({expected})"
            )


def _torch_profiler_active() -> bool:
    """Use torch's live profiler state; do not infer recording from env flags."""
    try:
        import torch

        state = torch.autograd.profiler._is_profiler_enabled
        return bool(state() if callable(state) else state)
    except (AttributeError, ImportError, RuntimeError):
        return False


@contextlib.contextmanager
def _annotation(name: str) -> Iterator[None]:
    """Emit a UserAnnotation only while torch's profiler is actively recording."""
    if not _torch_profiler_active():
        yield
        return
    try:
        from torch.autograd.profiler import record_function

        scope = record_function(name)
    except Exception:
        # Instrumentation must not alter inference behavior if torch withdraws
        # profiler support between the active-state check and scope creation.
        yield
        return
    try:
        scope.__enter__()
    except Exception:
        yield
        return
    try:
        yield
    except BaseException as error:
        try:
            scope.__exit__(type(error), error, error.__traceback__)
        except BaseException:
            pass
        raise
    else:
        try:
            scope.__exit__(None, None, None)
        except BaseException:
            pass


def _install(
    GPUModelRunner: type[Any],
    EagleSpeculator: type[Any],
    Worker: type[Any],
    compute_details: Callable[[Any], Any],
) -> None:
    if all(getattr(target, _PATCHED_ATTR, False) for target in (GPUModelRunner, EagleSpeculator, Worker)):
        return

    execute_model = GPUModelRunner.execute_model
    eagle_propose = EagleSpeculator.propose
    worker_init_device = Worker.init_device

    @functools.wraps(execute_model)
    def execute_with_phase(self: Any, scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
        if not _torch_profiler_active():
            return execute_model(self, scheduler_output, *args, **kwargs)
        with _annotation(_phase(compute_details, scheduler_output)):
            return execute_model(self, scheduler_output, *args, **kwargs)

    @functools.wraps(eagle_propose)
    def propose_with_mtp(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not _torch_profiler_active():
            return eagle_propose(self, *args, **kwargs)
        if getattr(self, "method", None) != "mtp":
            return eagle_propose(self, *args, **kwargs)
        with _annotation("host_insight::phase::mtp"):
            return eagle_propose(self, *args, **kwargs)

    @functools.wraps(worker_init_device)
    def init_device_with_identity(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = worker_init_device(self, *args, **kwargs)
        runner = getattr(self, "model_runner", None)
        if type(runner) is not GPUModelRunner:
            raise RuntimeError(
                f"{_PLUGIN_NAME}: unsupported active model runner "
                f"{type(runner).__module__}.{type(runner).__qualname__}"
            )
        speculator = getattr(runner, "speculator", None)
        if speculator is not None and type(speculator) is not EagleSpeculator:
            raise RuntimeError(
                f"{_PLUGIN_NAME}: unsupported active speculative proposer "
                f"{type(speculator).__module__}.{type(speculator).__qualname__}"
            )
        _LOGGER.info(
            "%s installed runner=%s speculator=%s rank=%s",
            _PLUGIN_NAME,
            f"{type(runner).__module__}.{type(runner).__qualname__}",
            (
                f"{type(speculator).__module__}.{type(speculator).__qualname__}"
                if speculator is not None
                else "none"
            ),
            getattr(self, "rank", "unknown"),
        )
        return result

    GPUModelRunner.execute_model = execute_with_phase
    EagleSpeculator.propose = propose_with_mtp
    Worker.init_device = init_device_with_identity
    for target in (GPUModelRunner, EagleSpeculator, Worker):
        setattr(target, _PATCHED_ATTR, True)


def check_compatibility() -> None:
    """Fail before service restart if the actual import target differs."""
    _load_reviewed_targets()


def install() -> None:
    """vLLM ``general_plugins`` entry point; no-op outside profile activation."""
    if not _enabled():
        return
    runner, eagle, worker, compute_details = _load_reviewed_targets()
    _install(runner, eagle, worker, compute_details)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify the reviewed vLLM import contract")
    arguments = parser.parse_args(argv)
    if arguments.check:
        check_compatibility()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
