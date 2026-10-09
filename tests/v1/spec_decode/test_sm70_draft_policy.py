# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drafter policy is per engine and independent of later env changes."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.config import KernelConfig
from vllm.config.sm70_draft import Sm70DraftConfig, parse_legacy_units
from vllm.model_executor.layers import sm70_draft47
from vllm.model_executor.layers.sm70_topk_gather import capture_top1_transport
from vllm.v1.worker.gpu.spec_decode.eagle.sm70_draft_ops import (
    capture_pad_ops,
    extend_decode_graphs,
)


@pytest.mark.parametrize("raw", ["", "0", "off", "false", "no", "none", "unknown"])
def test_disabled_legacy_spellings(raw):
    assert parse_legacy_units(raw) == ()


@pytest.mark.parametrize("raw", ["1", "ALL", "on", "true", "yes"])
def test_enabled_legacy_spellings(raw):
    assert parse_legacy_units(raw) == ("d2a", "d2b", "d1a")


def test_legacy_list_preserves_known_units_and_deduplicates():
    assert parse_legacy_units(" D1A , d2a, unknown, d1a ") == ("d2a", "d1a")


def test_each_engine_captures_legacy_policy_once(monkeypatch):
    monkeypatch.setenv("ONECAT_DRAFT47", "all")
    first = KernelConfig()
    explicit_off = KernelConfig(sm70_draft=Sm70DraftConfig(units=()))
    monkeypatch.setenv("ONECAT_DRAFT47", "0")
    second = KernelConfig()
    assert first.sm70_draft.units == ("d2a", "d2b", "d1a")
    assert second.sm70_draft.units == explicit_off.sm70_draft.units == ()
    assert second.compute_hash() == explicit_off.compute_hash()
    assert first.compute_hash() != second.compute_hash()


def test_top1_transport_retains_its_engine_policy(monkeypatch):
    import vllm.config as config_module

    first = SimpleNamespace(
        kernel_config=KernelConfig(sm70_draft=Sm70DraftConfig(units=("d1a",)))
    )
    second = SimpleNamespace(
        kernel_config=KernelConfig(sm70_draft=Sm70DraftConfig(units=()))
    )
    monkeypatch.setattr(config_module, "get_current_vllm_config_or_none", lambda: first)
    enabled = capture_top1_transport()
    monkeypatch.setattr(
        config_module, "get_current_vllm_config_or_none", lambda: second
    )
    disabled = capture_top1_transport()
    monkeypatch.setenv("ONECAT_DRAFT47", "all")
    monkeypatch.setattr(
        sm70_draft47, "enabled", Mock(side_effect=AssertionError("forward env read"))
    )
    result = object()
    kernel = Mock(return_value=result)
    monkeypatch.setattr(sm70_draft47, "top_tokens", kernel)
    logits = object()
    assert enabled(logits, 256, 4) is result
    assert disabled(logits, 256, 4) is None
    assert enabled(logits, 256, 1) is None
    kernel.assert_called_once_with(logits, 256, 4)


def test_graph_and_padding_adapters_use_explicit_engine_policy(monkeypatch):
    monkeypatch.setenv("ONECAT_DRAFT47", "0")
    config = SimpleNamespace(
        kernel_config=KernelConfig(sm70_draft=Sm70DraftConfig(units=("d2a", "d2b")))
    )
    manager = SimpleNamespace(
        cudagraph_mode=True,
        decode_query_len=1,
        max_num_reqs=8,
        _capture_sizes=[1, 2, 4],
        _init_candidates=Mock(),
    )
    extend_decode_graphs(manager, config)
    assert manager._capture_sizes == [1, 2, 4, 6, 7, 8]
    manager._init_candidates.assert_called_once()
    assert capture_pad_ops(config) is not None
    disabled = SimpleNamespace(
        kernel_config=KernelConfig(sm70_draft=Sm70DraftConfig(units=()))
    )
    monkeypatch.setenv("ONECAT_DRAFT47", "all")
    assert capture_pad_ops(disabled) is None
    untouched = SimpleNamespace(
        cudagraph_mode=True,
        decode_query_len=1,
        max_num_reqs=8,
        _capture_sizes=[1, 2, 4],
        _init_candidates=Mock(),
    )
    extend_decode_graphs(untouched, disabled)
    assert untouched._capture_sizes == [1, 2, 4]
    untouched._init_candidates.assert_not_called()
