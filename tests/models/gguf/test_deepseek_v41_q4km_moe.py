# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gguf
import pytest

from vllm.model_executor.layers.quantization.gguf_native import packed_tp_span
from vllm.model_executor.layers.quantization.gguf_v41_moe import (
    V41_FP16_EXPERT_TYPES,
)


@pytest.mark.parametrize(
    "kind", [gguf.GGMLQuantizationType.Q4_K, gguf.GGMLQuantizationType.Q6_K]
)
def test_q4km_down_tp_spans_cover_768_physical_columns(kind):
    assert int(kind) in V41_FP16_EXPERT_TYPES
    block, size = gguf.GGML_QUANT_SIZES[kind]
    for rank in range(4):
        first, last, left, physical = packed_tp_span(2304, int(kind), 4, rank)
        assert physical == 768
        assert (last - first) == 768 // block * size
        assert 0 <= left < block
        assert left + 576 <= physical
