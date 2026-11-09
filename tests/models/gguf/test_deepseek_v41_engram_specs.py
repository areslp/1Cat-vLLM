# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gguf
import numpy as np
import pytest

from vllm.model_executor.layers.deepseek_v41_engram_io import (
    engram_table_specs,
)


def _write_table(path, name, kind, rows):
    block, size = gguf.GGML_QUANT_SIZES[kind]
    width_blocks = 256 // block
    packed = np.random.default_rng(rows).integers(
        0, 256, (rows, width_blocks, size), dtype=np.uint8
    )
    writer = gguf.GGUFWriter(path, "deepseek41-engram")
    writer.add_tensor(name, packed.reshape(rows, -1), raw_dtype=kind)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.mark.parametrize(
    "kind", [gguf.GGMLQuantizationType.IQ4_NL, gguf.GGMLQuantizationType.Q4_K]
)
def test_engram_specs_use_q4_k_or_iq4_nl_rows_across_shards(tmp_path, kind):
    _write_table(tmp_path / "a.gguf", "blk.1.engram_embd.weight", kind, 3)
    _write_table(tmp_path / "b.gguf", "blk.14.engram_embd.weight", kind, 5)
    specs = engram_table_specs(tmp_path)
    assert [spec.rows for spec in specs] == [3, 5]
    assert {spec.source_type for spec in specs} == {int(kind)}
    assert [spec.path for spec in specs] == [
        str((tmp_path / "a.gguf").resolve()),
        str((tmp_path / "b.gguf").resolve()),
    ]


def test_engram_specs_reject_non_144_byte_rows(tmp_path):
    path = tmp_path / "bad.gguf"
    writer = gguf.GGUFWriter(path, "deepseek41-engram")
    # Q4_0 rows are 8 * 18 = 144 bytes wide, so the writer accepts them but
    # the V4.1 Engram path must reject the unsupported source type.
    writer.add_tensor(
        "blk.1.engram_embd.weight",
        np.zeros((2, 8 * 18), dtype=np.uint8),
        raw_dtype=gguf.GGMLQuantizationType.Q4_0,
    )
    writer.add_tensor(
        "blk.14.engram_embd.weight",
        np.zeros((2, 8 * 18), dtype=np.uint8),
        raw_dtype=gguf.GGMLQuantizationType.Q4_0,
    )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    with pytest.raises(ValueError, match="IQ4_NL or Q4_K"):
        engram_table_specs(path)
