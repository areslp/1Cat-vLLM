# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU validation of the non-SM70 V4.1 shared-SwiGLU fallback."""

import torch

from vllm.models.deepseek_v41.activations import _portable_swiglu, _swiglu


def _kernel_reference(gate, up):
    # Mirrors v41_swiglu in gguf_dp4a_sm70.cu: gate clamps only above +10,
    # up clamps to [-10, 10], and every arithmetic step stays in FP32 before
    # the final round-to-nearest FP16 store.
    gate_f = gate.float()
    gate_f = torch.where(gate_f > 10.0, torch.full_like(gate_f, 10.0), gate_f)
    up_f = up.float().clamp(min=-10.0, max=10.0)
    activated = gate_f / (1.0 + torch.exp(-gate_f))
    return (activated * up_f).half()


def test_portable_swiglu_matches_original_kernel_boundaries_bit_exact():
    boundary = torch.tensor(
        [-65504, -10.01, -10, -9.99, -0.0, 0.0, 0.001, 9.99, 10, 10.01, 65504],
        dtype=torch.float16,
    )
    gate = boundary.repeat(4)[:, None].expand(44, 576).contiguous()
    up = boundary.flip(0).repeat(4)[:, None].expand(44, 576).contiguous()
    actual = _portable_swiglu(gate, up)
    expected = _kernel_reference(gate, up)
    torch.testing.assert_close(
        actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
    )


def test_swiglu_uses_portable_path_without_cuda():
    gate = torch.tensor([[0.0, 10.01, -10.01, 3.5]], dtype=torch.float16)
    up = torch.tensor([[0.0, 10.01, -10.01, -3.5]], dtype=torch.float16)
    actual = _swiglu(gate, up)
    expected = _kernel_reference(gate, up)
    torch.testing.assert_close(
        actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
    )
