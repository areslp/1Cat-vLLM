# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP32 clamped SwiGLU with the original FP16 output boundary."""

import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op


def _portable_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Original FP32 clamp/activation, without the SM70-only CUDA kernel."""
    if (
        gate.dtype != torch.float16
        or up.dtype != torch.float16
        or gate.shape != up.shape
        or gate.device != up.device
    ):
        raise ValueError("V4.1 shared SwiGLU requires matching FP16 matrices")
    gate_f = gate.float().clamp(max=10.0)
    up_f = up.float().clamp(min=-10.0, max=10.0)
    activated = gate_f / (1.0 + torch.exp(-gate_f))
    return (activated * up_f).to(torch.float16)


def _swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    if not (
        gate.is_cuda
        and current_platform.is_device_capability(70)
        and hasattr(torch.ops._C, "deepseek_v41_swiglu_sm70_out")
    ):
        return _portable_swiglu(gate, up)
    output = torch.empty_like(gate)
    torch.ops._C.deepseek_v41_swiglu_sm70_out(output, gate, up)
    return output


def _fake(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(gate)


direct_register_custom_op(
    op_name="deepseek_v41_swiglu", op_func=_swiglu, fake_impl=_fake
)
swiglu = torch.ops.vllm.deepseek_v41_swiglu
