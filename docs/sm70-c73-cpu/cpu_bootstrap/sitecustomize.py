# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep model-inspection child interpreters on the real CPU platform."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import vllm.platforms as platforms
from vllm.platforms.cpu import CpuPlatform

platforms._current_platform = CpuPlatform()
