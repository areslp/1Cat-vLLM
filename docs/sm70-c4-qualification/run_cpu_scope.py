# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only test bootstrap; no native-op substitution or model requests."""

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import vllm.platforms as platforms
from vllm.platforms.cpu import CpuPlatform

# Platform auto-detection on a CUDA host imports its CUDA extension even with
# CUDA_VISIBLE_DEVICES empty. Select the real CPU platform for CPU route tests.
platforms._current_platform = CpuPlatform()
import pytest  # noqa: E402 - platform bootstrap precedes pytest imports

raise SystemExit(pytest.main(sys.argv[1:]))
