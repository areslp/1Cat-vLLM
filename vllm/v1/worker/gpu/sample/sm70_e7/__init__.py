# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, small-batch-safe SM70 E7 sampling integration."""

from .runtime import ENABLED, record_request, try_sample

__all__ = ["ENABLED", "record_request", "try_sample"]
