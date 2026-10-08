# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reject apparent gains from unequal work, incomplete streams or bad pairing."""

import copy
import unittest

from analyze_retest import per_group


def observation():
    return {
        "group": "main_1024_1_0",
        "concurrency": 1,
        "successful": True,
        "actual_output_tokens": 128,
        "wall_s": 2.0,
        "group_output_tok_s": 64.0,
        "requests": [
            {
                "id": "case",
                "input_token_sha256": "input",
                "request_sha256": "wire",
                "input_expected": 1024,
                "requested_output": 128,
                "status": "completed",
                "usage": {"prompt_tokens": 1024, "completion_tokens": 128},
                "fixed_output_budget_completed": True,
                "finish_reason": "length",
                "text": "synthetic response",
            }
        ],
        "server_phases": {
            key: {"mean_s": value}
            for key, value in (
                ("request_prefill_time_seconds", 0.3),
                ("request_decode_time_seconds", 1.7),
                ("request_queue_time_seconds", 0.001),
            )
        },
        "server_metric_delta": {},
    }


class AnalysisAdmission(unittest.TestCase):
    def test_equal_fixed_work_qualifies(self):
        old = observation()
        fixed = copy.deepcopy(old)
        fixed.update(wall_s=1.0, group_output_tok_s=128.0)
        result = per_group(old, fixed)
        self.assertTrue(result["fixed_budget_pair"])
        self.assertEqual(result["throughput_ratio_fixed_over_old"], 2.0)

    def test_early_eos_cannot_become_speedup_even_with_forged_flag(self):
        old = observation()
        fixed = copy.deepcopy(old)
        r = fixed["requests"][0]
        r["usage"]["completion_tokens"] = 3
        r["finish_reason"] = "stop"
        fixed.update(actual_output_tokens=3, wall_s=0.1, group_output_tok_s=30)
        result = per_group(old, fixed)
        self.assertFalse(result["fixed_budget_pair"])
        self.assertIsNone(result["throughput_ratio_fixed_over_old"])

    def test_different_wire_or_tokenized_input_never_qualifies(self):
        for key in ("request_sha256", "input_token_sha256", "requested_output"):
            old = observation()
            fixed = copy.deepcopy(old)
            fixed["requests"][0][key] = "different"
            self.assertFalse(per_group(old, fixed)["fixed_budget_pair"])

    def test_failed_stream_never_qualifies(self):
        old = observation()
        fixed = copy.deepcopy(old)
        fixed["requests"][0]["status"] = "failed"
        self.assertFalse(per_group(old, fixed)["fixed_budget_pair"])

    def test_aggregate_output_and_throughput_must_recompute(self):
        for changes in ({"actual_output_tokens": 129}, {"group_output_tok_s": 1000}):
            old = observation()
            fixed = copy.deepcopy(old)
            fixed.update(changes)
            self.assertFalse(per_group(old, fixed)["fixed_budget_pair"])

    def test_actual_prompt_count_must_match(self):
        old = observation()
        fixed = copy.deepcopy(old)
        fixed["requests"][0]["usage"]["prompt_tokens"] = 1
        self.assertFalse(per_group(old, fixed)["fixed_budget_pair"])


if __name__ == "__main__":
    unittest.main()
