# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only admission tests: no model, tokenizer, network or hardware."""

import copy
import gzip
import json
import tempfile
import unittest
from pathlib import Path

import review_evidence as review


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.case = {
            "id": "unit_0",
            "messages": [{"role": "user", "content": "synthetic"}],
            "max_output_tokens": 4,
            "input_token_sha256": "synthetic-token-hash",
            "expected_input_tokens": 2,
        }
        with gzip.open(self.root / "synthetic-fixtures.jsonl.gz", "wt") as f:
            f.write(json.dumps(self.case) + "\n")
        (self.root / "MATRIX.json").write_text(json.dumps({"groups": ["main_2_1_0"]}))
        phases = {
            k: {"mean_s": 0.1, "sum_s": 0.1, "count": 1}
            for k in (
                "request_prefill_time_seconds",
                "request_decode_time_seconds",
                "request_queue_time_seconds",
            )
        }
        request = {
            "id": "unit_0",
            "request_sha256": review.digest(review.expected_body(self.case)),
            "input_token_sha256": self.case["input_token_sha256"],
            "input_expected": 2,
            "requested_output": 4,
            "status": "completed",
            "usage": {"prompt_tokens": 2, "completion_tokens": 4},
            "text": "abcd",
        }
        self.group = {
            "group": "main_2_1_0",
            "concurrency": 1,
            "requests": [request],
            "successful": True,
            "fixed_output_budget_completed": True,
            "wall_s": 0.5,
            "actual_output_tokens": 4,
            "group_output_tok_s": 8,
            "server_phases": phases,
            "server_metric_delta": {},
        }
        for arm in ("main", "patched"):
            self.save(arm, copy.deepcopy(self.group))

    def tearDown(self):
        self.tmp.cleanup()

    def save(self, arm, data):
        p = self.root / "observations" / arm / "main_2_1_0"
        p.mkdir(parents=True, exist_ok=True)
        (p / "GROUP_RESULT.json").write_text(json.dumps(data))

    def test_matching_work_has_ratio(self):
        r = review.review(self.root)
        self.assertEqual(r["wire_errors"], [])
        self.assertEqual(r["pairs"][0]["wall_speedup"], 1)

    def test_missing_arm_never_has_ratio(self):
        (self.root / "observations/patched/main_2_1_0/GROUP_RESULT.json").unlink()
        r = review.review(self.root)
        self.assertFalse(r["all_planned_groups_paired"])
        self.assertNotIn("wall_speedup", r["pairs"][0])

    def test_corrupted_wire_excluded(self):
        d = copy.deepcopy(self.group)
        d["requests"][0]["request_sha256"] = "wrong"
        self.save("patched", d)
        r = review.review(self.root)
        self.assertTrue(r["wire_errors"])
        self.assertNotIn("wall_speedup", r["pairs"][0])

    def test_early_eos_excluded_without_relabeling_failure(self):
        d = copy.deepcopy(self.group)
        d["requests"][0]["usage"]["completion_tokens"] = 2
        d.update(
            actual_output_tokens=2,
            group_output_tok_s=4,
            fixed_output_budget_completed=False,
        )
        self.save("patched", d)
        r = review.review(self.root)
        self.assertEqual(r["wire_errors"], [])
        self.assertNotIn("wall_speedup", r["pairs"][0])

    def test_forged_budget_flag_excluded(self):
        d = copy.deepcopy(self.group)
        d["requests"][0]["usage"]["completion_tokens"] = 2
        d.update(actual_output_tokens=2, group_output_tok_s=4)
        self.save("patched", d)
        r = review.review(self.root)
        self.assertTrue(r["wire_errors"])
        self.assertNotIn("wall_speedup", r["pairs"][0])

    def test_forged_throughput_excluded(self):
        d = copy.deepcopy(self.group)
        d["group_output_tok_s"] = 999
        self.save("patched", d)
        r = review.review(self.root)
        self.assertTrue(r["wire_errors"])
        self.assertNotIn("wall_speedup", r["pairs"][0])

    def test_integer_gold_signed_unicode_and_multiple_numbers(self):
        self.assertTrue(
            review.gold_matches("answer -42", {"match": "integer", "expected": -42})
        )
        self.assertTrue(review.gold_matches("٤٢", {"match": "integer", "expected": 42}))
        self.assertFalse(
            review.gold_matches("42.0", {"match": "integer", "expected": 42})
        )

    def test_performance_key_requires_known_family_and_all_dimensions(self):
        self.assertEqual(review.performance_dimensions("main_1024_8_2"), (1024, 8))
        self.assertIsNone(review.performance_dimensions("retrieval_1024_8"))
        self.assertIsNone(review.performance_dimensions("main_1024_8_bad"))


if __name__ == "__main__":
    unittest.main()
