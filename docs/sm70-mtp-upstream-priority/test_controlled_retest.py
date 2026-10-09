# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prevent partial launch/graph alignment from qualifying performance pairs."""

import copy
import unittest

from audit_controlled_retest import analyze


def complete():
    arm = {
        "healthy": True,
        "finished": True,
        "gold_passed": True,
        "startup_failure": None,
        "timed_groups": 2,
        "autotuner_runs": [
            {
                "rank": 0,
                "records": [
                    {
                        "source_ast_sha256": "a",
                        "kernel_sha256": "k",
                        "config": {"kwargs": {"R0_BLOCK": 2048}, "num_warps": 16},
                    }
                ],
            }
        ],
    }
    return {
        "expected_ranks": 1,
        "expected_unique_runs_per_rank": 1,
        "expected_graph_pairs": 1,
        "expected_timed_groups_per_arm": 2,
        "expected_performance_groups_per_arm": 2,
        "request_pair_validation_passed": True,
        "expected_graph_keys": [[0, "drafter", 0]],
        "arms": [arm, copy.deepcopy(arm)],
        "graph_pairs": [
            {
                "rank": 0,
                "graph": "drafter",
                "subgraph": 0,
                "ast_hashes": [["r", "b"], ["r", "b"]],
            }
        ],
    }


class TestControlledRetest(unittest.TestCase):
    def test_complete_positive_control(self):
        self.assertTrue(analyze(complete())["performance_pair_admitted"])

    def test_config_alignment_does_not_replace_request_validation(self):
        data = complete()
        data["request_pair_validation_passed"] = False
        self.assertFalse(analyze(data)["performance_pair_admitted"])

    def test_uncovered_default_config_blocks_admission(self):
        data = complete()
        data["arms"][1]["autotuner_runs"][0]["records"][0]["config"] = {}
        self.assertFalse(analyze(data)["performance_pair_admitted"])

    def test_runtime_and_benchmark_sources_each_matter(self):
        for position in (0, 1):
            with self.subTest(position=position):
                data = complete()
                data["graph_pairs"][0]["ast_hashes"][1][position] = "different"
                self.assertFalse(analyze(data)["performance_pair_admitted"])

    def test_missing_or_duplicate_rank_blocks_admission(self):
        for rank_rows in ([], [{"rank": 0}, {"rank": 0}]):
            data = complete()
            data["arms"][1]["autotuner_runs"] = rank_rows
            self.assertFalse(analyze(data)["performance_pair_admitted"])

    def test_incomplete_error_free_driver_is_not_success(self):
        data = complete()
        data["arms"][1]["finished"] = False
        self.assertFalse(analyze(data)["performance_pair_admitted"])

    def test_startup_failure_or_missing_gold_or_groups_blocks_admission(self):
        for key, value in (
            ("startup_failure", "CUDA error"),
            ("gold_passed", False),
            ("timed_groups", 0),
        ):
            data = complete()
            data["arms"][1][key] = value
            self.assertFalse(analyze(data)["performance_pair_admitted"])

    def test_missing_graph_cannot_pass_equal_empty_lists(self):
        data = complete()
        data["graph_pairs"] = []
        self.assertFalse(analyze(data)["performance_pair_admitted"])


if __name__ == "__main__":
    unittest.main()
