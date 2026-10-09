# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard the static proof boundary: comments can vanish, arithmetic cannot."""

import hashlib
import unittest

from audit_saved_artifacts import analyze, canonical_parts, kernel_names


def module(comment, multiplier=2, benchmark_multiplier=3):
    return (
        'r"""\nCompile-time auto-tuning block:\n'
        f"# {comment}\nvalue = {benchmark_multiplier}\n"
        '"""\n'
        f"# {comment}\n"
        "kernel = async_compile.triton('kernel', '''\n"
        f"# {comment}\n@triton.jit\ndef kernel(x):\n return x * {multiplier}\n"
        "''', device_str='cuda')\n"
    )


class ProofBoundaryTests(unittest.TestCase):
    def test_comments_in_wrapper_benchmark_and_embedded_kernel(self):
        self.assertEqual(canonical_parts(module("one")), canonical_parts(module("two")))

    def test_kernel_arithmetic_change_is_retained(self):
        self.assertNotEqual(
            canonical_parts(module("one", 2))[0],
            canonical_parts(module("one", 4))[0],
        )

    def test_benchmark_is_audited_separately(self):
        a, b = (
            canonical_parts(module("one", 2, 3)),
            canonical_parts(module("one", 2, 5)),
        )
        self.assertEqual(a[0], b[0])
        self.assertNotEqual(a[1], b[1])

    def test_cache_mapping_includes_binary_torch_key_and_tag(self):
        source = (
            "# kernel path: /example/ab/cabc.py\n"
            "# comment\ntriton_red_example = async_compile.triton('example', '')\n"
        )
        digest, tag = b"\x00\xff", "test-tag"
        first = hashlib.sha256(b"cabc.py:test-tag").hexdigest()
        key = hashlib.sha256(first.encode() + digest).hexdigest()
        names = kernel_names(source, digest, tag)
        self.assertEqual(
            names["ab/" + key + ".best_config"]["kernel"], "triton_red_example"
        )
        self.assertNotEqual(names, kernel_names(source, digest, ""))

    def test_saved_config_difference_does_not_become_a_launch_observation(self):
        observations = {
            "pairs": [
                {
                    "rank": 0,
                    "graph": "backbone",
                    "subgraph": "subgraph0",
                    "source_ast_hashes": [["a", "b"], ["a", "b"]],
                    "copy_matches_reference": [True, True],
                    "reference_in_active_directory": [False, True],
                    "config_pairs": [
                        {
                            "runtime_reference": {"kernel": "triton_red_example"},
                            "old": {"R0_BLOCK": 32},
                            "fixed": {"R0_BLOCK": 64},
                            "configs_hash": ["same", "same"],
                        }
                    ],
                }
            ],
        }
        result = analyze(observations)
        self.assertEqual(result["runtime_source_referenced_reduction_differences"], 1)
        self.assertFalse(result["actual_launches_observed"])
        self.assertFalse(result["current_cohort_causality_proven"])


if __name__ == "__main__":
    unittest.main()
