# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit exported compile sources/configs as data; never execute them."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

LAUNCH_KEYS = {"XBLOCK", "R0_BLOCK", "num_warps", "num_stages"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


class EmbeddedKernelAST(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str) and "@triton.jit" in node.value:
            node.value = ast.dump(ast.parse(node.value), include_attributes=False)
        return node


def canonical_parts(source):
    tree = ast.parse(source)
    benchmark = None
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
        and tree.body[0].value.value.startswith("\nCompile-time auto-tuning block:")
    ):
        text = tree.body.pop(0).value.value
        benchmark = ast.parse(text.split("Compile-time auto-tuning block:", 1)[1])
    runtime = ast.dump(EmbeddedKernelAST().visit(tree), include_attributes=False)
    tuning = (
        ast.dump(EmbeddedKernelAST().visit(benchmark), include_attributes=False)
        if benchmark is not None
        else None
    )
    return [sha(runtime.encode()), sha(tuning.encode()) if tuning else None]


def kernel_names(source, torch_digest, tag=""):
    result = {}
    pending_path = None
    for line in source.splitlines():
        if line.startswith("# kernel path: "):
            pending_path = line.removeprefix("# kernel path: ")
            continue
        if not pending_path or " = async_compile.triton(" not in line:
            continue
        name = line.split(" = async_compile.triton(", 1)[0]
        if not name.startswith("triton_") or not name.isidentifier():
            continue
        path = Path(pending_path)
        key = sha((path.name + ":" + tag).encode())
        best = sha(key.encode() + torch_digest) + ".best_config"
        result[path.parent.name + "/" + best] = {
            "kernel": name,
            "source_basename": path.name,
        }
        pending_path = None
    return result


def export_observations(root, portable=True):
    inventory = json.loads((root / "ARTIFACT_INVENTORY_PRIVATE.json").read_text())
    maps = {
        arm: {
            (r["rank"], r["graph"], r["filename"]): r
            for r in inventory["rows"]
            if r["arm"] == arm
        }
        for arm in ("main", "patched")
    }
    assert maps["main"].keys() == maps["patched"].keys()
    observations = []
    for key, old in maps["main"].items():
        fixed = maps["patched"][key]
        sources = [
            [
                root / v["relative_file"]
                for a in r["artifacts"]
                for v in a.get("source_files", [])
            ]
            for r in (old, fixed)
        ]
        assert len(sources[0]) == len(sources[1]) == 1
        texts = [s[0].read_text() for s in sources]
        names = kernel_names(texts[0], bytes.fromhex(old["torch_key_hex"]))
        old_choices, fixed_choices = [
            {a["key"]: a["config"] for a in r["artifacts"] if a["type"] == "autotune"}
            for r in (old, fixed)
        ]
        assert old_choices.keys() == fixed_choices.keys()
        pairs = []
        for choice_key, a in old_choices.items():
            b = fixed_choices[choice_key]
            pairs.append(
                {
                    "choice_key": choice_key,
                    "runtime_reference": names.get(choice_key),
                    "old": {k: v for k, v in a.items() if k in LAUNCH_KEYS},
                    "fixed": {k: v for k, v in b.items() if k in LAUNCH_KEYS},
                    "configs_hash": [a["configs_hash"], b["configs_hash"]],
                    "compiled_triton_hash": [
                        a["triton_cache_hash"],
                        b["triton_cache_hash"],
                    ],
                }
            )
        observations.append(
            {
                "rank": key[0],
                "graph": key[1],
                "subgraph": key[2],
                "artifact_sha256": [r["referenced_sha256"] for r in (old, fixed)],
                "reference_in_active_directory": [
                    r["reference_is_in_active_directory"] for r in (old, fixed)
                ],
                "copy_matches_reference": [
                    r["referenced_sha256"] == r["copied_sha256"] for r in (old, fixed)
                ],
                "torch_digest_equal": old["torch_key_hex"] == fixed["torch_key_hex"],
                "source_ast_hashes": [canonical_parts(t) for t in texts],
                "config_pairs": pairs,
            }
        )
    result = {
        "schema": 1,
        "pairs": observations,
        "scope": "Post-run static audit of explicitly referenced saved artifacts",
        "source_normalization": (
            "AST comparison ignores comments and line locations, including embedded "
            "Triton code; compares runtime and compile-time benchmark separately"
        ),
        "private_material_excluded": [
            "absolute cache and host paths",
            "generated full source",
            "weights",
            "raw service/control logs",
            "autotune benchmark durations",
        ],
    }
    if portable:
        for graph in result["pairs"]:
            for pair in graph["config_pairs"]:
                if ref := pair["runtime_reference"]:
                    ref["source_basename_utf8_hex"] = (
                        ref.pop("source_basename").encode().hex()
                    )
                hashes = pair.pop("compiled_triton_hash")
                pair["compiled_triton_hash_utf8_hex"] = [
                    value.encode().hex() for value in hashes
                ]
        result["identifier_encoding"] = (
            "Source basenames and Triton hash strings are UTF-8 hex. "
            "Both transformations preserve the exact original bytes."
        )
    return result


def analyze(observations):
    pairs = observations["pairs"]
    config_pairs = [p for graph in pairs for p in graph["config_pairs"]]
    differences = []
    for graph in pairs:
        for p in graph["config_pairs"]:
            if p["old"] != p["fixed"]:
                differences.append(
                    {
                        **{k: graph[k] for k in ("rank", "graph", "subgraph")},
                        **p,
                    }
                )
    referenced = [p for p in differences if p["runtime_reference"]]
    reductions = [p for p in referenced if "_red_" in p["runtime_reference"]["kernel"]]
    return {
        "source_graph_pairs": len(pairs),
        "equal_runtime_and_benchmark_ast_pairs": sum(
            p["source_ast_hashes"][0] == p["source_ast_hashes"][1] for p in pairs
        ),
        "referenced_artifacts": len(pairs) * 2,
        "copied_reference_mismatches": sum(
            not equal for p in pairs for equal in p["copy_matches_reference"]
        ),
        "references_outside_copied_active_directory": sum(
            not own for p in pairs for own in p["reference_in_active_directory"]
        ),
        "saved_config_pairs": len(config_pairs),
        "candidate_config_hash_mismatches": sum(
            p["configs_hash"][0] != p["configs_hash"][1] for p in config_pairs
        ),
        "saved_launch_config_differences": len(differences),
        "runtime_source_referenced_config_pairs": sum(
            bool(p["runtime_reference"]) for p in config_pairs
        ),
        "runtime_source_referenced_differences": len(referenced),
        "runtime_source_referenced_reduction_differences": len(reductions),
        "runtime_source_referenced_difference_rows": referenced,
        "actual_launches_observed": False,
        "current_cohort_causality_proven": False,
        "limits": [
            "Saved choices are not direct observations of launches or CUDA graphs",
            "Unmapped saved entries can be compile-time candidates, not runtime work",
            "Equal source ASTs do not imply equal floating reduction trees",
            "Different valid launch choices do not prove output values differ",
            "No new GPU experiment or production change was performed",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--private-root", type=Path)
    source.add_argument("--observations", type=Path)
    parser.add_argument("--export-observations", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = (
        export_observations(args.private_root)
        if args.private_root
        else json.loads(args.observations.read_text())
    )
    if args.export_observations:
        args.export_observations.write_text(json.dumps(data, indent=2) + "\n")
    result = analyze(data)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if not k.endswith("rows")}, indent=2
        )
    )
