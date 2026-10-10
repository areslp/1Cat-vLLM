# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read saved Torch 2.10 graph choices after timing; no live loader changes."""

import argparse
import ast
import base64
import hashlib
import json
import os
import pickle
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def groups(reader):
    name = reader.read_str()
    rows = [
        (reader.read_str(), reader.read_bytes()) for _ in range(reader.read_uint64())
    ]
    return name, rows


def compiled_records(root, namespace, role, tp):
    from torch.utils._appending_byte_serializer import (
        AppendingByteSerializer,
        BytesReader,
    )

    found, counts, unresolved = {}, {}, []
    prefix = "backbone" if role == "backbone" else "eagle_head"
    for rank in range(tp):
        directory = (
            root / "vllm/torch_compile_cache" / namespace / f"rank_{rank}_0" / prefix
        )
        index = ast.literal_eval((directory / "vllm_compile_cache.py").read_text())
        for value in index.values():
            reader = BytesReader((directory / value["graph_handle"][0]).read_bytes())
            reader.read_bytes()
            reader.read_bytes()
            reader.read_str()
            payload = reader.read_bytes()
            for kind, artifacts in AppendingByteSerializer.to_list(
                payload, deserialize_fn=groups
            ):
                if kind != "inductor":
                    continue
                for _, content in artifacts:
                    graph = pickle.loads(content)
                    bundle = graph._triton_bundle
                    for saved in bundle.static_autotuners if bundle is not None else []:
                        tuner = saved.kernel
                        results = tuner.compile_results or []
                        counts[str(len(results))] = counts.get(str(len(results)), 0) + 1
                        key = f"{rank}/{Path(tuner.filename).name}"
                        entries = []
                        for result in results:
                            kernel = result.kernel
                            config = {
                                **result.config.kwargs,
                                "num_warps": result.config.num_warps,
                                "num_stages": result.config.num_stages,
                            }
                            encoded = (
                                base64.b32encode(bytes.fromhex(kernel.hash))
                                .decode()
                                .rstrip("=")
                            )
                            binary = (
                                root / "triton" / encoded / (kernel.name + ".cubin")
                            )
                            metadata = binary.with_suffix(".json")
                            row = {
                                "config": config,
                                "kernel_hash": kernel.hash,
                                "kernel": kernel.name,
                                "cubin_sha256": digest(binary)
                                if binary.exists()
                                else None,
                                "metadata_sha256": digest(metadata)
                                if metadata.exists()
                                else None,
                            }
                            if not binary.exists() or not metadata.exists():
                                unresolved.append(
                                    {"key": key, "kernel_hash": kernel.hash}
                                )
                            entries.append(row)
                        if key in found:
                            assert found[key] == entries, (
                                role,
                                key,
                                "inconsistent saved choices",
                            )
                        found[key] = entries
    return {"records": found, "candidate_histogram": counts, "unresolved": unresolved}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--u-namespace")
    parser.add_argument("--e-namespace")
    parser.add_argument(
        "--output", type=Path, default=Path("compiler-choice-review.json")
    )
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    arms, namespaces = {}, {}
    for arm in ("U", "E"):
        root = args.cache_root / arm
        compiled = root / "vllm/torch_compile_cache"
        found = {
            path.relative_to(compiled).parts[0]
            for path in compiled.rglob("vllm_compile_cache.py")
            if path.parent.name == "backbone"
        }
        selected = getattr(args, arm.lower() + "_namespace")
        if selected is None:
            assert len(found) == 1, (arm, sorted(found))
            selected = next(iter(found))
        assert selected in found, (arm, selected, sorted(found))
        namespaces[arm] = selected
        arms[arm] = {
            role: compiled_records(root, namespaces[arm], role, args.tp)
            for role in ("backbone", "draft")
        }
    roles = {}
    for role in ("backbone", "draft"):
        left, right = arms["U"][role]["records"], arms["E"][role]["records"]
        shared = left.keys() & right.keys()
        roles[role] = {
            "common_helpers_by_rank": len(shared),
            "different_configurations": [
                key
                for key in sorted(shared)
                if [row["config"] for row in left[key]]
                != [row["config"] for row in right[key]]
            ],
            "different_kernel_hashes": [
                key
                for key in sorted(shared)
                if [row["kernel_hash"] for row in left[key]]
                != [row["kernel_hash"] for row in right[key]]
            ],
            "different_helpers": [
                key for key in sorted(shared) if left[key] != right[key]
            ],
            "unique_helpers": {
                "U": len(left.keys() - shared),
                "E": len(right.keys() - shared),
            },
            "unresolved_references": {
                arm: len(arms[arm][role]["unresolved"]) for arm in arms
            },
        }
    result = {
        "normal_compiler_and_loader": True,
        "runtime_call_trace": False,
        "scope": (
            "Saved configurations and referenced binary/metadata digests "
            "after measurement."
        ),
        "namespaces": namespaces,
        "roles": roles,
        "details": arms,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(roles, indent=2))


if __name__ == "__main__":
    main()
