# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare ordinary tuning caches; align only identical source/candidate sets."""

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

ROOT = None
CACHE = None
IGNORED = {"time_taken_ms", "triton_cache_hash"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(arm):
    from torch._inductor.codecache import torch_key
    from torch.compiler import config

    root = CACHE / arm
    helpers = {}
    for path in root.rglob("*.py"):
        if "inductor_cache" not in path.parts and root / "inductor" not in path.parents:
            continue
        prepared = hashlib.sha256(
            f"{path.name}:{config.cache_key_tag}".encode()
        ).hexdigest()
        key = (
            hashlib.sha256(prepared.encode() + torch_key()).hexdigest() + ".best_config"
        )
        value = (path.name, digest(path))
        helpers.setdefault(key, set()).add(value)
    found = {}
    for path in sorted(root.rglob("*.best_config")):
        data = json.loads(path.read_text())
        candidates = helpers.get(path.name, set())
        helper = next(iter(candidates)) if len(candidates) == 1 else None
        found.setdefault(path.name, []).append(
            {"relative": str(path.relative_to(root)), "config": data, "helper": helper}
        )
    return found


def main():
    global ROOT, CACHE
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply only after stopping both serving arms",
    )
    parser.add_argument("--label", default="warm")
    args = parser.parse_args()
    CACHE = args.cache_root.resolve()
    ROOT = CACHE.parent
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    arms = {arm: inspect(arm) for arm in ("U", "E")}
    shared = sorted(set(arms["U"]) & set(arms["E"]))
    matched, unmatched, different = {}, [], []
    for key in shared:
        rows = arms["U"][key] + arms["E"][key]
        identities = {
            (tuple(row["helper"] or ()), row["config"].get("configs_hash"))
            for row in rows
        }
        if len(identities) != 1 or not all(
            row["helper"] and row["config"].get("configs_hash") for row in rows
        ):
            unmatched.append(key)
            continue
        canonical = arms["U"][key][0]["config"]
        clean = lambda value: {k: v for k, v in value.items() if k not in IGNORED}
        if any(clean(row["config"]) != clean(canonical) for row in rows):
            different.append(key)
        matched[key] = canonical
    result = {
        "label": args.label,
        "epoch": time.time(),
        "ordinary_loader": True,
        "matching_source_and_candidate_sets": len(matched),
        "different_choices": different,
        "unmatched_shared_entries": unmatched,
        "unique_entries": {
            a: sorted(set(rows) - set(shared)) for a, rows in arms.items()
        },
        "applied": False,
        "canonical_arm": "U",
        "details": arms,
    }
    if args.apply and different:
        backup = ROOT / ("cache-before-alignment-" + args.label)
        backup.mkdir()
        for arm in ("U", "E"):
            root = CACHE / arm
            payloads = {}
            for key, rows in arms[arm].items():
                for row in rows:
                    payloads[row["relative"]] = json.dumps(
                        matched.get(key, row["config"]), sort_keys=True
                    ).encode()
            dest = backup / arm
            dest.mkdir()
            # Rebuild graph containers normally, so they pick up saved choices.
            # Keep the ordinary compiled Triton cache and all unmatched choices.
            for relative in (
                "vllm/torch_compile_cache",
                "inductor/fxgraph",
                "inductor/aotautograd",
            ):
                path = root / relative
                if path.exists():
                    target = dest / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(path), str(target))
            for relative, payload in payloads.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
        result["applied"] = True
        result["graphs_rebuilt_on_next_start"] = True
    path = ROOT / (
        "KERNEL_CHOICES_" + args.label + ("_APPLIED" if args.apply else "") + ".json"
    )
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("details", "unique_entries", "different_choices")
            },
            indent=2,
        )
    )
    print("different_choices", len(different))


if __name__ == "__main__":
    main()
