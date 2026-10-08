# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit the real MTP tile selector and native tile admission without Torch.

This inspects source contracts only. It does not measure GPU performance or
prove that a particular tile explains an end-to-end acceptance-rate change.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def audit(path):
    source = path.read_text()
    tree = ast.parse(source)
    selector = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_get_sm70_mtp_moe_decode_config"
    )
    constants = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(
            isinstance(target, ast.Name)
            and target.id.startswith("_SM70_QWEN38_MTP_MOE_")
            for target in targets
        ):
            constants.append(node)
    namespace = {
        "envs": SimpleNamespace(VLLM_SM70_MTP_MOE_TUNED_CONFIG=True),
        "_force_sm70_mtp_moe_legacy_config": False,
    }
    exec(
        compile(
            ast.Module(body=[*constants, selector], type_ignores=[]),
            "tile-selector",
            "exec",
        ),
        namespace,
    )
    dispatch = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "dispatch_fused_moe_kernel"
    )
    admission = next(
        node.test
        for node in dispatch.body
        if isinstance(node, ast.If) and "sm70_mtp_moe_fp16" in ast.unparse(node.test)
    )
    terms = [term for term in admission.values if "config.get(" in ast.unparse(term)]
    assert terms, "Native tile admission must be present"
    guard = ast.Expression(body=ast.BoolOp(op=ast.And(), values=terms))
    ast.fix_missing_locations(guard)
    code = compile(guard, "native-tile-guard", "eval")
    shapes = {}
    for rows in (1, 2, 4, 5, 8, 10, 20, 40):
        config = namespace[selector.name](rows, 512, 160, 2560, 10)
        shapes[str(rows)] = {
            "config": config,
            "native_tile_guard_admitted": bool(
                config is not None and eval(code, {"config": config})
            ),
        }
    return {
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "shape": {"experts": 512, "intermediate": 160, "hidden": 2560, "topk": 10},
        "scope": "tile admission only; other device/shape/operator guards still apply",
        "native_tile_guard": [ast.unparse(term) for term in terms],
        "shapes": shapes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps([audit(source) for source in args.sources], indent=2))


if __name__ == "__main__":
    main()
