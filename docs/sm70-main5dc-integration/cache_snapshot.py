# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Record task-cache file bytes; run outside the timed request interval."""

import argparse
import hashlib
import json
from pathlib import Path


def snapshot(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        before = path.stat()
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Cache changed while hashing: {path.relative_to(root)}")
        result[str(path.relative_to(root))] = {
            "bytes": after.st_size,
            "sha256": digest,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error("--root must be an existing task cache directory")
    args.out.write_text(json.dumps(snapshot(args.root), indent=2) + "\n")


if __name__ == "__main__":
    main()
