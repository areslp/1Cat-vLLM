"""Create-only bounded CPU evidence. Aggregate monitoring is not an FS quota."""
import hashlib
import json
from pathlib import Path

WINDOW_CAP = 8 * 1024**3
REQUEST_RECEIVED_CAP = 16 * 1024**2
REQUEST_TREE_CAP = 34 * 1024**2
LINE_CAP = 512 * 1024
LINE_COUNT_CAP = 4096


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value, cap=1024 * 1024):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
    if len(raw) > cap:
        raise ValueError('bounded JSON payload too large')
    with Path(path).open('xb') as stream:
        stream.write(raw)
    return {'path': str(path), 'bytes': len(raw), 'sha256': sha(path)}


def tree_bytes(root):
    total = 0
    for path in Path(root).rglob('*'):
        if path.is_symlink():
            raise ValueError('unapproved artifact symlink')
        if path.is_file():
            total += path.stat().st_size
    return total


def aggregate_gate(root):
    total = tree_bytes(root)
    if total > WINDOW_CAP:
        raise ValueError('aggregate 8 GiB measurement budget exhausted')
    return total
