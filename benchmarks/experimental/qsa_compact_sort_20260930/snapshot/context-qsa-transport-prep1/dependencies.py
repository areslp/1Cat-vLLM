"""Activate only fixed CPU client sources; no model imports."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def activate():
    receipt = json.loads((HERE / 'DEPENDENCIES.json').read_text())
    for row in receipt['files']:
        path = BASE / row['relative_path']
        if path.is_symlink() or not path.is_relative_to(BASE):
            raise ValueError('dependency path escapes fixed workspace')
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('CPU dependency bytes differ: ' + str(path))
    for name in ('service-w2-control-prep4', 'mixed-diagnostic-20261002'):
        path = str(BASE / name)
        if path not in sys.path:
            sys.path.insert(0, path)
    return receipt
