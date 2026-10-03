"""Exact external W2 recipe; imports add no model or service observer."""
import hashlib
import importlib
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent.parent
PREP = BASE / 'service-w2-prep'
PINS = {
    'sse_parser.py': '7f2752c8e8b2099e7ed8486da18e7edb35ea1b865459535d79f9ecc1d82475a4',
    'stream_ids.py': 'b76b0fac9b5c40769ca44992d2299363b161f70697ce572892a558c0b404eb09',
    'group_analyzer.py': '4d8a897cd78da947b3117d559ba9a0279facc5cddf4632c7915864f278904fe5',
    'ANALYSIS-CONTRACT.md': '697ecdc0b089f036f2299bda654841ebbf5f4e7d59151d7162d864ce16eb54d0',
    'contract.json': 'ba1373f35f7f7c8b2a041520c5546470ebdbddff27b137d818d916ad26ebaf53',
    'matrix.frozen.json': '41cd10f72fcafe849d39756cfe12960354cf89f8a33abb483ff5b01ed3e9cc9a',
    'stability-pool.frozen.json': 'e4dffe56b8a27b8f4bdb61fe4cd41fdb22438d0a7e13e86ee1d015a5a3dcd35b',
    'manifest.json': 'f8e2b6ecdfedbeb1dc337fbe87e8fbeb8b17793cd998de4b4b997280ed91faca',
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(name):
    path = PREP / name
    if sha(path) != PINS[name]:
        raise ValueError('frozen W2 bytes changed: ' + name)
    return path


def module(name):
    require(name + '.py')
    # Frozen parser imports these exact sibling modules.
    require('stream_ids.py')
    require('group_analyzer.py')
    sys.path.insert(0, str(PREP))
    try:
        result = importlib.import_module(name)
    finally:
        sys.path.pop(0)
    for key in (name, 'stream_ids', 'group_analyzer'):
        loaded = sys.modules.get(key)
        if loaded is not None and Path(loaded.__file__).resolve() != PREP / (key + '.py'):
            raise ValueError('same-name module is not frozen W2 source: ' + key)
    return result
