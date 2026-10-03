"""Execute exact AST bodies from the pinned production source, in isolation.

No vLLM import, model import, environment mutation, or implementation rewrite.
Outer routing tests use host mocks; CUDA W1 uses the real forward dispatcher.
"""
import ast
import hashlib
from pathlib import Path
import re

SOURCE_SHA = 'ee03f537412a5063d5f280a017bc6fffaf13a9af8e818fc3df63e364205b2694'
FUNCTIONS = (
    '_qsa_grouped_page4_abi_version',
    '_qsa_grouped_page4_supported',
    '_qsa_grouped_page4_workspace',
    '_qsa_grouped_page4_forward',
    '_qsa_sparse_paged_attention_sm70_xqa_page4',
)


class QuietLogger:
    def info_once(self, *args):
        pass

    def warning_once(self, *args):
        pass


def load_namespace(torch_api, source=None, split='split', grouped=True):
    source = source or Path(__file__).parent / 'source/qsa.production.py'
    raw = Path(source).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA, 'source SHA changed'
    tree = ast.parse(raw)
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name in FUNCTIONS]
    assert {n.name for n in selected} == set(FUNCTIONS)
    future = ast.parse('from __future__ import annotations').body[0]
    module = ast.Module(body=[future, *selected], type_ignores=[])
    ns = {
        'torch': torch_api, 're': re, 'logger': QuietLogger(),
        '_ONECAT_QSA48': split, '_QSA48_SPLIT_MAX_ROWS': 64,
        '_SM70_QSA_GROUPED_PAGE4': grouped,
        '_SM70_QSA_GROUPED_PAGE4_QUERIES': 8,
        '_SM70_QSA_GROUPED_PAGE4_OUTPUT_PAGES': 4160,
        '_SM70_QSA_GROUPED_PAGE4_WORKSPACES': {},
        '_SM70_QSA_GROUPED_PAGE4_ABI_CACHE': None,
    }
    exec(compile(module, str(source), 'exec'), ns)
    ns['source_extraction'] = [
        {'name': n.name, 'line': n.lineno, 'end_line': n.end_lineno}
        for n in selected]
    return ns
