"""Owned cross-context A0/B/A2 identities; no live PID is frozen here."""
from pathlib import Path

import json

from io_tools import save, sha

BASE = Path('/home/l/work/flash-next/perf-20260923-resume/step58-qsa-compact-sort-20260930')
PACKAGE = BASE / 'context-qsa-control-prep4'
WINDOW = BASE / 'context-qsa-run4'
SERVICE = BASE / 'service-implementation-retry2'
PY = '/home/l/work/1Cat-vLLM/.venv/bin/python'
MODELS = {arm: f'step58-contextqsa4-{arm.lower()}.service' for arm in ('A0', 'B', 'A2')}
HTTP = {arm: f'step58-contextqsa4-http-{arm.lower()}.service' for arm in MODELS}
UNITS = [*MODELS.values(), *HTTP.values()]
TIMER = 'step58-contextqsa4-expiry'
OUTER = 'flash-next-recovery-orchestrator-step58-contextqsa4.service'


def read(path, cap=4 * 1024**2):
    path = Path(path)
    if path.stat().st_size > cap:
        raise ValueError('bounded context controller JSON input')
    return json.loads(path.read_text())
