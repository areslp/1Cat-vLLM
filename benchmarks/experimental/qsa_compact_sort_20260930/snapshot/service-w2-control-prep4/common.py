"""Exact new initial W2 ownership; no source-dependent live PID is frozen here."""
import json
import os
from pathlib import Path

from io_tools import save, sha, aggregate_gate

BASE = Path('/home/l/work/flash-next/perf-20260923-resume/step58-qsa-compact-sort-20260930')
PACKAGE = BASE / 'service-w2-control-prep4'
WINDOW = BASE / 'service-w2-run4'
SERVICE = BASE / 'service-implementation-retry2'
PY = '/home/l/work/1Cat-vLLM/.venv/bin/python'
MODELS = {arm: f'step58-w2-{arm.lower()}4.service' for arm in ('A0', 'B', 'A2')}
HTTP = {arm: f'step58-w2-http-{arm.lower()}4.service' for arm in ('A0', 'B', 'A2')}
HTTP['stability'] = 'step58-w2-http-b-stability4.service'
UNITS = {**MODELS, **{name + '_HTTP': unit for name, unit in HTTP.items()}}
TIMER = 'step58-w2-4-expiry'
OUTER = 'flash-next-recovery-orchestrator-step58-w2-4.service'


def read(path, cap=1024**2):
    path = Path(path)
    if path.stat().st_size > cap:
        raise ValueError('bounded controller JSON input')
    return json.loads(path.read_text())


def cpu_env():
    # stdlib controller/helpers only; HTTP workers use an even narrower env.
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('STEP58_', 'PHYSICAL_FIXTURE_'))}
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
               PYTHONNOUSERSITE='1')
    return env


def artifact_gate(root):
    return aggregate_gate(root)
