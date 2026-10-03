"""Persist existing host NVML query returns before unchanged native validation.

No queries, model hooks, GPU operations or gate overrides are added here.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

from identity_contract import PROCESS_NVML_MIB, DEVICE_NVML_MIB


def classify(argv, raw):
    process = '--query-compute-apps=pid,gpu_uuid,used_gpu_memory' in argv
    device = '--query-gpu=uuid,memory.used' in argv
    if not process and not device:
        return {'kind': 'OTHER_EXISTING_NVML_QUERY', 'native_gate_replaced': False}
    rows, issues = [], []
    for line in raw.splitlines():
        fields = [v.strip() for v in line.split(',')]
        if process:
            if (len(fields) != 3 or not re.fullmatch(r'[0-9]+', fields[0])
                    or not re.fullmatch(r'[0-9]+ MiB', fields[2])):
                issues.append({'kind': 'UNKNOWN_FIELD', 'raw_line': line})
                continue
            memory = int(fields[2].split()[0])
            row = {'pid': int(fields[0]), 'uuid': fields[1], 'used_MiB': memory}
            limit = PROCESS_NVML_MIB
        else:
            if len(fields) != 2 or not re.fullmatch(r'[0-9]+', fields[1]):
                issues.append({'kind': 'UNKNOWN_FIELD', 'raw_line': line})
                continue
            memory = int(fields[1])
            row = {'uuid': fields[0], 'used_MiB': memory}
            limit = DEVICE_NVML_MIB
        rows.append(row)
        if memory == 0:
            issues.append({'kind': 'ZERO_CONTEXT_MEMORY', **row})
        elif memory > limit:
            issues.append({'kind': 'OVER_CAP', 'limit_MiB': limit, **row})
    if not raw.strip():
        issues.append({'kind': 'EMPTY_CONTEXT_SET'})
    return {'kind': 'PROCESS_MEMORY' if process else 'DEVICE_MEMORY',
            'issues': issues, 'parsed_rows': rows,
            'process_limit_MiB': PROCESS_NVML_MIB,
            'device_limit_MiB': DEVICE_NVML_MIB,
            'native_gate_replaced': False,
            'identity_bijection_and_all_other_gates': 'UNCHANGED_NATIVE_VALIDATOR'}


def capture_output(original, root, scope):
    if getattr(original, '_resource_nvml_capture', False):
        return original
    root = Path(root)
    ordinal = 0

    def wrapped(*argv, **kwargs):
        nonlocal ordinal
        args = list(argv[0]) if len(argv) == 1 and isinstance(argv[0], (list, tuple)) else list(argv)
        if not args or Path(str(args[0])).name != 'nvidia-smi':
            return original(*argv, **kwargs)
        began = time.time_ns()
        try:
            raw = original(*argv, **kwargs)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            raw = getattr(error, 'stdout', None) or getattr(error, 'output', None) or ''
            stderr = getattr(error, 'stderr', None) or ''
            raw = raw.decode(errors='replace') if isinstance(raw, bytes) else raw
            stderr = stderr.decode(errors='replace') if isinstance(stderr, bytes) else stderr
            persist(args, raw, began, {'kind': 'ORIGINAL_QUERY_COMMAND_FAILED',
                'error': repr(error), 'stderr': stderr})
            raise
        persist(args, raw, began)
        return raw

    def persist(args, raw, began, failure=None):
        nonlocal ordinal
        ordinal += 1
        if ordinal > 1024 or len(raw.encode()) > 2 * 1024**2:
            raise ValueError('finite raw NVML record cap; native STOP required')
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        name = f'{scope}-pid{os.getpid()}-ns{began}-q{ordinal:04d}'
        raw_path = root / (name + '.raw.json')
        value = {'status': 'RAW_RECORDED_BEFORE_NATIVE_VALIDATION', 'argv': args,
                 'query_started_ns': began, 'query_returned_ns': time.time_ns(),
                 'raw': raw, 'additional_query': False, 'query_calls': 1}
        with raw_path.open('x') as stream:
            json.dump(value, stream, indent=2)
            stream.write('\n')
        # Deliberately classify only after the full raw receipt is on disk.
        result = failure if failure is not None else classify(args, raw)
        result.update(raw_path=str(raw_path),
            raw_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            classified_ns=time.time_ns(), native_gate_replaced=False)
        with (root / (name + '.classification.json')).open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')

    wrapped._resource_nvml_capture = True
    return wrapped


def install_guard(root):
    import original_guard
    original_guard.output = capture_output(original_guard.output, root, 'guard')
