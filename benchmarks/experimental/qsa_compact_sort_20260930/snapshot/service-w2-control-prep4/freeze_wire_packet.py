"""Freeze the single CPU-only synthetic real-wire plan before its one run."""
import ast
import json
from pathlib import Path
import shutil
import sys

from frozen import PINS, require
from io_tools import save, sha

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / 'service-w2-wire-prep1'
D = Path('/home/l/work/flash-next/perf-20260923-resume/step58-qsa-compact-sort-20260930')
REMOTE = D / OUT.name
FILES = ['frozen.py', 'io_tools.py', 'transport_worker.py',
         'transport_deadline.py', 'wire_preflight.py']


def main():
    OUT.mkdir(mode=0o700)
    for name in FILES:
        ast.parse((ROOT / name).read_text(), filename=name)
        with (OUT / name).open('xb') as stream:
            stream.write((ROOT / name).read_bytes())
    actual = ROOT.parent / 'service-w2-prep/deployed-http-source-receipt.json'
    with (OUT / 'runtime-pins.json').open('xb') as stream:
        stream.write(actual.read_bytes())
    native = json.loads(actual.read_text())
    external = {str(D / 'service-w2-prep' / name): value for name, value in PINS.items()}
    for name in PINS:
        require(name)
    external.update({row['path']: row['sha256'] for row in native['files']})
    script = f'''#!/bin/bash
set -u
umask 077
D={REMOTE}
cd "$D" || exit 1
test ! -e attempt1/cases || exit 125
set +e
env -i PATH=/usr/bin:/bin CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 /usr/bin/timeout --kill-after=5 60 /home/l/work/1Cat-vLLM/.venv/bin/python -I -B "$D/wire_preflight.py" --output "$D/attempt1/cases" > "$D/attempt1/child.stdout" 2> "$D/attempt1/child.stderr"
w2_child_rc=$?
printf '%s\\n' "$w2_child_rc" > "$D/attempt1/child.exit"
exit "$w2_child_rc"
'''
    with (OUT / 'run_cpu.sh').open('x') as stream:
        stream.write(script)
    argv = ['sudo', '-n', 'systemd-run', '--wait', '--collect',
        '--unit=step58-w2-wire1.service', '--property=User=l', '--property=Group=l',
        '--property=WorkingDirectory=' + str(REMOTE),
        '--property=AllowedCPUs=14,42', '--property=Nice=15',
        '--property=CPUQuota=100%', '--property=MemoryMax=1G',
        '--property=MemorySwapMax=0', '--property=RuntimeMaxSec=70s',
        '--property=TimeoutStopSec=5s', '--property=KillMode=control-group',
        '/bin/bash', str(REMOTE / 'run_cpu.sh')]
    save(OUT / 'PLAN.json', {'schema': 1, 'status': 'FROZEN_CPU_PLAN_NOT_EXECUTED',
        'argv': argv, 'cwd': str(REMOTE), 'unit': 'step58-w2-wire1.service',
        'external_pins': external, 'actual_dependency_source_receipt_sha256': sha(actual),
        'cases': ['normal', 'cancel', 'trickle-no-newline'],
        'synthetic_requests': 3, 'requests_to_model_or_8200_8201': 0,
        'normal_cancel_deadline_seconds': 5, 'trickle_absolute_deadline_seconds': 0.6,
        'parent_process_cleanup_seconds': 4, 'term_to_kill_seconds': 1,
        'artifact_cap_bytes': 1024**2, 'host_memory_cgroup_bytes': 1024**3,
        'model_GPU_or_Torch': 'NOT_IMPORTED_NO_OPERATION',
        'URL_adaptation': 'Only exact worker URL8201 is explicitly redirected in this synthetic wrapper to its newly owned 127.0.0.1 ephemeral server; recorded real socket peer must equal that port. No listener/request on8200/8201.',
        'returned_line_budget': 'Original16MiB iter_lines returned byte sum, not wire hard cap; no-newline continuous recv must terminate/reap by independent parent deadline.',
        'in_memory_tests': 'None substituted for real requests/socket; actual dependency versions/fileSHA required.',
        'on_failure': 'Preserve raw process exits/partial bytes/no rerun; original model service is never stopped or touched.',
        'runtime_profile': '/home/l/work/1Cat-vLLM/.venv/bin/python; uv-managed3.12.13; no package installation',
        'local_AST_check': {'files': FILES, 'executable': sys.executable, 'Python': sys.version}},
        1024**2)
    files = {str(path.relative_to(OUT)): {'bytes': path.stat().st_size,
                'sha256': sha(path)} for path in sorted(OUT.rglob('*')) if path.is_file()}
    save(OUT / 'manifest.json', {'files': files, 'count': len(files),
        'bytes': sum(row['bytes'] for row in files.values())})
    pins = dict(external)
    pins.update({str(REMOTE / name): row['sha256'] for name, row in files.items()})
    pins[str(REMOTE / 'manifest.json')] = sha(OUT / 'manifest.json')
    evidence = ROOT / 'evidence/wire1-transport-attempt1'
    evidence.mkdir(mode=0o700)
    with (evidence / 'host-pins.sha256').open('x') as stream:
        stream.writelines(f'{value}  {path}\n' for path, value in sorted(pins.items()))
    print(json.dumps({'root': str(OUT), 'manifest_sha256': sha(OUT / 'manifest.json'),
        'plan_sha256': sha(OUT / 'PLAN.json'), 'pins': len(pins)}))


if __name__ == '__main__':
    main()
