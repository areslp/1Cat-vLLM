"""One local CPU batch; raw stdout/stderr and actual exits remain create-only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def main(attempt):
    os.umask(0o077)
    root = HERE / f'evidence/cpu-attempt{attempt}'
    root.mkdir(mode=0o700, parents=True)
    tested = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}
    commands = [[sys.executable, '-I', '-B', str(HERE / 'check_cpu.py')],
                [sys.executable, '-I', '-B', str(HERE / 'check_memory.py'),
                 '--output', str(root / 'MEMORY.json')]]
    records = []
    for index, argv in enumerate(commands):
        began = time.perf_counter()
        with (root / f'command{index}.stdout').open('xb') as out, \
                (root / f'command{index}.stderr').open('xb') as err:
            result = subprocess.run(argv, stdout=out, stderr=err, timeout=120,
                                    env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        (root / f'command{index}.exit').open('x').write(str(result.returncode) + '\n')
        records.append({'argv': argv, 'actual_exit': result.returncode,
                        'elapsed_s': time.perf_counter() - began})
        if result.returncode != 0:
            break
    uv = subprocess.run(['/Users/l/.local/bin/uv', '--version'],
                         capture_output=True, text=True, timeout=10)
    value = {'status': 'PASS_LOCAL_CPU_NOT_SERVICE_OR_GPU' if
             len(records) == 2 and all(r['actual_exit'] == 0 for r in records) else
             'FAILED_LOCAL_CPU_PRESERVED', 'commands': records, 'tested_source_sha256': tested,
             'Python': sys.version, 'executable': sys.executable,
             'profile': 'uv --no-project --managed-python --python3.12.13 --no-python-downloads',
             'uv_actual_exit': uv.returncode, 'uv_version': uv.stdout.strip(),
             'HTTP_requests': 0, 'GPU_operations': 0}
    (root / 'RECEIPT.json').open('x').write(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value, sort_keys=True))
    if value['status'] != 'PASS_LOCAL_CPU_NOT_SERVICE_OR_GPU':
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=int, required=True)
    args = parser.parse_args()
    main(args.attempt)
