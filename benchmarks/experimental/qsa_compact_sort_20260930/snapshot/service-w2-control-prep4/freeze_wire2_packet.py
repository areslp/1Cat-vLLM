"""Create the distinct corrective wire2 source plan; never executes a unit."""
import ast
import json
from pathlib import Path
import sys

from io_tools import save, sha

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / 'service-w2-wire-prep2'
ORIGINAL = ROOT.parent / 'service-w2-wire-prep1'
CORRECTED = ROOT.parent / 'service-w2-transport-prep2'
D = Path('/home/l/work/flash-next/perf-20260923-resume/step58-qsa-compact-sort-20260930')


def main():
    OUT.mkdir(mode=0o700)
    paths = ['frozen.py', 'io_tools.py', 'transport_deadline.py', 'transport_worker.py']
    for name in paths:
        source = CORRECTED / name
        ast.parse(source.read_text(), filename=name)
        with (OUT / name).open('xb') as stream:
            stream.write(source.read_bytes())
    source = ROOT / 'wire_preflight.py'
    ast.parse(source.read_text(), filename=source.name)
    with (OUT / source.name).open('xb') as stream:
        stream.write(source.read_bytes())
    with (OUT / 'runtime-pins.json').open('xb') as stream:
        stream.write((ORIGINAL / 'runtime-pins.json').read_bytes())
    script = (ORIGINAL / 'run_cpu.sh').read_text().replace('service-w2-wire-prep1', OUT.name)
    with (OUT / 'run_cpu.sh').open('x') as stream:
        stream.write(script)
    old = json.loads((ORIGINAL / 'PLAN.json').read_text())
    value = dict(old)
    value['argv'] = [v.replace('service-w2-wire-prep1', OUT.name)
                     .replace('step58-w2-wire1.service', 'step58-w2-wire2.service')
                     for v in old['argv']]
    value.update(cwd=str(D / OUT.name), unit='step58-w2-wire2.service',
        status='FROZEN_CORRECTIVE_CPU_PLAN_NOT_EXECUTED_NO_CURRENT_UNIT_AUTHORIZATION',
        predecessor={'manifest_sha256': sha(ORIGINAL / 'manifest.json'),
                     'execution': 'wire1 attempt1 failed on normal EOF; only one HTTP sent'},
        corrected_transport_manifest_sha256=sha(CORRECTED / 'manifest.json'),
        normal_additional_required_checks={
            'live_socket_timeout_updates': '>0',
            'confirmed_native_closed_buffer_iterations': '>0',
            'usage_finish_DONE': 'exact frozen completion and actual child exit0/reap'},
        revision_scope='Only pinned worker EOF descriptor correction and explicit branch coverage; Pump/frozen/IO byte-identical to wire1.',
        local_AST_check={'files': paths + ['wire_preflight.py'],
                         'executable': sys.executable, 'Python': sys.version})
    save(OUT / 'PLAN.json', value)
    files = {str(p.relative_to(OUT)): {'bytes': p.stat().st_size, 'sha256': sha(p)}
             for p in sorted(OUT.rglob('*')) if p.is_file()}
    save(OUT / 'manifest.json', {'files': files, 'count': len(files),
                               'bytes': sum(v['bytes'] for v in files.values())})
    print(json.dumps({'root': str(OUT), 'manifest_sha256': sha(OUT / 'manifest.json'),
                      'plan_sha256': sha(OUT / 'PLAN.json'), 'files': len(files)}))


if __name__ == '__main__':
    main()
