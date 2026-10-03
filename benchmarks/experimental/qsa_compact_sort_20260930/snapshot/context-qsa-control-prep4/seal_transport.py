"""Seal stable transport sources and closed small receipts, not raw-tree copies."""
import ast
import hashlib
import json
from pathlib import Path
import sys

P = Path(__file__).resolve().parent.parent / 'context-qsa-transport-prep1'
sys.path.insert(0, str(Path(__file__).resolve().parent))
from assemble import EXPECTED


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def main():
    for name, digest in EXPECTED.items():
        assert sha(P / name) == digest
        if name.endswith('.py'):
            ast.parse((P / name).read_text())
    receipts = []
    for index in (1, 2, 3):
        path = P / f'cpu-wire{index}/RESULT.json'
        row = json.loads(path.read_text())
        receipts.append({
            'path': str(path.relative_to(P)), 'sha256': sha(path),
            'bytes': path.stat().st_size, 'status': row['status'],
            'groups': row['groups'], 'full_A0': row['full_A0_transport_inventory'],
            'elapsed_s': row['elapsed_s'], 'python': row['runtime']['python'],
            'requests': row['runtime']['requests'], 'urllib3': row['runtime']['urllib3'],
            'not_512MiB_capacity_admission': True,
            'raw_tree_retained_in_place_not_duplicated': True,
        })
    save(P / 'LOCAL-CPU-EVIDENCE.json', {
        'status': 'CLOSED_LOCAL_LOOPBACK_SOURCE_EVIDENCE_ONLY',
        'prior_runs': receipts, 'raw_loopback_trees_outside_source_manifest': True,
        'new_CPU_runs': 0, 'no_GPU_model_or_target_network_claim': True,
    })
    save(P / 'CONTRACT.json', {
        'status': 'SOURCE_SEALED_NOT_SERVICE_OR_MEMORY_ADMISSION',
        'stable_sources': EXPECTED,
        'line_cap_bytes': 2 * 1024**2,
        'received_cap_bytes': 16 * 1024**2,
        'request_tree_cap_bytes': 34 * 1024**2,
        'result_cap_bytes': 256 * 1024,
        'packet_cap_bytes': 4 * 1024**2,
        'group_cap_bytes': 8 * 1024**2,
        'line_count_cap': 4096, 'rejected_prefix_cap_bytes': 65536,
        'request_deadline_s': 600, 'group_deadline_s': 660, 'arm_deadline_s': 7200,
        'Pump_and_dependency_source_bytes_unchanged_from_context_qsa_prep1': True,
        'original_parser_bytes_unchanged': True,
        'received_line_before_limit_check_not_retained_in_run3': True,
        'worker_new_failure_receipt_precedes_parse_and_error': True,
        'runtime_capacity_decision': 'external native CPU cgroup receipt and new control contract',
        'raw_CPU_evidence': 'retained in existing cpu-wire*/cpu-server*; source seal lists only closed result/log receipts',
    })
    with (P / 'REPORT.md').open('x') as stream:
        stream.write('''# Context transport correction

The five stable source files retain their reviewed SHA values. The Pump and
dependency loader match the frozen context matrix bytes; the Pump's relative
WORKER now resolves to this package. The worker increases only the line limit
from 512KiB to 2MiB and writes a bounded rejected-line receipt before parsing or
raising: full length/SHA/cap reasons, at most 64KiB prefix. The success path has
no new per-line dict. Received16MiB/tree34MiB/result256KiB/packet4MiB/group8MiB,
line-count4096, parser/IDs/deadlines/reap/cleanup remain unchanged.

Three closed local loopback trials are retained, including earlier source
versions and failures. Only the last sealed-source trial proves the final five
source bytes; none proves model behavior, service capacity or GPU performance.
The full native A0 capacity stress is a separate root-owned evidence scope.
Its wire success cannot turn a 512MiB cgroup peak failure into a budget PASS.

This is a source seal with compact closed logs/results, not a recursive copy
of the raw CPU trees. Existing raw evidence remains in place and is identified
by the result receipts. The future control package must explicitly bind the
frozen runner.Pump to this package's Pump and verify both Pump and worker SHA.
Original 798 requests/body/order/salt/metrics and all old FAIL states remain.
''')
    paths = [path for path in P.iterdir() if path.is_file()
             and path.name != 'MANIFEST.json']
    paths.extend(P / item['path'] for item in receipts)
    for index in (1, 2, 3):
        for name in ('CLOSED.json', 'SERVER.json'):
            path = P / f'cpu-server{index}' / name
            if path.is_file():
                paths.append(path)
    files = []
    for path in sorted(set(paths)):
        assert not path.is_symlink()
        files.append({'path': str(path.relative_to(P)), 'sha256': sha(path),
                      'bytes': path.stat().st_size})
    save(P / 'MANIFEST.json', {
        'status': 'SOURCE_CPU_ONLY_NOT_TARGET_SERVICE_ADMISSION',
        'files': files, 'stable_sources': EXPECTED,
        'raw_evidence_trees_retained_not_covered_by_this_source_payload_seal':
            [f'cpu-wire{i}' for i in (1, 2, 3)] + [f'cpu-server{i}' for i in (1, 2, 3)],
        'source_manifest_excludes_itself': True,
    })
    print(json.dumps({'files': len(files), 'bytes': sum(r['bytes'] for r in files),
                      'manifest_sha256': sha(P / 'MANIFEST.json')}))


if __name__ == '__main__':
    main()
