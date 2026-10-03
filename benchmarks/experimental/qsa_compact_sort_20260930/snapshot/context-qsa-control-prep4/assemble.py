"""One-time local assembly from verified prep3; no live actions."""
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent
BASE = P.parent
OLD = BASE / 'context-qsa-control-prep3'
TRANSPORT = BASE / 'context-qsa-transport-prep1'
EXPECTED = {
    'check_wire.py': '738297f1529826dd9796d95c2f9d22f0444908d486e3c8443855b28bbbdbeff3',
    'transport_worker.py': 'a309a920992e9be021026b293cf4fd7a40349b1755103acd53dde9ea5aa2b998',
    'transport_deadline_context.py': 'a60c9910a492b2666a33e32ed705a6085644c9626065c313a6d4474954ea7c14',
    'dependencies.py': '24d0fc96214ab14208543a5e84db21496779dbbaaa04e7002e294aaf8ef46149',
    'DEPENDENCIES.json': 'c7509ace6279d3048d55c9815ac0d131c6e25fa6ed1ca597d36004a2b2ec7b04',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def main():
    assert sha(OLD / 'MANIFEST.json') == 'e8c9432b2d45394f94d2a74d748da1ddfcec45e41596ffc0e793a2f203a926a0'
    for name, digest in EXPECTED.items():
        assert sha(TRANSPORT / name) == digest, name
    for row in json.loads((OLD / 'MANIFEST.json').read_text())['files']:
        source = OLD / row['path']
        assert source.stat().st_size == row['bytes'] and sha(source) == row['sha256']
        name = row['path']
        if (name.split('/')[0] == 'evidence' or name in (
                'MANIFEST.json', 'DEPENDENCIES.json', 'CONTRACT.json', 'REPORT.md',
                'check_binding_cpu.py', 'BINDING-CPU.exit', 'BINDING-CPU.stdout',
                'BINDING-CPU.stderr') or name.startswith('source/INTEGRATION')
                or name == 'source/REUSE.json'):
            continue
        data = source.read_bytes()
        if source.suffix in ('.py', '.sh'):
            for before, after in (
                    (b'context-qsa-control-prep3', b'context-qsa-control-prep4'),
                    (b'context-qsa-run3', b'context-qsa-run4'),
                    (b'step58-contextqsa3', b'step58-contextqsa4'),
                    (b'STEP58_CONTEXTQSA3', b'STEP58_CONTEXTQSA4')):
                data = data.replace(before, after)
        target = P / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(data)
    contract = json.loads((OLD / 'CONTRACT.json').read_text())
    contract['schema'] = 'CONTEXT_QSA_PRE4_OWNED_TRANSPORT_STRICT_CORE_BINDING'
    contract['window'] = 'context-qsa-run4'
    contract['old_negative_runs'].append('context-qsa-run3')
    contract['prior_evidence_pins'].append({
        'relative_path': 'context-qsa-control-prep3/MANIFEST.json',
        'sha256': sha(OLD / 'MANIFEST.json')})
    contract['transport_repair'] = {
        'source_package': TRANSPORT.name,
        'only_transport_capacity_change': 'line cap 524288 -> 2097152 bytes',
        'rejected_line_receipt': 'before parse/throw; at most 65536B prefix + full line SHA/length/cap reasons',
        'Pump_source_bytes_equal_frozen_matrix': True,
        'explicit_frozen_runner_Pump_binding_required': True,
        'core_binding_extra_keys': 0,
        'metadata_location': 'SHA-bound startup resource_guard_identity.transport_identity plus source_files',
        'received_bytes': 16 * 1024**2,
        'request_tree_bytes': 34 * 1024**2,
        'result_bytes': 256 * 1024,
        'packet_bytes': 4 * 1024**2,
        'group_bytes': 8 * 1024**2,
        'line_count': 4096,
        'requests_replaced_or_extended': False,
        'old_NO_COMMON_WINDOW_preserved': '64K-c8 repeat0/repeat1; no output-length/sample changes',
        'client_budget_finalization': 'PENDING_ROOT_NATIVE_CAPACITY_DECISION',
    }
    save(P / 'CONTRACT.json', contract)
    (P / 'evidence').mkdir()
    save(P / 'source/ASSEMBLY.json', {
        'source_manifest': str(OLD / 'MANIFEST.json'),
        'sha256': sha(OLD / 'MANIFEST.json'),
        'stable_transport_sources': EXPECTED,
        'prior_CPU_not_reexecuted': str(OLD / 'evidence/RESULT.json'),
        'old_packages_modified': False,
    })
    print('ASSEMBLED_LOCAL_ONLY_NOT_SEALED')


if __name__ == '__main__':
    main()
