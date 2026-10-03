"""Create-only local controller3 clone and explicit time-only amendment."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
OLD = BASE / 'service-w2-control-prep3'
REMOTE = Path('/home/l/work/flash-next/perf-20260923-resume/step58-qsa-compact-sort-20260930')
OLD_MANIFEST = '59cc6850c6f35e66db4a7fc48aa5752b8dda262e8efc4e7bbd6702b4153a523b'

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
def publish(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(row, stream, indent=2, sort_keys=True)
        stream.write('\n')

def main():
    assert sha(OLD / 'manifest.json') == OLD_MANIFEST
    original = json.loads((OLD / 'manifest.json').read_text())['files']
    historical = {'CPU-RECEIPT.json', 'contract.json', 'command-contract.json',
                  'prepare_revision3.py', 'freeze_revision3.py'}
    mapping = {}
    for row in original:
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('foreign original payload')
        source = OLD / relative
        if source.is_symlink() or source.stat().st_size != row['bytes'] or sha(source) != row['sha256']:
            raise ValueError('old immutable source changed: ' + str(relative))
        dest = ROOT / ('evidence/previous-source-packet3' if str(relative) in historical else '') / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('xb') as stream:
            stream.write(source.read_bytes())
        mapping[str(dest.relative_to(ROOT))] = row
    dest = ROOT / 'evidence/previous-source-packet3/manifest.json'
    with dest.open('xb') as stream:
        stream.write((OLD / 'manifest.json').read_bytes())
    identities = [('service-w2-control-prep3', 'service-w2-control-prep4'),
        ('service-w2-run3', 'service-w2-run4'), ('step58-w2-3', 'step58-w2-4'),
        ('{arm.lower()}3.service', '{arm.lower()}4.service'),
        ('step58-w2-http-b-stability3.service', 'step58-w2-http-b-stability4.service'),
        ('step58-w2-a03.service', 'step58-w2-a04.service'),
        ('step58-w2-b3.service', 'step58-w2-b4.service'),
        ('step58-w2-a23.service', 'step58-w2-a24.service'),
        ('step58-w2-http-a03.service', 'step58-w2-http-a04.service'),
        ('step58-w2-http-b3.service', 'step58-w2-http-b4.service'),
        ('step58-w2-http-a23.service', 'step58-w2-http-a24.service')]
    active = [ROOT / name for name in ('common.py', 'service_binding.py')]
    active += list((ROOT / 'control').glob('*.py')) + list((ROOT / 'control').glob('*.sh'))
    active += [ROOT / 'restoration/verify_original.sh']
    for path in active:
        source = path.read_text()
        for old, new in identities:
            source = source.replace(old, new)
        path.write_text(source)
    old_contract_path = BASE / 'service-w2-prep/contract.json'
    assert sha(old_contract_path) == 'ba1373f35f7f7c8b2a041520c5546470ebdbddff27b137d818d916ad26ebaf53'
    old_budgets = json.loads(old_contract_path.read_text())['initial_budgets_seconds']
    replacements = {'matrix_per_arm': 3600, 'inner': 15300, 'expiry': 15420, 'outer': 18300}
    effective = old_budgets | replacements
    evidence_path = ROOT / 'evidence/budget-draft1/actual-cycle-evidence.json'
    revision = {
        'schema': 'step58-w2-execution-time-amendment-v1',
        'scope': 'Only four execution time budgets; fixed original contract/matrix/statistics/counts/resources/per-request deadlines unchanged',
        'original_contract': {'name': 'contract.json', 'sha256': sha(old_contract_path)},
        'original_matrix': {'name': 'matrix.frozen.json', 'sha256': '41cd10f72fcafe849d39756cfe12960354cf89f8a33abb483ff5b01ed3e9cc9a'},
        'original_budgets_seconds': old_budgets, 'replacements_seconds': replacements,
        'effective_budgets_seconds': effective,
        'evidence': {'relative_path': str(evidence_path.relative_to(ROOT)), 'sha256': sha(evidence_path)},
        'estimate': {'conservative_observed131_full_cycle_sum_s': 2084,
            'remaining_shared3_s': 3 * 21, 'unobserved_mixed4_proxy_s': 4 * (2 * 36 + 18),
            'unobserved_cancel4_proxy_s': 4 * 15, 'empirical_full_arm_estimate_s': 2567,
            'new_matrix_margin_over_estimate_s': 1033,
            'limitations': 'Not a worst-case completion guarantee; file intervals have rsync timestamp precision limits; mixed/cancel proxies were not observed in run3'},
        'time_audit': {'outer_chain_s': 18120, 'outer_margin_s': 180,
            'outer_terms_s': [15300, 20, 555, 25, 2120, 65, 35],
            'expiry_restore_body_and_trap_upper_s': 452,
            'expiry_TimeoutStartSec_s': 470, 'expiry_TimeoutStopSec_s': 30,
            'expiry_timer_minutes': 257,
            'expiry_offset_after_inner_s': 120},
        'new_HTTP_requests': 0, 'partial_run3_reused': False,
        'launch_authorized_by_this_file': False}
    publish(ROOT / 'execution-budget-revision.json', revision)
    # Actual frozen source contracts remain intact; only execution consumers change.
    matrix = ROOT / 'matrix.py'
    source = matrix.read_text()
    assert source.count('deadline = began + 1800') == 1
    source = source.replace('from frozen import require',
                            'from frozen import require\nfrom execution_budget import BUDGETS')
    source = source.replace('deadline = began + 1800', "deadline = began + BUDGETS['matrix_per_arm']")
    matrix.write_text(source)
    ops = ROOT / 'ops.py'
    source = ops.read_text()
    source = source.replace('from common import ', 'from execution_budget import BUDGETS\nfrom common import ', 1)
    assert source.count("script, duration, unit = 'matrix.py', 1800, HTTP[arm]") == 1
    source = source.replace("script, duration, unit = 'matrix.py', 1800, HTTP[arm]",
                            "script, duration, unit = 'matrix.py', BUDGETS['matrix_per_arm'], HTTP[arm]")
    source = source.replace('inside the9900s inner cap', 'inside the15300s inner cap')
    ops.write_text(source)
    for relative, old, new in (
        ('control/run_and_restore.sh', '20 9900 bash', '20 15300 bash'),
        ('control/offline_window.sh', '--on-active=167m', '--on-active=257m'),
        ('control/launch_outer.sh', 'RuntimeMaxSec=12900s', 'RuntimeMaxSec=18300s')):
        path = ROOT / relative
        source = path.read_text()
        assert source.count(old) == 1
        path.write_text(source.replace(old, new))
    publish(ROOT / 'evidence/initial-copy-map4.json', {
        'old_manifest_sha256': OLD_MANIFEST, 'copied_files': mapping,
        'identity_replacements': identities,
        'new_amendment_sha256': sha(ROOT / 'execution-budget-revision.json')})
    print(json.dumps({'status': 'LOCAL_PREP4_CLONE_WITH_TIME_AMENDMENT_NOT_HOST_STAGE',
        'copied_original_payloads': len(mapping),
        'amendment_sha256': sha(ROOT / 'execution-budget-revision.json'), 'effective': effective}))

if __name__ == '__main__':
    main()
