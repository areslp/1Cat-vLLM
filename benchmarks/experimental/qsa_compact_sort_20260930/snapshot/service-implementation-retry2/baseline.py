"""Create-only shadow config derivation: only baseline_off_receipts may differ."""
import argparse
import json
from pathlib import Path
from io_contract import save, sha256, require_hash, validate_config

BINDING_FIELDS = ('rank', 'path', 'sha256', 'pid', 'config_sha256',
                  'candidate_binary_sha256', 'original_binary_sha256',
                  'device_uuid', 'kv_num_blocks')


def derive(template, bindings):
    validate_config(template)
    assert template['mode'] == 'shadow' and template['baseline_off_receipts'] == []
    assert sorted(r['rank'] for r in bindings) == [0, 1, 2, 3]
    assert len({r['pid'] for r in bindings}) == 4
    assert len({r['device_uuid'] for r in bindings}) == 4
    for b in bindings:
        assert set(b) == set(BINDING_FIELDS)
        path = Path(template['baseline_off_capture_dir']) / f"capture-ready-rank{b['rank']}.json"
        assert Path(b['path']).resolve() == path.resolve()
        require_hash(path, b['sha256'])
        off = json.loads(path.read_text())
        for key in ('rank', 'pid', 'config_sha256', 'candidate_binary_sha256',
                    'original_binary_sha256', 'kv_num_blocks'):
            assert off[key] == b[key]
        assert off['runtime']['device_uuid'] == b['device_uuid']
        assert off['mode'] == 'off' and off['status'] == 'CAPTURE_READY_SERVICE_UNVERIFIED'
        assert off['source_pins'] == template['source_pins']
        assert off['candidate_binary_sha256'] == template['candidate_binary_sha256']
        assert off['original_binary_sha256'] == template['original_binary_sha256']
        assert off['private_bytes'] == off['failure_bank_bytes'] == 0
        assert off['counter_shape'] is None
        assert len(off['nodes']) == 36 and all(n['planner'] == 'original' for n in off['nodes'])
    runtime = {**template, 'baseline_off_receipts': bindings}
    assert {k: v for k, v in runtime.items() if k != 'baseline_off_receipts'} == {
        k: v for k, v in template.items() if k != 'baseline_off_receipts'}
    return runtime


def main():
    p = argparse.ArgumentParser(); p.add_argument('--template', required=True)
    p.add_argument('--bindings', required=True); p.add_argument('--output', required=True)
    p.add_argument('--receipt', required=True); a = p.parse_args()
    cfg = derive(json.loads(Path(a.template).read_text()), json.loads(Path(a.bindings).read_text()))
    save(a.output, cfg)
    save(a.receipt, {'status': 'PASS_ONLY_BASELINE_BINDINGS_DERIVED',
         'template_path': str(Path(a.template).resolve()), 'template_sha256': sha256(a.template),
         'bindings_path': str(Path(a.bindings).resolve()), 'bindings_sha256': sha256(a.bindings),
         'runtime_path': str(Path(a.output).resolve()), 'runtime_sha256': sha256(a.output),
         'changed_fields': ['baseline_off_receipts'], 'bindings': cfg['baseline_off_receipts']})


if __name__ == '__main__': main()
