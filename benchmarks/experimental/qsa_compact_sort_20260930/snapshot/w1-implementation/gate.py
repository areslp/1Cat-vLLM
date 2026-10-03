"""Small CPU completeness gate, distinct from service admission."""
import argparse
import json
from pathlib import Path

from fixtures import BOUNDARIES, TRANSITIONS, VARIANTS, WITNESSES
from headers import stream_sha


def check(result, manifest):
    assert result['status'] == 'OFFLINE_W1_PASS_SERVICE_UNVERIFIED'
    assert result['service_admission'] == 'UNVERIFIED'
    assert result['performance_claim'] is False
    assert [len(result[key]) for key in ('captures', 'synthetic', 'boundaries',
                                        'witnesses')] == [194, 72, 9, 5]
    assert [row['path'] for row in result['captures']] == [
        row['path'] for row in manifest['captures']]
    assert all(row['sha256'] == pinned['sha256']
               for row, pinned in zip(result['captures'], manifest['captures']))
    assert all(row['reference_reproduces_saved_effective_metadata_and_out_lse']
               for row in result['captures'])
    for key in ('captures', 'synthetic', 'boundaries', 'witnesses'):
        for row in result[key]:
            assert row['pass']
            assert [phase['phase'] for phase in row['phases']] == ['A', 'B', 'A', 'A']
            expected = 'split' if row['rows'] <= 64 else 'packed'
            for phase in row['phases']:
                assert phase['all_full_buffers_bit_exact']
                assert phase['input_unchanged']
                assert phase['original_forward_routes'] == [expected] * 4
    assert [row['case'] for row in result['boundaries']] == [f'bucket{n}' for n in BOUNDARIES]
    assert [row['case'] for row in result['witnesses']] == ['witness-' + n for n in WITNESSES]
    assert [row['hash_entries'] for row in result['full_chain_bucket_transitions']['steps']] == list(TRANSITIONS)
    assert result['full_chain_bucket_transitions']['pass']
    assert result['workspace']['pass']
    assert [row['capacity'] for row in result['workspace']['growth_reuse_stream_slices']] == [1, 4, 8, 8] * 2
    assert [row['case'] for row in result['synthetic']] == [
        f'seed{541000 + g * 100 + index}-g{g}-{variant}'
        for g in range(1, 9) for index, variant in enumerate(VARIANTS)]
    stats = result['memory']
    assert stats['gpu_allocated'] <= 2147483648
    assert stats['gpu_reserved'] <= 3221225472
    assert stats['cpu_peak_rss'] <= 3221225472
    assert stats['artifact_bytes'] <= 268435456
    return {'status': 'PASS_OFFLINE_W1_SERVICE_UNVERIFIED',
            'gpu': result['gpu'], 'counts': [194, 72, 9, 5, 11],
            'service_admission': False, 'performance_claim': False}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--result', required=True)
    p.add_argument('--input-manifest', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    result = json.loads(Path(a.result).read_text())
    manifest = json.loads(Path(a.input_manifest).read_text())
    try:
        verdict = check(result, manifest)
    except Exception as error:
        verdict = {'status': 'FAIL', 'error': repr(error),
                   'service_admission': False}
    verdict['result_sha256'] = stream_sha(a.result)
    with Path(a.output).open('x') as stream:
        json.dump(verdict, stream, indent=2)
        stream.write('\n')
    print(json.dumps(verdict))
    return int(verdict['status'] == 'FAIL')


if __name__ == '__main__':
    raise SystemExit(main())
