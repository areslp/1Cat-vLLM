"""CPU-only, streaming SHA + strict header preflight. No tensor loads."""
import argparse
import json
from pathlib import Path
import time
import traceback

from headers import read_header, stream_sha

ROOT = Path('/home/l/work/flash-next/perf-20260923-resume')
CPU_CAP = 3 * 1024 ** 3
GPU_CAP = 2 * 1024 ** 3


def inspect(path):
    row = read_header(path)
    headers, meta = row['tensor_headers'], row['meta']
    required = ('logical_indices', 'block_table', 'token_to_req',
                'query_positions', 'sequence_lengths', 'q', 'ids',
                'k_blocks', 'v_blocks', 'out', 'lse', 'grouped_pages',
                'token_masks', 'grouped_seq_lens')
    assert set(required) <= set(headers), f'missing tensor fields {path}'
    rows = meta['rows']
    assert headers['q']['shape'] == [rows, 6, 256]
    assert rows % 8 == 0 and rows == headers['logical_indices']['shape'][0]
    assert headers['logical_indices']['shape'][1] == 2051
    assert headers['k_blocks']['shape'] == headers['v_blocks']['shape']
    assert headers['k_blocks']['shape'][1:] == [4, 1, 256]
    assert headers['ids']['shape'] == [headers['k_blocks']['shape'][0]]
    assert meta['kind'] in ('prefill', 'verify')
    assert meta['pad_fix'] is False, 'unexpected production padding flag'
    assert meta['kv_cache_dtype'] == 'fp8_e4m3'
    assert headers['k_blocks']['element_width'] == 1
    assert row['retained_storage_bytes'] <= CPU_CAP
    # Shared Q/KV/input + four guarded metadata/workspace branches and remap
    # temporaries. A conservative estimate; CUDA allocated/reserved hard gate
    # remains mandatory because graph allocator lifetime is not proven here.
    groups = rows // 8
    plan_bytes = (groups + 2) * 4160 * 8 + (groups + 2) * 4
    result_bytes = (rows + 2) * 6 * (256 * 2 + 4)
    input_bytes = sum(headers[k]['bytes'] for k in required[:10]
                      if k not in ('out',))
    estimate = input_bytes + 4 * (plan_bytes + result_bytes)
    estimate += 8 * groups * 4160 * 8 + 64 * 1024 ** 2
    assert estimate <= GPU_CAP, f'GPU preallocation estimate exceeds cap {path}'
    return {'path': str(path), 'sha256': stream_sha(path), **row,
            'gpu_live_estimate_bytes': estimate,
            'service_role': 'UNVERIFIED_TARGET_DRAFT',
            'scope': ('POTENTIAL_VERIFY_INTEGER_STRESS'
                      if meta['kind'] == 'verify'
                      else 'PREFILL_INTEGER_STRESS_ORIGINAL_PACKED')}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    a = p.parse_args()
    destination = Path(a.output)
    assert not destination.exists()
    result = {'status': 'RUNNING', 'start': time.time(), 'captures': [],
              'tensor_loads': 0, 'torch_imported': False}
    try:
        paths = sorted((ROOT / 'step48-w0/cap48').glob('cap-*.pt'))
        paths += sorted((ROOT / 'step48-w1cap/cap48').glob('qsa-*.pt'))
        assert len(paths) == len(set(paths)) == 194
        for path in paths:
            result['captures'].append(inspect(path))
        counts = {kind: sum(r['meta']['kind'] == kind
                           for r in result['captures'])
                  for kind in ('verify', 'prefill')}
        assert counts == {'verify': 180, 'prefill': 14}
        result.update(status='PASS_HEADERS_AND_HASHES_NOT_GPU', counts=counts,
                      maximum_cpu_storage_bytes=max(
                          r['retained_storage_bytes'] for r in result['captures']),
                      maximum_gpu_live_estimate_bytes=max(
                          r['gpu_live_estimate_bytes'] for r in result['captures']))
    except BaseException:
        result.update(status='FAIL', error=traceback.format_exc())
    finally:
        result['finished'] = time.time()
        with destination.open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'captures'}))
    return int(result['status'] == 'FAIL')


if __name__ == '__main__':
    raise SystemExit(main())
