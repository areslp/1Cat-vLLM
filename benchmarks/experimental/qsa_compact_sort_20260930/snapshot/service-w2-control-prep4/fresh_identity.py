"""One read-only original PID/Invocation/starttime/worker binding for a plan."""
import argparse
from pathlib import Path
import re

from common import WINDOW, read, save
from original_guard import Observation
from identity_contract import gpu_identity, UUIDS
import ops


def validate_live(identity):
    pid = identity['api_pid']
    fields = dict(row.split('=', 1) for row in ops.native.output('systemctl', 'show',
        ops.native.S, '-pMainPID', '-pInvocationID', '-pNRestarts').splitlines())
    if (int(fields['MainPID']) != pid or fields['InvocationID'] != identity['invocation_id']
            or fields['NRestarts'] != '0'
            or Observation.pid_start(pid) != identity['api_starttime_ticks']):
        raise ValueError('live original API starttime/Invocation differs from approved plan')
    for worker in identity['workers']:
        if (Observation.pid_start(worker['pid']) != worker['starttime_ticks']
                or (Path('/proc') / str(worker['pid']) / 'cgroup').read_text().strip()
                != worker['cgroup_raw']):
            raise ValueError('live original worker identity differs from approved plan')
    gpu_identity(ops.native.output('nvidia-smi',
        '--query-compute-apps=pid,gpu_uuid,used_gpu_memory', '--format=csv,noheader'),
        ops.native.output('nvidia-smi', '--query-gpu=uuid,memory.used',
                          '--format=csv,noheader,nounits'), identity['workers'])


def capture(expected_pid):
    before = read(WINDOW / 'baseline/before-snapshot.json', 4 * 1024**2)
    fields = dict(row.split('=', 1) for row in ops.native.output('systemctl', 'show',
        ops.native.S, '-pMainPID', '-pInvocationID', '-pActiveState', '-pSubState',
        '-pNRestarts', '-pControlGroup', '-pMemoryCurrent', '-pMemoryPeak',
        '-pMemorySwapCurrent').splitlines())
    if (int(fields['MainPID']) != expected_pid or int(before['systemd']['MainPID']) != expected_pid
            or fields['ActiveState'] != 'active' or fields['SubState'] != 'running'
            or fields['NRestarts'] != '0'
            or not re.fullmatch('[0-9a-f]{32}', fields['InvocationID'])
            or any(before['queue'].values())):
        raise ValueError('actual fresh original identity/queue differs')
    workers = []
    for row in before['workers']:
        rank, pid = row['rank'], row['pid']
        if rank not in range(4):
            raise ValueError('original rank unknown')
        workers.append({'rank': rank, 'pid': pid,
            'starttime_ticks': Observation.pid_start(pid), 'physical_uuid': UUIDS[rank],
            'cgroup_raw': (Path('/proc') / str(pid) / 'cgroup').read_text().strip()})
    workers.sort(key=lambda row: row['rank'])
    if ([r['rank'] for r in workers] != list(range(4))
            or any(r['cgroup_raw'] != '0::/system.slice/' + ops.native.S for r in workers)):
        raise ValueError('fresh original worker cgroup/rank binding incomplete')
    gpu = gpu_identity(ops.native.output('nvidia-smi',
        '--query-compute-apps=pid,gpu_uuid,used_gpu_memory', '--format=csv,noheader'),
        ops.native.output('nvidia-smi', '--query-gpu=uuid,memory.used',
                          '--format=csv,noheader,nounits'), workers)
    value = {'status': 'FRESH_ORIGINAL_READ_ONLY_NO_REQUEST',
        'api_pid': expected_pid, 'api_starttime_ticks': Observation.pid_start(expected_pid),
        'invocation_id': fields['InvocationID'], 'systemd': fields,
        'workers': workers, 'actual_GPU_identity': gpu,
        'HEAD': before['head'], 'tracked_diff': before['tracked_diff'], 'queue': before['queue'],
        'native_raw_flags_boolean': before['checks']['flags']}
    save(WINDOW / 'baseline/fresh-identity.json', value)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--expected-pid', type=int, required=True)
    args = parser.parse_args()
    capture(args.expected_pid)
    print('FRESH_ORIGINAL_IDENTITY_BOUND_NO_GENERATION')
