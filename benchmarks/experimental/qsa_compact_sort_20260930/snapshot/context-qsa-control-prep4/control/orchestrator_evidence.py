"""Explicit CPU recovery inventory; final unit cleanup is a separate receipt."""

import os
from pathlib import Path
import shlex
import sys
import time

import step58_control as c


def state(unit):
    return dict(row.split('=', 1) for row in c.output('systemctl', 'show', unit,
        '-p', 'LoadState', '-p', 'ActiveState', '-p', 'SubState', '-p', 'MainPID',
        '-p', 'ControlGroup', '-p', 'MemoryMax', '-p', 'MemorySwapMax',
        '-p', 'RuntimeMaxUSec', '-p', 'AllowedCPUs', '-p', 'ExecStart').splitlines())


def ended(row):
    return (row['ActiveState'] in ('inactive', 'failed')
            and int(row.get('MainPID', '0')) == 0)


def cuda_hidden(environment):
    values = dict(row.split('=', 1) for row in shlex.split(environment)
                  if '=' in row)
    return values.get('CUDA_VISIBLE_DEVICES') == ''


def main():
    os.umask(0o077)
    mode = sys.argv[1]
    if mode not in ('restore', 'post'):
        raise ValueError('restore/post required')
    outer = state(c.OUTER)
    units = {unit: state(unit) for unit in c.UNITS}
    timer = state(c.TIMER + '.timer')
    active = c.output('systemctl', 'list-units', '--state=active', '--no-legend',
        '--no-pager', 'step*', 'perf-profile*', '*guardian*',
        'flash-next-recovery-orchestrator-*')
    active_names = [row.split()[0] for row in active.splitlines() if row.strip()]
    gpu = c.output('nvidia-smi', '--query-compute-apps=pid,process_name',
                   '--format=csv,noheader')
    processes = []
    for row in gpu.splitlines():
        pid, name = row.split(',', 1)
        pid = int(pid.strip())
        processes.append({'pid': pid, 'name': name.strip(),
            'in_original_cgroup': '/system.slice/flash-next-vllm.service'
                in (Path('/proc') / str(pid) / 'cgroup').read_text()})
    gates = {'owned_pilots_ended': all(ended(row) for row in units.values()),
        'expiry_timer_inactive': timer['ActiveState'] == 'inactive',
        'all_GPU_original': len(processes) == 4 and
            all(row['in_original_cgroup'] for row in processes)}
    fds = {}
    if mode == 'restore':
        gates['exact_CPU_orchestrator_active'] = (
            outer['ActiveState'] == 'active' and int(outer['MainPID']) > 0
            and 'control/run_and_restore.sh' in outer['ExecStart'])
        gates['no_other_experiment_or_orchestrator'] = active_names == [c.OUTER]
        env = c.output('systemctl', 'show', c.OUTER, '-p', 'Environment', '--value')
        gates['orchestrator_CUDA_hidden'] = cuda_hidden(env)
        cg = Path('/sys/fs/cgroup') / outer['ControlGroup'].lstrip('/')
        for pid in (cg / 'cgroup.procs').read_text().split():
            leaked = False
            try:
                descriptors = list((Path('/proc') / pid / 'fd').iterdir())
            except FileNotFoundError:
                continue
            for fd in descriptors:
                try:
                    leaked |= os.readlink(fd) == (
                        '/home/l/.local/state/host-insight-mcp/gpu-maintenance.lock')
                except FileNotFoundError:
                    pass
            fds[pid] = {'maintenance_lock_inherited': leaked}
        gates['CPU_recovery_does_not_hold_maintenance_lock'] = not any(
            row['maintenance_lock_inherited'] for row in fds.values())
        path = c.D / 'restoration/owned-cpu-orchestrator.json'
        meaning = 'OWNED_CPU_RECOVERY_ACTIVE; not an all-jobs-ended claim'
    else:
        gates['owned_CPU_outer_ended'] = ended(outer)
        gates['no_active_experiment_or_orchestrator'] = not active_names
        gates['global_artifact_budget'] = c.artifact_budget('post-outer') == 0
        path = c.D / 'control/post-outer-cleanup.json'
        meaning = 'INDEPENDENT_POST_UNIT_CLEANUP'
    receipt = {'status': 'PASS' if all(gates.values()) else 'FAIL',
        'scope': meaning, 'epoch': time.time(), 'gates': gates, 'outer': outer,
        'pilots': units, 'timer': timer, 'active_units_raw': active,
        'GPU_processes': processes, 'CPU_cgroup_fds': fds}
    c.save(path, receipt)
    print(receipt['status'] + ' ' + meaning)
    raise SystemExit(0 if receipt['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
