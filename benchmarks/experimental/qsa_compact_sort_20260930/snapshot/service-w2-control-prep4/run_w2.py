"""One initial pure W2 A0/B/A2; no diagnostic model hook or automatic repeat."""
import argparse
import os
from pathlib import Path
import traceback

from common import BASE, PACKAGE, WINDOW, PY, read, save, sha, cpu_env
from flow import Flow
from service_binding import derive, write
import ops


def config(arm):
    arguments = {}
    if arm == 'B':
        startup_path = ops.native.A / 'A0/startup-gate.json'
        startup = read(startup_path)
        rows = []
        off_dir = Path(startup['service_capture_dir'])
        for rank in range(4):
            path = off_dir / f'capture-ready-rank{rank}.json'
            row = read(path)
            rows.append({key: row[key] for key in ('rank', 'pid', 'config_sha256',
                'candidate_binary_sha256', 'original_binary_sha256', 'kv_num_blocks')} |
                {'path': str(path), 'sha256': sha(path), 'device_uuid': row['runtime']['device_uuid']})
        arguments.update(off_bindings=rows, off_dir=off_dir,
                         off_startup={'path': str(startup_path), 'sha256': sha(startup_path)})
    path = WINDOW / 'control/configs' / (arm + '.service.json')
    write(path, derive(arm, WINDOW, **arguments))
    return path


def run():
    from plan import validate
    path = WINDOW / 'control/plan.json'
    expected = os.environ.get('STEP58_W2_APPROVED_PLAN_SHA256')
    if (not expected or sha(path) != expected
            or read(ops.native.A / 'preflight.json')['plan_sha256'] != expected
            or not (WINDOW / 'restoration/stopped.txt').exists()
            or not (ops.native.A / 'maintenance-lock-acquired.json').exists()):
        raise ValueError('only one parent-approved, stopped, locked exact W2 window')
    validate(read(path))
    flow = Flow()
    try:
        for arm in ('A0', 'B', 'A2'):
            flow.begin(arm)
            configuration = config(arm)
            startup = ops.start(arm, configuration)
            ops.native.phase('STEP58_W2_PHASE=' + arm + '_PURE_READY')
            ops.timed(arm)
            if arm == 'B':
                ops.timed(arm, 'stability')
            proof = ops.stop(arm, startup)
            flow.stop(arm, proof)
            ops.native.phase('STEP58_W2_PHASE=' + arm + '_STOPPED')
        flow.analyzed()
        result, decision_exit = ops.analyze()
        save(WINDOW / 'W2-RESULT.json', {'status': result['status'],
            'scope': 'initial matched groups only; confirmation/service admission not executed',
            'analysis_sha256': sha(WINDOW / 'analysis/RESULT.json'),
            'arms_started': flow.started, 'arms_stopped': flow.stopped,
            'analysis_actual_exit': decision_exit,
            'restoration': 'REQUIRED_BY_SURVIVING_OUTER_AFTER_THIS_RESULT'})
        return decision_exit
    except BaseException as error:
        flow.fail(error)
        save(WINDOW / 'W2-FAILURE.json', {'status': 'STOP_NO_RETRY_NO_REPLACEMENT',
            'error': repr(error)[:4096], 'traceback': traceback.format_exc()[-16384:],
            'arms_started': flow.started, 'arms_stopped': flow.stopped,
            'active_arm': flow.active,
            'restoration': 'REQUIRED_BY_SURVIVING_OUTER_BEFORE_ERROR_AGGREGATION'})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'run', 'drain', 'cleanup'))
    parser.add_argument('--plan')
    parser.add_argument('--expected-pid')
    parser.add_argument('--label', choices=('pre-original-stop', 'exit-final', 'outer-before-restore'))
    args = parser.parse_args()
    if args.action == 'prepare':
        from plan import validate
        path = Path(args.plan)
        if (path != WINDOW / 'control/plan.json'
                or sha(path) != os.environ.get('STEP58_W2_APPROVED_PLAN_SHA256')):
            raise ValueError('exact parent-approved initial W2 plan required')
        validate(read(path))
        ops.native.validate_plan = validate
        os.environ['STEP58_SERVICE_APPROVED_PLAN_SHA256'] = sha(path)
        ops.native.prepare(path, args.expected_pid)
        from fresh_identity import validate_live
        validate_live(read(WINDOW / 'baseline/fresh-identity.json'))
        save(ops.native.A / 'fresh-disk-resource.json',
             __import__('artifact_budget').free_space_preflight(WINDOW))
        props = dict(line.split('=', 1) for line in ops.native.output('systemctl', 'show',
            ops.native.S, '-pMemoryCurrent', '-pMemoryPeak', '-pMemorySwapCurrent').splitlines())
        if (any(int(props[key]) >= 120 * 1024**3 for key in ('MemoryCurrent', 'MemoryPeak'))
                or int(props['MemorySwapCurrent']) != 0):
            raise ValueError('fresh original current/peak/swap outside reviewed diagnostic cap')
        save(ops.native.A / 'fresh-original-resource.json', {
            'original_properties': props, 'experiment_MemoryMax_bytes': 120 * 1024**3,
            'original_cgroup_changed': False,
            'host_meminfo_raw': Path('/proc/meminfo').read_text(),
            'host_global_swap_zero_claim': False})
    elif args.action == 'drain':
        if args.label != 'pre-original-stop':
            raise ValueError('unique finite drain label required')
        ops.native.drain(args.label)
    elif args.action == 'cleanup':
        if args.label not in ('exit-final', 'outer-before-restore'):
            raise ValueError('unique finite cleanup label required')
        raise SystemExit(ops.cleanup(args.label, drain_gpu=args.label == 'exit-final'))
    else:
        raise SystemExit(run())


if __name__ == '__main__':
    main()
