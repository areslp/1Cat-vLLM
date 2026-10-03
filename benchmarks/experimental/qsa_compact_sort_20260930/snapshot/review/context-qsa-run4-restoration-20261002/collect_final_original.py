"""Read-only post-window context-QSA A0 resource diagnostic restoration collection.

Adapted from collect_final_original.rev2.py. No service mutations or completions.
Only main() performs live read-only health/metrics, /proc, systemd and NVML reads.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import time

B = Path('/home/l/work/flash-next/perf-20260923-resume/'
         'step58-qsa-compact-sort-20260930')
D = B / 'context-qsa-run4'
H = B / 'review/context-qsa-run4-restoration-20261002'
SOURCE_SHA = '9e6544def28b62cd8b8ad8cc2d3fcd524a184fc495852058129ad588ffbfce11'
OLD_API = 166245
OLD_INV = 'e3ca051687274273a7c61b0861841142'
OLD_WORKERS = {166757, 166758, 166759, 166760}
GATES = {
    'controller_restore_exit', 'readiness_maturity', 'stable_pid', 'no_restarts',
    'source_config_kv_swap_cleanup_worker', 'original_unit', 'requests',
    'six_fast_rounds', 'all_24_fast', 'fixed_exact', 'finite_concurrency',
    'e7_rank_equal_compressed', 'no_active_jobs', 'no_guardian', 'no_oom',
    'timer_cleared', 'cgroup_oom_zero',
}
RESTORE_EXITS = (
    'restoration/verify.exit', 'restoration/evaluate.exit',
    'control/post-outer.exit', 'control/artifact-final.exit',
)
EXPERIMENT_EXITS = ('control/window.exit', 'control/outer-launch.exit')
OPTIONAL_EXITS = (
    'control/ROOT-LAUNCH-EXIT.txt', 'control/attempt1/outer-cleanup.exit',
    'control/attempt1/outer-start.exit',
)


def require(value, message):
    if not value:
        raise ValueError(message)


def check_exits(exits):
    for name in RESTORE_EXITS:
        require(type(exits.get(name)) is int and exits[name] == 0,
                'strict restoration exit: ' + name)
    for name in EXPERIMENT_EXITS:
        require(type(exits.get(name)) is int and 0 <= exits[name] <= 255,
                'missing/invalid terminal experiment exit: ' + name)
    for name in OPTIONAL_EXITS:
        require(exits.get(name) is None or (
            type(exits[name]) is int and 0 <= exits[name] <= 255),
            'invalid optional native exit: ' + name)


def identity(receipt, after, systemd):
    require(receipt['status'] == 'PASS' and set(receipt['gates']) == GATES
            and all(v is True for v in receipt['gates'].values()),
            'strict named 17-gate restoration receipt')
    pid = receipt['pid']
    require(type(pid) is int and pid > 0 and pid != OLD_API,
            'RESTORATION must name a new API PID')
    require(systemd['MainPID'] == str(pid)
            and after['systemd']['MainPID'] == str(pid),
            'RESTORATION, fresh systemd and closure PID differ')
    inv = systemd['InvocationID']
    require(len(inv) == 32 and all(c in '0123456789abcdef' for c in inv)
            and inv != OLD_INV, 'fresh InvocationID must differ from original')
    require(systemd['ActiveState'] == 'active' and systemd['SubState'] == 'running'
            and systemd['NRestarts'] == '0', 'fresh original unit is not stable')
    workers = sorted(after['workers'], key=lambda row: row['rank'])
    require([row['rank'] for row in workers] == [0, 1, 2, 3], 'four exact ranks')
    pids = tuple(row['pid'] for row in workers)
    require(all(type(p) is int and p > 0 for p in pids)
            and len(set(pids)) == 4 and pid not in pids
            and not set(pids) & (OLD_WORKERS | {OLD_API}),
            'restored four worker PIDs must all be new')
    return pid, inv, pids


def main():
    H.mkdir(parents=True, exist_ok=True)
    names = ('FINAL-ORIGINAL-BINDING.json', 'FINAL-ORIGINAL-OBSERVED.json',
             'FINAL-ORIGINAL-SYSTEMD.json', 'FINAL-ORIGINAL-READBACK.json')
    require(not any((H / name).exists() for name in names),
            'create-only final collector outputs already exist')
    require(Path(__file__).resolve() == H / 'collect_final_original.py',
            'collector must run from its reviewed deployed path')
    pins = {str(Path(__file__).resolve()):
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def read(path, json_value=True):
        raw = path.read_bytes()
        pins[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw) if json_value else raw.decode().strip()

    exits = {name: int(read(D / name, False)) for name in
             (*RESTORE_EXITS, *EXPERIMENT_EXITS)}
    exits.update({name: int(read(D / name, False)) if (D / name).exists()
                  else None for name in OPTIONAL_EXITS})
    check_exits(exits)
    receipt_path = D / 'restoration/RESTORATION.json'
    restored = read(receipt_path)
    after = read(D / 'restoration/original-check/after-snapshot.json')
    before = read(D / 'restoration/original-check/before-snapshot.json')
    source = B / 'mixed-order-diagnostic-control1/collect_binding.py'
    require(hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA,
            'frozen read-only collector SHA')
    pins[str(source)] = SOURCE_SHA
    spec = importlib.util.spec_from_file_location('context_final_original', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fields = ('MainPID', 'InvocationID', 'ActiveState', 'SubState', 'NRestarts')
    fresh = dict(line.split('=', 1) for line in module.command(
        'systemctl', 'show', module.UNIT,
        *['-p' + field for field in fields]).splitlines())
    pid, inv, pids = identity(restored, after, fresh)
    require(before['systemd']['MainPID'] == str(pid)
            and tuple(row['pid'] for row in sorted(
                before['workers'], key=lambda row: row['rank'])) == pids,
            'before/after closure identity differs')
    module.API, module.INV, module.PIDS = pid, inv, pids
    binding, observed = module.collect()
    require(binding['api_pid'] == pid and binding['invocation_id'] == inv
            and tuple(row['pid'] for row in binding['workers']) == pids,
            'actual live original binding differs')
    require(identity(restored, after, observed['systemd']) == (pid, inv, pids),
            'fresh systemd changed during read-only collection')
    require(read(receipt_path) == restored, 'RESTORATION changed during collection')
    result = {
        'status': 'PASS_FINAL_ORIGINAL_INDEPENDENT_IDENTITY', 'unix': time.time(),
        'pid': pid, 'invocation_id': inv, 'workers': list(pids),
        'old_api_pid': OLD_API, 'old_invocation_id': OLD_INV,
        'old_workers': sorted(OLD_WORKERS),
        'selected_API_flags': 'PASS_INDEPENDENT_RAW_PROC_READ',
        'raw_snapshot_flags_boolean': restored['raw_flags_boolean'],
        'native_exits': exits,
        'experiment_exit_status': ('NONZERO_EXPERIMENT_OR_OUTER_RETAINED'
            if any(exits[n] != 0 for n in EXPERIMENT_EXITS)
            else 'ZERO_WINDOW_AND_OUTER'),
        'no_unit_mutations': True, 'no_generation_requests': True, 'pins': pins,
    }
    for name, value in zip(names, (binding, observed, fresh, result)):
        with (H / name).open('x') as handle:
            json.dump(value, handle, indent=2)
            handle.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'pins'}))


if __name__ == '__main__':
    main()
