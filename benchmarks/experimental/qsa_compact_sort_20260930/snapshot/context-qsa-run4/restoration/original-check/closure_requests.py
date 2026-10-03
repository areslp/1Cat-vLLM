"""STEP-53 original-load recovery gate. No production writes or service restarts."""
import datetime
import hashlib
import json
from pathlib import Path
import statistics as st
import sys
import time
import traceback

import bench10 as b
import client as c
import perf38 as p
import snapshot as s
from snapshot import snapshot
from exporter_fence import wait_fence

D = Path(__file__).resolve().parent
REF = Path('/home/l/work/flash-next/perf-20260923-resume/step51-e7-service-20260930')
OFFSETS = (0, 1024, 2560, 4096)


def save(name, value):
    b.save(D / name, value)


def idle():
    raw = b.metrics()
    vals = [float(x.rsplit(' ', 1)[1]) for x in raw.splitlines()
            if x.startswith(('vllm:num_requests_running{', 'vllm:num_requests_waiting{'))]
    assert len(vals) == 2 and not any(vals), 'not idle'


def fenced_workers(name, baseline, last_completed):
    expected = baseline['workers']

    def check_idle():
        idle()
        actual = s.out('systemctl', 'show', s.S, '-p', 'MainPID',
                       '--value').strip()
        assert actual == baseline['systemd']['MainPID'], 'original PID changed'
        return time.time()

    def read_rows():
        pids = {int(x) for x in (s.CG / 'cgroup.procs').read_text().split()}
        stats = Path(s.BASE['selected_flags']['ONECAT_E7_STATS_DIR'])
        rows = [json.loads(f.read_text()) for f in stats.glob('e7-*.json')]
        return [row for row in rows if row['pid'] in pids]

    with (D / (name + '-candidates.jsonl')).open('x') as trace:
        def record(value):
            trace.write(json.dumps(value) + '\n')
            trace.flush()

        workers = wait_fence(name, expected, read_rows, check_idle, save,
                             last_completed, record)
    observed = snapshot(name)
    assert observed['systemd']['MainPID'] == baseline['systemd']['MainPID']
    assert [r['counters'] for r in observed['workers']] == [
        r['counters'] for r in workers], 'counter changed after passive fence'
    return workers


def timing(row):
    ts = [x[0] for x in row['chunks']]
    return st.median(1000 * (y - x) for x, y in zip(ts[1:], ts[2:]))


def getref(mode, rep, off):
    name = f'P-final-greedy-d1-o{off}.json' if mode == 'G1' else f'P-final-{rep}-N1-o{off}.json'
    path = REF / name
    return path, json.loads(path.read_text())


def run():
    before = json.loads((D / 'before-snapshot.json').read_text())
    assert before['checks']['health'] and before['checks']['source_config']
    assert before['checks']['kv_capacity'] and before['checks']['cleanup']
    result = {'status': 'RUNNING', 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'pid': before['systemd']['MainPID'], 'requests': [], 'rounds': [], 'concurrent': [], 'fixed': []}
    try:
        idle()
        for mode in ('G1', 'N1'):
            if mode == 'G1':
                b.isolated('warmup', mode, p.tok(0), 256)
            else:
                p.iso('warmup', mode, p.tok(0), p.N, True)
        for rep in range(3):
            for mode in (('G1', 'N1') if rep % 2 == 0 else ('N1', 'G1')):
                rr = []
                for off in OFFSETS:
                    idle()
                    label = f'{mode}-o{off}'
                    refpath, ref = getref(mode, rep, off)
                    raw_before = b.metrics()
                    (D / f'round{rep}-{label}.metrics_before.txt').write_text(raw_before)
                    if mode == 'G1':
                        row = b.isolated(f'round{rep}', label, p.tok(off), 256)
                    else:
                        row = p.iso(f'round{rep}', label, p.tok(off), p.N, True)
                    (D / f'round{rep}-{label}.metrics_after.txt').write_text(b.metrics())
                    ms = timing(row)
                    ref_ms = timing(ref)
                    spec = b.KEYS[4:]
                    checks = {'request': all(row['checks'].values()),
                              'usage': row['usage']['prompt_tokens'] == 512 and row['usage']['completion_tokens'] == 256,
                              'tokens': row['output_token_ids'] == ref['output_token_ids'],
                              'acceptance': all(row['metrics_delta'][k] == ref['metrics_delta'][k] for k in spec)}
                    item = {'rep': rep, 'mode': mode, 'offset': off, 'step_ms': ms, 'reference_step_ms': ref_ms,
                            'fast': ms <= ref_ms * 1.01 and (mode == 'G1' or ms < 34.0),
                            'checks': checks, 'reference': str(refpath), 'reference_sha256': hashlib.sha256(refpath.read_bytes()).hexdigest()}
                    result['requests'].append(item)
                    rr.append(item)
                    save(f'round{rep}-{label}-comparison.json', item)
                    if not all(checks.values()):
                        raise RuntimeError('Output/acceptance divergence: ' + label)
                measured = st.median(x['step_ms'] for x in rr)
                reference = st.median(x['reference_step_ms'] for x in rr)
                summary = {'rep': rep, 'mode': mode, 'median_ms': measured, 'reference_ms': reference,
                           'delta_pct': 100 * (measured / reference - 1), 'fast_count': sum(x['fast'] for x in rr),
                           'slow_count': sum(not x['fast'] for x in rr),
                           'pass': measured <= reference * 1.01 and (mode == 'G1' or measured < 34)}
                result['rounds'].append(summary)
                print(json.dumps(summary), flush=True)
        # Limited matched c4/c8 sampling; exact same prompts per request as P-matched.
        workers0 = fenced_workers('before-concurrent', before, time.time())
        for n in (4, 8):
            for rep in range(2):
                idle()
                row = p.grp('matched', f'N{n}-{rep}', [p.tok(1024 * (rep % 2))] * n, p.N, True)
                result['concurrent'].append({'n': n, 'rep': rep, 'step_ms': st.median(x['step_ms'] for x in row['rows']),
                                             'checks': row['checks'], 'metrics_delta': row['metrics_delta']})
        workers1 = fenced_workers('after-concurrent', before, time.time())
        result['e7_delta'] = [{'rank': a['rank'], 'pid': a['pid'], 'counters': {
            k: a['counters'].get(k, 0) - z['counters'].get(k, 0)
            for k in set(a['counters']) | set(z['counters'])}} for a, z in zip(workers1, workers0)]
        e7 = result['e7_delta']
        result['e7_pass'] = [x['rank'] for x in e7] == list(range(4)) and len({json.dumps(x['counters'], sort_keys=True) for x in e7}) == 1 and all(x['counters'].get('compressed_steps', 0) > 0 for x in e7)
        result['e7_pass'] &= all(v >= 0 for r in e7 for v in r['counters'].values())
        ids = c.prompt()
        fixed_ref = json.loads((D / 'reference-outputs.json').read_text())['A']
        for mode in ('natural', 'full', 'mask'):
            idle()
            if mode == 'natural':
                row = c.run('fixed-natural', ids, 16, logprobs=10)
            elif mode == 'full':
                row = c.run('fixed-full', ids + fixed_ref[:5], 1, logprobs=10)
            else:
                row = c.run('fixed-mask', ids, 16, allowed=[1000])
            ref = json.loads((REF / f'P-fixed-{mode}.json').read_text())
            checks = {'tokens': row['output_token_ids'] == ref['output_token_ids'],
                      'logprobs': row['logprobs'] == ref['logprobs'],
                      'acceptance': all(row['metrics_delta'][k] == ref['metrics_delta'][k] for k in b.KEYS[4:])}
            result['fixed'].append({'mode': mode, 'checks': checks})
            assert all(checks.values()), 'fixed divergence: ' + mode
        result['request_pass'] = all(all(x['checks'].values()) for x in result['requests'])
        result['fast_state_pass'] = len(result['rounds']) == 6 and all(x['pass'] for x in result['rounds'])
        result['status'] = 'PASS' if result['request_pass'] and result['fast_state_pass'] and result['e7_pass'] else 'BLOCKED'
    except BaseException:
        result['status'] = 'BLOCKED'
        result['error'] = traceback.format_exc()
    finally:
        result['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save('requests-result.json', result)
        snapshot('after')
    print(json.dumps({k: v for k, v in result.items() if k not in ('requests', 'concurrent', 'fixed', 'e7_delta')}), flush=True)
    return result['status'] == 'PASS'


if __name__ == '__main__':
    raise SystemExit(not run())
