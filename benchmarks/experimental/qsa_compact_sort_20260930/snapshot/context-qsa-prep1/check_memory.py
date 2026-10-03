"""Local stdlib peak observation; no network/model or target actions."""
import argparse
import importlib.util
import json
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)


def child(path):
    value = json.loads(Path(path).read_text())
    available = importlib.util.find_spec('requests') is not None
    if available:
        import requests
        import urllib3
        del requests, urllib3
    print(json.dumps({'pid': __import__('os').getpid(), 'peak_RSS_bytes': rss(),
                      'requests_urllib3_imported': available,
                      'prompt_tokens': len(value['body']['prompt'])}), flush=True)
    command = sys.stdin.readline(16)
    if command != 'STOP\n':
        raise ValueError('one owner STOP required')


def parent(output=None):
    matrix = json.loads((HERE / 'matrix.frozen.json').read_text())
    start = rss()
    max_tokens, max_file = 0, 0
    for pin in matrix['groups']:
        group = json.loads((HERE / pin['path']).read_text())
        max_tokens = max(max_tokens, sum(len(r['body']['prompt']) for r in group['requests'] + group['primes']))
        max_file = max(max_file, pin['bytes'])
        del group
    peak_load = rss()
    pin = next(g for g in matrix['groups'] if g['arm'] == 'A0' and g['ordinal'] == 0
               and '65536-greedy-c8' in g['row_id'])
    group = json.loads((HERE / pin['path']).read_text())
    children, observations = [], []
    with tempfile.TemporaryDirectory() as name:
        try:
            for index, packet in enumerate(group['requests']):
                path = Path(name) / f'packet{index}.json'
                path.write_text(json.dumps(packet))
                process = subprocess.Popen([sys.executable, '-I', '-B', str(Path(__file__).resolve()),
                                            '--child', str(path)], stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                children.append(process)
            for process in children:
                observations.append(json.loads(process.stdout.readline()))
            for process in children:
                process.stdin.write('STOP\n')
                process.stdin.flush()
                out, err = process.communicate(timeout=10)
                if process.returncode != 0 or out or err:
                    raise ValueError('memory child did not close cleanly')
        finally:
            for process in children:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
    value = {'status': 'LOCAL_CPU_MEMORY_OBSERVATION_NOT_TARGET_CGROUP_PROOF',
             'index_bytes': (HERE / 'matrix.frozen.json').stat().st_size,
             'group_files_scanned_once': len(matrix['groups']),
             'max_group_prompt_integer_count_including_prime_copies': max_tokens,
             'max_group_bytes': max_file, 'index_load_peak_RSS_bytes': start,
             'all_groups_lazy_load_parent_peak_RSS_bytes': peak_load,
             'simultaneous_eight_stdlib_children': observations,
             'actual_native_child_exits': [p.returncode for p in children],
             'observed_sum_parent_plus_individual_child_peaks_bytes':
                rss() + sum(r['peak_RSS_bytes'] for r in observations),
             'conservative_projection_bytes': rss() +
                sum(max(r['peak_RSS_bytes'], 40 * 1024**2) for r in observations) + 40 * 1024**2,
             'projection_rule': '40MiB floor per request child plus40MiB async scanner; actual requests imports presence shown perchild',
             'target_cgroup_cap_bytes': 512 * 1024**2,
             'target_actual_request_RSS_and_cgroup_peak': 'PENDING_TARGET_RUN_NO_HTTP_CPU_MEASUREMENT'}
    if value['conservative_projection_bytes'] > value['target_cgroup_cap_bytes']:
        raise ValueError(json.dumps(value))
    if output is not None:
        path = Path(output)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with path.open('x') as handle:
            handle.write(json.dumps(value, sort_keys=True) + '\n')
    print(json.dumps(value, sort_keys=True))


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--child')
    args.add_argument('--output')
    options = args.parse_args()
    child(options.child) if options.child else parent(options.output)
