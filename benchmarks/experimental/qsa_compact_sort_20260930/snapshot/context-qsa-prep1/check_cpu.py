"""Targeted CPU tests for this finite inventory, clocks and cleanup only."""
import ast
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dependencies import activate, sha
activate()
from context_metrics import measure
from freeze_matrix import rows
from runner import inventory, load_group, strict_checks
from analyze import token_comparison
from transport_deadline_context import Pump, worker_environment
from prime_guard import after_prime, PRIME_CHECKS

HERE = Path(__file__).resolve().parent


def stream(offset=0, chunk=1, spacing=.01):
    pieces = 256 // chunk
    chunks = [{'elapsed_s': .1 + i * chunk * spacing, 'tokens': chunk,
               'ordinal': i} for i in range(pieces)]
    return {'started_monotonic': 10 + offset,
            'finished_monotonic': 10 + offset + chunks[-1]['elapsed_s'] + .01,
            'status': 'COMPLETE', 'positive_chunks': chunks, 'ttft_s': .1,
            'prompt_tokens': 8192, 'step_ms': chunk * spacing * 1000,
            'output_token_ids': [42] * 256}


class Checks(unittest.TestCase):
    def test_actual_guard_two_primes_then_main_unique_files_and_hook0(self):
        from original_guard import OriginalGuard, Observation, METRICS
        from e7_counters import RUNTIME_SHA256
        state = {'length': 0, 'hook': 0}
        def e7():
            return [{'pid': 100 + rank, 'rank': rank, 'ranks': [0, 1, 2, 3],
                     'mode': 'on', 'module_path': 'CPU_FIXTURE_NOT_RUNTIME',
                     'package_hashes': {'runtime.py': RUNTIME_SHA256},
                     'counters': {'hook_steps': state['hook']}} for rank in range(4)]
        def metrics():
            values = {(n, None): 0 for n in METRICS if n != 'vllm:request_success_total'}
            values.update({('vllm:request_success_total', reason):
                           state['length'] if reason == 'length' else 0
                           for reason in ('length', 'stop', 'abort', 'error', 'repetition')})
            return 'CPU external-native-metrics fixture\n', values
        # Stub OS/telemetry boundaries explicitly; the actual OriginalGuard
        # before/after and actual after_prime file/save/delta logic run below.
        obs = object.__new__(Observation)
        obs.e7, obs.metrics = e7, metrics
        obs.identity = lambda: {'properties': {'InvocationID': 'cpu1', 'NRestarts': '0'}}
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            guard = OriginalGuard(obs, root / 'guard')
            def stable(label, completed):
                from io_tools import save
                save(guard.root / (label + '-cpu-fence.json'), {'boundary_stub': True})
                return e7()
            guard.stable = stable
            for index in range(2):
                packet = {'body': {'prompt': [1] * 2048, 'max_tokens': 1}}
                group = {'row_id': f'row-repeat0-prime{index}', 'phase': 'measured',
                         'ordinal': 0, 'requests': [packet],
                         'required_checks': PRIME_CHECKS, 'terminal_states': ['COMPLETE']}
                before = guard.before(group, time.perf_counter() + 10)
                state['length'] += 1
                row = {'status': 'COMPLETE', 'prompt_tokens': 2048,
                       'output_token_ids': [1], 'finished_epoch': time.time()}
                after, checks = after_prime(guard, group, before, row, time.perf_counter() + 10)
                strict_checks(checks, PRIME_CHECKS)
                self.assertEqual(after['full_E7_deltas'][0]['hook_steps'], 0)
            packet = {'body': {'prompt': [1] * 512, 'max_tokens': 256}, 'cancel_contract': None}
            group = {'row_id': 'main', 'phase': 'measured', 'ordinal': 0,
                     'requests': [packet], 'terminal_states': ['COMPLETE'],
                     'required_checks': ['requests_ok', 'input_tokens', 'output_tokens',
                         'terminal_states', 'queue_drained', 'no_restart',
                         'original_route_counters_valid', 'actual_zero_prefix_cache_hits']}
            before = guard.before(group, time.perf_counter() + 10)
            state.update(length=3, hook=1)
            row = {'status': 'COMPLETE', 'prompt_tokens': 512,
                   'output_token_ids': [1] * 256, 'finished_epoch': time.time()}
            _, checks = guard.after(group, before, [row], time.perf_counter() + 10)
            strict_checks(checks, group['required_checks'])
            names = {p.name for p in guard.root.glob('*-before-metrics.txt')}
            self.assertEqual(names, {'group-000-before-metrics.txt',
                                    'group-001-before-metrics.txt', 'group-002-before-metrics.txt'})

    def test_inventory_and_exact_cross_arm_tasks(self):
        matrix = json.loads((HERE / 'matrix.frozen.json').read_text())
        all_ids, salts, compared = [], [], {}
        for arm in ('A0', 'B', 'A2'):
            assigned = inventory(matrix, arm)
            for pin in assigned:
                group = load_group(pin)
                self.assertEqual({len(r['body']['prompt']) for r in group['requests']},
                                 {group['input_tokens']})
                self.assertEqual(group['prime_order'], list(range(len(group['primes']))))
                for i, request in enumerate(group['requests']):
                    body = request['body']
                    other = {k: v for k, v in body.items() if k != 'request_id'}
                    key = (group['row_id'], group['ordinal'], i)
                    digest = hashlib.sha256(json.dumps(other, sort_keys=True).encode()).hexdigest()
                    if arm == 'A0':
                        compared[key] = digest
                        salts.append(body['cache_salt'])
                    else:
                        self.assertEqual(compared[key], digest)
                    all_ids.append(body['request_id'])
                    self.assertLessEqual(len(body['prompt']) + 256 + 4, 262144)
                    if group['primes']:
                        prime = group['primes'][i]['body']
                        self.assertEqual(prime['cache_salt'], body['cache_salt'])
                        self.assertEqual(prime['prompt'], body['prompt'])
                        self.assertEqual(prime['max_tokens'], 1)
                all_ids += [r['body']['request_id'] for r in group['primes']]
        self.assertEqual(len(all_ids), 798)
        self.assertEqual(len(set(all_ids)), 798)
        self.assertEqual(len(salts), len(set(salts)))

    def test_context_caps_and_fallback_rows(self):
        accepted, excluded = rows()
        self.assertEqual(len(accepted), 19)
        self.assertEqual(len(excluded), 3)
        self.assertTrue(all(r['nominal_KV_reserved_tokens'] <= 663816 for r in accepted))
        self.assertTrue(all(r['nominal_KV_reserved_tokens'] > 663816 for r in excluded))
        for row in accepted:
            if row['concurrency'] < 4:
                self.assertEqual(row['candidate_route_expected'], 'fallback')
        self.assertEqual({r['input_tokens'] for r in accepted},
                         {512, 2048, 8192, 32768, 65536, 131072, 261632})

    def test_common_token_metric_ignores_partition_speed_illusion(self):
        fine = measure([stream(0, 1), stream(.013, 1)])
        coarse = measure([stream(0, 4), stream(.013, 4)])
        for value in (fine, coarse):
            self.assertEqual(value['common_status'], 'COMMON_WINDOW_OBSERVED')
            self.assertAlmostEqual(value['pooled_complete_interval_ms_per_output_token'], 10, places=8)
            self.assertGreater(sum(r['boundary_crossing_intervals_excluded'] for r in value['per_request']), 0)
        self.assertNotEqual(fine['all_request_frozen_perf38_median_ms'],
                            coarse['all_request_frozen_perf38_median_ms'])
        self.assertAlmostEqual(measure([stream(0, 4)])['pooled_complete_interval_ms_per_output_token'], 10)

    def test_empty_and_insufficient_are_preserved(self):
        value = measure([stream(0), stream(10)])
        self.assertEqual(value['common_status'], 'NO_COMMON_WINDOW')
        near = stream(2.545)
        self.assertEqual(measure([stream(), near])['common_status'], 'INSUFFICIENT_COMMON_EVENTS')
        invalid = stream()
        invalid['prompt_tokens'] = 32768
        with self.assertRaises(ValueError):
            measure([stream(), invalid])

    def test_check_and_token_comparison_are_not_vacuous(self):
        with self.assertRaises(ValueError):
            strict_checks({}, ['output_tokens'])
        with self.assertRaises(ValueError):
            strict_checks({'output_tokens': 1}, ['output_tokens'])
        normal = {'request_results': [stream()]}
        different = {'request_results': [stream()]}
        different['request_results'][0]['output_token_ids'][17] = 43
        self.assertEqual(token_comparison([normal, different, normal])['status'],
                         'FAIL_B_ONLY_OUTPUT_CHANGE')
        self.assertEqual(token_comparison([normal, normal, different])['status'],
                         'INCONCLUSIVE_AA_SELF_VARIATION')

    def test_native_eight_silent_children_deadline_and_reap(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            pump = Pump()
            program = ('import json,os,sys,time;'
                'print(json.dumps({"event":"READY","pid":os.getpid()}),flush=True);'
                'json.loads(sys.stdin.readline());time.sleep(10)')
            try:
                jobs = [pump.add(root / 'unused', root / f'job{i}',
                    argv=[sys.executable, '-I', '-B', '-c', program], env=worker_environment())
                        for i in range(8)]
                pump.wait_ready(jobs, time.perf_counter() + 10)
                deadline = time.perf_counter() + .2
                for job in jobs:
                    job.release(deadline)
                with self.assertRaises(RuntimeError):
                    pump.wait_finished(jobs, deadline + 4)
                self.assertTrue(all(j.reaped for j in jobs))
                self.assertTrue(all(j.abort_reason for j in jobs))
            finally:
                pump.close()

    def test_largest_packet_real_save_boundary_and_lazy_index(self):
        matrix = json.loads((HERE / 'matrix.frozen.json').read_text())
        self.assertLess((HERE / 'matrix.frozen.json').stat().st_size, 65536)
        biggest = max(matrix['groups'], key=lambda g: g['bytes'])
        group = load_group(biggest)
        from io_tools import save
        from transport_worker import refresh_timeout
        del refresh_timeout  # Imported actual worker only; no network action.
        near_pin = next(g for g in matrix['groups'] if '261632' in g['row_id'])
        packet = load_group(near_pin)['requests'][0]
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / 'request.json'
            save(path, packet, 4 * 1024**2)
            self.assertGreater(path.stat().st_size, 1024**2)
            self.assertLess(path.stat().st_size, 4 * 1024**2)
        for source in HERE.glob('*.py'):
            ast.parse(source.read_text(), filename=str(source))
        self.assertEqual(Path('/owned/run/control/attempt1/A0/http').parents[3], Path('/owned/run'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
