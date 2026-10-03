"""Bounded stdlib tests of new protocol gates; no model/Torch/CUDA imports."""
import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import service_hook as hook
from admission import evaluate, reuse_witness
from baseline import derive
from io_contract import sha256, save
from make_config import create
from policy import DESCRIPTORS, TARGET_NAMES

ROOT = Path(__file__).parent
INVENTORY = json.loads((ROOT / 'evidence/cohort-inventory.input.json').read_text())


def cfg(tmp, mode='shadow'):
    return create(INVENTORY, mode, str(Path(tmp) / mode / 'capture'), [], mode == 'off')


def gate_case(mode='shadow', c=4, route='eligible_consumer', sentinel_only=False):
    xs = [f'cmpl-case-r{i}-0' for i in range(c)]; sentinel = 'cmpl-case-sentinel-0'
    bindings = {x: x + '-1234abcd' for x in [*xs, sentinel]}
    spec = {'external_request_ids': xs, 'sentinel_external_request_id': sentinel,
            'declared_concurrency': c, 'route_expectation': route,
            'scenario_label_not_runtime_proof': 'short'}
    nodes = [{'planner': 'candidate' if mode == 'shadow' else 'original'} for _ in range(36)]
    ready = {'mode': mode, 'status': 'CAPTURE_READY_SERVICE_UNVERIFIED', 'rank': 0,
             'pid': 123, 'owners': sorted(TARGET_NAMES), 'nodes': nodes,
             'epoch_contract': {'e': spec}, 'counter_shape': [36, 6] if mode == 'shadow' else None,
             'outer_off_host_observer': mode == 'off', 'private_bytes': 0, 'failure_bank_bytes': 0}
    eligible = route == 'eligible_consumer'
    key = (20, 4, 5) if c == 4 else (40, 8, 5)
    events = [{'event': 'real_scheduler', 'request_ids': [bindings[x] for x in xs],
               'sentinel': False, 'scheduled_tokens': {bindings[x]: 5 for x in xs},
               'draft_counts': {bindings[x]: 4 for x in xs}}]
    if route == 'mixed_original_witness':
        events[0]['scheduled_tokens'][bindings[xs[0]]] = 7
        events[0]['draft_counts'][bindings[xs[0]]] = 0
    if route == 'fallback_original_only':
        events[0]['draft_counts'] = {}
    if not sentinel_only:
        events.append({'event': 'dispatch', 'manager_role': 'target',
            'actual_requests': c, 'actual_tokens': c * 5,
            'uniform': 5 if eligible or route == 'short_original' else None,
            'selected': {'mode': 'FULL' if eligible else 'NONE', 'bucket': None,
                'tokens': key[0] if eligible else c * 5,
                'requests': key[1] if eligible else c,
                'uniform': 5 if eligible else None}})
    events.extend([{'event': 'real_scheduler', 'request_ids': [bindings[sentinel]],
                    'sentinel': True, 'scheduled_tokens': {bindings[sentinel]: 1}, 'draft_counts': {}},
                   {'event': 'dispatch', 'manager_role': 'target', 'actual_requests': 1,
                    'actual_tokens': 1, 'uniform': None,
                    'selected': {'mode': 'FULL', 'bucket': None, 'tokens': 1,
                                 'requests': 1, 'uniform': None}}])
    counts = [[0]*6 for _ in range(36)] if mode == 'shadow' else None
    if eligible and mode == 'shadow':
        idx = sorted(DESCRIPTORS).index(key) * 12
        for row in counts[idx:idx+12]: row[0] = 1
    drain = {'mode': mode, 'rank': 0, 'pid': 123,
             'status': 'SHADOW_DRAIN_NOT_ADMISSION' if mode == 'shadow' else 'OFF_HOST_DRAIN_DEVICE_UNINSTRUMENTED',
             'mismatches_zero': True if mode == 'shadow' else None,
             'first_bad_node': None, 'first_divergence': None, 'counts': counts,
             'host_replays': [{'descriptor': list(key), 'count': 1}] if eligible else [],
             'epoch': {'epoch': 'e', **spec}, 'request_id_bindings': bindings,
             'sentinel_external_request_id': sentinel, 'events': events, 'lifecycle': [],
             'client_cohort': {'epoch': 'e', 'queue_drained': True,
                'completed_requests': [{'external_request_id': x, 'response_id': x[:-2],
                    'http_status': 200, 'complete': True, 'n': 1, 'prompt_count': 1} for x in xs]}}
    return ready, drain


class Protocol(unittest.TestCase):
    def test_real_routes_shadow_off_and_sentinel_exclusion(self):
        for mode in ('shadow', 'off'):
            ready, drain = gate_case(mode)
            row = evaluate(ready, drain)
            self.assertEqual(row['actual_consumers'], [4])
            self.assertEqual(row['candidate_hits'], 12 if mode == 'shadow' else None)
            self.assertEqual(row['fallback_dispatches'], 0)
            for route, c in (('short_original', 2), ('mixed_original_witness', 8),
                             ('fallback_original_only', 4)):
                self.assertEqual(evaluate(*gate_case(mode, c, route))['route_gate'], 'PASS')
            self.assertTrue(evaluate(*gate_case(mode,4,'fallback_original_only'))[
                'shape_only_verify_guard_witness'])
            self.assertFalse(evaluate(*gate_case(mode,1,'fallback_original_only'))[
                'shape_only_verify_guard_witness'])
            ready, drain = gate_case(mode,8,'mixed_original_witness')
            drain['events'][0]['draft_counts'] = {}  # unequal prefill only is insufficient
            with self.assertRaises(AssertionError): evaluate(ready,drain)
            with self.assertRaises(AssertionError):
                evaluate(*gate_case(mode, 4, 'fallback_original_only', sentinel_only=True))
        ready, drain = gate_case(); drain['counts'][0][0] = 0
        with self.assertRaises(AssertionError): evaluate(ready, drain)
        ready, drain = gate_case('off'); ready['private_bytes'] = 4
        with self.assertRaises(AssertionError): evaluate(ready, drain)

    def test_epoch_bind_reset_drained_and_foreign(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = cfg(tmp, 'off'); arm = Path(config['arm_file']); arm.parent.mkdir(parents=True)
            rt = object.__new__(hook.Runtime)
            rt.cfg, rt.audit, rt.mode = config, True, 'off'
            rt.first_epoch_time = None; rt.shadow_failed = None; rt.epoch = None
            rt.seen_epochs, rt.completed_epochs = set(), set(); rt.counter = None
            rt.events, rt.real_replays = [{'stale': True}], {(20,4,5): 2}
            from identity import Bindings
            rt.id_bindings = Bindings()
            rt.slot_last_owner, rt.slot_generation, rt.slot_initial_binding = {}, {}, {}
            rt.lifecycle_pending=[]; rt.lifecycle_count=rt.lifecycle_bytes=0
            rt.lifecycle_bootstrap_reset=None
            names = list(config['epochs']); spec = config['epochs'][names[0]]
            request = {'epoch': names[0], 'external_request_ids': spec['external_request_ids']}
            arm.write_text(json.dumps(request))
            internal = spec['external_request_ids'][0] + '-1234abcd'
            self.assertTrue(rt.begin_epoch(NS(num_scheduled_tokens={internal: 5})))
            self.assertEqual(rt.events, []); self.assertEqual(rt.real_replays, {})
            second = config['epochs'][names[1]]
            arm.write_text(json.dumps({'epoch': names[1], 'external_request_ids': second['external_request_ids']}))
            with self.assertRaises(AssertionError): rt.begin_epoch(NS(num_scheduled_tokens={second['external_request_ids'][0]+'-1234abcd': 5}))
            rt.completed_epochs.add(names[0])
            self.assertTrue(rt.begin_epoch(NS(num_scheduled_tokens={second['external_request_ids'][0]+'-1234abcd': 5})))
            with self.assertRaises(AssertionError): rt.begin_epoch(NS(num_scheduled_tokens={'cmpl-foreign-0-1234abcd': 1}))

    def test_stale_drain_destination_before_new_epoch_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'drain-arm.json'
            path.write_text(json.dumps({'id':'old', 'epoch':'old'}))
            (Path(tmp)/'drain-old-rank0.json').write_text('{}')
            rt=object.__new__(hook.Runtime); rt.audit=True; rt.directory=Path(tmp)
            rt.cfg={'drain_file':str(path),'drain_ids':['old']}; rt.rank=0
            rt.epoch={'epoch':'new'}
            rt.drain_if_requested()  # old already-drained sentinel ignored

    def test_actual_zero_token_ast_retains_finished_lifecycle(self):
        source = ROOT / 'source/model_runner.production.py'
        cls = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'GPUModelRunner')
        execute = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'execute_model')
        # Exact production first block includes finished processing BEFORE
        # zero-token return; no target model body is loaded or simulated.
        execute.decorator_list = []; execute.body = execute.body[:1]
        finish = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'finish_requests')
        ns = {}
        exec(compile(ast.Module(body=[ast.parse('from __future__ import annotations').body[0], execute, finish], type_ignores=[]), str(source), 'exec'), ns)
        class States:
            def __init__(self): self.req_id_to_index={'old':0}; self.index_to_req_id={0:'old'}
            def remove_request(self, req_id):
                slot=self.req_id_to_index.pop(req_id, None)
                if slot is not None: self.index_to_req_id.pop(slot)
                return slot
            def add_request(self, req_id):
                self.req_id_to_index[req_id]=0; self.index_to_req_id[0]=req_id
        class Runner:
            execute_model=ns['execute_model']; finish_requests=ns['finish_requests']
            def __init__(self):
                self.req_states=States(); self.kv_connector=NS(no_forward=lambda s:'ZERO_TOKEN_ORIGINAL')
                self.block_tables=NS(append_block_ids=lambda i,b,overwrite:None,
                    blocks_per_kv_block=[2], num_blocks=NS(np={(0,0):4}),
                    num_kv_cache_groups=1, apply_staged_writes=lambda:None)
            def _remove_request(self, identity): self.req_states.remove_request(identity)
            def update_pp_decode_requests(self): pass
            def free_states(self, s): pass
            def add_requests(self, s):
                for identity in s.new:
                    self.req_states.add_request(identity)
                    self.block_tables.append_block_ids(0, ([3,7],), overwrite=True)
            def update_requests(self, s): pass
            def sample_tokens(self): return 'original'
        rt=object.__new__(hook.Runtime); rt.runner=Runner(); rt.manager=NS(capture=lambda:None)
        rt.audit=True; rt.mode='off'; rt.epoch={'epoch':'old'}; rt.begin_epoch=lambda s:False
        rt.cfg={'max_lifecycle_events_per_rank':100,'max_lifecycle_bytes_per_rank':4096}
        rt.slot_last_owner={0:'old'}; rt.slot_generation={0:1}
        rt.slot_initial_binding={0:{'scheduler_id':'old','physical_mapping_sha256':'oldhash'}}
        rt.lifecycle_pending=[]; rt.lifecycle_context=None; rt.lifecycle_count=rt.lifecycle_bytes=0
        with patch.object(hook, 'install_shared_wrappers', lambda r:None): rt.install()
        def schedule(n, done=(), new=()):
            return NS(total_num_scheduled_tokens=n, finished_req_ids=set(done),
                      preempted_req_ids=set(), new_block_ids_to_zero=[], new=new)
        self.assertEqual(rt.runner.execute_model(schedule(0,['old'])), 'ZERO_TOKEN_ORIGINAL')
        self.assertTrue(rt.lifecycle_pending[0]['finished'])
        self.assertIsNone(hook.ACTIVE.get()); self.assertFalse(rt.sample_epoch_active)
        rt.runner.execute_model(schedule(1,new=['new']))
        witness = reuse_witness(rt.lifecycle_pending, {'new'}, {'old'})
        self.assertEqual(witness['old_scheduler_id'], 'old')
        self.assertEqual(witness['host_generation'], 2)
        bad=[dict(r) for r in rt.lifecycle_pending]; bad[0]['finished']=False
        with self.assertRaises(AssertionError): reuse_witness(bad, {'new'}, {'old'})

    def test_derived_config_only_bindings_and_negative_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            template=cfg(tmp); template['baseline_off_capture_dir']=str(Path(tmp)/'off/capture')
            rows=[]
            for rank in range(4):
                path=Path(template['baseline_off_capture_dir'])/f'capture-ready-rank{rank}.json'
                off={'rank':rank, 'pid':100+rank, 'config_sha256':'cfg',
                     'mode':'off','status':'CAPTURE_READY_SERVICE_UNVERIFIED',
                     'source_pins':template['source_pins'], 'kv_num_blocks':55,
                     'candidate_binary_sha256':template['candidate_binary_sha256'],
                     'original_binary_sha256':template['original_binary_sha256'],
                     'runtime':{'device_uuid':f'UUID{rank}'}, 'private_bytes':0,
                     'failure_bank_bytes':0,'counter_shape':None,
                     'nodes':[{'planner':'original'} for _ in range(36)]}
                save(path, off)
                rows.append({key:off[key] for key in ('rank','pid','config_sha256',
                    'candidate_binary_sha256','original_binary_sha256','kv_num_blocks')} |
                    {'path':str(path),'sha256':sha256(path),'device_uuid':f'UUID{rank}'})
            actual=derive(template,rows)
            self.assertEqual({k: v for k,v in actual.items() if k!='baseline_off_receipts'},
                             {k: v for k,v in template.items() if k!='baseline_off_receipts'})
            bad=[dict(r) for r in rows]; bad[0]['pid']=999
            with self.assertRaises(AssertionError): derive(template,bad)
            bad=[dict(r) for r in rows]; bad[0]['sha256']='0'*64
            with self.assertRaises(RuntimeError): derive(template,bad)
            altered=dict(template); altered['diagnostic_host_audit']=True
            with self.assertRaises(AssertionError): derive(altered,rows)


if __name__=='__main__': unittest.main(verbosity=2)
