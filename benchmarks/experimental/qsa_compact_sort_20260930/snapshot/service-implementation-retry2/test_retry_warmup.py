"""Actual-source CPU wrapper regression; no Torch/model/CUDA import or run."""
import ast
import copy
from dataclasses import dataclass
from functools import cached_property
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import service_hook as hook
from admission import completed_primary_ids, reuse_witness
from identity import Bindings
from make_config import create
from test_protocol import gate_case

ROOT = Path(__file__).parent


def scheduler_classes():
    source = ROOT / 'source/scheduler_output.production.py'
    tree = ast.parse(source.read_text())
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name in ('CachedRequestData', 'SchedulerOutput')]
    module = ModuleType('step58_retry_scheduler_cpu')
    module.__dict__.update(dataclass=dataclass, cached_property=cached_property)
    sys.modules[module.__name__] = module
    body = [ast.parse('from __future__ import annotations').body[0], *classes]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source), 'exec'), module.__dict__)
    return module.SchedulerOutput, module.CachedRequestData


SCHEDULER, CACHED = scheduler_classes()


def production_first_block():
    source = ROOT / 'source/model_runner.production.py'
    cls = next(n for n in ast.parse(source.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == 'GPUModelRunner')
    execute = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                   and n.name == 'execute_model')
    execute = copy.deepcopy(execute)
    execute.decorator_list = []
    execute.body = execute.body[:1]
    finish = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == 'finish_requests')
    namespace = {}
    body = [ast.parse('from __future__ import annotations').body[0], execute, finish]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace


PRODUCTION = production_first_block()


class States:
    def __init__(self):
        self.req_id_to_index, self.index_to_req_id = {}, {}

    def add_request(self, req_id):
        self.req_id_to_index[req_id] = 0
        self.index_to_req_id[0] = req_id

    def remove_request(self, req_id):
        slot = self.req_id_to_index.pop(req_id, None)
        if slot is not None:
            self.index_to_req_id.pop(slot)
        return slot


class Runner:
    execute_model = PRODUCTION['execute_model']
    finish_requests = PRODUCTION['finish_requests']

    def __init__(self):
        self.req_states = States()
        self.kv_connector = NS(no_forward=lambda s: 'ZERO_TOKEN_ORIGINAL')
        self.calls = []
        self.block_tables = NS(
            append_block_ids=lambda i, b, overwrite: None,
            blocks_per_kv_block=[2], num_blocks=NS(np={(0, 0): 4}),
            num_kv_cache_groups=1, apply_staged_writes=lambda: None)

    def _remove_request(self, identity):
        self.req_states.remove_request(identity)

    def update_pp_decode_requests(self):
        self.calls.append('original_execute_first_block')

    def free_states(self, schedule):
        pass

    def add_requests(self, schedule):
        for request in schedule.scheduled_new_reqs:
            self.req_states.add_request(request.req_id)
            self.block_tables.append_block_ids(0, ([3, 7],), overwrite=True)

    def update_requests(self, schedule):
        pass

    def sample_tokens(self, grammar=None):
        return 'ORIGINAL_SAMPLE'


def runtime(tmp, module=hook):
    inventory = json.loads((ROOT / 'evidence/cohort-inventory.input.json').read_text())
    cfg = create(inventory, 'off', str(Path(tmp) / 'capture'), [], True)
    rt = object.__new__(module.Runtime)
    rt.cfg, rt.runner, rt.manager = cfg, Runner(), NS(capture=lambda: None)
    rt.audit, rt.mode, rt.epoch = True, 'off', None
    rt.events, rt.real_replays = [], {}
    rt.seen_epochs, rt.completed_epochs = set(), set()
    rt.first_epoch_time = rt.shadow_failed = rt.counter = None
    rt.id_bindings = Bindings()
    rt.sample_epoch_active = False
    rt.slot_last_owner, rt.slot_generation, rt.slot_initial_binding = {}, {}, {}
    rt.lifecycle_pending, rt.lifecycle_context = [], None
    rt.lifecycle_count = rt.lifecycle_bytes = 0
    rt.lifecycle_bootstrap_reset = None
    with patch.object(module, 'install_shared_wrappers', lambda r: None):
        rt.install()
    return rt


def warmup_fragment(rt):
    """Execute exact active warmup AST assignments + original wrapper calls."""
    source = ROOT / 'source/warmup.production.py'
    tree = ast.parse(source.read_text())
    selected = []
    for low, high in ((305, 310), (344, 358), (361, 363)):
        within = [n for n in ast.walk(tree) if isinstance(n, ast.stmt)
                  and low <= n.lineno <= high and n.end_lineno <= high]
        children = {id(child) for parent in within for child in ast.walk(parent)
                    if child is not parent and isinstance(child, ast.stmt)}
        selected.extend(sorted(
            [n for n in within if id(n) not in children],
            key=lambda n: n.lineno))
    # These ranges contain only top-level assignment/call statements, not a
    # copied helper equivalent. The source file SHA is in the retry evidence.
    assert [n.lineno for n in selected] == [305, 306, 307, 308, 309, 310,
                                           344, 345, 346, 349, 353, 356, 358,
                                           361, 362, 363]
    req_ids = ['_warmup_0_0_']
    namespace = dict(SchedulerOutput=SCHEDULER, req_ids=req_ids, prompt_len=512,
                     new_reqs=[NS(req_id=req_ids[0])], num_reqs=1,
                     num_kv_cache_groups=1, num_spec_steps=4,
                     cached_req_data=CACHED.make_empty(),
                     worker_execute_model=rt.runner.execute_model)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace


def schedule(identity=None, *, finished=(), preempted=None, tokens=0):
    row = SCHEDULER.make_empty()
    row.finished_req_ids = set(finished)
    row.preempted_req_ids = preempted
    if identity:
        row.num_scheduled_tokens = {identity: tokens}
        row.total_num_scheduled_tokens = tokens
        row.scheduled_new_reqs = [NS(req_id=identity)]
    return row


class Retry(unittest.TestCase):
    def test_original_wrapper_reproduces_native_none_failure(self):
        original = ROOT.parent / 'service-implementation/service_hook.py'
        spec = importlib.util.spec_from_file_location('step58_before_retry', original)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            rt = runtime(tmp, module)
            with self.assertRaisesRegex(TypeError, 'NoneType.*not iterable'):
                warmup_fragment(rt)
            self.assertEqual(rt.runner.calls, [])

    def test_native_false_dummy_warmup_and_first_epoch_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt = runtime(tmp)
            rows = warmup_fragment(rt)
            self.assertIsNone(rows['prefill_output'].preempted_req_ids)
            self.assertIsNone(rows['decode_output'].preempted_req_ids)
            self.assertEqual(rt.runner.calls, ['original_execute_first_block'] * 3)
            self.assertEqual(rt.lifecycle_pending, [])
            self.assertEqual(rt.lifecycle_count, 0)
            self.assertEqual(rt.slot_last_owner[0], '_warmup_0_0_')
            epoch, spec = next(iter(rt.cfg['epochs'].items()))
            arm = Path(rt.cfg['arm_file']); arm.parent.mkdir(parents=True, exist_ok=True)
            arm.write_text(json.dumps({'epoch': epoch, 'external_request_ids': spec['external_request_ids']}))
            identity = spec['external_request_ids'][0] + '-1234abcd'
            rt.runner.execute_model(schedule(identity, tokens=5))
            self.assertEqual(rt.lifecycle_bootstrap_reset['discarded_slot_owners'], 1)
            self.assertEqual(rt.slot_generation[0], 1)
            self.assertEqual(rt.slot_last_owner[0], identity)
            self.assertIsNone(rt.lifecycle_pending[0]['previous_owner'])
            self.assertIsNone(rt.lifecycle_pending[0]['previous_initial_binding'])
            self.assertIsNone(hook.ACTIVE.get())

    def test_preempted_nonempty_retained_but_not_finished_witness(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt = runtime(tmp)
            rt.epoch = {'epoch': 'existing'}
            rt.begin_epoch = lambda s: False
            rt.runner.execute_model(schedule('old', tokens=1))
            rt.runner.execute_model(schedule(preempted={'old'}))
            remove = [r for r in rt.lifecycle_pending if r['event'] == 'slot_remove'][0]
            self.assertTrue(remove['preempted']); self.assertFalse(remove['finished'])
            rt.runner.execute_model(schedule('new', tokens=1))
            with self.assertRaises(AssertionError):
                reuse_witness(rt.lifecycle_pending, {'new'}, {'old'})

    def test_zero_token_finished_actual_client_reuse_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt = runtime(tmp); rt.epoch = {'epoch': 'existing'}
            rt.begin_epoch = lambda s: False
            rt.runner.execute_model(schedule('old', tokens=1))
            self.assertEqual(rt.runner.execute_model(schedule(finished={'old'})), 'ZERO_TOKEN_ORIGINAL')
            rt.runner.execute_model(schedule('new', tokens=1))
            self.assertEqual(reuse_witness(rt.lifecycle_pending, {'new'}, {'old'})['old_scheduler_id'], 'old')
            with self.assertRaises(AssertionError):
                reuse_witness(rt.lifecycle_pending, {'new'}, set())
            self.assertFalse(rt.sample_epoch_active)
            self.assertIsNone(rt.lifecycle_context); self.assertIsNone(hook.ACTIVE.get())

    def test_intervening_untracked_owner_invalidates_prior_primary(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt = runtime(tmp); rt.epoch = {'epoch': 'existing'}
            rt.begin_epoch = lambda s: False
            for identity in ('primary-A', 'untracked-U'):
                rt.runner.execute_model(schedule(identity, tokens=1))
                rt.runner.execute_model(schedule(finished={identity}))
            rt.runner.execute_model(schedule('primary-B', tokens=1))
            self.assertEqual(rt.lifecycle_pending[-2]['previous_owner'], 'untracked-U')
            self.assertEqual(rt.lifecycle_pending[-2]['previous_initial_binding']['scheduler_id'], 'untracked-U')
            with self.assertRaises(AssertionError):
                reuse_witness(rt.lifecycle_pending, {'primary-B'}, {'primary-A'})

    def test_completed_receipt_membership_and_optional_type_fail_closed(self):
        _, drain = gate_case()
        completed = completed_primary_ids(drain)
        self.assertNotIn(drain['request_id_bindings'][drain['sentinel_external_request_id']], completed)
        bad = copy.deepcopy(drain); bad['client_cohort']['completed_requests'][0]['complete'] = False
        with self.assertRaises(AssertionError): completed_primary_ids(bad)
        with tempfile.TemporaryDirectory() as tmp:
            rt = runtime(tmp)
            with self.assertRaises(TypeError): rt.runner.execute_model(schedule(preempted=123))
            row = schedule(); row.finished_req_ids = None
            with self.assertRaises(TypeError): rt.runner.execute_model(row)


if __name__ == '__main__': unittest.main(verbosity=2)
