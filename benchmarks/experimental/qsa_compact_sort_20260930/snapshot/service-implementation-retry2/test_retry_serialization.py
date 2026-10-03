"""Actual NumPy/source/wrapper/drain CPU regression; CUDA/model never imported."""
import ast
import copy
from dataclasses import dataclass
from enum import Enum
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

import event_contract
import service_hook as hook
from io_contract import save, sha256
from test_retry_warmup import (PRODUCTION, Runner as LifecycleRunner,
                               runtime, schedule)

ROOT = Path(__file__).parent
PINS = {
    'eagle_speculator.production.py': '9cae2d54f70ed54ef1f5416014cc19862685e6d33ec7833a3fba629f048b8f61',
    'dp_utils.production.py': '1400a8cb2c37bfb0fb49efe9a26e26f56a18329248cdc9c365732a656115967a',
    'cudagraph_utils.production.py': '477d892d04d6983bc320e3fa9a2f1b905b3a475b7b3da9fedca24122dc044a24',
}


class Mode(Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2


def native_route_fragments():
    """Compile unchanged production AST statements, not copied equivalents."""
    cg_source = ROOT / 'source/cudagraph_utils.production.py'
    cg_tree = ast.parse(cg_source.read_text())
    cls = next(n for n in cg_tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'CudaGraphManager')
    descriptor = next(n for n in cg_tree.body if isinstance(n, ast.ClassDef)
                      and n.name == 'BatchExecutionDescriptor')
    dispatch = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'dispatch')
    functions = [n for n in cg_tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('get_uniform_token_count', '_is_compatible')]
    ns = {'dataclass': dataclass, 'CUDAGraphMode': Mode,
          'is_profile': False, '__name__': 'cpu_native_desc'}
    module = ModuleType(ns['__name__']); module.__dict__.update(ns)
    sys.modules[module.__name__] = module
    ns = module.__dict__
    future = ast.parse('from __future__ import annotations').body[0]
    exec(compile(ast.Module(body=[future, descriptor, *functions], type_ignores=[]),
                 str(cg_source), 'exec'), ns)
    exec(compile(ast.Module(body=[future, dispatch], type_ignores=[]),
                 str(cg_source), 'exec'), ns)
    dp_source = ROOT / 'source/dp_utils.production.py'
    dp_fn = next(n for n in ast.parse(dp_source.read_text()).body
                 if isinstance(n, ast.FunctionDef) and n.name == 'dispatch_cg_and_sync_dp')
    exec(compile(ast.Module(body=[future, dp_fn], type_ignores=[]), str(dp_source), 'exec'), ns)
    eagle_source = ROOT / 'source/eagle_speculator.production.py'
    tree = ast.parse(eagle_source.read_text())
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
             and n.lineno in (779, 780, 781, 823, 830)]
    assert sorted(n.lineno for n in nodes) == [779, 780, 781, 823, 830]
    fn = ast.parse('def source_propose(self, input_batch):\n    pass').body[0]
    fn.body = sorted(copy.deepcopy(nodes), key=lambda n: n.lineno)
    fn.body.append(ast.Return(value=ast.Name(id='prefill_batch_desc', ctx=ast.Load())))
    ast.fix_missing_locations(fn)
    exec(compile(ast.Module(body=[future, fn], type_ignores=[]), str(eagle_source), 'exec'), ns)
    return ns


NATIVE = native_route_fragments()


def cg_module():
    class Manager:
        dispatch = NATIVE['dispatch']
        def __init__(self):
            self._graphs_captured = False
            self._candidates = []
        def run_fullgraph(self, desc):
            return 'GPU_REPLAY_EXPLICITLY_STUBBED'
        def capture(self):
            return None
    return NS(CudaGraphManager=Manager)


def configured(tmp, module=hook, mode='off'):
    rt = runtime(tmp, module)
    cg = cg_module()
    target, draft = cg.CudaGraphManager(), cg.CudaGraphManager()
    class Runner(LifecycleRunner):
        def execute_model(self, scheduler_output, intermediate_tensors=None,
                          dummy_run=False, skip_attn_for_dummy_run=False,
                          is_profile=False):
            result = PRODUCTION['execute_model'](self, scheduler_output,
                intermediate_tensors, dummy_run, skip_attn_for_dummy_run, is_profile)
            if scheduler_output.total_num_scheduled_tokens:
                values = list(scheduler_output.num_scheduled_tokens.values())
                self.batch = NS(num_tokens_after_padding=sum(values),
                    num_tokens=sum(values), num_reqs=len(values),
                    num_scheduled_tokens=np.array(values, dtype=np.int32))
                target.dispatch(len(values), sum(values), max(values))
            return result
        def sample_tokens(self, grammar=None):
            desc = NATIVE['source_propose'](NS(prefill_cudagraph_manager=draft,
                dp_size=1, dp_rank=0), self.batch)
            self.last_native_descriptor = desc
            return 'ORIGINAL_SAMPLE_OUTPUT_IDENTITY'
    rt.runner = Runner(); rt.manager = target
    rt.mode = mode; rt.cfg['mode'] = mode
    rt.directory = Path(tmp) / 'capture'; rt.directory.mkdir(parents=True, exist_ok=True)
    rt.rank = 0; rt.drains = 0; rt.first_failure = None
    rt.check_budget = lambda: {'allocator_gpu_explicitly_stubbed': True}
    counts = [[0] * 6 for _ in range(36)]
    rt.counter = NS(zero_=lambda: None, cpu=lambda: NS(tolist=lambda: counts)) if mode == 'shadow' else None
    rt.torch = NS(cuda=NS(synchronize=lambda device: None))
    rt.runner.device = 'GPU_IS_STUBBED'
    rt.first_failure = NS(reset=lambda: None) if mode == 'shadow' else None
    rt.install_lifecycle_observers = lambda: None  # Already covered unchanged retry1; no fake GPU state.
    with patch.object(module, 'install_shared_wrappers', lambda r: None):
        rt.install()
    module.RUNTIMES[id(target)] = rt
    module.install_shadow_observers(cg)
    name, spec = next(iter(rt.cfg['epochs'].items()))
    arm = Path(rt.cfg['arm_file']); arm.parent.mkdir(parents=True, exist_ok=True)
    arm.write_text(json.dumps({'epoch': name, 'external_request_ids': spec['external_request_ids']}))
    return rt, name, spec


def complete_main_then_sentinel(rt, name, spec):
    identity = spec['external_request_ids'][0] + '-1234abcd'
    main = schedule(identity, tokens=256)
    rt.runner.execute_model(main)
    assert rt.runner.sample_tokens() == 'ORIGINAL_SAMPLE_OUTPUT_IDENTITY'
    # Completed primary request proof is an explicit CPU fixture, not HTTP.
    cohort = Path(rt.cfg['cohort_dir']) / 'completed-main.json'
    save(cohort, {'epoch': name, 'queue_drained': True,
        'completed_requests': [{'external_request_id': external,
            'response_id': external[:-2], 'http_status': 200,
            'complete': True, 'n': 1, 'prompt_count': 1}
            for external in spec['external_request_ids']]})
    save(rt.cfg['drain_file'], {'id': name, 'epoch': name,
        'cohort_receipt': str(cohort), 'cohort_sha256': sha256(cohort)})
    sentinel = spec['sentinel_external_request_id'] + '-1234abcd'
    rt.runner.execute_model(schedule(sentinel, tokens=1))
    return rt.runner.sample_tokens()


class Serialization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert 'torch' not in sys.modules
        cls.out = Path(os.environ['STEP58_CPU_EVIDENCE_DIR'])
        cls.out.mkdir(parents=True, exist_ok=False)
        for file, expected in PINS.items():
            assert sha256(ROOT / 'source' / file) == expected

    def tearDown(self):
        hook.RUNTIMES.clear()
        self.assertIsNone(hook.ACTIVE.get())
        self.assertNotIn('torch', sys.modules)

    def test_old_real_source_chain_fails_before_drain_creation(self):
        path = ROOT / 'evidence/original-retry1-freeze/service_hook.py'
        self.assertEqual(sha256(path), 'e26c082c8529c2ac9237e1f8a3fa04871a4fede434f2ae66c1153c6138ae210b')
        spec = importlib.util.spec_from_file_location('step58_retry1_serialization_old', path)
        old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
        rt, epoch, cohort = configured(self.out / 'old', old)
        with self.assertRaisesRegex(TypeError, 'int32.*not JSON serializable'):
            complete_main_then_sentinel(rt, epoch, cohort)
        bad = [(i, row) for i, row in enumerate(rt.events)
               if row['event'] == 'dispatch' and isinstance(row['uniform'], np.integer)]
        self.assertTrue(bad)
        self.assertTrue(all(row['manager_role'] == 'other' for _, row in bad))
        self.assertFalse((rt.directory / f'drain-{epoch}-rank0.json').exists())
        save(self.out / 'old/source-bad-fields.json', {
            'expected_error': 'TypeError int32 not JSON serializable',
            'drain_created': False,
            'bad_fields': [{'event_index': i, 'field': 'uniform',
                'type': type(row['uniform']).__module__ + '.' + type(row['uniform']).__name__,
                'lossless_integer_value': int(row['uniform']),
                'manager_role': row['manager_role']} for i, row in bad]})
        old.RUNTIMES.clear()

    def test_actual_installed_wrapper_full_off_shadow_drain_roundtrip(self):
        for mode in ('off', 'shadow'):
            with self.subTest(mode=mode):
                rt, epoch, spec = configured(self.out / mode, mode=mode)
                self.assertEqual(complete_main_then_sentinel(rt, epoch, spec),
                                 'ORIGINAL_SAMPLE_OUTPUT_IDENTITY')
                path = rt.directory / f'drain-{epoch}-rank0.json'
                row = json.loads(path.read_text())
                self.assertEqual(row['mode'], mode)
                self.assertEqual(row['epoch']['epoch'], epoch)
                self.assertEqual(row['request_id_bindings'][spec['sentinel_external_request_id']],
                                 spec['sentinel_external_request_id'] + '-1234abcd')
                other = [r for r in row['events'] if r['event'] == 'dispatch'
                         and r['manager_role'] == 'other']
                self.assertEqual([r['uniform'] for r in other], [256, 1])
                self.assertTrue(all(type(r['uniform']) is int for r in other))
                self.assertEqual(row['events'], json.loads(json.dumps(rt.events)))
                self.assertEqual(row['memory'], {'allocator_gpu_explicitly_stubbed': True})
                self.assertEqual(row['mismatches_zero'], True if mode == 'shadow' else None)
                self.assertEqual(row['counts'], [[0]*6 for _ in range(36)] if mode == 'shadow' else None)
                hook.RUNTIMES.clear()

    def test_every_dispatch_integer_field_strict_and_lossless(self):
        base = {'event':'dispatch','manager_role':'target',
            'actual_requests':np.int32(8),'actual_tokens':np.int32(40),'uniform':np.int32(5),
            'selected':{'mode':'FULL','tokens':np.int32(40),'requests':np.int32(8),
                        'uniform':np.int32(5),'bucket':np.int32(0)}}
        result = event_contract.normalize_event(base)
        self.assertEqual(json.loads(json.dumps(result)), {
            'event':'dispatch','manager_role':'target','actual_requests':8,'actual_tokens':40,'uniform':5,
            'selected':{'mode':'FULL','tokens':40,'requests':8,'uniform':5,'bucket':0}})
        class IntLike:
            def __int__(self): raise AssertionError('no custom conversion')
        class TensorLike:
            def item(self): raise AssertionError('no GPU item')
        class IntSubclass(int): pass
        invalid = [True, False, np.bool_(True), 1.0, np.float32(1),
                   np.array([1],dtype=np.int32), IntLike(), TensorLike(), IntSubclass(1)]
        fields = [(False, k) for k in ('actual_requests','actual_tokens','uniform')]
        fields += [(True,k) for k in ('tokens','requests','uniform','bucket')]
        for nested, key in fields:
            for value in invalid:
                row = copy.deepcopy(base); (row['selected'] if nested else row)[key] = value
                with self.subTest(field=key,type=type(value).__name__):
                    with self.assertRaises(AssertionError): event_contract.normalize_event(row)
        self.assertEqual(event_contract.integer(np.int64(2**31-1)), 2**31-1)
        for value in (-1, np.int32(-1), np.uint64(2**63), 2**31):
            with self.assertRaises(AssertionError): event_contract.integer(value)
        for key in ('uniform',):
            row=copy.deepcopy(base); row[key]=None
            self.assertIsNone(event_contract.normalize_event(row)[key])
        row=copy.deepcopy(base); row['actual_requests']=None
        with self.assertRaises(AssertionError): event_contract.normalize_event(row)

    def test_all_producers_schema_immutable_and_foreign_fields_fail(self):
        rows = [
            {'event':'real_scheduler','request_ids':['r'], 'sentinel':False,
             'scheduled_tokens':{'r':np.int32(5)},'draft_counts':{'r':np.int32(4)}},
            {'event':'original_owner_call','owner':'mtp.layers.48.self_attn.attn',
             'role':'draft','route':'original_no_target_capture_ticket'},
            {'event':'unsupported_original','owner':'target','descriptor':(np.int32(20),4,5),
             'q_shape':[np.int32(16),6,256]}]
        for row in rows:
            checked = event_contract.normalize_event(row)
            json.dumps(checked)
            self.assertEqual(checked['event'],row['event'])
            extra=copy.deepcopy(row); extra['unexpected']=1
            with self.assertRaises(AssertionError): event_contract.normalize_event(extra)
        checked=event_contract.normalize_event(rows[0]); rows[0]['scheduled_tokens']['r']=9
        self.assertEqual(checked['scheduled_tokens']['r'],5)
        bad=copy.deepcopy(rows[0]); bad['draft_counts']['r']=np.int32(5)
        with self.assertRaises(AssertionError): event_contract.normalize_event(bad)
        with self.assertRaises(AssertionError): event_contract.normalize_event({'event':'new_kind'})

    def test_off_on_unarmed_event_remains_noop_and_io_bank_unchanged(self):
        rt=object.__new__(hook.Runtime); rt.audit=False
        rt.event({'event':'unvalidated_noop'})  # No numpy load/metadata allocation branch.
        rt.audit=True
        rt.event({'event':'unvalidated_noop'})  # ACTIVE absent: still startup noop.
        prior=ROOT.parent/'service-implementation-retry1'
        for file in ('io_contract.py','first_failure.py','shadow.py','smoke_graph.py','policy.py'):
            self.assertEqual(sha256(ROOT/file),sha256(prior/file))


if __name__ == '__main__':
    unittest.main(verbosity=2)
