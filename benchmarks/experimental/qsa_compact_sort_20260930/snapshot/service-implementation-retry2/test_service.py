"""Stdlib mocks/AST checks only. No Torch/model/CUDA import or runtime proof."""
import ast
import contextlib
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest

import service_hook as hook
from capture_site import formal_call_line
from io_contract import sha256
from policy import TARGET_NAMES, DESCRIPTORS, Proxy, bind_owners, descriptor_key
from identity import Bindings

ROOT = Path(__file__).parent
SOURCE = ROOT / 'source'


class Mode:
    def __init__(self, name): self.name = name
FULL, NONE, PW = Mode('FULL'), Mode('NONE'), Mode('PIECEWISE')


def desc(tokens=20, reqs=4, uniform=5, mode=FULL, bucket=None):
    return NS(num_tokens=tokens, num_reqs=reqs, uniform_token_count=uniform,
              cg_mode=mode, attention_context_bucket=bucket)


class Tests(unittest.TestCase):
    def tearDown(self):
        hook.RUNTIMES.clear()

    def test_actual_constructor_and_historical_names(self):
        owner = (SOURCE / 'qsa_owner.production.py').read_text()
        model = (SOURCE / 'model.production.py').read_text()
        mtp = (SOURCE / 'mtp.production.py').read_text()
        self.assertIn('self.layer_name = f"{prefix}.attn"', owner)
        self.assertIn('prefix=f"{prefix}.self_attn"', model)
        self.assertIn('prefix=maybe_prefix(prefix, "language_model")', model)
        self.assertIn('prefix=maybe_prefix(prefix, "model")', model)
        self.assertIn('prefix=maybe_prefix(prefix, "mtp")', mtp)
        evidence = json.loads((ROOT / 'evidence/owner-name-provenance.json').read_text())
        self.assertEqual(evidence['target_layer_names'], list(TARGET_NAMES))
        self.assertEqual(evidence['draft_layer_names'], ['mtp.layers.48.self_attn.attn'])

    def test_exact_object_binding_not_shape(self):
        class Owner:
            def __init__(self, name): self.layer_name = name
        targets = [Owner(name) for name in TARGET_NAMES]
        draft = Owner('mtp.layers.48.self_attn.attn')
        target_model = NS(named_modules=lambda: [(o.layer_name[:-5], o) for o in targets])
        draft_model = NS(named_modules=lambda: [('model.layers.0.self_attn', draft)])
        context = {o.layer_name: o for o in [*targets, draft]}
        a, b = bind_owners(target_model, draft_model, Owner, context)
        self.assertEqual(len(a), 12); self.assertEqual(len(b), 1)
        context[TARGET_NAMES[0]] = Owner(TARGET_NAMES[0])
        with self.assertRaises(AssertionError):
            bind_owners(target_model, draft_model, Owner, context)

    def test_descriptor_false_routes(self):
        self.assertEqual(descriptor_key(desc()), (20, 4, 5))
        for d in (desc(mode=NONE), desc(mode=PW), desc(bucket=8192),
                  desc(tokens=15, reqs=3), desc(uniform=None), desc(tokens=30, reqs=5)):
            self.assertIsNone(descriptor_key(d))

    def test_proxy_changes_planner_only(self):
        original = NS(grouped_sparse_page4_plan_fwd=object(),
                      grouped_sparse_page4_fwd=object(),
                      grouped_sparse_page4_split_fwd=object())
        candidate = object(); proxy = Proxy(original, candidate)
        self.assertIs(proxy.grouped_sparse_page4_plan_fwd, candidate)
        self.assertIs(proxy.grouped_sparse_page4_split_fwd,
                      original.grouped_sparse_page4_split_fwd)
        self.assertIs(proxy.grouped_sparse_page4_fwd, original.grouped_sparse_page4_fwd)

    def install_fake(self, mode):
        # Compile only the ACTUAL pinned generic capture method AST, with CPU
        # mocks for graph contexts. Body/line numbers are not rewritten.
        raw = (SOURCE / 'cudagraph_utils.production.py').read_text()
        cls = next(n for n in ast.parse(raw).body if isinstance(n, ast.ClassDef)
                   and n.name == 'CudaGraphManager')
        capture = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                       and n.name == 'capture')
        capture.decorator_list = []
        capturing = [False]
        @contextlib.contextmanager
        def graph(*args):
            capturing[0] = True
            try: yield
            finally: capturing[0] = False
        cuda = NS(is_current_stream_capturing=lambda: capturing[0],
                  CUDAGraph=lambda: object(), graph=graph)
        offloader = NS(sync_prev_onload=lambda: None, join_after_forward=lambda: None)
        ns = {'torch': NS(cuda=cuda), 'CUDAGraphMode': NS(FULL=FULL, PIECEWISE=PW, NONE=NONE),
              'graph_capture': lambda **kwargs: contextlib.nullcontext(),
              'is_global_first_rank': lambda: False,
              'logger': NS(debug=lambda *args: None),
              'envs': NS(VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH=False),
              'get_offloader': lambda: offloader,
              'compilation_counter': NS(num_cudagraph_captured=0)}
        module = ast.Module(body=[ast.parse('from __future__ import annotations').body[0],
                                  capture], type_ignores=[])
        exec(compile(module, str(SOURCE / 'cudagraph_utils.production.py'), 'exec'), ns)
        class CG:
            capture = ns['capture']
            def dispatch(self, *args): return desc()
            def run_fullgraph(self, d): return 'original_replay'
        class Impl:
            def forward_qsa(self, layer, *args, **kwargs): return layer
        ops = NS(_qsa_sparse_paged_attention_sm70_grouped_page4=lambda: 'original',
                 _qsa_grouped_page4_workspace=lambda q: 'original_workspace')
        manager = CG()
        manager._capture_descs = {FULL: [desc()]}
        # Descriptors need hashing, matching production dataclass semantics.
        from collections import namedtuple
        D = namedtuple('D', 'cg_mode num_tokens num_reqs uniform_token_count attention_context_bucket')
        manager._capture_descs = {FULL: [D(FULL, 20, 4, 5, None)]}
        manager.device, manager.pool, manager.graphs = 'cpu_mock', None, {}
        rt = NS(cg=NS(CudaGraphManager=CG, __file__=str(SOURCE / 'cudagraph_utils.production.py')),
                qsa=NS(Qwen4ExpQSAFlashAttentionImpl=Impl), ops=ops,
                torch=NS(cuda=cuda), manager=manager, mode=mode)
        original_dispatch, original_replay = CG.dispatch, CG.run_fullgraph
        original_workspace = ops._qsa_grouped_page4_workspace
        hook.RUNTIMES[id(manager)] = rt
        hook.install_shared_wrappers(rt)
        return rt, original_dispatch, original_replay, original_workspace

    def test_actual_formal_capture_site(self):
        self.assertEqual(formal_call_line(SOURCE / 'cudagraph_utils.production.py'), 378)
        rt, _, _, _ = self.install_fake('on')
        calls = []
        def factory(d):
            def forward(mode):
                calls.append(hook.TICKET.get().capturing)
            return forward, 'same_state'
        result = rt.manager.capture(factory)
        self.assertEqual(calls, [False, True])
        self.assertEqual(list(result.values()), ['same_state'])
        self.assertIsNone(hook.TICKET.get())

    def test_off_on_have_no_runtime_observers(self):
        for mode in ('off', 'on'):
            rt, dispatch, replay, workspace = self.install_fake(mode)
            self.assertIs(rt.cg.CudaGraphManager.dispatch, dispatch)
            self.assertIs(rt.cg.CudaGraphManager.run_fullgraph, replay)
            self.assertIs(rt.ops._qsa_grouped_page4_workspace, workspace)
            self.assertEqual(rt.manager.run_fullgraph(desc()), 'original_replay')
            helper = rt.ops._qsa_sparse_paged_attention_sm70_grouped_page4
            rt.performance_uninstall()
            self.assertIsNot(rt.ops._qsa_sparse_paged_attention_sm70_grouped_page4, helper)
            self.assertEqual(rt.ops._qsa_sparse_paged_attention_sm70_grouped_page4(), 'original')
            hook.RUNTIMES.clear()
        tree = ast.parse((ROOT / 'service_hook.py').read_text())
        runtime = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Runtime')
        init = next(n for n in runtime.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        counter_branch = [n for n in ast.walk(init) if isinstance(n, ast.If)
                          and ast.unparse(n.test) == "self.mode == 'shadow'"]
        self.assertEqual(len(counter_branch), 1)
        self.assertIn('torch.zeros', ast.unparse(counter_branch[0]))

    def test_external_internal_bijection(self):
        b = Bindings(); allowed = ['cmpl-cohort-A-0', 'cmpl-sentinel-A-0']
        self.assertEqual(b.bind('cmpl-cohort-A-0-1234abcd', allowed), allowed[0])
        self.assertEqual(b.bind('cmpl-cohort-A-0-1234abcd', allowed), allowed[0])
        for bad in ('cmpl-cohort-A-0-87654321', 'cmpl-cohort-A-0-ABCD1234',
                    'cmpl-cohort-A-1-1234abcd', 'cmpl-foreign-0-1234abcd'):
            with self.assertRaises(AssertionError): b.bind(bad, allowed)

    def test_stage_source_gate_preserved(self):
        marker_path = SOURCE / 'host_insight_stage_markers.production.py'
        spec = importlib.util.spec_from_file_location('test_actual_markers', marker_path)
        markers = importlib.util.module_from_spec(spec); spec.loader.exec_module(markers)
        tree = ast.parse((SOURCE / 'model_runner.production.py').read_text())
        runner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GPUModelRunner')
        fn = next(n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name == 'execute_model')
        ns = {'torch': NS(inference_mode=lambda: lambda f: f)}
        module = ast.Module(body=[ast.parse('from __future__ import annotations').body[0], fn], type_ignores=[])
        exec(compile(module, str(SOURCE / 'model_runner.production.py'), 'exec'), ns)
        original = ns['execute_model']
        markers._require_reviewed_source('execute_model', original)
        wrapper_ns = {}
        exec(compile('def wrapper(*args, **kwargs):\n return None\n', str(marker_path), 'exec'), wrapper_ns)
        wrapper = wrapper_ns['wrapper']; wrapper.__wrapped__ = original
        cls = type('Runner', (), {'execute_model': wrapper,
                                  '__host_insight_stage_markers_v2__': True})
        receipt = hook.stage_contract(cls, markers, sha256(marker_path))
        self.assertTrue(receipt['class_execute_model_unmodified_by_step58'])
        self.assertIs(cls.execute_model.__wrapped__, original)

    def test_first_bank_has_integer_copy_and_immutable_flag(self):
        raw = (ROOT / 'first_failure.py').read_text()
        tree = ast.parse(raw)
        self.assertIn('bad & (self.flag[0] == 0)', raw)
        self.assertIn('self.torch.where(take, source_bits, target_bits)', raw)
        self.assertIn('INTEGER_REPRODUCIBLE_FLOAT_KV_MISSING', (ROOT / 'service_hook.py').read_text()
                      + 'INTEGER_REPRODUCIBLE_FLOAT_KV_MISSING')
        # Scope checks, not evidence these tensor ops work in CUDA graph.
        self.assertFalse(any(isinstance(n, ast.Import) and
                             any(x.name == 'torch' for x in n.names) for n in ast.walk(tree)))


if __name__ == '__main__':
    unittest.main(verbosity=2)

class Domains(unittest.TestCase):
    def test_token_units_and_invalid_lengths(self):
        from domains import pages_for_tokens
        for tokens, pages in ((0, 0), (4, 1), (32, 8), (16640, 4160)):
            self.assertEqual(pages_for_tokens(tokens), pages)
        for bad in (-4, -1, 1, 31, 16641, 16644):
            with self.assertRaises(AssertionError): pages_for_tokens(bad)
        raw = (ROOT / 'shadow.py').read_text()
        self.assertIn('self.columns < rl[:, None] // 4', raw)
        self.assertIn('self.columns < cl[:, None] // 4', raw)
        self.assertIn('cl % 4 != 0', raw)

    def test_forward_writes_full_segments_padding(self):
        raw = (SOURCE / 'forward.production.cu').read_text()
        self.assertEqual(sha256(SOURCE / 'forward.production.cu'),
                         'ec1b977a218f4663a8f92a892c603696164455cb886235e9a603615d8b12dd7a')
        start = raw.index('void grouped_sparse_page4_split_kernel(')
        end = raw.index('}  // namespace', start)
        body = raw[start:end]
        self.assertIn('if (total_kv <= 0)', body)
        self.assertIn('__float2half_rn(0.0f)', body)
        self.assertIn('token >= t_lo && token < t_hi', body)
        self.assertNotIn('request_idx < 0', body)
        self.assertIn('output_seq_lens[group_idx] = min(category_offsets[0], output_width) * 4;',
                      (SOURCE / 'planner.production.inc').read_text())
