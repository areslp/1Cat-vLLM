"""Deferred, target-bound STEP58 insertion; startup mode is immutable.

No changes to scheduler guards, attention float code, graph lists or descriptors.
"""
import contextvars
import functools
import importlib.util
import inspect
import os
from pathlib import Path

from io_contract import require_hash, save, validate_config
from policy import DESCRIPTORS, Proxy, TARGET_NAMES, Ticket, audit_dispatch
from policy import bind_owners, descriptor_key

TICKET = contextvars.ContextVar('step58_graph_ticket', default=None)
OWNER = contextvars.ContextVar('step58_qsa_owner', default=None)
PRIVATE = contextvars.ContextVar('step58_private_workspace', default=None)
ACTIVE = contextvars.ContextVar('step58_real_epoch', default=None)
RUNTIMES = {}


def stage_contract(cls, markers, expected_sha):
    require_hash(markers.__file__, expected_sha)
    assert cls.__dict__.get('__host_insight_stage_markers_v2__') is True
    actual = cls.execute_model
    assert Path(actual.__code__.co_filename).resolve() == (
        Path(markers.__file__).resolve()), 'stage wrapper identity'
    markers._require_reviewed_source('execute_model', inspect.unwrap(actual))
    return {'stage_installed_before_step58': True,
            'stage_wrapper_file': actual.__code__.co_filename,
            'stage_source_sha256': expected_sha,
            'class_execute_model_unmodified_by_step58': True}


def install_early(cfg):
    """Only KV initializer wrapped before the general plugin source gate."""
    cfg = validate_config(cfg)
    for pin in cfg['source_pins'].values():
        require_hash(pin['path'], pin['sha256'])
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    assert not getattr(GPUModelRunner, '_step58_installed', False)
    original = GPUModelRunner.initialize_kv_cache

    @functools.wraps(original)
    def initialize(runner, *args, **kwargs):
        result = original(runner, *args, **kwargs)
        try:
            runtime = Runtime(runner, cfg)
            runtime.install()
            RUNTIMES[id(runner.cudagraph_manager)] = runtime
        except BaseException as error:
            save(Path(cfg['output_dir']) / f'deferred-error-pid{os.getpid()}.json',
                 {'status': 'ERROR', 'error': repr(error)})
            raise
        return result

    GPUModelRunner.initialize_kv_cache = initialize
    GPUModelRunner._step58_installed = True
    save(Path(cfg['output_dir']) / f'early-installed-pid{os.getpid()}.json',
         {'status': 'EARLY_INSTALLED', 'execute_model_wrapped': False,
          'install_order': 'AFTER_KV_INIT_RETURN_BEFORE_CAPTURE'})


class Runtime:
    def __init__(self, runner, cfg):
        import torch
        import host_insight_stage_markers as markers
        from vllm.distributed import get_tensor_model_parallel_rank
        from vllm.models.qwen4_exp.nvidia import qsa
        from vllm.models.qwen4_exp.nvidia.ops import qsa as ops
        from vllm.v1.worker.gpu import cudagraph_utils as cg
        import flash_attn_v100.flash_attn_v100_cuda as ext
        self.torch, self.qsa, self.ops, self.cg = torch, qsa, ops, cg
        self.runner, self.manager = runner, runner.cudagraph_manager
        self.cfg, self.mode = cfg, cfg['mode']
        self.audit = self.mode == 'shadow' or cfg['diagnostic_host_audit']
        self.rank = get_tensor_model_parallel_rank()
        self.directory = Path(cfg['output_dir'])
        self.stage = stage_contract(
            type(runner), markers, cfg['source_pins']['marker']['sha256'])
        for name, value in cfg['required_flags'].items():
            assert os.environ.get(name) == value, (name, os.environ.get(name))
        assert runner.decode_query_len == 5
        assert runner.max_num_reqs == 8
        assert runner.parallel_config.tensor_parallel_size == 4
        assert runner.parallel_config.pipeline_parallel_size == 1
        assert runner.dp_size == 1
        assert runner.cache_config.cache_dtype == 'fp8_e4m3'
        assert torch.cuda.get_device_capability(runner.device) == (7, 0)
        self.target, self.draft = bind_owners(
            runner.model, runner.speculator.model, qsa.Qwen4ExpQSAAttention,
            runner.compilation_config.static_forward_context)
        from vllm.models.qwen4_exp.nvidia import model as model_module, mtp as mtp_module
        from vllm.v1.worker.gpu import input_batch, states, block_table, model_runner
        loaded = {'model': model_module, 'mtp': mtp_module, 'owner': qsa, 'ops': ops,
                  'cg': cg, 'marker': markers, 'input_batch': input_batch,
                  'states': states, 'block_table': block_table, 'runner': model_runner}
        self.loaded_sources = {}
        for name, module in loaded.items():
            pin = cfg['source_pins'][name]
            assert Path(module.__file__).resolve() == Path(pin['path']).resolve()
            self.loaded_sources[name] = require_hash(module.__file__, pin['sha256'])
        require_hash(ext.__file__, cfg['original_binary_sha256'])
        assert Path(ext.__file__).resolve() == Path(cfg['original_binary']).resolve()
        assert ops._qsa_grouped_page4_abi_version(ext) >= 2
        assert callable(ext.grouped_sparse_page4_split_fwd)
        assert ops._ONECAT_QSA48 == 'split'
        assert ops._SM70_QSA_GROUPED_PAGE4
        assert not ops._SM70_QSA_GROUPED_PAD_FIX
        assert ops._QSA48_SPLIT_MAX_ROWS == 64
        assert ops._SM70_QSA_GROUPED_PAGE4_QUERIES == 8
        assert ops._SM70_QSA_GROUPED_PAGE4_OUTPUT_PAGES == 4160
        require_hash(cfg['candidate_binary'], cfg['candidate_binary_sha256'])
        spec = importlib.util.spec_from_file_location(
            'qsa_planner58', cfg['candidate_binary'])
        self.candidate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.candidate)
        assert callable(self.candidate.plan_fwd)
        self.ext, self.proxy = ext, Proxy(ext, self.candidate.plan_fwd)
        self.nodes, self.prepared, self.private, self.events = {}, {}, {}, []
        self.real_replays, self.drains = {}, 0
        self.epoch = None
        self.seen_epochs = set()
        self.completed_epochs = set()
        self.shadow_failed = None
        self.last_step_ids = set()
        self.sample_epoch_active = False
        self.first_epoch_time = None
        self.slot_last_owner = {}
        self.slot_generation = {}
        self.slot_initial_binding = {}
        self.lifecycle_pending = []
        self.lifecycle_context = None
        self.lifecycle_count = self.lifecycle_bytes = 0
        self.lifecycle_bootstrap_reset = None
        from identity import Bindings
        self.id_bindings = Bindings()
        self.memory_start = self.memory()
        self.first_failure = None
        self.counter = None
        if self.mode == 'shadow':
            self.counter = torch.zeros((36, 6), dtype=torch.int64,
                                       device=runner.device)
        self.private_bytes = 0 if self.counter is None else (
            self.counter.numel() * self.counter.element_size())

    def memory(self):
        t = self.torch
        return {'allocated': t.cuda.memory_allocated(self.runner.device),
                'reserved': t.cuda.memory_reserved(self.runner.device)}

    def check_budget(self, final=False):
        current = self.memory()
        assert self.private_bytes <= self.cfg['max_private_bytes_per_rank']
        if self.first_failure is not None:
            assert self.first_failure.bytes <= self.cfg['max_failure_bank_bytes_per_rank']
        if final and self.mode != 'off':
            import json
            binding = next(row for row in self.cfg['baseline_off_receipts']
                           if row['rank'] == self.rank)
            baseline_path = Path(self.cfg['baseline_off_capture_dir']) / (
                f'capture-ready-rank{self.rank}.json')
            assert baseline_path.resolve() == Path(binding['path']).resolve()
            require_hash(baseline_path, binding['sha256'])
            baseline = json.loads(baseline_path.read_text())
            for key in ('rank', 'pid', 'config_sha256', 'candidate_binary_sha256',
                        'original_binary_sha256', 'kv_num_blocks'):
                assert baseline[key] == binding[key], ('off receipt binding', key)
            assert baseline['runtime']['device_uuid'] == binding['device_uuid']
            assert binding['device_uuid'] == str(self.torch.cuda.get_device_properties(
                self.runner.device).uuid), 'off/shadow physical device binding'
            assert baseline['rank'] == self.rank
            assert baseline['candidate_binary_sha256'] == self.cfg['candidate_binary_sha256']
            assert baseline['original_binary_sha256'] == self.cfg['original_binary_sha256']
            assert baseline['mode'] == 'off'
            assert baseline['source_pins'] == self.cfg['source_pins']
            assert baseline['kv_num_blocks'] == self.runner.kv_cache_config.num_blocks
            for field, cap in (('allocated', 'max_added_allocated_bytes_per_rank'),
                               ('reserved', 'max_added_reserved_bytes_per_rank')):
                assert current[field] - baseline['memory'][field] <= self.cfg[cap]
        return current

    def event(self, record):
        if not self.audit or ACTIVE.get() is not self:
            return
        assert len(self.events) < self.cfg['max_host_events_per_rank']
        from event_contract import normalize_event
        self.events.append(normalize_event(record))

    def node_key(self, ticket, layer):
        return (ticket.key, layer.layer_name)

    def index(self, node):
        key, owner = node
        return sorted(DESCRIPTORS).index(key) * 12 + TARGET_NAMES.index(owner)

    def grouped(self, original, args, kwargs):
        ticket, layer = TICKET.get(), OWNER.get()
        if (ticket is None or ticket.manager_id != id(self.manager)
                or ticket.key not in DESCRIPTORS or id(layer) not in self.target):
            return original(*args, **kwargs)
        bound = inspect.signature(original).bind(*args, **kwargs)
        a = bound.arguments
        q = a['q']
        supported = (q.device.type == 'cuda' and q.dtype == self.torch.float16
                     and q.shape[0] == DESCRIPTORS[ticket.key]
                     and a['kv_cache_dtype'] == 'fp8_e4m3'
                     and a['flash_attn_v100_cuda'] is self.ext
                     and a['k_cache'].dtype == self.torch.uint8
                     and a['v_cache'].dtype == self.torch.uint8)
        if not supported:
            self.event({'event': 'unsupported_original', 'owner': layer.layer_name,
                        'descriptor': ticket.key, 'q_shape': list(q.shape)})
            return original(*args, **kwargs)
        node = self.node_key(ticket, layer)
        stream = self.torch.cuda.current_stream(q.device).cuda_stream
        if not ticket.capturing:
            if self.torch.cuda.is_current_stream_capturing():
                return original(*args, **kwargs)
            # Explicit descriptor-bound preparation; ordinary model warmup
            # output/state is still original. No preparation counted as hits.
            out = original(*args, **kwargs)
            if self.mode != 'off' and node not in self.prepared:
                if self.mode == 'shadow':
                    from shadow import Shadow
                    private = Shadow(self.torch, q)
                    self.private[node] = private
                    self.private_bytes += private.bytes
                    self.check_budget()
                    self.private_forward(original, a, private)
                    from first_failure import FirstFailure, values
                    example = values(a, self.ops._step58_original_workspace(q), private)
                    if self.first_failure is None:
                        assert ticket.key == (40, 8, 5), 'largest descriptor must prepare first'
                        self.first_failure = FirstFailure(
                            self.torch, example, self.cfg['max_failure_bank_bytes_per_rank'])
                    self.first_failure.prepare(self.index(node), example)
                else:
                    # Temporary integer-only prewarm, released before capture.
                    p = self.torch.empty((q.shape[0] // 8, 4160),
                                         dtype=self.torch.int32, device=q.device)
                    m = self.torch.empty_like(p, dtype=self.torch.uint32)
                    n = self.torch.empty((q.shape[0] // 8,),
                                         dtype=self.torch.int32, device=q.device)
                    self.candidate.plan_fwd(
                        a['logical_indices'], a['block_table'], a['token_to_req'],
                        a['query_positions'], a['sequence_lengths'], p, m, n,
                        a['k_cache'].shape[1],
                        a['k_cache'].stride(0) // (4 * q.shape[2]),
                        a['k_cache'].shape[0])
                self.prepared[node] = {'stream': stream, 'q_shape': list(q.shape)}
            return out
        if self.mode == 'off':
            self.nodes[node] = {'planner': 'original', 'stream': stream,
                                'q_shape': list(q.shape)}
            return original(*args, **kwargs)
        assert node in self.prepared, 'candidate must be prewarmed outside graph'
        assert node not in self.nodes, 'duplicate owner/planner node in graph'
        if self.mode == 'on':
            a['flash_attn_v100_cuda'] = self.proxy
            out = original(**a)
        else:
            out = original(*args, **kwargs)
            reference_workspace = self.ops._step58_original_workspace(q)
            private = self.private[node]
            self.private_forward(original, a, private)
            bad = private.compare(reference_workspace, out,
                                  self.counter[self.index(node)])
            from first_failure import values
            self.first_failure.capture(
                self.index(node), values(a, reference_workspace, private), bad)
        self.nodes[node] = {
            'planner': 'candidate', 'stream': stream,
            'prewarm_stream': self.prepared[node]['stream'],
            'q_shape': list(q.shape),
            'planner_scalars': {'page_size': a['k_cache'].shape[1],
                'physical_page_stride': a['k_cache'].stride(0) // (4 * q.shape[2]),
                'num_cache_blocks': a['k_cache'].shape[0]},
            'attention_scalars': {'k_scale': a['k_scale'], 'v_scale': a['v_scale'],
                'kv_cache_dtype': a['kv_cache_dtype'], 'head_dim': q.shape[2]},
            'candidate_binding': 'qsa_planner58.plan_fwd',
            'candidate_sha256': self.cfg['candidate_binary_sha256'],
            'forward_binding': 'original grouped_sparse_page4_split_fwd',
            'forward_sha256': self.cfg['original_binary_sha256'],
            'counter_row': self.index(node) if self.mode == 'shadow' else None,
            'kv_identity': {name: {'address': a[name].data_ptr(),
                'storage_address': a[name].untyped_storage().data_ptr(),
                'shape': list(a[name].shape), 'stride': list(a[name].stride()),
                'storage_offset': a[name].storage_offset()}
                for name in ('k_cache', 'v_cache')},
            'private': self.private[node].layout if self.mode == 'shadow' else None}
        return out

    def private_forward(self, original, arguments, private):
        private.reset()
        args = dict(arguments)
        args['out'] = private.out.value
        args['flash_attn_v100_cuda'] = self.proxy
        handle = PRIVATE.set(private)
        try:
            return original(**args)
        finally:
            PRIVATE.reset(handle)

    def after_capture(self):
        expected = {(key, name) for key in DESCRIPTORS for name in TARGET_NAMES}
        assert set(self.nodes) == expected, (
            'capture node coverage', sorted(expected - self.nodes.keys()))
        rows = audit_dispatch(self.manager.dispatch)
        if self.counter is not None:
            self.counter.zero_()  # Startup/capture is explicitly excluded.
        self.events.clear()
        self.real_replays.clear()
        if self.first_failure is not None:
            self.first_failure.reset()
        current = self.check_budget(final=True)
        self.capture_memory = current
        if not self.audit:
            # CUDA graph nodes retain their chosen planner binding. Subsequent
            # eager/unsupported paths use exact original Python callables.
            self.performance_uninstall()
        save(self.directory / f'capture-ready-rank{self.rank}.json',
             {'status': 'CAPTURE_READY_SERVICE_UNVERIFIED', 'mode': self.mode,
              'rank': self.rank, 'pid': os.getpid(), 'stage': self.stage,
              'config_path': os.environ['STEP58_SERVICE_CONFIG'],
              'config_sha256': __import__('io_contract').sha256(
                  os.environ['STEP58_SERVICE_CONFIG']),
              'epoch_contract': self.cfg['epochs'],
              'loaded_source_sha256': self.loaded_sources,
              'owners': sorted(obj.layer_name for obj in self.target.values()),
              'draft_owners_excluded': [o.layer_name for o in self.draft.values()],
              'consumer_table': rows, 'counter_shape': [36, 6]
              if self.counter is not None else None,
              'private_bytes': self.private_bytes, 'memory': current,
              'failure_bank_bytes': self.first_failure.bytes
              if self.first_failure is not None else 0,
              'kv_num_blocks': self.runner.kv_cache_config.num_blocks,
              'memory_before_capture': self.memory_start,
              'nodes': [{'descriptor': key, 'owner': name, **value}
                        for (key, name), value in sorted(self.nodes.items())],
              'source_pins': self.cfg['source_pins'],
              'candidate_binary_sha256': self.cfg['candidate_binary_sha256'],
              'original_binary_sha256': self.cfg['original_binary_sha256'],
              'owner_bindings': [{'module_path': name, 'object_id': id(obj),
                  'layer_name': obj.layer_name, 'class': type(obj).__qualname__,
                  'static_context_object_id': id(self.runner.compilation_config.
                      static_forward_context[obj.layer_name])}
                  for name, obj in self.runner.model.named_modules()
                  if id(obj) in self.target],
              'class_stage_chain_preserved': True,
              'outer_shadow_execute_observer': self.mode == 'shadow',
              'formal_capture_call_line': 378,
              'outer_off_host_observer': self.mode == 'off' and self.audit,
              'performance_observers_absent': not self.audit,
              'performance_capture_wrappers_removed': not self.audit,
              'runtime': {'python': __import__('sys').version,
                          'torch': self.torch.__version__,
                          'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                          'device': str(self.runner.device),
                          'device_uuid': str(self.torch.cuda.get_device_properties(
                              self.runner.device).uuid),
                          'allocator_stats': self.torch.cuda.memory_stats(
                              self.runner.device)}})


    def drain_if_requested(self):
        if not self.audit:
            return
        import json
        path = Path(self.cfg['drain_file'])
        if not path.exists():
            return
        assert path.stat().st_size <= 4096
        request = json.loads(path.read_text())
        name = request['id']
        assert name in self.cfg['drain_ids']
        destination = self.directory / f'drain-{name}-rank{self.rank}.json'
        if destination.exists():
            return  # A stale, already-drained sentinel cannot poison next epoch.
        if self.epoch is None:
            return
        assert request['epoch'] == self.epoch['epoch']
        sentinel = self.cfg['epochs'][request['epoch']]['sentinel_external_request_id']
        if self.last_step_ids != {sentinel}:
            return
        assert self.drains < self.cfg['max_drains_per_rank']
        # One bounded transfer after proposal/sample completion, not per kernel.
        from cohort import validate_receipt
        cohort = validate_receipt(
            request['cohort_receipt'], request['cohort_sha256'],
            self.epoch, self.cfg['cohort_dir'])
        # Predeclared c1 sentinel sample drains after cohort completion.
        # Synchronization is once per completed cohort, never per kernel.
        if self.mode == 'shadow':
            self.torch.cuda.synchronize(self.runner.device)
            counts = self.counter.cpu().tolist()
            bad = next((i for i, row in enumerate(counts) if any(row[1:])), None)
        else:
            counts, bad = None, None  # off audit has NO device counter measurement
        if bad is not None:
            self.shadow_failed = {'counter_row': bad, 'counts': counts[bad]}
        first_bad_node = None
        snapshot = None
        if bad is not None:
            from io_contract import sha256
            snapshot_path = self.directory / f'first-divergence-{name}-rank{self.rank}.pt'
            assert not snapshot_path.exists()
            snapshot = self.first_failure.export(
                snapshot_path, self.cfg['max_failure_artifact_bytes_per_rank'],
                {'epoch': self.epoch,
                 'capture_ready_sha256': sha256(self.directory / f'capture-ready-rank{self.rank}.json'),
                 'config_sha256': sha256(os.environ['STEP58_SERVICE_CONFIG']),
                 'source_pins': self.cfg['source_pins'],
                 'candidate_binary_sha256': self.cfg['candidate_binary_sha256'],
                 'original_binary_sha256': self.cfg['original_binary_sha256'],
                 'nodes': [
                    {'descriptor': key, 'owner': owner, **record}
                    for (key, owner), record in sorted(self.nodes.items())]})
            snapshot['sha256'] = sha256(snapshot_path)
            first_index = snapshot['node_index']
            first_bad_node = {
                'counter_row': first_index,
                'descriptor': sorted(DESCRIPTORS)[first_index // 12],
                'owner': TARGET_NAMES[first_index % 12], 'counts': counts[first_index],
                'provenance': 'immutable first-failure bank flag node_index'}
        self.drains += 1
        self.completed_epochs.add(self.epoch['epoch'])
        save(destination, {'status': 'SHADOW_DRAIN_NOT_ADMISSION' if self.mode == 'shadow'
                           else 'OFF_HOST_DRAIN_DEVICE_UNINSTRUMENTED',
                           'rank': self.rank, 'pid': os.getpid(), 'mode': self.mode,
                           'epoch': self.epoch,
                           'mismatches_zero': all(not any(row[1:]) for row in counts)
                           if counts is not None else None,
                           'first_bad_node': first_bad_node,
                           'lowest_bad_counter_row': bad,
                           'request_id_bindings': self.id_bindings.receipt(
                               self.epoch['external_request_ids'] + [sentinel]),
                           'sentinel_external_request_id': sentinel,
                           'client_cohort_sha256': request['cohort_sha256'],
                           'client_cohort': cohort,
                           'first_divergence': snapshot,
                           'diagnostic_scope': 'first device divergence retained; integer exact replay, float KV missing',
                           'counts': counts, 'events': self.events,
                           'host_replays': [{'descriptor': key, 'count': count}
                                            for key, count in self.real_replays.items()],
                           'lifecycle': self.lifecycle_pending,
                           'lifecycle_total_count': self.lifecycle_count,
                           'lifecycle_total_bytes': self.lifecycle_bytes,
                           'lifecycle_bootstrap_reset': self.lifecycle_bootstrap_reset,
                           'memory': self.check_budget()})
        self.lifecycle_pending = []

    def begin_epoch(self, scheduler):
        import json
        import time
        arm = Path(self.cfg['arm_file'])
        if not arm.exists():
            return False
        if self.first_epoch_time is None:
            self.first_epoch_time = time.monotonic()
        assert time.monotonic() - self.first_epoch_time <= self.cfg['max_shadow_seconds']
        assert arm.stat().st_size <= 4096
        assert self.shadow_failed is None, 'shadow mismatch: no next cohort'
        request = json.loads(arm.read_text())
        spec = self.cfg['epochs'][request['epoch']]
        assert request['external_request_ids'] == spec['external_request_ids']
        actual = set(scheduler.num_scheduled_tokens)
        allowed = set(spec['external_request_ids']) | {spec['sentinel_external_request_id']}
        assert actual
        external_actual = {self.id_bindings.bind(identity, allowed) for identity in actual}
        assert len(external_actual) == len(actual), 'duplicate scheduler cohort identity'
        self.last_step_ids = external_actual
        if self.epoch is None or request['epoch'] != self.epoch['epoch']:
            assert request['epoch'] not in self.seen_epochs, 'epoch cannot be reused'
            if self.epoch is not None:
                assert self.epoch['epoch'] in self.completed_epochs, (
                    'previous epoch must be drained before replacing arm')
            else:
                # capture-ready precedes native kernel warmup. The first
                # strictly bound client epoch is the startup-history boundary.
                self.lifecycle_bootstrap_reset = {
                    'boundary': 'FIRST_STRICTLY_BOUND_CLIENT_EPOCH',
                    'discarded_slot_owners': len(self.slot_last_owner),
                    'discarded_lifecycle_records': len(self.lifecycle_pending)}
                self.slot_last_owner.clear()
                self.slot_generation.clear()
                self.slot_initial_binding.clear()
                self.lifecycle_pending.clear()
                self.lifecycle_count = self.lifecycle_bytes = 0
            self.seen_epochs.add(request['epoch'])
            if self.counter is not None:
                self.counter.zero_()
                self.first_failure.reset()
            self.events.clear()
            self.real_replays.clear()
            self.epoch = {**request, 'route_expectation': spec['route_expectation'],
                          'declared_concurrency': spec['declared_concurrency']}
        return True

    def lifecycle_event(self, record):
        """CPU-only ledger also records real zero-token finished removals.

        It is independent of ACTIVE/candidate attribution. Dummy/profile calls
        never install lifecycle_context. Each drain publishes only new records.
        """
        if not self.audit or self.lifecycle_context is None or self.epoch is None:
            return
        import json
        identity = record['scheduler_id']
        ctx = self.lifecycle_context
        row = {**record, 'sequence': self.lifecycle_count,
               'epoch_context': ctx['epoch'],
               'finished': identity in ctx['finished'],
               'preempted': identity in ctx['preempted']}
        size = len(json.dumps(row, sort_keys=True).encode())
        assert self.lifecycle_count < self.cfg['max_lifecycle_events_per_rank']
        assert self.lifecycle_bytes + size <= self.cfg['max_lifecycle_bytes_per_rank']
        self.lifecycle_count += 1
        self.lifecycle_bytes += size
        self.lifecycle_pending.append(row)

    def install_lifecycle_observers(self):
        states = self.runner.req_states
        original_add, original_remove = states.add_request, states.remove_request
        signature = inspect.signature(original_add)
        @functools.wraps(original_add)
        def add(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            identity = bound.arguments['req_id']
            result = original_add(*args, **kwargs)
            slot = states.req_id_to_index[identity]
            previous = self.slot_last_owner.get(slot)
            self.slot_generation[slot] = self.slot_generation.get(slot, 0) + 1
            self.slot_last_owner[slot] = identity
            self.lifecycle_event({'event': 'slot_add', 'scheduler_id': identity, 'slot': slot,
                        'previous_owner': previous,
                        'previous_initial_binding': self.slot_initial_binding.get(slot),
                        'host_slot_generation': self.slot_generation[slot]})
            return result
        @functools.wraps(original_remove)
        def remove(identity):
            result = original_remove(identity)
            if result is not None:
                self.lifecycle_event({'event': 'slot_remove', 'scheduler_id': identity,
                            'slot': result})
            return result
        states.add_request, states.remove_request = add, remove
        tables = self.runner.block_tables
        original_append = tables.append_block_ids
        @functools.wraps(original_append)
        def append(req_index, new_block_ids, overwrite):
            result = original_append(req_index, new_block_ids, overwrite)
            record = {'event': 'block_append', 'slot': req_index,
                'scheduler_id': states.index_to_req_id[req_index],
                'overwrite': overwrite,
                'scheduler_physical_ids_by_group': [list(x) for x in new_block_ids],
                'expansion_by_group': list(tables.blocks_per_kv_block),
                'expanded_counts_after': [int(tables.num_blocks.np[g, req_index])
                                          for g in range(tables.num_kv_cache_groups)],
                'identity_scope': 'host lifecycle/address mapping; no immutable KV generation proof'}
            import hashlib, json
            record['physical_mapping_sha256'] = hashlib.sha256(json.dumps(
                record['scheduler_physical_ids_by_group'], separators=(',', ':')).encode()).hexdigest()
            if overwrite:
                self.slot_initial_binding[req_index] = {
                    'scheduler_id': record['scheduler_id'],
                    'physical_mapping_sha256': record['physical_mapping_sha256'],
                    'expanded_counts_after': record['expanded_counts_after']}
            self.lifecycle_event(record)
            return result
        tables.append_block_ids = append

    def install(self):

        install_shared_wrappers(self)
        original = self.manager.capture
        @functools.wraps(original)
        def capture(*args, **kwargs):
            result = original(*args, **kwargs)
            self.after_capture()
            return result
        self.manager.capture = capture
        if self.audit:
            self.install_lifecycle_observers()
            original_execute = self.runner.execute_model
            signature = inspect.signature(original_execute)
            @functools.wraps(original_execute)
            def execute(*args, **kwargs):
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                a = bound.arguments
                if a.get('dummy_run') or a.get('is_profile'):
                    self.sample_epoch_active = False
                    return original_execute(*args, **kwargs)
                scheduler = a['scheduler_output']
                active = bool(scheduler.total_num_scheduled_tokens) and self.begin_epoch(scheduler)
                self.sample_epoch_active = active
                previous_context = self.lifecycle_context
                self.lifecycle_context = {
                    'epoch': self.epoch['epoch'] if active else None,
                    'finished': set(scheduler.finished_req_ids),
                    'preempted': set(() if scheduler.preempted_req_ids is None
                                     else scheduler.preempted_req_ids)}
                token = ACTIVE.set(self if active else None)
                try:
                    if active:
                        self.event({'event': 'real_scheduler',
                            'request_ids': sorted(scheduler.num_scheduled_tokens),
                            'sentinel': self.last_step_ids == {self.cfg['epochs'][
                                self.epoch['epoch']]['sentinel_external_request_id']},
                            'scheduled_tokens': scheduler.num_scheduled_tokens,
                            'draft_counts': {k: len(v) for k, v in
                                scheduler.scheduled_spec_decode_tokens.items()}})
                    return original_execute(*args, **kwargs)
                finally:
                    ACTIVE.reset(token)
                    self.lifecycle_context = previous_context
            # Instance wrapper only, AFTER the actual stage plugin source gate.
            self.runner.execute_model = execute
            original_sample = self.runner.sample_tokens

            @functools.wraps(original_sample)
            def sample(*args, **kwargs):
                token = ACTIVE.set(self if self.sample_epoch_active else None)
                try:
                    result = original_sample(*args, **kwargs)
                    if self.sample_epoch_active:
                        self.drain_if_requested()
                    return result
                finally:
                    ACTIVE.reset(token)
            self.runner.sample_tokens = sample


def install_shared_wrappers(runtime):
    cg, qsa, ops = runtime.cg, runtime.qsa, runtime.ops
    if getattr(cg.CudaGraphManager, '_step58_wrapped', False):
        return
    original_capture = cg.CudaGraphManager.capture
    from capture_site import formal_call_line, is_formal
    formal_line = formal_call_line(cg.__file__)
    capture_code = inspect.unwrap(original_capture).__code__
    @functools.wraps(original_capture)
    def capture(manager, factory, *args, **kwargs):
        rt = RUNTIMES.get(id(manager))
        if rt is None:
            return original_capture(manager, factory, *args, **kwargs)
        def bind(desc):
            forward, state = factory(desc)
            key = descriptor_key(desc)
            @functools.wraps(forward)
            def call(*args, **kwargs):
                caller = inspect.currentframe().f_back
                formal = is_formal(
                    caller, capture_code, formal_line,
                    rt.torch.cuda.is_current_stream_capturing())
                del caller
                ticket = Ticket(id(manager), key, formal)
                token = TICKET.set(ticket)
                try:
                    return forward(*args, **kwargs)
                finally:
                    TICKET.reset(token)
            return call, state
        return original_capture(manager, bind, *args, **kwargs)
    cg.CudaGraphManager.capture = capture
    if getattr(runtime, 'audit', False):
        install_shadow_observers(cg)
    original_owner = qsa.Qwen4ExpQSAFlashAttentionImpl.forward_qsa
    @functools.wraps(original_owner)
    def owner(impl, layer, *args, **kwargs):
        ticket = TICKET.get()
        if ticket is None:
            active = ACTIVE.get()
            if active is not None:
                active.event({'event': 'original_owner_call',
                              'owner': layer.layer_name,
                              'role': 'target' if id(layer) in active.target else 'draft',
                              'route': 'original_no_target_capture_ticket'})
            return original_owner(impl, layer, *args, **kwargs)
        token = OWNER.set(layer)
        try:
            return original_owner(impl, layer, *args, **kwargs)
        finally:
            OWNER.reset(token)
    qsa.Qwen4ExpQSAFlashAttentionImpl.forward_qsa = owner
    original_helper = ops._qsa_sparse_paged_attention_sm70_grouped_page4
    @functools.wraps(original_helper)
    def grouped(*args, **kwargs):
        ticket = TICKET.get()
        rt = RUNTIMES.get(ticket.manager_id) if ticket else None
        if rt is None:
            return original_helper(*args, **kwargs)
        return rt.grouped(original_helper, args, kwargs)
    ops._qsa_sparse_paged_attention_sm70_grouped_page4 = grouped
    def performance_uninstall():
        cg.CudaGraphManager.capture = original_capture
        qsa.Qwen4ExpQSAFlashAttentionImpl.forward_qsa = original_owner
        ops._qsa_sparse_paged_attention_sm70_grouped_page4 = original_helper
    runtime.performance_uninstall = performance_uninstall
    if runtime.mode == 'shadow':
        original_workspace = ops._qsa_grouped_page4_workspace
        ops._step58_original_workspace = original_workspace
        @functools.wraps(original_workspace)
        def workspace(q):
            private = PRIVATE.get()
            return private.workspace() if private is not None else original_workspace(q)
        ops._qsa_grouped_page4_workspace = workspace
    cg.CudaGraphManager._step58_wrapped = True


def install_shadow_observers(cg):
    original_dispatch = cg.CudaGraphManager.dispatch
    @functools.wraps(original_dispatch)
    def dispatch(manager, num_reqs, num_tokens, uniform_token_count):
        result = original_dispatch(manager, num_reqs, num_tokens, uniform_token_count)
        for rt in RUNTIMES.values():
            rt.event({'event': 'dispatch', 'manager_role': 'target'
                      if manager is rt.manager else 'other',
                      'actual_requests': num_reqs, 'actual_tokens': num_tokens,
                      'uniform': uniform_token_count,
                      'selected': {'mode': result.cg_mode.name,
                                   'tokens': result.num_tokens,
                                   'requests': result.num_reqs,
                                   'uniform': result.uniform_token_count,
                                   'bucket': result.attention_context_bucket}})
        return result
    cg.CudaGraphManager.dispatch = dispatch
    original_replay = cg.CudaGraphManager.run_fullgraph
    @functools.wraps(original_replay)
    def replay(manager, desc):
        result = original_replay(manager, desc)
        rt = RUNTIMES.get(id(manager))
        if rt is not None and rt.audit and ACTIVE.get() is rt:
            key = descriptor_key(desc)
            if key is not None:
                rt.real_replays[key] = rt.real_replays.get(key, 0) + 1
        return result
    cg.CudaGraphManager.run_fullgraph = replay
