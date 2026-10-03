"""Complete-chain offline W1. No service admission, role or performance claim."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from headers import stream_sha
from fixtures import Case, VARIANTS, BOUNDARIES, TRANSITIONS, WITNESSES
from fixtures import synthetic, boundary, witness

PROD_SHA = 'a2c3c845572e36a6eb788e53c1861734285e2512ffb12bc5b3c19909871e433b'
CAND_SHA = 'e5ac0b418ebb9a387e0838b230dd15114185f637804e9d14b336897ab24bb7e3'
PROD = '/home/l/work/flash-next/prod-w48/flash_attn_v100_cuda.cpython-312-x86_64-linux-gnu.so'


def capture_case(torch, row):
    path = Path(row['path'])
    # Already pinned once in CPU preflight; fail closed on per-file drift.
    assert stream_sha(path) == row['sha256']
    data = torch.load(path, map_location='cpu', weights_only=True)
    meta = data['meta']
    assert meta == row['meta']
    inp = [data[k].to(dtype=dtype).contiguous() for k, dtype in (
        ('logical_indices', torch.int32), ('block_table', torch.int32),
        ('token_to_req', torch.int32), ('query_positions', torch.int64),
        ('sequence_lengths', torch.int32))]
    inp += [meta['k_shape'][1], meta['k_stride'][0] // (4 * data['q'].shape[2]),
            meta['k_shape'][0]]
    return Case(path.parent.parent.name + '/' + path.name,
                inp, data['q'], data['k_blocks'], data['v_blocks'], data['ids'],
                meta, {k: data[k] for k in ('out', 'lse', 'grouped_pages',
                                           'token_masks', 'grouped_seq_lens')})


def exercise(torch, case, reference, candidate, destination, device, context):
    from runtime import State, Branch, equal, compare_branches, witness_check, budget
    state = State(torch, case, device)
    branches = []
    for graph in (False, True):
        for role, planner in (('original', reference.grouped_sparse_page4_plan_fwd),
                              ('candidate', candidate.plan_fwd)):
            name = role + ('/graph' if graph else '/eager')
            context(case.name, 'SETUP', None, name)
            branches.append(Branch(torch, state, reference, planner, graph=graph, name=name))
    phases = []
    for iteration, phase in enumerate(('A', 'B', 'A', 'A')):
        context(case.name, phase, iteration, 'INPUT_UPDATE')
        changed = state.update(phase)
        for branch in branches:
            context(case.name, phase, iteration, branch.name)
            branch.execute()
        torch.cuda.synchronize()
        routes = []
        for branch in branches:
            context(case.name, phase, iteration, branch.name + '/VALIDATE')
            routes.append(branch.validate())
        compare_branches(torch, branches, destination,
                         f'{case.name}/iteration{iteration}/phase{phase}')
        state.check_inputs_unchanged()
        if iteration == 0 and case.saved:
            # Producer inactive suffix has an unknown prior workspace state.
            # Compare all saved effective entries, preserving the complete
            # cold-sentinel byte oracle between our four controlled branches.
            width = case.saved['grouped_pages'].shape[1]
            saved_valid = (torch.arange(width)[None, :] <
                           case.saved['grouped_seq_lens'][:, None] // 4).to(device)
            for name, actual, saved in (
                    ('saved_pages', branches[0].plan[0][:, :width][saved_valid],
                     case.saved['grouped_pages'].to(device)[saved_valid]),
                    ('saved_masks', branches[0].plan[1][:, :width].view(torch.int32)[saved_valid],
                     case.saved['token_masks'].to(device).view(torch.int32)[saved_valid]),
                    ('saved_lengths', branches[0].plan[2],
                     case.saved['grouped_seq_lens']),
                    ('saved_out', branches[0].out, case.saved['out']),
                    ('saved_lse', branches[0].lse, case.saved['lse'])):
                equal(torch, actual, saved.to(device), name, destination, case.name)
        witness_evidence = witness_check(torch, case, branches[0]) if phase == 'A' else {}
        phases.append({'phase': phase, 'changed_input_fields': changed,
                       'original_forward_routes': routes,
                       'all_full_buffers_bit_exact': True,
                       'input_unchanged': True,
                       'witness': witness_evidence,
                       'memory': budget(torch, destination)})
    return {'case': case.name, 'kind': case.meta['kind'],
            'reference_reproduces_saved_effective_metadata_and_out_lse': bool(case.saved),
            'rows': case.q.shape[0], 'groups': case.q.shape[0] // 8,
            'fixed_addresses': state.addresses,
            'phases': phases,
            'graph_route_evidence': 'CAPTURE_HOST_DISPATCH_NOT_RUNTIME_SERVICE_HIT',
            'target_draft_binding': 'UNVERIFIED_NOT_INFERRED_FROM_ROWS',
            'pass': True}


def transitions(torch, reference, candidate, destination, device, context):
    from runtime import State, Branch, compare_branches, budget
    case = boundary(torch, 1)
    state = State(torch, case, device)
    context('bucket-transition', 'SETUP', None, 'ALL_BRANCHES')
    branches = [Branch(torch, state, reference, planner, graph=graph,
                       name=role + ('/graph' if graph else '/eager'))
                for graph in (False, True)
                for role, planner in (('original', reference.grouped_sparse_page4_plan_fwd),
                                      ('candidate', candidate.plan_fwd))]
    rows = []
    for iteration, count in enumerate(TRANSITIONS):
        context('bucket-transition', f'count{count}', iteration, 'INPUT_UPDATE')
        changed = boundary(torch, count)
        assert changed.inputs[5:] == state.scalars
        # Reuse all inputs/outputs/Q/KV addresses, not only metadata graph.
        state.case = changed
        state.update('A')
        for branch in branches:
            context('bucket-transition', f'count{count}', iteration, branch.name)
            branch.execute()
        torch.cuda.synchronize()
        routes = [branch.validate() for branch in branches]
        compare_branches(torch, branches, destination,
                         f'bucket-transition/iteration{iteration}/count{count}')
        state.check_inputs_unchanged()
        rows.append({'hash_entries': count, 'full_chain_exact': True,
                     'routes': routes, 'memory': budget(torch, destination)})
    return {'steps': rows, 'input_addresses': state.addresses,
            'output_addresses': [b.bank.addresses() for b in branches],
            'fresh_all_sentinels': True, 'pass': True}


def workspace_witness(torch, device):
    from runtime import GuardBank, GuardTorch
    from production import load_namespace
    bank = GuardBank(torch)
    ns = load_namespace(GuardTorch(torch, bank), split='split')
    records = []
    streams = [torch.cuda.current_stream(), torch.cuda.Stream(device=device)]
    for stream in streams:
        previous = None
        with torch.cuda.stream(stream):
            for groups, expected_capacity in ((1, 1), (3, 4), (8, 8), (2, 8)):
                q = torch.empty((groups * 8, 6, 256),
                                dtype=torch.float16, device=device)
                tensors = ns['_qsa_grouped_page4_workspace'](q)
                key = (device.index, int(stream.cuda_stream))
                workspace = ns['_SM70_QSA_GROUPED_PAGE4_WORKSPACES'][key]
                assert workspace[0] == expected_capacity
                pointers = [t.data_ptr() for t in tensors]
                if groups == 2:
                    assert pointers == previous
                previous = pointers
                records.append({'stream': key[1], 'groups': groups,
                                'capacity': workspace[0], 'addresses': pointers})
    assert records[0]['addresses'] != records[4]['addresses']
    bank.reset()
    torch.cuda.synchronize()
    bank.check()
    return {'production_workspace_body_executed': True,
            'growth_reuse_stream_slices': records, 'pass': True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candidate', required=True)
    p.add_argument('--input-manifest', required=True)
    p.add_argument('--input-manifest-sha', required=True)
    p.add_argument('--production-flags', required=True)
    p.add_argument('--expected-device-uuid', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--gpu', required=True, type=int)
    a = p.parse_args()
    destination = Path(a.output)
    destination.mkdir(parents=True, exist_ok=False)
    result = {'status': 'RUNNING', 'start': time.time(), 'gpu': a.gpu,
              'captures': [], 'synthetic': [], 'boundaries': [], 'witnesses': [],
              'service_admission': 'UNVERIFIED',
              'service_role_c1_c3_switch_hits': 'RESERVED_FOR_INTEGRATION_SHADOW',
              'performance_claim': False}
    def persist():
        (destination / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    def context(case, phase, iteration, branch):
        result['current'] = {'case': case, 'phase': phase,
                             'iteration': iteration, 'branch': branch}
        (destination / 'current-case.json').write_text(json.dumps(result['current']) + '\n')
    try:
        assert stream_sha(a.input_manifest) == a.input_manifest_sha
        manifest = json.loads(Path(a.input_manifest).read_text())
        assert manifest['status'] == 'PASS_HEADERS_AND_HASHES_NOT_GPU'
        assert len(manifest['captures']) == 194
        assert manifest['counts'] == {'verify': 180, 'prefill': 14}
        flags = json.loads(Path(a.production_flags).read_text())
        assert flags['flags'] == {'ONECAT_QSA48': 'split',
                                 'VLLM_SM70_QSA_GROUPED_PAGE4': '1',
                                 'VLLM_SM70_QSA_GROUPED_PAD_FIX': '0'}
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
        assert visible == flags['physical_device_uuids']
        assert visible[a.gpu] == a.expected_device_uuid
        assert stream_sha(PROD) == PROD_SHA
        assert stream_sha(a.candidate) == CAND_SHA
        from production import SOURCE_SHA
        assert stream_sha('/home/l/work/1Cat-vLLM-kv-w12/vllm/models/qwen4_exp/nvidia/ops/qsa.py') == SOURCE_SHA
        import torch
        from runtime import load_extension, budget
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda', a.gpu)
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        properties = torch.cuda.get_device_properties(device)
        observed_uuid = str(properties.uuid)
        if not observed_uuid.startswith('GPU-'):
            observed_uuid = 'GPU-' + observed_uuid
        assert observed_uuid == a.expected_device_uuid
        assert (properties.major, properties.minor) == (7, 0)
        reference = load_extension('flash_attn_v100.flash_attn_v100_cuda', PROD)
        candidate = load_extension('qsa_planner58', a.candidate)
        assert reference.grouped_sparse_page4_abi_version() >= 2
        result.update(production_flags=flags, reference_sha256=PROD_SHA,
                      candidate_sha256=CAND_SHA,
                      input_manifest_sha256=a.input_manifest_sha,
                      runtime={'python': sys.version, 'executable': sys.executable,
                               'torch': torch.__version__,
                               'torch_cuda': torch.version.cuda,
                               'uv': subprocess.check_output([
                                   '/home/l/.local/bin/uv', '--version'],
                                   text=True, timeout=10).strip(),
                               'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'],
                               'visible_index': a.gpu, 'physical_uuid': observed_uuid,
                               'device_name': properties.name,
                               'device_total_memory': properties.total_memory,
                               'source_runner_sha256': stream_sha(__file__)})
        for row in manifest['captures']:
            case = capture_case(torch, row)
            observed = exercise(torch, case, reference, candidate, destination, device, context)
            observed.update(path=row['path'], sha256=row['sha256'], scope=row['scope'])
            result['captures'].append(observed)
            del case
            gc.collect()
            persist()
        for groups in range(1, 9):
            for index, variant in enumerate(VARIANTS):
                case = synthetic(torch, 541000 + groups * 100 + index, groups, variant)
                result['synthetic'].append(exercise(
                    torch, case, reference, candidate, destination, device, context))
                del case
                gc.collect()
                persist()
        for count in BOUNDARIES:
            case = boundary(torch, count)
            result['boundaries'].append(exercise(
                torch, case, reference, candidate, destination, device, context))
            del case
            gc.collect()
            persist()
        for name in WITNESSES:
            case = witness(torch, name)
            result['witnesses'].append(exercise(
                torch, case, reference, candidate, destination, device, context))
            del case
            gc.collect()
            persist()
        result['full_chain_bucket_transitions'] = transitions(
            torch, reference, candidate, destination, device, context)
        gc.collect()
        result['workspace'] = workspace_witness(torch, device)
        result['memory'] = budget(torch, destination)
        assert [len(result[k]) for k in ('captures', 'synthetic', 'boundaries',
                                         'witnesses')] == [194, 72, 9, 5]
        assert len(result['full_chain_bucket_transitions']['steps']) == 11
        assert [r['path'] for r in result['captures']] == [r['path'] for r in manifest['captures']]
        assert [r['case'] for r in result['boundaries']] == [f'bucket{n}' for n in BOUNDARIES]
        assert [r['case'] for r in result['witnesses']] == ['witness-' + n for n in WITNESSES]
        assert [r['hash_entries'] for r in result['full_chain_bucket_transitions']['steps']] == list(TRANSITIONS)
        assert [r['case'] for r in result['synthetic']] == [
            f'seed{541000 + g * 100 + index}-g{g}-{variant}'
            for g in range(1, 9) for index, variant in enumerate(VARIANTS)]
        assert all(row['pass'] for key in ('captures', 'synthetic', 'boundaries',
                                          'witnesses') for row in result[key])
        result.update(status='OFFLINE_W1_PASS_SERVICE_UNVERIFIED',
                      independent_capture_files=194,
                      correctness_only=True)
    except BaseException:
        result.update(status='FAIL', error=traceback.format_exc())
    finally:
        result['finished'] = time.time()
        persist()
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ('captures', 'synthetic', 'boundaries', 'witnesses')}))
    return int(result['status'] != 'OFFLINE_W1_PASS_SERVICE_UNVERIFIED')


if __name__ == '__main__':
    raise SystemExit(main())
