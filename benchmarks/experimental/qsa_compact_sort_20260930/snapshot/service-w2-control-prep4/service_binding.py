"""Pure-service off/on recipes and actual capture schema; no numeric imports."""
import copy
import json
from pathlib import Path

from frozen import BASE
from io_tools import sha, save

CANONICAL = Path('/home/l/work/flash-next/perf-20260923-resume/step58-qsa-compact-sort-20260930')
WINDOW = CANONICAL / 'service-w2-run4'

SERVICE_MANIFEST = 'd17f0dbd40c966e5277b1208af2577a3b118e83ef0840d2911b94912fb1120aa'
OFF_SOURCE = '9feb342bdc8570fca61e5996c66c47f9dc1b9bde96b6bb0eba9be52174036a56'
SERVICE = BASE / 'service-implementation-retry2'
OFF_TEMPLATE = BASE / 'numeric-run7/control/configs/A0.service.json'
CANDIDATE = 'e5ac0b418ebb9a387e0838b230dd15114185f637804e9d14b336897ab24bb7e3'
FORWARD = 'a2c3c845572e36a6eb788e53c1861734285e2512ffb12bc5b3c19909871e433b'
OWNERS = {f'language_model.model.layers.{i}.self_attn.attn' for i in range(3, 48, 4)}
SHAPES = {(20, 4, 5): 16, (30, 6, 5): 24, (40, 8, 5): 40}
CONSUMERS = [None, None, None, [20, 4, 5], [30, 6, 5], [30, 6, 5], [40, 8, 5], [40, 8, 5]]
LOADED = {'model', 'mtp', 'owner', 'ops', 'cg', 'marker', 'input_batch',
          'states', 'block_table', 'runner'}
BINDING_KEYS = {'rank', 'path', 'sha256', 'pid', 'config_sha256',
                'candidate_binary_sha256', 'original_binary_sha256',
                'device_uuid', 'kv_num_blocks'}


def pinned_inputs():
    if sha(SERVICE / 'manifest.json') != SERVICE_MANIFEST or sha(OFF_TEMPLATE) != OFF_SOURCE:
        raise ValueError('fixed pure service manifest/off template differs')
    manifest = json.loads((SERVICE / 'manifest.json').read_text())
    files = {}
    for row in manifest['files']:
        path = SERVICE / row['path']
        if not path.is_relative_to(SERVICE) or path.is_symlink():
            raise ValueError('unregistered service manifest path')
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('fixed service package dependency changed')
        files[str(path)] = row['sha256']
    return files


def pure_config(cfg):
    if (cfg['mode'] not in ('off', 'on') or cfg['diagnostic_host_audit'] is not False
            or cfg['epochs'] != {} or cfg['drain_ids'] != []
            or cfg['max_drains_per_rank'] != 0
            or cfg['candidate_binary_sha256'] != CANDIDATE
            or cfg['original_binary_sha256'] != FORWARD
            or cfg['required_kv_tokens'] != 663816):
        raise ValueError('timed W2 must use pure frozen off/on, no diagnostic epochs')
    return cfg


def capture_contract(row, cfg, rank, pid, config_path, config_sha):
    pure_config(cfg)
    stage = row['stage']
    marker = cfg['source_pins']['marker']
    if (row['status'] != 'CAPTURE_READY_SERVICE_UNVERIFIED'
            or row['rank'] != rank or row['pid'] != pid or row['mode'] != cfg['mode']
            or row['config_path'] != str(config_path) or row['config_sha256'] != config_sha
            or row['source_pins'] != cfg['source_pins']
            or row['epoch_contract'] != {}
            or row['private_bytes'] != 0 or row['failure_bank_bytes'] != 0
            or row['counter_shape'] is not None
            or row['performance_observers_absent'] is not True
            or row['performance_capture_wrappers_removed'] is not True
            or row['outer_shadow_execute_observer'] is not False
            or row['outer_off_host_observer'] is not False
            or stage['stage_source_sha256'] != marker['sha256']
            or stage['stage_wrapper_file'] != marker['path']
            or stage['stage_installed_before_step58'] is not True
            or stage['class_execute_model_unmodified_by_step58'] is not True
            or row['class_stage_chain_preserved'] is not True
            or set(row['loaded_source_sha256']) != LOADED
            or any(row['loaded_source_sha256'][key] != cfg['source_pins'][key]['sha256']
                   for key in LOADED)
            or row['candidate_binary_sha256'] != CANDIDATE
            or row['original_binary_sha256'] != FORWARD
            or type(row['kv_num_blocks']) is not int or row['kv_num_blocks'] <= 0
            or row['formal_capture_call_line'] != 378
            or row['runtime']['cuda_visible_devices'] != '0,1,2,3'
            or not row['runtime']['python'].startswith('3.12.13 ')
            or row['runtime']['torch'] != '2.10.0+cu128'):
        raise ValueError('actual pure capture rank/config/source/stage/resources mismatch')
    objects = row['owner_bindings']
    if (set(row['owners']) != OWNERS or row['draft_owners_excluded'] != ['mtp.layers.48.self_attn.attn']
            or len(objects) != 12 or {o['layer_name'] for o in objects} != OWNERS
            or len({o['object_id'] for o in objects}) != 12
            or any(o['object_id'] != o['static_context_object_id']
                   or o['module_path'] + '.attn' != o['layer_name']
                   or o['class'] != 'Qwen4ExpQSAAttention' for o in objects)):
        raise ValueError('actual target/static context/draft owner identity mismatch')
    expected_consumers = [{'actual_requests': n, 'actual_tokens': n * 5,
        'uniform': 5, 'candidate_key': CONSUMERS[n - 1]} for n in range(1, 9)]
    if row['consumer_table'] != expected_consumers:
        raise ValueError('unchanged actual consumer dispatch table differs')
    nodes = row['nodes']
    if (len(nodes) != 36 or {(tuple(node['descriptor']), node['owner']) for node in nodes}
            != {(key, owner) for key in SHAPES for owner in OWNERS}):
        raise ValueError('exact approved 36 target formal nodes required')
    for node in nodes:
        key = tuple(node['descriptor'])
        if node['q_shape'] != [SHAPES[key], 6, 256]:
            raise ValueError('grouped subshape differs from original consumer/tail')
        if cfg['mode'] == 'off':
            if node['planner'] != 'original':
                raise ValueError('off node called candidate')
        elif (node['planner'] != 'candidate' or node['candidate_binding'] != 'qsa_planner58.plan_fwd'
                or node['candidate_sha256'] != CANDIDATE
                or node['forward_binding'] != 'original grouped_sparse_page4_split_fwd'
                or node['forward_sha256'] != FORWARD
                or node['private'] is not None or node['counter_row'] is not None):
            raise ValueError('on must bind planner only, original forward and zero shadow')
    return row


def derive(arm, study, *, off_bindings=None, off_dir=None, off_startup=None):
    pinned_inputs()  # Authenticate fixed values before constructing/writing.
    if arm not in ('A0', 'B', 'A2'):
        raise ValueError('only initial pure-service A0/B/A2')
    if Path(study) != WINDOW:
        raise ValueError('exact separately owned first W2 study root required')
    cfg = json.loads(OFF_TEMPLATE.read_text())
    root = Path(study) / 'control/attempt1' / arm
    cfg.update(mode='on' if arm == 'B' else 'off',
        output_dir=str(root / 'service-capture'),
        arm_file=str(root / 'unused-epoch-arm.json'),
        drain_file=str(root / 'unused-drain.json'), cohort_dir=str(root / 'unused-client'))
    cfg['baseline_off_receipts'] = []
    if arm == 'B':
        if off_dir is None or off_bindings is None or len(off_bindings) != 4 or off_startup is None:
            raise ValueError('fresh exact A0 off capture-ready baseline required')
        startup_path = Path(off_startup['path'])
        local_startup = BASE / startup_path.relative_to(CANONICAL)
        if sha(local_startup) != off_startup['sha256']:
            raise ValueError('independent pure A0 startup receipt changed')
        startup = json.loads(local_startup.read_text())
        if (startup['status'] != 'PASS_W2_PURE_SERVICE_STARTUP_NOT_PERFORMANCE'
                or startup['arm'] != 'A0' or startup['mode'] != 'off'
                or startup['service_capture_dir'] != str(off_dir)):
            raise ValueError('numeric/admission historical baseline cannot be W2 off binding')
        cfg['baseline_off_capture_dir'] = str(off_dir)
        uuids = set()
        for rank, binding in enumerate(off_bindings):
            if set(binding) != BINDING_KEYS or binding['rank'] != rank:
                raise ValueError('exact nine-field/rank binding required')
            path = Path(off_dir) / f'capture-ready-rank{rank}.json'
            local_path = BASE / path.relative_to(CANONICAL)
            if (binding['path'] != str(path) or sha(local_path) != binding['sha256']
                    or startup['capture_receipts'][rank] != {
                        'rank': rank, 'path': str(path), 'sha256': binding['sha256']}
                    or startup['worker_pids'][str(rank)] != binding['pid']
                    or startup['service_config_sha256'] != binding['config_sha256']):
                raise ValueError('fresh original capture bytes/path changed')
            row = json.loads(local_path.read_text())
            for key in ('rank', 'pid', 'config_sha256', 'candidate_binary_sha256',
                        'original_binary_sha256', 'kv_num_blocks'):
                if row[key] != binding[key]:
                    raise ValueError('off source identity binding field differs')
            if row['mode'] != 'off' or row['runtime']['device_uuid'] != binding['device_uuid']:
                raise ValueError('Torch original UUID binding differs')
            if (row['source_pins'] != cfg['source_pins'] or row['private_bytes'] != 0
                    or row['counter_shape'] is not None or row['failure_bank_bytes'] != 0
                    or row['performance_observers_absent'] is not True
                    or row['performance_capture_wrappers_removed'] is not True
                    or any(node['planner'] != 'original' for node in row['nodes'])):
                raise ValueError('off baseline is not pure source-bound original')
            if binding['device_uuid'] in uuids:
                raise ValueError('duplicate raw Torch UUID')
            uuids.add(binding['device_uuid'])
        cfg['baseline_off_receipts'] = copy.deepcopy(off_bindings)
    pure_config(cfg)
    return cfg


def write(path, cfg):
    pure_config(cfg)
    return save(path, cfg)
