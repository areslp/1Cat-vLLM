"""Generate runtime configs from exact controller inventory; no model imports."""
import argparse
import json
from pathlib import Path
from io_contract import save, sha256, validate_config
from request_ids import completion_ids

K = '/home/l/work/1Cat-vLLM-kv-w12'
D = '/home/l/work/flash-next/perf-20260923-resume/step58-qsa-compact-sort-20260930'
ROOT = Path(__file__).parent
SOURCES = {
    'cg': ('cudagraph_utils.production.py', 'vllm/v1/worker/gpu/cudagraph_utils.py'),
    'runner': ('model_runner.production.py', 'vllm/v1/worker/gpu/model_runner.py'),
    'input_batch': ('input_batch.production.py', 'vllm/v1/worker/gpu/input_batch.py'),
    'states': ('states.production.py', 'vllm/v1/worker/gpu/states.py'),
    'block_table': ('block_table.production.py', 'vllm/v1/worker/gpu/block_table.py'),
    'owner': ('qsa_owner.production.py', 'vllm/models/qwen4_exp/nvidia/qsa.py'),
    'ops': ('qsa_ops.production.py', 'vllm/models/qwen4_exp/nvidia/ops/qsa.py'),
    'model': ('model.production.py', 'vllm/models/qwen4_exp/nvidia/model.py'),
    'mtp': ('mtp.production.py', 'vllm/models/qwen4_exp/nvidia/mtp.py'),
    'input_processor': ('input_processor.production.py', 'vllm/v1/engine/input_processor.py'),
    'completion_serving': ('completion_serving.production.py', 'vllm/entrypoints/openai/completion/serving.py'),
    'engine_serving': ('engine_serving.production.py', 'vllm/entrypoints/openai/engine/serving.py'),
    'marker': ('host_insight_stage_markers.production.py',
               '/home/l/work/host-insight-mcp/python/host_insight_stage_markers.py'),
}


def external(base):
    # Frozen ID derivation only. Prompt values are owned/pinned by controller.
    return completion_ids({'request_id': base, 'prompt': [0], 'stream': False}, base)[
        'external_request_id']


def create(inventory, mode, output_dir, baseline_bindings, diagnostic_off):
    rows = [row for row in inventory['rows'] if row['arm'] == mode]
    assert len(rows) == 20 and len({row['epoch'] for row in rows}) == 20
    epochs = {row['epoch']: {
        'external_request_ids': [external(base) for base in row['client_request_ids']['main']],
        'sentinel_external_request_id': external(row['client_request_ids']['sentinel'][0]),
        'scenario_label_not_runtime_proof': row['scenario'],
        'route_expectation': ('mixed_original_witness' if row['scenario'] == 'mixed'
            else 'fallback_original_only' if row['scenario'] == 'fallback'
            else 'short_original' if row['main_concurrency'] <= 3
            else 'eligible_consumer'),
        'declared_concurrency': row['main_concurrency']} for row in rows}
    cfg = {
        'mode': mode, 'diagnostic_host_audit': diagnostic_off,
        'output_dir': output_dir, 'arm_file': str(Path(output_dir).parent / 'epoch-arm.json'),
        'drain_file': str(Path(output_dir).parent / 'drain-arm.json'),
        'cohort_dir': str(Path(output_dir).parent / 'client'),
        'epochs': epochs, 'drain_ids': list(epochs), 'max_drains_per_rank': 20,
        'max_shadow_seconds': 600, 'max_host_events_per_rank': 4096,
        'max_lifecycle_events_per_rank': 4096,
        'max_lifecycle_bytes_per_rank': 4 * 1024**2,
        'max_private_bytes_per_rank': 32 * 1024**2,
        'max_failure_bank_bytes_per_rank': 64 * 1024**2,
        'max_failure_artifact_bytes_per_rank': 64 * 1024**2,
        'max_added_allocated_bytes_per_rank': 256 * 1024**2,
        'max_added_reserved_bytes_per_rank': 512 * 1024**2,
        'required_kv_tokens': 663816,
        'source_pins': {name: {'path': rel if rel.startswith('/') else K + '/' + rel,
                              'sha256': sha256(ROOT / 'source' / file)}
                        for name, (file, rel) in SOURCES.items()},
        'original_shim': '/home/l/work/flash-next/prod-w12/sitecustomize.py',
        'original_shim_sha256': '96fb14138879162fdafd77a3078150198714c483b1cb6277f9b7907e78916953',
        'original_binary': '/home/l/work/flash-next/prod-w48/flash_attn_v100_cuda.cpython-312-x86_64-linux-gnu.so',
        'original_binary_sha256': 'a2c3c845572e36a6eb788e53c1861734285e2512ffb12bc5b3c19909871e433b',
        'candidate_binary': D + '/build/candidate1/qsa_planner58.so',
        'candidate_binary_sha256': 'e5ac0b418ebb9a387e0838b230dd15114185f637804e9d14b336897ab24bb7e3',
        'required_flags': {'ONECAT_QSA48': 'split', 'VLLM_SM70_QSA_GROUPED_PAGE4': '1',
                           'VLLM_SM70_QSA_GROUPED_PAD_FIX': '0', 'HOST_INSIGHT_STAGE_MARKERS': '1'},
        'baseline_off_receipts': baseline_bindings,
        'baseline_off_capture_dir': str(Path(output_dir).parent.parent / 'off' / 'capture'),
    }
    return validate_config(cfg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inventory', required=True)
    p.add_argument('--mode', choices=('off', 'shadow', 'on'), required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--baseline-bindings')
    p.add_argument('--diagnostic-off', action='store_true')
    a = p.parse_args()
    inventory = json.loads(Path(a.inventory).read_text())
    # on performance needs separately frozen HTTP inventory with exact rows.
    baseline = json.loads(Path(a.baseline_bindings).read_text()) if a.baseline_bindings else []
    cfg = create(inventory, a.mode, a.output_dir, baseline, a.diagnostic_off)
    cfg['inventory_path'] = str(Path(a.inventory).resolve())
    cfg['inventory_sha256'] = sha256(a.inventory)
    save(a.config, cfg)


if __name__ == '__main__': main()
