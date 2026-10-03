"""Apply the root's explicit future-only native1GiB capacity decision."""
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent
BASE = P.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    path = P / 'CONTRACT.json'
    value = json.loads(path.read_text())
    root_review = BASE / 'review/context-qsa-run4-execution-20261002/CPU1024-ROOT-REVIEW.json'
    assert sha(root_review) == 'dfaf7746333d294e5aaeb409d92520cf44fe9f7a1b3ef2fc6a364de6fb44e34e'
    decision = json.loads(root_review.read_text())
    assert decision['client_max_bytes'] == 1024**3
    assert decision['client_peak_bytes'] == 890236928
    assert all(decision['events'][k] == '0' for k in ('max', 'oom', 'oom_kill'))
    assert decision['swap_bytes'] == 0
    value['resources']['client_cgroup_bytes'] = 1024**3
    value['resources']['client_memory_stat_required'] = True
    value['resources']['client_events_max_required'] = 0
    value['transport_repair']['client_budget_finalization'] = 'ROOT_APPROVED_FUTURE_ONLY_1GiB'
    value['binding_repair']['same_inventory_resource_budgets_and_restoration'] = False
    value['binding_repair']['same_inventory_model_GPU_time_and_restoration'] = True
    value['client_capacity_override'] = {
        'old_client_cgroup_bytes': 512 * 1024**2,
        'new_client_cgroup_bytes': 1024**3,
        'scope': 'new run4 A0/B/A2 client units only; old matrix declaration and all old FAIL remain',
        'native_CPU_fixture': 'full38 groups/266HTTP +3negative +closed run3A0 prefix; synthetic loopback, not model/performance',
        'actual_new_peak_bytes': decision['client_peak_bytes'],
        'headroom_bytes': decision['headroom_bytes'],
        'events_max_zero_gate': True,
        'memory_stat_readback_required': True,
        'root_review_relative_path': str(root_review.relative_to(BASE)),
        'root_review_sha256': sha(root_review),
        'all_other_caps_times_inputs_metrics_unchanged': True,
        'not_long_context_success_or_performance_guarantee': True,
    }
    pins = [root_review]
    directory = BASE / 'review/context-qsa-transport-native1024-20261002'
    pins.extend(directory / name for name in decision['sources'])
    for source in pins:
        value['prior_evidence_pins'].append({
            'relative_path': str(source.relative_to(BASE)), 'sha256': sha(source)})
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'contract_sha256': sha(path), 'client_bytes': 1024**3}))


if __name__ == '__main__':
    main()
