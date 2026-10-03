"""A cold one-token prime need not exercise a decode/E7 eligible route."""
import time

from dependencies import activate
activate()
from io_tools import save
from original_guard import counter_delta, terminal_delta
from e7_counters import e7_deltas

PRIME_CHECKS = ['requests_ok', 'input_tokens', 'output_tokens', 'terminal_states',
                'queue_drained', 'no_restart', 'actual_zero_prefix_cache_hits',
                'original_full_E7_delta_equal_no_eligibility_claim']


def after_prime(guard, group, before, row, deadline):
    label = before['label']
    old = {(r['name'], r['reason']): r['value'] for r in before['metrics']}
    limit = min(deadline, time.perf_counter() + 15)
    observations = []
    raw, delta = None, None
    idle = False
    try:
        while time.perf_counter() < limit:
            raw, metrics = guard.obs.metrics()
            delta = counter_delta(old, metrics)
            idle = not any(metrics[(n, None)] for n in
                           ('vllm:num_requests_running', 'vllm:num_requests_waiting'))
            observations.append({'epoch': time.time(), 'idle': idle,
                'delta': [{'name': k[0], 'reason': k[1], 'value': v} for k, v in delta.items()]})
            if len(observations) > 64:
                raise ValueError('finite prime publication trace cap')
            if idle and terminal_delta(delta, ['COMPLETE']):
                break
            time.sleep(.25)
        else:
            raise TimeoutError('prime exact terminal/queue publication absent')
    finally:
        save(guard.root / (label + '-prime-native-publication.json'), observations, 256 * 1024)
        if raw is not None:
            with (guard.root / (label + '-prime-after-metrics.txt')).open('x') as handle:
                handle.write(raw)
    e7 = guard.stable(label + '-prime-after', row['finished_epoch'])
    vectors = e7_deltas(before['E7'], e7)
    identity = guard.obs.identity()
    checks = {'requests_ok': row['status'] == 'COMPLETE',
        'input_tokens': row['prompt_tokens'] == len(group['requests'][0]['body']['prompt']),
        'output_tokens': len(row['output_token_ids']) == 1,
        'terminal_states': terminal_delta(delta, ['COMPLETE']), 'queue_drained': idle,
        'no_restart': identity['properties']['NRestarts'] == '0',
        'actual_zero_prefix_cache_hits': delta[('vllm:prefix_cache_hits_total', None)] == 0,
        'original_full_E7_delta_equal_no_eligibility_claim': bool(vectors)}
    value = {'label': label, 'identity': identity, 'E7': e7, 'full_E7_deltas': vectors,
        'checks': checks, 'metric_deltas': [
            {'name': k[0], 'reason': k[1], 'value': v} for k, v in delta.items()],
        'scope': 'cold c1 prime completion/cache/drain/source; hook_steps may be0, no decode-route activation claim'}
    save(guard.root / (label + '-prime-after.json'), value, 4 * 1024**2)
    if time.perf_counter() >= deadline:
        raise TimeoutError('prime after guard exceeded fixed deadline')
    return value, checks
