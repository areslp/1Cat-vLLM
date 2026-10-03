"""Generate finite homogeneous input groups locally; sends no requests."""
import hashlib
import json
import os
from pathlib import Path

from dependencies import activate, sha
activate()
from frozen import module

HERE = Path(__file__).resolve().parent
ARMS = ('A0', 'B', 'A2')
LENGTHS = (512, 2048, 8192, 32768, 65536, 131072, 261632)
CAPACITY = 663816
BLOCK = 1632
REQUIRED = ['requests_ok', 'input_tokens', 'output_tokens', 'terminal_states',
            'queue_drained', 'no_restart', 'original_route_counters_valid',
            'actual_zero_prefix_cache_hits']


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       allow_nan=False) + '\n').encode()


def digest(value):
    return hashlib.sha256(encoded(value).rstrip(b'\n')).hexdigest()


def put(path, value):
    raw = encoded(value)
    with path.open('xb') as handle:
        handle.write(raw)
    return {'path': str(path.relative_to(HERE)), 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest()}


def rows():
    accepted, excluded = [], []
    for length in LENGTHS:
        consumers = (1, 2, 4, 8) if length == 261632 else (1, 4, 8)
        for count in consumers:
            reserved = ((length + 256 + 4 + BLOCK - 1) // BLOCK) * BLOCK
            row = {'row_id': f'context-{length}-greedy-c{count}',
                   'input_tokens': length, 'concurrency': count,
                   'output_tokens': 256, 'speculative_reserve_tokens': 4,
                   'block_rounding_tokens': BLOCK,
                   'nominal_KV_reserved_tokens': reserved * count,
                   'repeats': 2, 'metric': 'context-client-observation',
                   'kind': 'diagnostic-length-row',
                   'context_state': 'cold' if length == 512 else 'native-primed',
                   'candidate_route_expected': 'fallback' if count < 4 else
                       {'actual_FULL_consumer_required': count,
                        'descriptor': [count * 5, count, 5]},
                   'dynamic_candidate_activation': 'UNVERIFIED_NO_ADDED_DEVICE_COUNTER'}
            if reserved * count > CAPACITY:
                excluded.append({**row, 'status': 'EXCLUDED_BEFORE_EXECUTION',
                    'reason': 'block-rounded nominal KV exceeds663816; no request generated'})
            else:
                accepted.append(row)
    if len(accepted) != 19 or len(excluded) != 3:
        raise ValueError('finite matrix shape differs')
    return accepted, excluded


def build_group(row, arm, ordinal, sources):
    length = row['input_tokens']
    source_length = 8192 if length <= 8192 else 32768
    original = sources[source_length]
    prompt = (original * ((length + source_length - 1) // source_length))[:length]
    # >32K is a new, explicitly synthetic periodic token input. It is not an
    # existing historical long-context fixture or natural-document claim.
    requests = []
    for index in range(row['concurrency']):
        salt = f'context-qsa1-{row["row_id"]}-repeat{ordinal}-r{index}'
        tag = f'step58-context1-{arm}-{row["row_id"]}-n{ordinal}-r{index}'
        body = {'request_id': tag, 'model': 'flash-next', 'prompt': prompt,
                'max_tokens': 256, 'ignore_eos': True,
                'cache_salt': hashlib.sha256(salt.encode()).hexdigest(),
                'stream': True, 'stream_options': {'include_usage': True},
                'return_token_ids': True, 'n': 1, 'best_of': 1, 'seed': 1729,
                'temperature': 0, 'top_p': 1, 'top_k': -1}
        if length + 256 + 4 > 262144:
            raise ValueError('model context/output/spec reserve exceeds limit')
        requests.append({'body': body, 'header': {'X-Request-Id': tag},
            'identity': module('stream_ids').ids(body, tag),
            'body_sha256': digest(body), 'kind': 'main',
            'expected_terminal': 'COMPLETE', 'cancel_contract': None,
            'source_length': source_length,
            'input_derivation': {'type': 'prefix' if length <= source_length
                else 'synthetic_periodic_tile', 'length': length,
                'base_sha256': sha(HERE / f'sources/input-{source_length}.json')}})
    arrival = {'type': 'barrier_all_main_requests', 'replacements': False,
               'all_child_READY_before_release': True, 'mid_group_injection': False}
    primes = []
    if length > 512:
        for request in requests:
            prime = json.loads(json.dumps(request))
            prime['body']['request_id'] += '-prime'
            prime['body']['max_tokens'] = 1
            prime['header']['X-Request-Id'] = prime['body']['request_id']
            prime['identity'] = module('stream_ids').ids(prime['body'], prime['header']['X-Request-Id'])
            prime['body_sha256'] = digest(prime['body'])
            prime['kind'] = 'prime'
            primes.append(prime)
    checks = [k for k in REQUIRED if k != 'actual_zero_prefix_cache_hits']
    if length == 512:
        checks.append('actual_zero_prefix_cache_hits')
    return {**row, 'arm': arm, 'ordinal': ordinal, 'phase': 'measured',
            'requests': requests, 'prime': None, 'arrival_rule': arrival,
            'primes': primes, 'prime_order': list(range(len(primes))),
            'prime_rule': 'sequential unique salt cold1; queue empty after each; same measured salt',
            'arrival_rule_sha256': digest(arrival),
            'request_ids': [r['body']['request_id'] for r in requests],
            'body_sha256s': [r['body_sha256'] for r in requests],
            'terminal_states': ['COMPLETE'] * row['concurrency'],
            'required_checks': checks,
            'metric_member_indices': list(range(row['concurrency']))}


def main():
    os.umask(0o077)
    sources = {}
    for length in (8192, 32768):
        value = json.loads((HERE / f'sources/input-{length}.json').read_text())
        if value['length'] != length or len(value['tokens']) != length:
            raise ValueError('base fixture length differs')
        if any(type(token) is not int or token < 0 for token in value['tokens']):
            raise ValueError('base token fixture invalid')
        sources[length] = value['tokens']
    accepted, excluded = rows()
    directory = HERE / 'groups'
    directory.mkdir(mode=0o700)
    inventory, request_count, prime_count, packet_max = [], 0, 0, 0
    for arm in ARMS:
        for row in accepted:
            for ordinal in range(2):
                group = build_group(row, arm, ordinal, sources)
                name = f'{arm}-{row["row_id"]}-repeat{ordinal}.json'
                pin = put(directory / name, group)
                inventory.append({**pin, 'arm': arm, 'row_id': row['row_id'],
                                  'ordinal': ordinal, 'requests': row['concurrency'],
                                  'primes': len(group['primes'])})
                request_count += row['concurrency']
                prime_count += len(group['primes'])
                packet_max = max(packet_max, *(len(encoded(p)) for p in group['requests']))
                if pin['bytes'] > 8 * 1024**2 or packet_max > 4 * 1024**2:
                    raise ValueError('predeclared group/packet size cap exceeded')
    if request_count != 438 or prime_count != 360 or len(inventory) != 114:
        raise ValueError('finite request/group inventory differs')
    value = {'schema': 'HOMOGENEOUS_CONTEXT_QSA_MATRIX_V1', 'arms': list(ARMS),
             'rows': accepted, 'excluded_rows': excluded, 'groups': inventory,
             'groups_per_arm': 38, 'requests_per_arm': 146,
             'measured_HTTP_requests': 438, 'prime_HTTP_requests': 360,
             'HTTP_requests': 798, 'max_output_tokens': 112488,
             'output_tokens_per_arm': 37496, 'repeats': 2,
             'request_s': 600, 'group_s': 660, 'arm_s': 7200,
             'prime_block_s': 'c*(600+30) limited by the same absolute arm deadline',
             'model_max_tokens': 262144, 'nominal_KV_tokens': CAPACITY,
             'fresh_actual_capacity_required': True,
             'max_packet_bytes': packet_max,
             'source_files': {f'sources/input-{n}.json': sha(HERE / f'sources/input-{n}.json')
                              for n in sources},
             'input_semantics': 'prefixes through32K; explicitly synthetic tiled tokens above32K',
             'HTTP_requests_executed': 0, 'client_GPU_operations': 0}
    print(json.dumps(put(HERE / 'matrix.frozen.json', value), sort_keys=True))


if __name__ == '__main__':
    main()
