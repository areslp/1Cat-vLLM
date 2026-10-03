"""Independent stdlib source/raw review. Never imports the measured analyzer."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import traceback

ARMS = ('A0', 'B', 'A2')
MANIFEST_SHA = '728a58ebe0269cb48e9d3dd57dd60fe640a6598c5659066185450a2b20435b59'
MAIN_CHECKS = {'requests_ok', 'input_tokens', 'output_tokens', 'terminal_states',
               'queue_drained', 'no_restart', 'original_route_counters_valid'}
PRIME_CHECKS = {'requests_ok', 'input_tokens', 'output_tokens', 'terminal_states',
                'queue_drained', 'no_restart', 'actual_zero_prefix_cache_hits',
                'original_full_E7_delta_equal_no_eligibility_claim'}


def require(condition, text):
    if not condition:
        raise ValueError(text)


def read(path):
    require(not path.is_symlink(), 'symlink input: ' + str(path))
    return json.loads(path.read_bytes())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode()


def pin(path, expected):
    require(path.is_file() and not path.is_symlink(), 'missing/nonregular: ' + str(path))
    require(path.stat().st_size == expected['bytes'] and
            sha(path) == expected['sha256'], 'SHA/extent: ' + str(path))


def packet_check(packet, length, count, tag, salt, prompt):
    body = packet['body']
    expected = {'request_id': tag, 'model': 'flash-next', 'prompt': prompt,
                'max_tokens': count, 'ignore_eos': True, 'cache_salt': salt,
                'stream': True, 'stream_options': {'include_usage': True},
                'return_token_ids': True, 'n': 1, 'best_of': 1, 'seed': 1729,
                'temperature': 0, 'top_p': 1, 'top_k': -1}
    require(body == expected and len(body['prompt']) == length, 'body/generation/prompt differs')
    require(packet['body_sha256'] == hashlib.sha256(canonical(body)).hexdigest(), 'body SHA')
    require(packet['header'] == {'X-Request-Id': tag}, 'header ID')
    require(packet['identity'] == {'client_id': tag, 'response_id': 'cmpl-' + tag,
            'external_engine_id': 'cmpl-' + tag + '-0'}, 'three identities')
    require(packet['expected_terminal'] == 'COMPLETE' and
            packet['cancel_contract'] is None, 'unexpected cancel/terminal')


def source_review(prep):
    require(sha(prep / 'MANIFEST.json') == MANIFEST_SHA, 'reviewed manifest changed')
    manifest = read(prep / 'MANIFEST.json')
    require(len(manifest['files']) == 156 and manifest['bytes'] == 163571776,
            'payload manifest count/extent')
    seen = set()
    for item in manifest['files']:
        relative = Path(item['path'])
        require(not relative.is_absolute() and '..' not in relative.parts and
                str(relative) not in seen, 'manifest path/duplicate')
        seen.add(str(relative))
        pin(prep / relative, item)
    require(sum(x['bytes'] for x in manifest['files']) == manifest['bytes'], 'manifest total')
    for dep in manifest['dependencies']:
        relative = Path(dep['relative_path'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'dependency path')
        pin(prep.parent / relative, dep)
    contract, matrix = read(prep / 'CONTRACT.json'), read(prep / 'matrix.frozen.json')
    require(contract['matrix_sha256'] == sha(prep / 'matrix.frozen.json'), 'contract matrix SHA')
    require(matrix['arms'] == list(ARMS) and matrix['repeats'] == 2 and
            matrix['HTTP_requests'] == 798 and matrix['max_output_tokens'] == 112488,
            'matrix inventory')
    require(contract['inventory'] == {'arms': list(ARMS), 'measured_groups_per_arm': 38,
            'measured_requests_per_arm': 146, 'measured_outputs_per_arm': 37376,
            'prime_requests_per_arm': 120, 'prime_outputs_per_arm': 120,
            'HTTP_requests_per_arm': 266, 'HTTP_requests_total': 798,
            'outputs_total': 112488, 'warmups': 0, 'replacements': 0, 'retries': 0},
            'contract inventory')
    require(contract['old_run4_NO_GO'] == 'UNCHANGED' and contract['performance_admission'] ==
            'NONE_N2_DESCRIPTIVE_OLD_THRESHOLDS_UNCHANGED', 'admission boundary')
    allowed = [(n, c) for n in (512, 2048, 8192, 32768, 65536) for c in (1, 4, 8)]
    allowed += [(131072, 1), (131072, 4), (261632, 1), (261632, 2)]
    require([(x['input_tokens'], x['concurrency']) for x in matrix['rows']] == allowed,
            'length/concurrency inventory')
    require(contract['rows'] == matrix['rows'], 'contract rows')
    require({(r['input_tokens'], r['concurrency']) for r in matrix['excluded_rows']} ==
            {(131072, 8), (261632, 4), (261632, 8)}, 'capacity exclusions')
    for row in matrix['rows'] + matrix['excluded_rows']:
        n, c = row['input_tokens'], row['concurrency']
        reserved = ((n + 256 + 4 + 1631) // 1632) * 1632 * c
        require(row['nominal_KV_reserved_tokens'] == reserved and n + 260 <= 262144,
                'capacity/model extent')
        require((reserved <= 663816) == ((n, c) in allowed), 'capacity predicate')
        route = 'fallback' if c < 4 else {'actual_FULL_consumer_required': c,
                                        'descriptor': [5 * c, c, 5]}
        require(row['candidate_route_expected'] == route and
                row['dynamic_candidate_activation'] == 'UNVERIFIED_NO_ADDED_DEVICE_COUNTER',
                'static route/unknown dynamic activation')
    bases = {n: read(prep / f'sources/input-{n}.json')['tokens'] for n in (8192, 32768)}
    assignments, normalized, all_ids, salts = [], {}, set(), {}
    totals = {a: [0, 0, 0] for a in ARMS}
    for arm in ARMS:
        for row in matrix['rows']:
            for ordinal in range(2):
                assignments.append((arm, row['row_id'], ordinal))
    require([(x['arm'], x['row_id'], x['ordinal']) for x in matrix['groups']] == assignments,
            'fixed arm/row/repeat ordering')
    for item in matrix['groups']:
        pin(prep / item['path'], item)
        g = read(prep / item['path'])
        arm, row_id, ordinal = g['arm'], g['row_id'], g['ordinal']
        n, c = g['input_tokens'], g['concurrency']
        require((arm, row_id, ordinal) == (item['arm'], item['row_id'], item['ordinal']), 'group identity')
        require(len(g['requests']) == c == item['requests'] and
                len(g['primes']) == (0 if n == 512 else c) == item['primes'], 'group lengths')
        required = MAIN_CHECKS | ({'actual_zero_prefix_cache_hits'} if n == 512 else set())
        require(set(g['required_checks']) == required, 'main check inventory')
        require(g['arrival_rule'] == {'type': 'barrier_all_main_requests', 'replacements': False,
                'all_child_READY_before_release': True, 'mid_group_injection': False}, 'homogeneous arrival')
        base = bases[8192 if n <= 8192 else 32768]
        prompt = (base * ((n + len(base) - 1) // len(base)))[:n]
        for i, packet in enumerate(g['requests']):
            tag = f'step58-context1-{arm}-{row_id}-n{ordinal}-r{i}'
            salt = hashlib.sha256(f'context-qsa1-{row_id}-repeat{ordinal}-r{i}'.encode()).hexdigest()
            packet_check(packet, n, 256, tag, salt, prompt)
            key = (row_id, ordinal, i)
            norm = {k: v for k, v in packet['body'].items() if k != 'request_id'}
            digest = hashlib.sha256(canonical(norm)).hexdigest()
            require(key not in normalized or normalized[key] == digest, 'cross-arm body/salt changes')
            normalized[key] = digest
            require(tag not in all_ids, 'duplicate request ID')
            all_ids.add(tag)
            salt_set = salts.setdefault(arm, set())
            require(salt not in salt_set, 'within-arm measured salt reuse')
            salt_set.add(salt)
            if n > 512:
                prime = g['primes'][i]
                packet_check(prime, n, 1, tag + '-prime', salt, prompt)
                require(tag + '-prime' not in all_ids, 'prime ID reused')
                all_ids.add(tag + '-prime')
        totals[arm][0] += 1
        totals[arm][1] += c
        totals[arm][2] += len(g['primes'])
    require(len(all_ids) == 798 and all(v == [38, 146, 120] for v in totals.values()), 'actual packet totals')
    return {'status': 'PASS_LOCAL_FROZEN_SOURCE_REVIEW_NOT_EXECUTION',
            'manifest_sha256': MANIFEST_SHA, 'payload_files': 156,
            'payload_bytes': 163571776, 'dependencies': len(manifest['dependencies']),
            'unique_HTTP_IDs': len(all_ids), 'HTTP_requests': 798, 'outputs': 112488,
            'arms': totals, 'rows': matrix['rows'], 'n_per_row': 2,
            'candidate_near_max': 'c1/c2 fallback only; no actual QSA activation proof',
            'analyzer_scope': 'descriptive n2; token equality only; SSE receipt timing not GPU steps',
            'prime_scope': 'sequential cold-c1 prime TTFT separate from native-primed main TTFT; hits may be0'}


def number(value):
    require(type(value) in (int, float) and math.isfinite(value), 'nonfinite time')
    return value


def raw_stream(path, packet, row):
    identity = packet['identity']
    body = packet['body']
    tokens, positive, finish, usage, done = [], [], None, None, False
    event_index, previous, received = 0, -1.0, 0
    with path.open('rb') as f:
        for saved in f:
            value = json.loads(saved)
            when = number(value['elapsed_s'])
            require(previous <= when <= row['elapsed_s'] and when >= 0, 'raw clock order')
            previous = when
            line = bytes.fromhex(value['raw_hex'])
            received += len(line)
            if not line.startswith(b'data:'):
                continue
            require(not done, 'data after DONE')
            text = line[5:].strip().decode('utf-8')
            current = event_index
            event_index += 1
            if text == '[DONE]':
                done = True
                continue
            chunk = json.loads(text)
            require('error' not in chunk and chunk['id'] == identity['response_id'], 'foreign/error SSE')
            choices = chunk.get('choices', [])
            require(type(choices) is list and len(choices) <= 1, 'choice extent')
            if chunk.get('usage') is not None:
                require(not choices and usage is None, 'duplicate/nonseparate usage')
                usage = (current, chunk['usage'])
            if choices:
                require(usage is None and choices[0]['index'] == 0, 'choice after usage/index')
                choice = choices[0]
                output = choice.get('token_ids') or []
                require(type(output) is list and all(type(t) is int and t >= 0 for t in output), 'token dtype')
                if output:
                    require(finish is None, 'token after finish')
                    tokens.extend(output)
                    positive.append({'elapsed_s': when, 'tokens': len(output), 'ordinal': current})
                if choice.get('finish_reason') is not None:
                    require(finish is None and choice['finish_reason'] == 'length', 'duplicate/bad finish')
                    finish = current
    require(done and finish is not None and usage is not None and finish < usage[0] < event_index - 1,
            'missing/order finish/usage/DONE')
    require(usage[1]['prompt_tokens'] == len(body['prompt']) and
            usage[1]['completion_tokens'] == len(tokens) == body['max_tokens'] and
            usage[1]['total_tokens'] == len(body['prompt']) + len(tokens), 'usage/token extent')
    require(row['output_token_ids'] == tokens and row['positive_chunks'] == positive and
            row['raw_received_bytes'] == received and row['ttft_s'] == positive[0]['elapsed_s'], 'raw/parsed mismatch')
    require(row['identity'] == identity and row['body_sha256'] == packet['body_sha256'] and
            row['status'] == 'COMPLETE' and row['terminal'] == 'eof', 'row identity/completion')
    return positive


def common_metrics(rows):
    traces = []
    for row in rows:
        require(row['status'] == 'COMPLETE' and len(row['output_token_ids']) == 256, 'complete256 only')
        started, finished = number(row['started_monotonic']), number(row['finished_monotonic'])
        require(finished > started, 'request duration')
        trace = [(started + number(x['elapsed_s']), x['tokens'], x['ordinal']) for x in row['positive_chunks']]
        require(trace and sum(x[1] for x in trace) == 256 and
                all(started <= t[0] <= finished for t in trace) and
                all(a[0] < b[0] for a, b in zip(trace, trace[1:])), 'positive clock/token extent')
        traces.append(trace)
    lo, hi = max(t[0][0] for t in traces), min(t[-1][0] for t in traces)
    counts, durations, pairs = [], [], []
    for trace in traces:
        inside = [(a, b) for a, b in zip(trace, trace[1:]) if lo <= a[0] < b[0] <= hi]
        counts.append(sum(b[1] for a, b in inside))
        durations.append(sum(b[0] - a[0] for a, b in inside))
        pairs.append([[a[2], b[2]] for a, b in inside])
    status = 'NO_COMMON_WINDOW' if hi <= lo else ('INSUFFICIENT_COMMON_EVENTS' if
             any(not p for p in pairs) else 'COMMON_WINDOW_OBSERVED')
    total, seconds = sum(counts), sum(durations)
    span = max(r['finished_monotonic'] for r in rows) - min(r['started_monotonic'] for r in rows)
    return {'common_status': status, 'common_start_monotonic': lo, 'common_end_monotonic': hi,
            'common_complete_interval_tokens': total, 'common_all_requests_interval_time_s': seconds,
            'pooled_complete_interval_ms_per_output_token': 1000 * seconds / total if total else None,
            'median_TTFT_s': statistics.median(r['ttft_s'] for r in rows),
            'HTTP_makespan_s': span, 'end_to_end_client_tokens_s': len(rows) * 256 / span,
            'counts': counts, 'durations': durations, 'pairs': pairs}


def close_value(actual, expected, label):
    if expected is None or type(expected) is str or type(expected) is int:
        require(actual == expected, label)
    else:
        require(type(actual) in (int, float) and math.isfinite(actual) and
                math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8), label)


def runtime_review(prep, local, canonical_root):
    def mapped(remote):
        p = Path(remote)
        require(p.is_absolute() and p.is_relative_to(canonical_root), 'runtime path escapes window')
        result = local / p.relative_to(canonical_root)
        require(not result.is_symlink(), 'runtime symlink')
        return result
    matrix = read(prep / 'matrix.frozen.json')
    results, groups, total, output = {}, [], 0, 0
    for arm in ARMS:
        root = local / 'control/attempt1' / arm / 'http'
        final = read(root / 'RESULT.json')
        require(final['status'] == 'COMPLETE_HTTP_MATRIX_NOT_PERFORMANCE_ADMISSION' and
                final['arm'] == arm and final['completed_groups'] == 38 and
                final['HTTP_requests'] == 266 and final['outputs'] == 37496,
                'incomplete arm remains failure; no replacement')
        assigned = [p for p in matrix['groups'] if p['arm'] == arm]
        require(len(final['group_records']) == 38, 'record count')
        for item, record in zip(assigned, final['group_records']):
            require((record['row_id'], record['ordinal']) == (item['row_id'], item['ordinal']), 'record order')
            path = mapped(record['path'])
            require(sha(path) == record['sha256'], 'GROUP SHA')
            g, frozen = read(path), read(prep / item['path'])
            require(g['status'] == 'COMPLETE_HTTP_GROUP_NOT_PERFORMANCE_ADMISSION' and
                    (g['arm'], g['row_id'], g['ordinal']) == (arm, item['row_id'], item['ordinal']), 'GROUP status/identity')
            require(set(g['checks']) == set(frozen['required_checks']) and
                    all(v is True for v in g['checks'].values()), 'main checks')
            require(len(g['request_results']) == len(frozen['requests']) and
                    len(g['priming_results']) == len(frozen['primes']), 'request count')
            streams = [(p, r) for p, r in zip(frozen['requests'], g['request_results'])]
            for p, prime in zip(frozen['primes'], g['priming_results']):
                require(set(prime['checks']) == PRIME_CHECKS and all(v is True for v in prime['checks'].values()), 'prime checks')
                streams.append((p, prime['request_result']))
            for packet, row in streams:
                raw = mapped(row['raw_response']['path'])
                pin(raw, row['raw_response'])
                require(read(raw.parent / 'request.json') == packet, 'saved actual packet')
                raw_stream(raw, packet, row)
                process_path = mapped(row['process_exit_path'])
                require(sha(process_path) == row['process_exit_sha256'], 'child process SHA')
                proc, close = read(process_path), read(raw.parent / 'native-close.json')
                require(proc['actual_child_exit'] == 0 and proc['process_reaped'] is True and
                        proc['abort_reason'] is None, 'child not successfully reaped')
                require(close['response_close_returned'] is True and close['response_existed'] is True and
                        close['torch_imported'] is False and close['CVD'] == '' and
                        close['pid'] == proc['pid'], 'native close/CPU-only client')
                require(proc['release_monotonic'] <= row['started_monotonic'] <
                        row['finished_monotonic'] <= proc['deadline_monotonic'], 'native request deadline')
                total += 1
                output += len(row['output_token_ids'])
            computed = common_metrics(g['request_results'])
            for key, value in computed.items():
                if key not in ('counts', 'durations', 'pairs'):
                    close_value(g['metrics'][key], value, 'metric ' + key)
            require(len(g['metrics']['per_request']) == len(g['request_results']), 'per-request metric cardinality')
            for i, value in enumerate(g['metrics']['per_request']):
                require(value['common_tokens'] == computed['counts'][i] and
                        value['common_interval_ordinal_pairs'] == computed['pairs'][i], 'per-request common tokens/ordinals')
                close_value(value['common_covered_interval_s'], computed['durations'][i], 'per-request common time')
            key = (arm, g['row_id'], g['ordinal'])
            results[key] = g
            groups.append({'arm': arm, 'row_id': g['row_id'], 'ordinal': g['ordinal'], **computed})
    require(total == 798 and output == 112488, 'actual raw totals')
    equality = []
    for row in matrix['rows']:
        for ordinal in range(2):
            triple = [results[(a, row['row_id'], ordinal)] for a in ARMS]
            tokens = [[r['output_token_ids'] for r in g['request_results']] for g in triple]
            aa, b = tokens[0] == tokens[2], tokens[1] == tokens[0]
            status = 'PASS_OUTPUT_TOKENS_ONLY' if aa and b else ('FAIL_B_ONLY_OUTPUT_CHANGE' if aa else 'INCONCLUSIVE_AA_SELF_VARIATION')
            equality.append({'row_id': row['row_id'], 'ordinal': ordinal, 'status': status})
    comparison_exit = (2 if any(x['status'] == 'FAIL_B_ONLY_OUTPUT_CHANGE' for x in equality)
                       else 3 if any(x['status'] == 'INCONCLUSIVE_AA_SELF_VARIATION' for x in equality) else 0)
    return {'status': 'COMPLETE_RAW_RECOMPUTE_NOT_PERFORMANCE_ADMISSION',
            'output_comparison_exit': comparison_exit,
            'HTTP_requests': total, 'outputs': output, 'groups': groups, 'token_equality': equality,
            'n_per_row': 2, 'old_run4_NO_GO': 'UNCHANGED',
            'scope': 'actual emitted token counts/client receipts; not GPU scheduling, logits, or dynamic activation'}


def main():
    os.umask(0o077)
    a = argparse.ArgumentParser()
    a.add_argument('--prep', required=True, type=Path)
    a.add_argument('--window', type=Path)
    a.add_argument('--runtime-root', type=Path)
    a.add_argument('--output', required=True, type=Path)
    args = a.parse_args()
    result = {'runtime': sys.version, 'checker_sha256': sha(Path(__file__))}
    code = 0
    try:
        result['source'] = source_review(args.prep)
        if args.window:
            require(args.runtime_root is not None, 'explicit canonical runtime root required')
            result['actual'] = runtime_review(args.prep, args.window, args.runtime_root)
    except Exception as error:
        result.update(status='FAIL_RETAINED_NO_INFERENCE', error=repr(error), traceback=traceback.format_exc())
        code = 1
    result['exit'] = code
    with args.output.open('xb') as f:
        f.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False).encode() + b'\n')
    print(json.dumps({'exit': code, 'output': str(args.output), 'sha256': sha(args.output)}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
