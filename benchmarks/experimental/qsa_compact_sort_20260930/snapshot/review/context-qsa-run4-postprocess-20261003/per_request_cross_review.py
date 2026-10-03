"""Read closed raw SSE; distinguish group classification from request triples."""
import collections
import hashlib
import json
import math
import os
from pathlib import Path

L = Path('/private/tmp/flash-next-step58-20260930')
D = Path('/home/l/work/flash-next/perf-20260923-resume/step58-qsa-compact-sort-20260930')
WINDOW = L / 'context-qsa-run4'
HERE = Path(__file__).resolve().parent
ARMS = ('A0', 'B', 'A2')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def raw_tokens(folder, stored):
    path = folder / 'raw-lines.jsonl'
    assert path.stat().st_size <= 34 * 1024 * 1024
    tokens, finish, usages, done = [], 0, [], 0
    with path.open('rb') as stream:
        for encoded in stream:
            raw = bytes.fromhex(json.loads(encoded)['raw_hex']).strip()
            if not raw:
                continue
            assert not done and raw.startswith(b'data:')
            payload = raw[5:].strip()
            if payload == b'[DONE]':
                done += 1
                continue
            value = json.loads(payload)
            assert value['id'] == stored['identity']['response_id']
            for choice in value['choices']:
                assert choice['index'] == 0
                actual = choice.get('token_ids') or []
                assert all(type(t) is int for t in actual)
                tokens.extend(actual)
                finish += choice.get('finish_reason') is not None
            if value.get('usage') is not None:
                usages.append(value['usage'])
    assert len(tokens) == 256 and tokens == stored['output_token_ids']
    assert finish == done == len(usages) == 1
    assert usages[0] == {'completion_tokens': 256,
                         'prompt_tokens': stored['prompt_tokens'],
                         'total_tokens': stored['prompt_tokens'] + 256}
    return tokens, {'path': str(path.relative_to(L)), 'bytes': path.stat().st_size,
                    'sha256': sha(path), 'client_id': stored['identity']['client_id']}


def main():
    analysis_path = WINDOW / 'analysis/RESULT.json'
    analysis = load(analysis_path)
    groups = {}
    for arm in ARMS:
        result_path = WINDOW / f'control/attempt1/{arm}/http/RESULT.json'
        for record in load(result_path)['group_records']:
            path = L / Path(record['path']).relative_to(D)
            assert sha(path) == record['sha256']
            group = load(path)
            streams, pins = [], []
            for index, stored in enumerate(group['request_results']):
                tokens, pin = raw_tokens(path.parent / f'request-{index:02d}', stored)
                streams.append(tokens)
                pins.append(pin)
            groups[arm, record['row_id'], record['ordinal']] = (streams, pins)
    counts, group_counts = collections.Counter(), collections.Counter()
    rows, b_only = [], []
    main_requests = 0
    for row in analysis['rows']:
        row_counts = collections.Counter()
        repeat_counts = []
        for ordinal in (0, 1):
            trip = [groups[a, row['row_id'], ordinal] for a in ARMS]
            outputs = [g[0] for g in trip]
            assert all(len(s) == row['concurrency'] for s in outputs)
            aa, ba0, ba2 = outputs[0] == outputs[2], outputs[1] == outputs[0], outputs[1] == outputs[2]
            group_status = ('PASS_OUTPUT_TOKENS_ONLY' if aa and ba0 else
                            'FAIL_B_ONLY_OUTPUT_CHANGE' if aa else 'INCONCLUSIVE_AA_SELF_VARIATION')
            original = row['token_equality'][ordinal]
            assert (group_status, aa, ba0, ba2) == (original['status'], original['A0_A2_equal'],
                                                   original['B_A0_equal'], original['B_A2_equal'])
            group_counts[group_status] += 1
            per_repeat = collections.Counter()
            for index, (a0, b, a2) in enumerate(zip(*outputs)):
                main_requests += 3
                status = ('ALL_THREE_EQUAL' if a0 == a2 == b else
                          'A0_A2_EQUAL_B_DIFFERS' if a0 == a2 else 'AA_SELF_VARIATION')
                counts[status] += 1
                row_counts[status] += 1
                per_repeat[status] += 1
                if status == 'A0_A2_EQUAL_B_DIFFERS':
                    first = next(j for j, (x, y) in enumerate(zip(a0, b)) if x != y)
                    b_only.append({'row_id': row['row_id'], 'input_tokens': row['input_tokens'],
                        'concurrency': row['concurrency'], 'repeat': ordinal, 'request_index': index,
                        'first_different_token_index': first, 'A0_A2_token': a0[first],
                        'B_token': b[first], 'group_status': group_status,
                        'raw_streams': {arm: trip[k][1][index] for k, arm in enumerate(ARMS)}})
            repeat_counts.append(dict(per_repeat))
        metric = row['common_decode']
        if metric['status'] == 'DESCRIPTIVE_ONLY_N2':
            gains = [100 * (1 - b / ((a0 + a2) / 2)) for a0, b, a2 in metric['paired_values']]
            drift = [100 * abs(a2 - a0) / ((a0 + a2) / 2) for a0, b, a2 in metric['paired_values']]
            for actual, stored in zip(gains + drift, metric['improvement_pct_per_pair'] + metric['AA_drift_pct_per_pair']):
                assert math.isclose(actual, stored, rel_tol=1e-12, abs_tol=1e-12)
        rows.append({'row_id': row['row_id'], 'counts': dict(row_counts),
                     'repeat_counts': repeat_counts, 'common_statuses': row['common_statuses'],
                     'gains_pct': metric.get('improvement_pct_per_pair'),
                     'AA_drift_pct': metric.get('AA_drift_pct_per_pair'),
                     'candidate_route_expected': row['candidate_route_expected']})
    assert main_requests == 438 and sum(counts.values()) == 146
    report = {'scope': 'Read-only reconstruction of main-request raw tokens; descriptive n2 only; no causal attribution',
        'analysis_sha256': sha(analysis_path),
        'render_source_sha256': sha(L / 'review/context-qsa-closure-20261002/render_results.py'),
        'main_raw_requests': main_requests, 'main_output_tokens': main_requests * 256,
        'request_triples': sum(counts.values()), 'request_classification_counts': dict(counts),
        'unchanged_group_classification_counts': dict(group_counts),
        'group_scope_limitation': 'A/A inequality elsewhere in one group masks a request with A0=A2 but B differs',
        'A0_A2_EQUAL_B_DIFFERS': b_only, 'rows': rows,
        'frozen_diagnostic_exit_unchanged': analysis['diagnostic_exit'],
        'new_runtime_or_tests_executed': False}
    target = HERE / 'PER-REQUEST-CROSS-REVIEW.json'
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'request_counts': dict(counts), 'group_counts': dict(group_counts),
                      'sha256': sha(target), 'bytes': target.stat().st_size}))


if __name__ == '__main__':
    main()
