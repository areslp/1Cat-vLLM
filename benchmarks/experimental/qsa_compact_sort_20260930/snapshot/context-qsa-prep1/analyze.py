"""Length rows remain separate; n2 gives descriptions, no statistical admission."""
import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dependencies import activate, sha
activate()
from io_tools import save
from runner import inventory, load_group, strict_checks

HERE = Path(__file__).resolve().parent


def validate_complete_arm(root, arm):
    root = Path(root)
    final = json.loads((root / 'RESULT.json').read_text())
    if (final['status'] != 'COMPLETE_HTTP_MATRIX_NOT_PERFORMANCE_ADMISSION'
            or final['arm'] != arm or final['completed_groups'] != 38
            or final['measured_requests'] != 146 or final['prime_requests'] != 120
            or final['HTTP_requests'] != 266 or final['outputs'] != 37496):
        raise ValueError('complete fixed arm totals absent')
    matrix = json.loads((HERE / 'matrix.frozen.json').read_text())
    assigned = inventory(matrix, arm)
    if len(final['group_records']) != len(assigned):
        raise ValueError('complete assigned records required')
    results = {}
    for pin, record in zip(assigned, final['group_records']):
        path = Path(record['path'])
        if not path.is_relative_to(root) or sha(path) != record['sha256']:
            raise ValueError('group result ownership/SHA differs')
        value = json.loads(path.read_text())
        group = load_group(pin)
        if (value['arm'] != arm or value['row_id'] != pin['row_id']
                or value['ordinal'] != pin['ordinal']
                or value['status'] != 'COMPLETE_HTTP_GROUP_NOT_PERFORMANCE_ADMISSION'
                or len(value['request_results']) != pin['requests']
                or len(value['priming_results']) != pin['primes']):
            raise ValueError('group/assignment/result count differs')
        strict_checks(value['checks'], group['required_checks'])
        for packet, row in zip(group['requests'], value['request_results']):
            if (row['body_sha256'] != packet['body_sha256']
                    or row['status'] != 'COMPLETE' or len(row['output_token_ids']) != 256
                    or row['prompt_tokens'] != group['input_tokens']):
                raise ValueError('complete measured body/output binding differs')
        for packet, row in zip(group['primes'], value['priming_results']):
            result = row['request_result']
            if (result['body_sha256'] != packet['body_sha256']
                    or result['status'] != 'COMPLETE' or len(result['output_token_ids']) != 1):
                raise ValueError('complete prime body/output binding differs')
            if not row['checks'] or any(v is not True for v in row['checks'].values()):
                raise ValueError('prime native checks incomplete')
        key = (pin['row_id'], pin['ordinal'])
        if key in results:
            raise ValueError('duplicate group ordinal')
        results[key] = value
    return results


def token_comparison(triple):
    outputs = [[r['output_token_ids'] for r in group['request_results']] for group in triple]
    aa = outputs[0] == outputs[2]
    b0, b2 = outputs[1] == outputs[0], outputs[1] == outputs[2]
    status = ('PASS_OUTPUT_TOKENS_ONLY' if aa and b0 else
              'FAIL_B_ONLY_OUTPUT_CHANGE' if aa else 'INCONCLUSIVE_AA_SELF_VARIATION')
    differences = []
    for index, streams in enumerate(zip(*outputs)):
        for pair, a, b in (('A0_A2', streams[0], streams[2]),
                           ('A0_B', streams[0], streams[1]), ('A2_B', streams[2], streams[1])):
            if a != b:
                differences.append({'request_index': index, 'pair': pair,
                    'first_different_token_index': next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)})
    return {'status': status, 'A0_A2_equal': aa, 'B_A0_equal': b0,
            'B_A2_equal': b2, 'first_differences': differences,
            'scope': 'API generated token equality; no expanded logits validation or same GPU execution-plan proof'}


def paired_summary(values):
    if any(v is None or v <= 0 for triple in values for v in triple):
        return {'status': 'INCOMPLETE_METRIC_NO_FILTERING', 'paired_values': values}
    improvements = [100 * (1 - b / ((a0 + a2) / 2)) for a0, b, a2 in values]
    drifts = [100 * abs(a2 - a0) / ((a0 + a2) / 2) for a0, b, a2 in values]
    return {'status': 'DESCRIPTIVE_ONLY_N2', 'paired_values': values,
            'improvement_pct_per_pair': improvements, 'AA_drift_pct_per_pair': drifts,
            'median_paired_improvement_pct': statistics.median(improvements),
            'CI': None, 'performance_admission': 'NOT_EVALUATED_NO_THRESHOLD_CHANGE'}


def run(window):
    window = Path(window)
    arms = {arm: validate_complete_arm(window / 'control/attempt1' / arm / 'http', arm)
            for arm in ('A0', 'B', 'A2')}
    matrix = json.loads((HERE / 'matrix.frozen.json').read_text())
    reports = []
    exits = []
    for row in matrix['rows']:
        triples = [[arms[a][(row['row_id'], ordinal)] for a in ('A0', 'B', 'A2')]
                   for ordinal in range(2)]
        token_rows = [token_comparison(triple) for triple in triples]
        exits.extend(2 if r['status'] == 'FAIL_B_ONLY_OUTPUT_CHANGE' else
                     3 if r['status'] == 'INCONCLUSIVE_AA_SELF_VARIATION' else 0 for r in token_rows)
        all_valid = all(g['metrics']['common_status'] == 'COMMON_WINDOW_OBSERVED'
                        for triple in triples for g in triple)
        common = [[g['metrics']['pooled_complete_interval_ms_per_output_token'] for g in t]
                  for t in triples]
        report = {**row, 'token_equality': token_rows,
            'common_statuses': [[g['metrics']['common_status'] for g in t] for t in triples],
            'common_decode': paired_summary(common) if all_valid else {
                'status': 'NO_COMPLETE_COMMON_COMPARISON_NO_FILTERING', 'paired_values': common},
            'TTFT_client': paired_summary([[g['metrics']['median_TTFT_s'] for g in t] for t in triples]),
            'end_to_end_makespan': paired_summary([[g['metrics']['HTTP_makespan_s'] for g in t] for t in triples]),
            'raw_group_paths': [[str(window / 'control/attempt1' / g['arm'] / 'http')
                                for g in t] for t in triples],
            'priming_cold_c1_TTFT_s': [[[p['request_result']['ttft_s'] for p in g['priming_results']]
                                      for g in t] for t in triples]}
        reports.append(report)
    exit_code = 2 if 2 in exits else 3 if 3 in exits else 0
    output = window / 'analysis'
    output.mkdir(mode=0o700)
    value = {'status': 'COMPLETE_CONTEXT_DESCRIPTIONS_NO_PERFORMANCE_ADMISSION' if exit_code == 0
             else 'STOP_OUTPUT_CHANGE_OR_AA_ATTRIBUTION_UNRESOLVED',
             'diagnostic_exit': exit_code, 'rows': reports, 'n_per_row': 2,
             'HTTP_requests': 798, 'outputs': 112488, 'old_run4_NO_GO': 'UNCHANGED',
             'measurement_scope': 'homogeneous client SSE token-normalized intervals; no GPU timing',
             'strict_logits_scope': 'prior finite numeric evidence unchanged; not extended to these contexts'}
    save(output / 'RESULT.json', value, 4 * 1024**2)
    return exit_code


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', required=True)
    options = parser.parse_args()
    raise SystemExit(run(options.window))
