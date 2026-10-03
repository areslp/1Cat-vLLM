"""Partial evidence review: complete raw streams never imply an arm/guard PASS."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
from independent_review import common_metrics, pin, raw_stream, read, require, sha

FULL_CHECKER_SHA = '80ffc5e2d03e907aa1629368ce2d5a66ad5b9a5cb7c3db05e60f754f4ddfffcf'


def actual_request(folder, packet):
    for name in ('request.json', 'result.json', 'raw-lines.jsonl',
                 'process-exit.json', 'native-close.json'):
        require((folder / name).is_file(), 'request evidence missing: ' + str(folder / name))
    require(read(folder / 'request.json') == packet, 'submitted packet differs from frozen packet')
    row = read(folder / 'result.json')
    pin(folder / 'raw-lines.jsonl', row['raw_response'])
    raw_stream(folder / 'raw-lines.jsonl', packet, row)
    proc, close = read(folder / 'process-exit.json'), read(folder / 'native-close.json')
    require(proc['actual_child_exit'] == 0 and proc['process_reaped'] is True and
            proc['abort_reason'] is None, 'request child exit/reap')
    require(close['pid'] == proc['pid'] and close['response_existed'] is True and
            close['response_close_returned'] is True and close['torch_imported'] is False and
            close['CVD'] == '', 'request close/identity/CPU client')
    require(proc['release_monotonic'] <= row['started_monotonic'] < row['finished_monotonic']
            <= proc['deadline_monotonic'], 'native child deadline')
    return row


def run(prep, window, canonical_root):
    helper = Path(__file__).with_name('independent_review.py')
    require(sha(helper) == FULL_CHECKER_SHA, 'independent full checker changed')
    matrix = read(prep / 'matrix.frozen.json')
    records, verified_requests, verified_outputs = [], 0, 0
    guard_closed = 0
    for arm in matrix['arms']:
        root = window / 'control/attempt1' / arm / 'http'
        assigned = [p for p in matrix['groups'] if p['arm'] == arm]
        for index, frozen_pin in enumerate(assigned):
            group_root = root / f'{index:03d}-{frozen_pin["row_id"]}-repeat{frozen_pin["ordinal"]}'
            if not group_root.exists():
                continue
            pin(prep / frozen_pin['path'], frozen_pin)
            frozen = read(prep / frozen_pin['path'])
            value = {'arm': arm, 'index': index, 'row_id': frozen_pin['row_id'],
                     'ordinal': frozen_pin['ordinal'], 'group_guard_closed': False,
                     'streams': [], 'errors': [], 'failure_files': []}
            expected_remote = canonical_root / group_root.relative_to(window)
            main_rows = []
            for kind, packets in (('prime', frozen['primes']), ('request', frozen['requests'])):
                for j, packet in enumerate(packets):
                    folder = group_root / f'{kind}-{j:02d}'
                    entry = {'kind': kind, 'index': j, 'request_id': packet['body']['request_id']}
                    try:
                        row = actual_request(folder, packet)
                        require(Path(row['raw_response']['path']) ==
                                expected_remote / folder.name / 'raw-lines.jsonl', 'raw path identity')
                        entry.update(status='PASS_RAW_STREAM_ONLY', tokens=len(row['output_token_ids']),
                                     prompt_tokens=row['prompt_tokens'], TTFT_s=row['ttft_s'],
                                     raw_sha256=sha(folder / 'raw-lines.jsonl'))
                        verified_requests += 1
                        verified_outputs += len(row['output_token_ids'])
                        if kind == 'request':
                            main_rows.append(row)
                    except Exception as error:
                        entry.update(status='UNVERIFIED_OR_FAILED_STREAM', error=repr(error))
                    value['streams'].append(entry)
            if len(main_rows) == len(frozen['requests']):
                value['raw_common_metrics'] = common_metrics(main_rows)
            if (group_root / 'GROUP.json').exists():
                try:
                    group = read(group_root / 'GROUP.json')
                    receipt = read(root / f'group-complete-{index:03d}.json')
                    require(Path(receipt['path']) == expected_remote / 'GROUP.json' and
                            sha(group_root / 'GROUP.json') == receipt['sha256'], 'closed GROUP receipt')
                    require(group['status'] == 'COMPLETE_HTTP_GROUP_NOT_PERFORMANCE_ADMISSION' and
                            (group['arm'], group['row_id'], group['ordinal']) ==
                            (arm, frozen_pin['row_id'], frozen_pin['ordinal']), 'GROUP identity')
                    require(set(group['checks']) == set(frozen['required_checks']) and
                            all(v is True for v in group['checks'].values()), 'GROUP guard incomplete')
                    require(all(x['status'] == 'PASS_RAW_STREAM_ONLY' for x in value['streams']), 'GROUP raw incomplete')
                    value['group_guard_closed'] = True
                    guard_closed += 1
                except Exception as error:
                    value['errors'].append(repr(error))
            for name in ('FAILURE.json', 'CLEANUP-FAILURE.json'):
                p = group_root / name
                if p.exists():
                    value['failure_files'].append({'path': str(p), 'sha256': sha(p), 'value': read(p)})
            value['status'] = ('CLOSED_GROUP_SUBGATE_ONLY' if value['group_guard_closed'] else
                               'GUARD_UNCLOSED_OR_FAILED_NO_GROUP_PASS')
            records.append(value)
    return {'status': 'PARTIAL_RAW_REVIEW_NOT_ARM_PASS_OR_FULL_MATRIX_PASS',
            'group_directories_observed': len(records), 'guard_closed_groups': guard_closed,
            'raw_verified_requests': verified_requests, 'raw_verified_outputs': verified_outputs,
            'expected_HTTP_requests': 798, 'expected_outputs': 112488, 'groups': records,
            'old_run4_NO_GO': 'UNCHANGED', 'performance_admission': 'NONE',
            'scope': 'local mirrored stream/child evidence; missing GPU-memory failure sample remains missing'}


def main():
    os.umask(0o077)
    a = argparse.ArgumentParser()
    a.add_argument('--prep', required=True, type=Path)
    a.add_argument('--window', required=True, type=Path)
    a.add_argument('--runtime-root', required=True, type=Path)
    a.add_argument('--output', required=True, type=Path)
    opt = a.parse_args()
    result = {'script_sha256': sha(Path(__file__)), 'runtime': sys.version}
    code = 0
    try:
        result['actual'] = run(opt.prep, opt.window, opt.runtime_root)
    except Exception as error:
        result.update(status='PARTIAL_REVIEW_FAILURE_PRESERVED', error=repr(error),
                      traceback=traceback.format_exc())
        code = 1
    result['checker_exit'] = code
    result['checker_exit_scope'] = 'reviewer execution only, never service/window success'
    with opt.output.open('xb') as f:
        f.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False).encode() + b'\n')
    print(json.dumps({'checker_exit': code, 'output': str(opt.output), 'sha256': sha(opt.output)}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
