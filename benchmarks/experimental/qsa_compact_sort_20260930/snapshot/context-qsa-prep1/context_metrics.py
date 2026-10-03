"""Count output tokens in exact client-positive intervals, never GPU steps."""
import math
import statistics


def finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('nonfinite client time')
    return value


def request_trace(row):
    started = finite(row['started_monotonic'])
    finished = finite(row['finished_monotonic'])
    if finished <= started or row['status'] != 'COMPLETE':
        raise ValueError('unfinished/nonpositive request elapsed')
    chunks = row['positive_chunks']
    if not chunks or sum(c['tokens'] for c in chunks) != 256:
        raise ValueError('complete256token positive trace required')
    absolute = []
    for chunk in chunks:
        if type(chunk['tokens']) is not int or chunk['tokens'] <= 0:
            raise ValueError('positive output token extent required')
        point = started + finite(chunk['elapsed_s'])
        if not started <= point <= finished or absolute and point <= absolute[-1]['at']:
            raise ValueError('client chunk clock nonmonotonic/outside HTTP')
        absolute.append({'at': point, 'tokens': chunk['tokens'], 'ordinal': chunk['ordinal']})
    return absolute


def measure(rows):
    if not 1 <= len(rows) <= 8:
        raise ValueError('finite c1..8 homogeneous group required')
    if len({r['prompt_tokens'] for r in rows}) != 1:
        raise ValueError('mixed prompt lengths prohibited')
    traces = [request_trace(row) for row in rows]
    start = max(trace[0]['at'] for trace in traces)
    end = min(trace[-1]['at'] for trace in traces)
    per_request = []
    complete_tokens, full_interval_s = 0, 0.0
    for index, (row, trace) in enumerate(zip(rows, traces)):
        intervals = [(a, b) for a, b in zip(trace, trace[1:])]
        full_tokens = sum(b['tokens'] for a, b in intervals)
        full_s = trace[-1]['at'] - trace[0]['at']
        common = [(a, b) for a, b in intervals if start <= a['at'] < b['at'] <= end]
        crossing = [(a, b) for a, b in intervals
                    if max(a['at'], start) < min(b['at'], end)
                    and not start <= a['at'] < b['at'] <= end]
        tokens = sum(b['tokens'] for a, b in common)
        seconds = sum(b['at'] - a['at'] for a, b in common)
        complete_tokens += tokens
        full_interval_s += seconds
        per_request.append({'index': index, 'ttft_s': row['ttft_s'],
            'HTTP_elapsed_s': row['finished_monotonic'] - row['started_monotonic'],
            'full_decode_client_s': full_s,
            'full_decode_tokens_after_first_chunk': full_tokens,
            'full_decode_client_tokens_s': full_tokens / full_s if full_s > 0 else None,
            'full_decode_client_ms_per_output_token': 1000 * full_s / full_tokens if full_tokens else None,
            'whole_frozen_perf38_inter_chunk_median_ms': row['step_ms'],
            'common_complete_intervals': len(common), 'common_tokens': tokens,
            'common_covered_interval_s': seconds,
            'common_client_tokens_s': tokens / seconds if seconds > 0 else None,
            'common_client_ms_per_output_token': 1000 * seconds / tokens if tokens else None,
            'boundary_crossing_intervals_excluded': len(crossing),
            'common_interval_ordinal_pairs': [[a['ordinal'], b['ordinal']] for a, b in common]})
    status = ('NO_COMMON_WINDOW' if end <= start else
              'INSUFFICIENT_COMMON_EVENTS' if any(r['common_complete_intervals'] == 0
                                                for r in per_request) else
              'COMMON_WINDOW_OBSERVED')
    duration = max(0.0, end - start)
    makespan = max(r['finished_monotonic'] for r in rows) - min(r['started_monotonic'] for r in rows)
    return {'common_status': status, 'common_start_monotonic': start,
            'common_end_monotonic': end, 'common_client_window_s': duration,
            'common_complete_interval_tokens': complete_tokens,
            'common_aggregate_client_tokens_s': complete_tokens / duration if duration else None,
            'common_aggregate_client_ms_per_output_token': 1000 * duration / complete_tokens
                if complete_tokens and duration else None,
            'common_all_requests_interval_time_s': full_interval_s,
            'primary_decode_metric': 'pooled_complete_interval_ms_per_output_token',
            'pooled_complete_interval_ms_per_output_token':
                1000 * full_interval_s / complete_tokens if complete_tokens else None,
            'pooled_complete_interval_client_tokens_s':
                complete_tokens / full_interval_s if full_interval_s > 0 else None,
            'per_request': per_request,
            'median_TTFT_s': statistics.median(r['ttft_s'] for r in rows),
            'HTTP_makespan_s': makespan,
            'end_to_end_client_tokens_s': len(rows) * 256 / makespan,
            'all_request_frozen_perf38_median_ms': statistics.median(r['step_ms'] for r in rows)
                if all(r['step_ms'] is not None for r in rows) else None,
            'interpretation': 'client receipt times and actual emitted token counts; not GPU steps or GPU busy time',
            'common_boundary_policy': 'only complete positive-to-positive intervals inside max(first)..min(last); no token interpolation',
            'aggregate_full_window_scope': 'descriptive conservative receipt throughput; excluded boundary intervals may bias it; not the main decode comparison',
            'uncounted_first_chunk_tokens': [t[0]['tokens'] for t in traces]}
