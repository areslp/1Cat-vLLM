"""Create-only recipe after genuine prelive wire proof and fresh baseline."""
import argparse
from common import PACKAGE, WINDOW, UNITS, TIMER, OUTER, read, save
from plan import mandatory, validate, BUDGETS, RESOURCES
from execution_budget import binding as budget_binding
from identity_contract import UUIDS


def prepare(expected_pid):
    before = read(WINDOW / 'baseline/before-snapshot.json', 4 * 1024**2)
    value = {'schema': 'step58-pure-w2-initial-v1', 'reviewed': False,
             'window': str(WINDOW), 'package': str(PACKAGE),
             'expected_pid': expected_pid, 'arms': ['A0', 'B', 'A2'],
             'units': UNITS, 'outer': OUTER, 'timer': TIMER,
             'device_uuids': list(UUIDS), 'budgets': BUDGETS, 'resources': RESOURCES,
             'execution_budget_amendment': budget_binding(),
             'matrix_requests': 2010, 'maximum_HTTP_requests': 4066,
             'confirmation_authorized': False, 'new_observer_counter_profiler': False,
             'engine_cancel_ack': 'UNVERIFIED', 'engine_cancelled_count': None,
             'native_flags_boolean': before['checks']['flags'],
             'files': mandatory(),
             'restore': 'unchanged 53 requests/17 gates/24 fast/fresh full E7; all best-effort recovery precedes cleanup failure aggregation',
             'parent_approval': 'this source/recipe is not launch permission; exact plan SHA still required'}
    validate(value)
    save(WINDOW / 'control/plan.json', value)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--expected-pid', type=int, required=True)
    args = parser.parse_args()
    prepare(args.expected_pid)
    print('CREATE_ONLY_PURE_W2_INITIAL_PLAN_NOT_LAUNCH_APPROVAL')
