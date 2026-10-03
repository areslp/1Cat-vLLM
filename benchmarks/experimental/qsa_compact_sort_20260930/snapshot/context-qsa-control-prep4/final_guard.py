"""Full assigned HTTP inventory plus independent real queue/identity closure."""
import argparse
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE / 'control'))
import step58_control as c
from transport_activation import owned
owned()  # Select the reviewed loader before frozen analyze imports dependencies.
sys.path.insert(0, str(PACKAGE.parent / 'context-qsa-prep1'))
from analyze import validate_complete_arm
from original_guard import Observation
from guard_activation import require_binding
from nvml_capture import install_guard
from http_runner import validate_client_resource


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('A0', 'B', 'A2'), required=True)
    arm = parser.parse_args().arm
    root = c.A / arm
    results = validate_complete_arm(root / 'http', arm)
    binding = c.read(root / 'model/binding.json')
    resource_guard = require_binding(binding)
    client_resource = validate_client_resource(root / 'http', arm)
    install_guard(root / 'nvml-final')
    observation = Observation(binding)
    identity = observation.identity()
    queue_empty_epoch = observation.idle()
    e7 = observation.e7()
    c.require(len(e7) == 4 and {r['rank'] for r in e7} == set(range(4)),
              'exact four final original E7 ranks required')
    client = c.state(c.HTTP[arm])
    c.require(client['MainPID'] == '0' and client['ActiveState'] in ('inactive', 'failed'),
              'client closure required')
    c.save(root / 'final-guard.json', {'status': 'PASS_COMPLETE_HTTP_REAL_QUEUE_IDENTITY',
        'arm': arm, 'assigned_groups_validated': len(results), 'identity': identity,
        'resource_guard_identity': resource_guard, 'client_resource': client_resource,
        'queue_empty_epoch': queue_empty_epoch, 'E7_identity': binding['E7_identity'],
        'client_state': client, 'HTTP_result_sha256': c.sha(root / 'http/RESULT.json'),
        'not_candidate_replay_or_performance_admission': True})


if __name__ == '__main__':
    main()
