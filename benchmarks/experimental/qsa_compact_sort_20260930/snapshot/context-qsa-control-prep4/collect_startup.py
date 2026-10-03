"""Isolated CPU process for the unchanged capture gates plus explicit deltas."""
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE.parent / 'service-w2-control-prep4'))
sys.path.insert(0, str(PACKAGE))
from guard_activation import activate
activate()
from nvml_capture import install_guard
from common import WINDOW

def recorded_main():
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--arm', choices=('A0', 'B', 'A2'), required=True)
    args, _ = parser.parse_known_args()
    install_guard(WINDOW / 'control/attempt1' / args.arm / 'nvml-startup')
    from startup import main
    main()

if __name__ == '__main__':
    recorded_main()
