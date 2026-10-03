"""Measured complete-window cap; never delete evidence or interrupt recovery."""
import argparse
from ops import artifact_budget

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('label', choices=('final',))
    args = parser.parse_args()
    raise SystemExit(artifact_budget(args.label))
