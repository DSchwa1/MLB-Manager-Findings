"""Master runner for the MLB Manager Firings analysis pipeline.

Usage:
    python run_analysis.py --all            # run all phases 0-10 in order
    python run_analysis.py --phase 0        # run a single phase
    python run_analysis.py --phase 3        # re-run any phase individually
"""

import argparse
import importlib
import sys


PHASE_MODULES = {
    0:  'phase0_audit',
    1:  'phase1_event_table',
    2:  'phase2_game_logs',
    3:  'phase3_metrics',
    4:  'phase4_projections',
    5:  'phase5_control_group',
    6:  'phase6_regression',
    7:  'phase7_secondary',
    8:  'phase8_robustness',
    9:  'phase9_visualizations',
    10: 'phase10_summary',
}


def run_phase(n):
    module_name = PHASE_MODULES[n]
    try:
        mod = importlib.import_module(module_name)
    except ImportError as exc:
        print(f"ERROR: could not import {module_name}: {exc}", file=sys.stderr)
        sys.exit(1)
    if not hasattr(mod, 'run'):
        print(f"ERROR: {module_name} has no run() function.", file=sys.stderr)
        sys.exit(1)
    mod.run()


def main():
    parser = argparse.ArgumentParser(
        description='MLB Manager Firings — analysis pipeline runner'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--all',
        action='store_true',
        help='Run all phases 0–10 in order.',
    )
    group.add_argument(
        '--phase',
        type=int,
        choices=sorted(PHASE_MODULES.keys()),
        metavar='N',
        help=f'Run a single phase (0–{max(PHASE_MODULES)}).',
    )
    args = parser.parse_args()

    if args.all:
        for n in sorted(PHASE_MODULES.keys()):
            print(f"\n{'='*60}")
            print(f"  STARTING PHASE {n}: {PHASE_MODULES[n]}")
            print(f"{'='*60}")
            run_phase(n)
    else:
        run_phase(args.phase)


if __name__ == '__main__':
    main()
