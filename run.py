"""
Handwritten Digit Recognition Pipeline -- Master Runner
========================================================
Datasets  : MNIST, EMNIST (digits), USPS  (all auto-downloaded)
Models    : Logistic Regression, KNN, RFC, SVM, CNN Baseline, CNN Improved
New in v2 : multi-dataset training, data augmentation, GradCAM,
            weighted soft ensemble, cross-dataset transfer matrix.

Usage:
  python run.py                # run all steps 1 to 11
  python run.py --from 5       # run steps 5 to 11 (cascade from step 5)
  python run.py --only 5       # run only step 5 in isolation
  python run.py --list         # show all steps
"""

import argparse
import importlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config


def print_header():
    print("\n" + "=" * 60)
    print("  MNIST Handwritten Digit Recognition -- Pipeline")
    print("=" * 60)


def list_steps():
    print_header()
    print("\n  Available steps:\n")
    for num, name in config.STEPS.items():
        label = name.replace("step_0", "").replace("step_", "").replace("_", " ").title()
        print(f"    {num:>2}.  {label}")
    print()


def run_step(step_num):
    module_name = config.STEPS[step_num]
    label = module_name.replace("_", " ").title()

    print(f"\n{'=' * 60}")
    print(f"  Step {step_num:>2} | {label}")
    print(f"{'=' * 60}")

    t0 = time.time()
    try:
        mod = importlib.import_module(f"pipeline.{module_name}")
        importlib.reload(mod)
        mod.main()
        elapsed = time.time() - t0
        print(f"\n  OK  Step {step_num} completed in {elapsed:.1f}s")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  FAIL  Step {step_num} failed after {elapsed:.1f}s")
        print(f"    Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_from(start):
    steps = [n for n in sorted(config.STEPS) if n >= start]
    print_header()
    print(f"\n  Running steps {start} to {steps[-1]}  ({len(steps)} steps)\n")
    t0 = time.time()

    for step_num in steps:
        ok = run_step(step_num)
        if not ok:
            print(f"\n  Pipeline stopped at step {step_num}.")
            print("  Fix the error and re-run from this step:\n")
            print(f"    python run.py --from {step_num}\n")
            sys.exit(1)

    total = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  All steps complete in {total:.1f}s")
    print(f"  Outputs saved to: {config.OUTPUTS_DIR}")
    print("=" * 60 + "\n")


def run_only(step_num):
    print_header()
    print(f"\n  Running step {step_num} in isolation\n")
    ok = run_step(step_num)
    if ok:
        print(f"\n  Outputs saved to: {config.OUTPUTS_DIR}\n")
    else:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="MNIST Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--from", dest="from_step", type=int, metavar="N",
        help="run steps N through 11 (cascade)",
    )
    group.add_argument(
        "--only", dest="only_step", type=int, metavar="N",
        help="run only step N",
    )
    group.add_argument(
        "--list", action="store_true",
        help="list all available steps",
    )
    args = parser.parse_args()

    all_steps = sorted(config.STEPS.keys())

    if args.list:
        list_steps()

    elif args.only_step is not None:
        if args.only_step not in config.STEPS:
            print(f"  Error: step {args.only_step} does not exist.")
            print(f"  Valid steps: {all_steps}")
            sys.exit(1)
        run_only(args.only_step)

    elif args.from_step is not None:
        if args.from_step not in config.STEPS:
            print(f"  Error: step {args.from_step} does not exist.")
            print(f"  Valid steps: {all_steps}")
            sys.exit(1)
        run_from(args.from_step)

    else:
        run_from(1)


if __name__ == "__main__":
    main()
