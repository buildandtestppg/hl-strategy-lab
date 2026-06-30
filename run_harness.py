"""
HL Strategy Lab — Harness Runner
Single entry point that runs the full harness pipeline:
  1. Test suite (regression tests)
  2. Evaluator (backtest → leaderboard → auto-assign)
  3. Paper trader (live cycle)
  4. Bundler (package data for dashboard)

Usage:
  python3 run_harness.py              # Full pipeline
  python3 run_harness.py --skip-eval  # Skip evaluator (just trade + tests)
  python3 run_harness.py --dry-run    # Dry run evaluator (no config changes)
  python3 run_harness.py --tests-only # Just run tests
"""
import sys
import os
import subprocess
import time
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)

STEPS = {
    "tests": True,
    "evaluator": True,
    "trader": True,
    "bundler": True,
}


def run_step(name, cmd, timeout=600):
    """Run a subprocess step, return (success, output)."""
    print(f"\n{'='*60}")
    print(f"  🔧 Step: {name}")
    print(f"  ⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC")
    print(f"{'='*60}")
    start = time.time()
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout,
            cwd=BASE
        )
        elapsed = time.time() - start
        # Print output
        if result.stdout:
            print(result.stdout.rstrip())
        if result.stderr and "Traceback" in result.stderr:
            print(f"STDERR: {result.stderr.rstrip()}")
        success = result.returncode == 0
        status = "✅" if success else "❌"
        print(f"\n{status} {name} completed in {elapsed:.1f}s (exit {result.returncode})")
        return success, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"\n⏰ {name} timed out after {elapsed:.1f}s")
        return False, f"TIMEOUT after {elapsed:.1f}s"
    except Exception as e:
        print(f"\n❌ {name} failed: {e}")
        return False, str(e)


def main():
    args = sys.argv[1:]
    skip_eval = "--skip-eval" in args
    dry_run = "--dry-run" in args
    tests_only = "--tests-only" in args

    results = {}
    errors = []

    # Step 1: Tests
    if STEPS["tests"] and not tests_only:
        ok, out = run_step("Test Suite", "python3 test_suite.py --save")
        results["tests"] = ok
        if not ok:
            errors.append("Test suite failed")

    # Step 2: Evaluator
    if STEPS["evaluator"] and not skip_eval and not tests_only:
        eval_cmd = "python3 evaluator.py" + (" --dry-run" if dry_run else "")
        ok, out = run_step("Evaluator", eval_cmd)
        results["evaluator"] = ok
        if not ok:
            errors.append("Evaluator failed")

    # Step 3: Paper Trader
    if STEPS["trader"] and not tests_only:
        ok, out = run_step("Paper Trader", "python3 paper_trader.py")
        results["trader"] = ok
        if not ok:
            errors.append("Paper trader failed")

    # Step 4: Bundler (always run if any data changed)
    if STEPS["bundler"] and not tests_only:
        ok, out = run_step("Harness Bundler", "python3 harness_bundler.py")
        results["bundler"] = ok

    # Summary
    print(f"\n{'='*60}")
    print(f"  HARNESS RUN SUMMARY")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"{'='*60}")
    for step, ok in results.items():
        print(f"  {'✅' if ok else '❌'} {step}")
    if errors:
        print(f"\n  ⚠️  Errors: {', '.join(errors)}")
    print(f"{'='*60}")

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
