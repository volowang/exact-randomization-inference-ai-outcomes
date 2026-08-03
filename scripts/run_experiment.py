#!/usr/bin/env python3
"""Run exact randomization inference simulations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exact_ai_inference.simulation import run_simulation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-units", type=int, default=10)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = run_simulation(args.n_units, args.trials, args.alpha, args.seed)
    _print_summary(result["summary"])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nwrote {args.output}")


def _print_summary(rows):
    print(f"{'scenario':<18} {'method':<10} {'coverage':>9} {'avg_width':>10} {'runtime_ms':>11}")
    for row in rows:
        print(
            f"{row['scenario']:<18} "
            f"{row['method']:<10} "
            f"{row['coverage']:>9.3f} "
            f"{row['avg_width']:>10.3f} "
            f"{row['avg_runtime_ms']:>11.1f}"
        )


if __name__ == "__main__":
    main()
