#!/usr/bin/env python3
"""AIME benchmark runner: baseline vs GEPA-optimized evaluation.

Mirrors the DSPy AIME tutorial using Kedi's optimization framework.
Evaluates GPT-4.1 Mini on AIME 2025 problems before and after GEPA optimization.

Usage:
    uv run examples/aime.py
    uv run examples/aime.py --optimizer-model openai/gpt-4.1 --max-metric-calls 560
"""

import argparse
from importlib.util import find_spec
import json
import os
import sys
from pathlib import Path

# Check dependencies before importing Kedi
if find_spec("datasets") is None:
    print("Error: 'datasets' package is required but not installed.", file=sys.stderr)
    print("Install it with: uv add datasets", file=sys.stderr)
    sys.exit(1)

if find_spec("dspy") is None:
    print("Error: 'dspy-ai' package is required but not installed.", file=sys.stderr)
    print("Install it with: uv add dspy-ai", file=sys.stderr)
    sys.exit(1)

if not os.environ.get("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
    print("Set it with: export OPENAI_API_KEY=your_key_here", file=sys.stderr)
    print("Continuing anyway (may fail during optimization)...\n", file=sys.stderr)

from kedi.app import run_evals_cli, run_optimization_cli

DEFAULT_SOURCE = str(Path(__file__).with_name("aime.kedi"))


def read_metric_score(text: str) -> float:
    """Parse metric score from eval output.

    Expected format: "- solve_aime::accuracy: 0.5667"
    """
    for line in text.splitlines():
        if line.strip().startswith("- ") and "::" in line and ":" in line:
            try:
                return float(line.rsplit(":", 1)[-1].strip())
            except Exception:
                pass
    return float("nan")


def print_optimized_prompt(source: Path):
    """Print the optimized prompt from .optimized.json file."""
    opt = source.with_suffix(source.suffix + ".optimized.json")
    if opt.exists():
        data = json.loads(opt.read_text(encoding="utf-8"))
        span = data.get("solve_aime", {}).get("predict")
        if span:
            print("\n" + "=" * 80)
            print("Optimized prompt for solve_aime[predict]:")
            print("=" * 80)
            print(span)
            print("=" * 80 + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="Run AIME benchmark with baseline and GEPA-optimized evaluation"
    )
    ap.add_argument("--source", default=DEFAULT_SOURCE, help="Path to aime.kedi")
    ap.add_argument(
        "--optimizer-model",
        default="openai/gpt-4.1",
        help="Model for GEPA optimizer (default: openai/gpt-4.1)",
    )
    ap.add_argument(
        "--reflection-model",
        default="openai/gpt-4.1",
        help="Model for GEPA reflection (default: openai/gpt-4.1)",
    )
    ap.add_argument(
        "--max-metric-calls",
        type=int,
        default=560,
        help="GEPA optimization budget (default: 560)",
    )
    args = ap.parse_args()

    src = Path(args.source).resolve()

    if not src.exists():
        print(f"Error: Source file not found: {src}", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print("AIME Benchmark: Baseline vs GEPA-Optimized")
    print("=" * 80)
    print(f"Source: {src.name}")
    print(f"Optimizer model: {args.optimizer_model}")
    print(f"Reflection model: {args.reflection_model}")
    print(f"Max metric calls: {args.max_metric_calls}")
    print("=" * 80 + "\n")

    # Step 1: Baseline evaluation (test set preferred via test_data: train)
    print("Step 1: Running baseline evaluation...")
    print("-" * 80)
    code, baseline_out = run_evals_cli(str(src))
    print(baseline_out)
    baseline = read_metric_score(baseline_out)
    print(f"\nBaseline score: {baseline:.4f}")
    print("=" * 80 + "\n")

    # Step 2: Optimize with GEPA
    print("Step 2: Optimizing with GEPA...")
    print("-" * 80)
    code, msg = run_optimization_cli(
        str(src),
        optimizer="gepa",
        optimizer_model=args.optimizer_model,
        reflection_model=args.reflection_model,
        max_metric_calls=args.max_metric_calls,
    )
    if code != 0:
        print(f"Optimization failed: {msg}", file=sys.stderr)
        sys.exit(1)
    print(msg)
    print_optimized_prompt(src)

    # Step 3: Re-run evals with optimized prompts injected
    print("Step 3: Running optimized evaluation...")
    print("-" * 80)
    code, optimized_out = run_evals_cli(str(src))
    print(optimized_out)
    optimized = read_metric_score(optimized_out)
    print(f"\nOptimized score: {optimized:.4f}")
    print("=" * 80 + "\n")

    # Step 4: Report results
    if not (baseline != baseline or optimized != optimized):  # both are valid numbers
        delta = optimized - baseline
        pct_change = (delta / baseline * 100) if baseline > 0 else 0

        print("=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        print(f"Baseline:  {baseline:.4f} ({baseline * 100:.2f}%)")
        print(f"Optimized: {optimized:.4f} ({optimized * 100:.2f}%)")
        print(f"Delta:     {delta:+.4f} ({pct_change:+.2f}%)")
        print("=" * 80)

        if delta > 0:
            print(f"\n✓ GEPA improved performance by {abs(delta):.4f} ({abs(pct_change):.2f}%)")
        elif delta < 0:
            print(f"\n✗ Performance decreased by {abs(delta):.4f} ({abs(pct_change):.2f}%)")
        else:
            print("\n→ No change in performance")
    else:
        print("Warning: Could not parse metric scores", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
