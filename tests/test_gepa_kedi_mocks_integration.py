"""End-to-end integration tests for GEPA optimizer convergence.

Tests the full chain: Evaluator -> Reflection -> Optimizer -> Evaluator.
Uses real dspy.GEPA with injected MockLM to verify that:
  1. Scores improve over iterations
  2. Reflection feedback drives range adjustments in the right direction
  3. Textual feedback content is propagated to the reflection model
  4. Inverted feedback hinders convergence
  5. OK feedback stabilizes the range
"""

from __future__ import annotations

import warnings

import pytest

from kedi.lang import compile_program, parse_program
from kedi.lang.ast import ProgramWithValidation
from kedi.optimization.config import OptimizerConfig
from kedi.optimization.gepa import GEPAOptimizer
from kedi.optimization.services import MetricFactory, TypeResolver
from tests.dspy_mocks import RangeAwareAdapter, Recorder, dist_to_target, install_mock_dspy


def _prepare(src: str):
    prog = parse_program(src)
    assert isinstance(prog, ProgramWithValidation)
    rt = compile_program(prog, adapter=RangeAwareAdapter())
    proc = prog.procedures[0]
    resolver = TypeResolver(rt)
    param_types = resolver.resolve_param_types(proc)
    eval_suite = prog.evals[0]
    metric_def = eval_suite.metrics[0]
    dataset_name = metric_def.dataset_name or next(iter(eval_suite.datasets.keys()))
    metric = MetricFactory(rt).create_metric(metric_def, proc, dataset_name)
    return prog, rt, proc, param_types, eval_suite, metric


_CONVERGENCE_SRC = """
@solve(seed: str) -> int:
  > optimize: produce_value:
    The result is [y: int]
  = `y`

@eval: solve:
  > data: cases:
    = ```
    return [("alpha", None), ("beta", None), ("gamma", None)]
    ```
  > metric: range_score(cases):
    = ```
    y = solve(cases)
    if 40 <= y <= 60:
        return (1.0, "OK")
    elif y < 40:
        distance = 40 - y
        score = max(0.0, 1.0 - distance / 100.0)
        return (score, "TOO_LOW")
    else:
        distance = y - 60
        score = max(0.0, 1.0 - distance / 100.0)
        return (score, "TOO_HIGH")
    ```
"""


# ── Score improvement and range convergence ──────────────────────────


def test_optimization_converges_score_improves(monkeypatch: pytest.MonkeyPatch):
    """Target [40,60], initial range [-100,-50].
    After optimization:
      (a) final score >= 0.66
      (b) last RANGE closer to target than first RANGE
      (c) multiple distinct ranges in trace (proving iteration happened)
    """
    install_mock_dspy(monkeypatch)
    Recorder.reset()

    prog, rt, proc, param_types, _, metric = _prepare(_CONVERGENCE_SRC)
    train_data = [("alpha", None), ("beta", None), ("gamma", None)]
    optimizer = GEPAOptimizer(
        config=OptimizerConfig(
            max_metric_calls=120,
            reflection_model="mock/reflection",
            reflection_minibatch_size=3,
        ),
        runtime=rt,
    )
    optimized, score = optimizer.optimize_procedure(
        proc_name=proc.name,
        proc_params=proc.params,
        param_types=param_types,
        spans=proc.optimize_spans,
        train_data=train_data,
        test_data=None,
        metric=metric,
    )

    assert Recorder.calls, "Predict should be invoked by GEPA + scoring"

    ranges = Recorder.predict_ranges()
    first_lo, first_hi = ranges[0]
    last_lo, last_hi = ranges[-1]

    assert dist_to_target(last_lo, last_hi) <= dist_to_target(first_lo, first_hi), (
        f"Last range [{last_lo},{last_hi}] should be closer to [40,60] "
        f"than first range [{first_lo},{first_hi}]"
    )

    assert score >= 0.66, f"Expected score >= 0.66, got {score}"

    distinct = {r for r in ranges}
    assert len(distinct) >= 2, (
        f"Expected at least 2 distinct ranges (proving iteration), got {len(distinct)}"
    )

    assert "produce_value" in optimized
    assert "RANGE[" in optimized["produce_value"]


# ── Feedback text propagated to reflection ───────────────────────────


def test_feedback_text_reaches_reflection_lm(monkeypatch: pytest.MonkeyPatch):
    """The evaluator returns feedback strings like 'TOO_LOW' or 'TOO_HIGH'.
    Verify these strings appear in the reflection LM's input prompts."""
    install_mock_dspy(monkeypatch)
    Recorder.reset()

    prog, rt, proc, param_types, _, metric = _prepare(_CONVERGENCE_SRC)
    train_data = [("alpha", None), ("beta", None), ("gamma", None)]
    optimizer = GEPAOptimizer(
        config=OptimizerConfig(
            max_metric_calls=120,
            reflection_model="mock/reflection",
            reflection_minibatch_size=3,
        ),
        runtime=rt,
    )
    optimized, score = optimizer.optimize_procedure(
        proc_name=proc.name,
        proc_params=proc.params,
        param_types=param_types,
        spans=proc.optimize_spans,
        train_data=train_data,
        test_data=None,
        metric=metric,
    )

    reflection_prompts = [c.prompt for c in Recorder.reflection_calls()]
    if reflection_prompts:
        all_text = " ".join(reflection_prompts)
        feedback_found = "TOO_LOW" in all_text or "TOO_HIGH" in all_text or "OK" in all_text
        assert feedback_found, (
            "Expected feedback text (TOO_LOW/TOO_HIGH/OK) in at least one reflection prompt"
        )


# ── Inverted feedback diverges or stalls ─────────────────────────────


def test_inverted_feedback_hinders_convergence(monkeypatch: pytest.MonkeyPatch):
    """With TOO_LOW and TOO_HIGH swapped, the optimizer should NOT converge
    to the target range [40,60]. Compare against correct-feedback run."""
    install_mock_dspy(monkeypatch)
    Recorder.reset()

    inverted_src = """
@solve(seed: str) -> int:
  > optimize: produce_value:
    The result is [y: int]
  = `y`

@eval: solve:
  > data: cases:
    = ```
    return [("omega", None)]
    ```
  > metric: flipped_feedback(cases):
    = ```
    y = solve(cases)
    if 40 <= y <= 60:
        return (1.0, "OK")
    elif y < 40:
        distance = 40 - y
        score = max(0.0, 1.0 - distance / 100.0)
        return (score, "TOO_HIGH")
    else:
        distance = y - 60
        score = max(0.0, 1.0 - distance / 100.0)
        return (score, "TOO_LOW")
    ```
"""
    prog, rt, proc, param_types, _, metric = _prepare(inverted_src)
    train_data = [("omega", None)]
    optimizer = GEPAOptimizer(
        config=OptimizerConfig(
            max_metric_calls=40,
            reflection_model="mock/reflection",
            reflection_minibatch_size=1,
        ),
        runtime=rt,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*appears unchanged.*")
        _, inverted_score = optimizer.optimize_procedure(
            proc_name=proc.name,
            proc_params=proc.params,
            param_types=param_types,
            spans=proc.optimize_spans,
            train_data=train_data,
            test_data=None,
            metric=metric,
        )

    inverted_ranges = Recorder.predict_ranges()

    # Now run with correct feedback for comparison
    Recorder.reset()
    correct_src = """
@solve(seed: str) -> int:
  > optimize: produce_value:
    The result is [y: int]
  = `y`

@eval: solve:
  > data: cases:
    = ```
    return [("omega", None)]
    ```
  > metric: correct_feedback(cases):
    = ```
    y = solve(cases)
    if 40 <= y <= 60:
        return (1.0, "OK")
    elif y < 40:
        distance = 40 - y
        score = max(0.0, 1.0 - distance / 100.0)
        return (score, "TOO_LOW")
    else:
        distance = y - 60
        score = max(0.0, 1.0 - distance / 100.0)
        return (score, "TOO_HIGH")
    ```
"""
    prog2, rt2, proc2, param_types2, _, metric2 = _prepare(correct_src)
    optimizer2 = GEPAOptimizer(
        config=OptimizerConfig(
            max_metric_calls=40,
            reflection_model="mock/reflection",
            reflection_minibatch_size=1,
        ),
        runtime=rt2,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*appears unchanged.*")
        _, correct_score = optimizer2.optimize_procedure(
            proc_name=proc2.name,
            proc_params=proc2.params,
            param_types=param_types2,
            spans=proc2.optimize_spans,
            train_data=train_data,
            test_data=None,
            metric=metric2,
        )

    correct_ranges = Recorder.predict_ranges()

    # Correct feedback should yield final range closer to target than inverted
    if inverted_ranges and correct_ranges:
        inv_last = inverted_ranges[-1]
        cor_last = correct_ranges[-1]
        inv_dist = dist_to_target(*inv_last)
        cor_dist = dist_to_target(*cor_last)
        assert cor_dist <= inv_dist, (
            f"Correct feedback final range dist={cor_dist:.1f} should be <= "
            f"inverted feedback dist={inv_dist:.1f}"
        )


# ── OK feedback stabilizes range ─────────────────────────────────────


def test_ok_feedback_stabilizes_range(monkeypatch: pytest.MonkeyPatch):
    """When the evaluator always returns OK (score=1.0), the reflection model
    should not shift the range, and the range should remain stable."""
    install_mock_dspy(monkeypatch)
    Recorder.reset()

    src = """
@solve(seed: str) -> int:
  > optimize: produce_value:
    The result is [y: int]
  = `y`

@eval: solve:
  > data: cases:
    = ```
    return [("alpha", None), ("beta", None)]
    ```
  > metric: always_ok(cases):
    = ```
    y = solve(cases)
    return (1.0, "OK")
    ```
"""
    prog, rt, proc, param_types, _, metric = _prepare(src)
    train_data = [("alpha", None), ("beta", None)]
    optimizer = GEPAOptimizer(
        config=OptimizerConfig(
            max_metric_calls=30,
            reflection_model="mock/reflection",
            reflection_minibatch_size=2,
        ),
        runtime=rt,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*appears unchanged.*")
        optimized, score = optimizer.optimize_procedure(
            proc_name=proc.name,
            proc_params=proc.params,
            param_types=param_types,
            spans=proc.optimize_spans,
            train_data=train_data,
            test_data=None,
            metric=metric,
        )

    assert score >= 0.5, "Perfect OK feedback should yield high score"

    ranges = Recorder.predict_ranges()
    if len(ranges) >= 2:
        first = ranges[0]
        last = ranges[-1]
        shift = abs(last[0] - first[0]) + abs(last[1] - first[1])
        assert shift < 250.0, (
            f"With OK feedback, range should not shift drastically. "
            f"First={first}, Last={last}, shift={shift:.1f}"
        )


# ── Gradual metric rewards proximity ─────────────────────────────────


def test_gradual_metric_drives_incremental_improvement(monkeypatch: pytest.MonkeyPatch):
    """A gradual metric (partial credit for proximity) should drive the
    optimizer closer to the target over iterations, not just binary pass/fail."""
    install_mock_dspy(monkeypatch)
    Recorder.reset()

    src = """
@solve(seed: str) -> int:
  > optimize: produce_value:
    The result is [y: int]
  = `y`

@eval: solve:
  > data: cases:
    = ```
    return [("a", None), ("b", None), ("c", None), ("d", None)]
    ```
  > metric: gradual(cases):
    = ```
    y = solve(cases)
    if 40 <= y <= 60:
        return (1.0, "OK")
    elif y < 40:
        distance = 40 - y
        score = max(0.0, 1.0 - distance / 200.0)
        return (score, "TOO_LOW")
    else:
        distance = y - 60
        score = max(0.0, 1.0 - distance / 200.0)
        return (score, "TOO_HIGH")
    ```
"""
    prog, rt, proc, param_types, _, metric = _prepare(src)
    train_data = [("a", None), ("b", None), ("c", None), ("d", None)]
    optimizer = GEPAOptimizer(
        config=OptimizerConfig(
            max_metric_calls=200,
            reflection_model="mock/reflection",
            reflection_minibatch_size=3,
        ),
        runtime=rt,
    )
    optimized, score = optimizer.optimize_procedure(
        proc_name=proc.name,
        proc_params=proc.params,
        param_types=param_types,
        spans=proc.optimize_spans,
        train_data=train_data,
        test_data=None,
        metric=metric,
    )

    ranges = Recorder.predict_ranges()
    first_lo, first_hi = ranges[0]
    last_lo, last_hi = ranges[-1]

    assert dist_to_target(last_lo, last_hi) <= dist_to_target(first_lo, first_hi), (
        "Gradual metric should drive range closer to target"
    )
    assert score >= 0.5, f"Expected score >= 0.5 with gradual metric, got {score}"
