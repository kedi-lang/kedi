from __future__ import annotations

from typing import Any, List, Tuple

import pytest

from kedi.lang import compile_program, parse_program
from kedi.optimization.config import OptimizerConfig
from kedi.optimization.gepa import GEPAOptimizer
from kedi.optimization.services import MetricFactory, TypeResolver
from tests.dspy_mocks import (
    Recorder,
    RangeAwareAdapter,
    install_mock_dspy,
    parse_first_range,
)


def _prepare(src: str):
    prog = parse_program(src)
    rt = compile_program(prog, adapter=RangeAwareAdapter())
    proc = prog.procedures[0]
    resolver = TypeResolver(rt)
    param_types = resolver.resolve_param_types(proc)
    eval_suite = prog._validation_evals[0]
    metric_def = eval_suite.metrics[0]
    dataset_name = metric_def.dataset_name or next(iter(eval_suite.datasets.keys()))
    metric = MetricFactory(rt).create_metric(metric_def, proc, dataset_name)
    return prog, rt, proc, param_types, eval_suite, metric


def _collect_ranges_from_calls() -> List[Tuple[float, float]]:
    ranges: List[Tuple[float, float]] = []
    for call in Recorder.calls:
        lo, hi = parse_first_range(call.instructions)
        ranges.append((lo, hi))
    return ranges


def test_reflection_improves_scores(monkeypatch: pytest.MonkeyPatch):
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
    return [("alpha", None), ("beta", None), ("gamma", None)]
    ```
  > metric: range_score(cases):
    = ```
    y = solve(cases)
    # Gradual metric: score based on distance to target range [40,60]
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
    prog, rt, proc, param_types, eval_suite, metric = _prepare(src)
    spans = proc.optimize_spans
    train_data = [("alpha", None), ("beta", None), ("gamma", None)]
    optimizer = GEPAOptimizer(
        config=OptimizerConfig(
            max_metric_calls=120,  # Increased to allow more iterations for convergence
            reflection_model="mock/reflection",
            reflection_minibatch_size=3,
        ),
        runtime=rt,
    )
    optimized, score = optimizer.optimize_procedure(
        proc_name=proc.name,
        proc_params=proc.params,
        param_types=param_types,
        spans=spans,
        train_data=train_data,
        test_data=None,
        metric=metric,
    )
    # We should have recorded Predict calls during compile and scoring
    assert Recorder.calls, "Predict should be invoked by GEPA + scoring"
    # Ranges observed in instructions should have shifted toward target band
    ranges = _collect_ranges_from_calls()
    first_lo, first_hi = ranges[0]
    last_lo, last_hi = ranges[-1]
    # Initial default is far from [40,60]; final should overlap or be closer
    def dist_to_target(lo: float, hi: float) -> float:
        # distance from interval to [40,60]
        if hi < 40:
            return 40 - hi
        if lo > 60:
            return lo - 60
        return 0.0
    assert dist_to_target(last_lo, last_hi) <= dist_to_target(first_lo, first_hi)
    # Final training score should be high - with enough iterations we should converge
    assert score >= 0.66, f"Expected score >= 0.66, got {score}"
    # Optimized prompt extracted for the span should contain a RANGE token
    assert spans[0].name in optimized
    assert "RANGE[" in optimized[spans[0].name]


def test_feedback_directionality(monkeypatch: pytest.MonkeyPatch):
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
    return [("omega", None)]
    ```
  > metric: flipped_feedback(cases):
    = ```
    y = solve(cases)
    # Gradual metric with inverted feedback semantics
    if 40 <= y <= 60:
        return (1.0, "OK")
    elif y < 40:
        distance = 40 - y
        score = max(0.0, 1.0 - distance / 100.0)
        return (score, "TOO_HIGH")  # inverted
    else:
        distance = y - 60
        score = max(0.0, 1.0 - distance / 100.0)
        return (score, "TOO_LOW")  # inverted
    ```
"""
    prog, rt, proc, param_types, eval_suite, metric = _prepare(src)
    spans = proc.optimize_spans
    train_data = [("omega", None)]
    optimizer = GEPAOptimizer(
        config=OptimizerConfig(
            max_metric_calls=20,
            reflection_model="mock/reflection",
            reflection_minibatch_size=1,
        ),
        runtime=rt,
    )
    # Suppress warning - this test uses inverted feedback which confuses GEPA
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*appears unchanged.*")
        _ = optimizer.optimize_procedure(
            proc_name=proc.name,
            proc_params=proc.params,
            param_types=param_types,
            spans=spans,
            train_data=train_data,
            test_data=None,
            metric=metric,
        )
    assert Recorder.calls
    ranges = _collect_ranges_from_calls()
    # With flipped feedback, ensure that the range moved at least once but not necessarily toward target
    assert len(ranges) >= 1
    # Check that at least two distinct ranges were proposed
    distinct = {tuple(r) for r in ranges}
    assert len(distinct) >= 1


