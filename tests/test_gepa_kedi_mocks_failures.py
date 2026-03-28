"""Failure and error-path tests for the GEPA optimizer with mock LMs.

Every test here verifies that the optimizer raises the right error or
recovers gracefully from bad inputs, and assertions match the test name.
"""
from __future__ import annotations

import warnings

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
    reflection_fail_once,
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


# ── Missing output in span → ValueError ─────────────────────────────

def test_missing_output_in_span_raises_valueerror(monkeypatch: pytest.MonkeyPatch):
    """A span with no [output] placeholders must raise ValueError."""
    install_mock_dspy(monkeypatch)
    src = """
@p(x: str):
  > optimize: bad_span:
    This span defines no outputs at all.
  = done

@eval: p:
  > data: ds:
    = ```
    return [("z", None)]
    ```
  > metric: ok(ds):
    = ```
    _ = p(ds)
    return (1.0, None)
    ```
"""
    prog = parse_program(src)
    rt = compile_program(prog, adapter=RangeAwareAdapter())
    proc = prog.procedures[0]
    param_types = TypeResolver(rt).resolve_param_types(proc)
    metric_def = prog._validation_evals[0].metrics[0]
    dataset_name = metric_def.dataset_name or next(iter(prog._validation_evals[0].datasets.keys()))
    metric = MetricFactory(rt).create_metric(metric_def, proc, dataset_name)
    optimizer = GEPAOptimizer(
        config=OptimizerConfig(max_metric_calls=1, reflection_model="mock/reflection"),
        runtime=rt,
    )
    with pytest.raises(ValueError, match="output"):
        optimizer.optimize_procedure(
            proc_name=proc.name,
            proc_params=proc.params,
            param_types=param_types,
            spans=proc.optimize_spans,
            train_data=[("z", None)],
            test_data=None,
            metric=metric,
        )


# ── Empty training data → ValueError ────────────────────────────────

def test_empty_train_data_raises_valueerror(monkeypatch: pytest.MonkeyPatch):
    """Empty training data must raise ValueError with a clear message."""
    install_mock_dspy(monkeypatch)
    src = """
@p(x: str) -> int:
  > optimize: span1:
    [y: int]
  = `y`

@eval: p:
  > data: ds:
    = ```
    return []
    ```
  > metric: ok(ds):
    = ```
    _ = p(ds)
    return (1.0, None)
    ```
"""
    prog = parse_program(src)
    rt = compile_program(prog, adapter=RangeAwareAdapter())
    proc = prog.procedures[0]
    param_types = TypeResolver(rt).resolve_param_types(proc)
    metric_def = prog._validation_evals[0].metrics[0]
    dataset_name = metric_def.dataset_name or next(iter(prog._validation_evals[0].datasets.keys()))
    metric = MetricFactory(rt).create_metric(metric_def, proc, dataset_name)
    optimizer = GEPAOptimizer(
        config=OptimizerConfig(max_metric_calls=1, reflection_model="mock/reflection"),
        runtime=rt,
    )
    with pytest.raises(ValueError, match="[Nn]o training data"):
        optimizer.optimize_procedure(
            proc_name=proc.name,
            proc_params=proc.params,
            param_types=param_types,
            spans=proc.optimize_spans,
            train_data=[],
            test_data=None,
            metric=metric,
        )


# ── Bad reflection NaN recovery ──────────────────────────────────────

def test_bad_reflection_nan_range_recovers(monkeypatch: pytest.MonkeyPatch):
    """When reflection produces RANGE[NaN,NaN], optimizer still completes
    without crashing. The NaN response is injected by reflection_fail_once()
    which affects the first MockLM._reflection_response call."""
    install_mock_dspy(monkeypatch)
    Recorder.reset()
    reflection_fail_once()

    src = """
@p(x: str) -> int:
  > optimize: s:
    The answer is [y: int]
  = `y`

@eval: p:
  > data: ds:
    = ```
    return [("zz", None)]
    ```
  > metric: score(ds):
    = ```
    y = p(ds)
    return (1.0, "OK")
    ```
"""
    prog, rt, proc, param_types, _, metric = _prepare(src)
    optimizer = GEPAOptimizer(
        config=OptimizerConfig(max_metric_calls=10, reflection_model="mock/reflection"),
        runtime=rt,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*appears unchanged.*")
        optimized, score = optimizer.optimize_procedure(
            proc_name=proc.name,
            proc_params=proc.params,
            param_types=param_types,
            spans=proc.optimize_spans,
            train_data=[("zz", None)],
            test_data=None,
            metric=metric,
        )

    assert isinstance(optimized, dict), "Optimizer must return a dict even after bad reflection"
    assert score >= 0.0, "Score must be non-negative"
    assert len(Recorder.lm_calls) > 0, "LM should have been called at least once"


# ── Span with no inputs still optimizes ──────────────────────────────

def test_span_without_inputs_still_optimizes(monkeypatch: pytest.MonkeyPatch):
    """A span that references no <inputs> from the procedure should still
    be optimizable. The procedure has a parameter (required by the formatter)
    but the span itself doesn't use it."""
    install_mock_dspy(monkeypatch)
    Recorder.reset()

    src = """
@generate(seed: str) -> int:
  > optimize: bare_span:
    The magic number is [y: int]
  = `y`

@eval: generate:
  > data: ds:
    = ```
    return [("x", None)]
    ```
  > metric: score(ds):
    = ```
    y = generate(ds)
    if 40 <= y <= 60:
        return (1.0, "OK")
    return (0.5, "TOO_LOW" if y < 40 else "TOO_HIGH")
    ```
"""
    prog, rt, proc, param_types, _, metric = _prepare(src)
    optimizer = GEPAOptimizer(
        config=OptimizerConfig(max_metric_calls=20, reflection_model="mock/reflection"),
        runtime=rt,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*appears unchanged.*")
        optimized, score = optimizer.optimize_procedure(
            proc_name=proc.name,
            proc_params=proc.params,
            param_types=param_types,
            spans=proc.optimize_spans,
            train_data=[("x", None)],
            test_data=None,
            metric=metric,
        )
    assert isinstance(optimized, dict)
    assert "bare_span" in optimized
    assert score >= 0.0

    for call in Recorder.calls:
        assert "seed" not in call.kwargs, (
            "Span has no <seed> reference, so seed should not be forwarded to Predict"
        )


# ── Empty spans list returns empty ───────────────────────────────────

def test_empty_spans_returns_empty(monkeypatch: pytest.MonkeyPatch):
    """Passing an empty spans list should return empty prompts and 0.0 score."""
    install_mock_dspy(monkeypatch)
    src = """
@p(x: str) -> int:
  > optimize: s:
    [y: int]
  = `y`

@eval: p:
  > data: ds:
    = ```
    return [("a", None)]
    ```
  > metric: ok(ds):
    = ```
    _ = p(ds)
    return (1.0, None)
    ```
"""
    prog, rt, proc, param_types, _, metric = _prepare(src)
    optimizer = GEPAOptimizer(
        config=OptimizerConfig(max_metric_calls=1, reflection_model="mock/reflection"),
        runtime=rt,
    )
    optimized, score = optimizer.optimize_procedure(
        proc_name=proc.name,
        proc_params=proc.params,
        param_types=param_types,
        spans=[],
        train_data=[("a", None)],
        test_data=None,
        metric=metric,
    )
    assert optimized == {}
    assert score == 0.0
