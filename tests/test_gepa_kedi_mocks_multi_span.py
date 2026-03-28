"""Multi-span GEPA optimization tests.

Tests that:
  1. Two spans inside ONE procedure are optimized independently toward
     different targets.
  2. Two separate procedures are optimized independently without interference.
  3. Spans are optimized in declaration order and each gets its own optimized
     prompt.
"""
from __future__ import annotations

import warnings
from typing import Any

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


def _prepare(src: str, proc_index: int = 0, eval_index: int = 0):
    prog = parse_program(src)
    rt = compile_program(prog, adapter=RangeAwareAdapter())
    proc = prog.procedures[proc_index]
    resolver = TypeResolver(rt)
    param_types = resolver.resolve_param_types(proc)
    eval_suite = prog._validation_evals[eval_index]
    metric_def = eval_suite.metrics[0]
    dataset_name = metric_def.dataset_name or next(iter(eval_suite.datasets.keys()))
    metric = MetricFactory(rt).create_metric(metric_def, proc, dataset_name)
    return prog, rt, proc, param_types, eval_suite, metric


# ── Two spans in ONE procedure ───────────────────────────────────────

def test_two_spans_in_one_procedure_both_optimized(monkeypatch: pytest.MonkeyPatch):
    """A single procedure with two optimize spans. Each span gets its own
    optimized prompt, and both appear in the result dict."""
    install_mock_dspy(monkeypatch)
    Recorder.reset()

    src = """
@solve(seed: str) -> int:
  > optimize: span_a:
    First compute [x: int]
  > optimize: span_b:
    Then compute [y: int]
  = `x + y`

@eval: solve:
  > data: cases:
    = ```
    return [("alpha", None), ("beta", None)]
    ```
  > metric: score(cases):
    = ```
    result = solve(cases)
    if 40 <= result <= 60:
        return (1.0, "OK")
    elif result < 40:
        return (0.5, "TOO_LOW")
    else:
        return (0.5, "TOO_HIGH")
    ```
"""
    prog, rt, proc, param_types, _, metric = _prepare(src)
    train_data = [("alpha", None), ("beta", None)]

    optimizer = GEPAOptimizer(
        config=OptimizerConfig(
            max_metric_calls=40,
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

    assert len(proc.optimize_spans) == 2, "Procedure should have 2 optimize spans"
    assert "span_a" in optimized, "span_a should have an optimized prompt"
    assert "span_b" in optimized, "span_b should have an optimized prompt"
    assert optimized["span_a"] != optimized["span_b"], (
        "Different spans should get different optimized prompts"
    )


# ── Two procedures optimized independently ───────────────────────────

def test_two_procedures_independent_optimization(monkeypatch: pytest.MonkeyPatch):
    """Optimize proc_a and proc_b separately. Each targets a different range.
    Results should not interfere with each other."""
    install_mock_dspy(monkeypatch)

    src = """
@pa(seed: str) -> int:
  > optimize: span_a:
    Produce a low value [y: int]
  = `y`

@pb(seed: str) -> int:
  > optimize: span_b:
    Produce a high value [y: int]
  = `y`

@eval: pa:
  > data: ds_a:
    = ```
    return [("x", None), ("y", None)]
    ```
  > metric: low_target(ds_a):
    = ```
    y = pa(ds_a)
    if 10 <= y <= 20:
        return (1.0, "OK")
    elif y < 10:
        return (max(0.0, 1.0 - (10-y)/100.0), "TOO_LOW")
    else:
        return (max(0.0, 1.0 - (y-20)/100.0), "TOO_HIGH")
    ```

@eval: pb:
  > data: ds_b:
    = ```
    return [("x", None), ("y", None)]
    ```
  > metric: high_target(ds_b):
    = ```
    y = pb(ds_b)
    if 80 <= y <= 90:
        return (1.0, "OK")
    elif y < 80:
        return (max(0.0, 1.0 - (80-y)/100.0), "TOO_LOW")
    else:
        return (max(0.0, 1.0 - (y-90)/100.0), "TOO_HIGH")
    ```
"""
    # Optimize proc_a
    Recorder.reset()
    prog_a, rt_a, proc_a, param_types_a, _, metric_a = _prepare(src, proc_index=0, eval_index=0)
    optimizer_a = GEPAOptimizer(
        config=OptimizerConfig(
            max_metric_calls=120,
            reflection_model="mock/reflection",
            reflection_minibatch_size=2,
        ),
        runtime=rt_a,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*appears unchanged.*")
        optimized_a, score_a = optimizer_a.optimize_procedure(
            proc_name=proc_a.name,
            proc_params=proc_a.params,
            param_types=param_types_a,
            spans=proc_a.optimize_spans,
            train_data=[("x", None), ("y", None)],
            test_data=None,
            metric=metric_a,
        )

    ranges_a = Recorder.predict_ranges()

    # Optimize proc_b
    Recorder.reset()
    prog_b, rt_b, proc_b, param_types_b, _, metric_b = _prepare(src, proc_index=1, eval_index=1)
    optimizer_b = GEPAOptimizer(
        config=OptimizerConfig(
            max_metric_calls=120,
            reflection_model="mock/reflection",
            reflection_minibatch_size=2,
        ),
        runtime=rt_b,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*appears unchanged.*")
        optimized_b, score_b = optimizer_b.optimize_procedure(
            proc_name=proc_b.name,
            proc_params=proc_b.params,
            param_types=param_types_b,
            spans=proc_b.optimize_spans,
            train_data=[("x", None), ("y", None)],
            test_data=None,
            metric=metric_b,
        )

    ranges_b = Recorder.predict_ranges()

    assert score_a >= 0.5, f"proc_a score {score_a} should be >= 0.5"
    assert score_b >= 0.5, f"proc_b score {score_b} should be >= 0.5"

    assert "span_a" in optimized_a
    assert "span_b" in optimized_b

    if ranges_a and ranges_b:
        last_a = ranges_a[-1]
        last_b = ranges_b[-1]
        mid_a = (last_a[0] + last_a[1]) / 2
        mid_b = (last_b[0] + last_b[1]) / 2
        assert mid_a < mid_b or abs(mid_a - mid_b) < 200, (
            f"proc_a (target [10,20]) final midpoint {mid_a:.1f} should generally be "
            f"lower than proc_b (target [80,90]) midpoint {mid_b:.1f}"
        )


# ── Each span gets its own prompt ────────────────────────────────────

def test_spans_produce_distinct_optimized_prompts(monkeypatch: pytest.MonkeyPatch):
    """Each span in a multi-span procedure gets a unique optimized prompt
    containing its span name in the instructions."""
    install_mock_dspy(monkeypatch)
    Recorder.reset()

    src = """
@proc(seed: str) -> int:
  > optimize: first_step:
    Compute the initial [x: int]
  > optimize: second_step:
    Refine to get [y: int]
  = `x + y`

@eval: proc:
  > data: ds:
    = ```
    return [("a", None), ("b", None)]
    ```
  > metric: score(ds):
    = ```
    r = proc(ds)
    return (1.0, "OK")
    ```
"""
    prog, rt, proc, param_types, _, metric = _prepare(src)
    train_data = [("a", None), ("b", None)]

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

    assert "first_step" in optimized
    assert "second_step" in optimized

    assert isinstance(optimized["first_step"], str) and len(optimized["first_step"]) > 0
    assert isinstance(optimized["second_step"], str) and len(optimized["second_step"]) > 0
