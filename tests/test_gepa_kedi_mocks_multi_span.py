from __future__ import annotations

import pytest

from kedi.lang import compile_program, parse_program
from kedi.optimization.config import OptimizerConfig
from kedi.optimization.gepa import GEPAOptimizer
from kedi.optimization.services import MetricFactory, TypeResolver
from tests.dspy_mocks import RangeAwareAdapter, install_mock_dspy, parse_first_range


def _setup(proc_src: str, eval_name: str):
    prog = parse_program(proc_src)
    rt = compile_program(prog, adapter=RangeAwareAdapter())
    # pick the matching procedure by name
    proc = next(p for p in prog.procedures if p.name == eval_name)
    param_types = TypeResolver(rt).resolve_param_types(proc)
    suite = next(s for s in prog._validation_evals if s.procedure_name == eval_name)
    metric_def = suite.metrics[0]
    dataset_name = metric_def.dataset_name or next(iter(suite.datasets.keys()))
    metric = MetricFactory(rt).create_metric(metric_def, proc, dataset_name)
    return prog, rt, proc, param_types, metric


def test_independent_optimization_for_two_procedures(monkeypatch: pytest.MonkeyPatch):
    install_mock_dspy(monkeypatch)
    src = """
@pa(seed: str) -> int:
  > optimize: span_a:
    The A value is [ya: int]
  = `ya`

@pb(seed: str) -> int:
  > optimize: span_b:
    The B value is [yb: int]
  = `yb`

@eval: pa:
  > data: cases_a:
    = ```
    return [("aa", None), ("ab", None)]
    ```
  > metric: score_a(cases_a):
    = ```
    y = pa(cases_a)
    # Gradual metric for range [10,20]
    if 10 <= y <= 20:
        return (1.0, "OK")
    elif y < 10:
        distance = 10 - y
        score = max(0.0, 1.0 - distance / 100.0)
        return (score, "TOO_LOW")
    else:
        distance = y - 20
        score = max(0.0, 1.0 - distance / 100.0)
        return (score, "TOO_HIGH")
    ```

@eval: pb:
  > data: cases_b:
    = ```
    return [("ba", None), ("bb", None)]
    ```
  > metric: score_b(cases_b):
    = ```
    y = pb(cases_b)
    # Gradual metric for range [80,90]
    if 80 <= y <= 90:
        return (1.0, "OK")
    elif y < 80:
        distance = 80 - y
        score = max(0.0, 1.0 - distance / 100.0)
        return (score, "TOO_LOW")
    else:
        distance = y - 90
        score = max(0.0, 1.0 - distance / 100.0)
        return (score, "TOO_HIGH")
    ```
"""
    # Optimize A
    _, rt, proc_a, param_types_a, metric_a = _setup(src, "pa")
    opt = GEPAOptimizer(
        config=OptimizerConfig(max_metric_calls=120, reflection_model="mock/reflection"),  # Increased for convergence
        runtime=rt,
    )
    spans_a = [s for s in proc_a.optimize_spans if s.name == "span_a"]
    optimized_a, score_a = opt.optimize_procedure(
        proc_name=proc_a.name,
        proc_params=proc_a.params,
        param_types=param_types_a,
        spans=spans_a,
        train_data=[("aa", None), ("ab", None)],
        test_data=None,
        metric=metric_a,
    )
    assert "span_a" in optimized_a
    lo_a, hi_a = parse_first_range(optimized_a["span_a"])
    # With enough iterations, we should achieve good convergence
    assert score_a >= 0.5, f"Expected score_a >= 0.5 with 120 iterations, got score={score_a}"
    # Optimize B
    _, rt2, proc_b, param_types_b, metric_b = _setup(src, "pb")
    opt2 = GEPAOptimizer(
        config=OptimizerConfig(max_metric_calls=120, reflection_model="mock/reflection"),  # Increased for convergence
        runtime=rt2,
    )
    spans_b = [s for s in proc_b.optimize_spans if s.name == "span_b"]
    optimized_b, score_b = opt2.optimize_procedure(
        proc_name=proc_b.name,
        proc_params=proc_b.params,
        param_types=param_types_b,
        spans=spans_b,
        train_data=[("ba", None), ("bb", None)],
        test_data=None,
        metric=metric_b,
    )
    assert "span_b" in optimized_b
    lo_b, hi_b = parse_first_range(optimized_b["span_b"])
    # With enough iterations, we should achieve good convergence
    assert score_b >= 0.5, f"Expected score_b >= 0.5 with 120 iterations, got score={score_b}"
    # Ranges should differ significantly (aimed at different target bands)
    # Target for A is [10,20], target for B is [80,90]
    # We expect the ranges to be on opposite sides of the midpoint (around 50)
    assert (hi_a <= 50 and lo_b >= 50) or (hi_b <= 50 and lo_a >= 50)


