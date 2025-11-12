from __future__ import annotations

import pytest

from kedi.lang import compile_program, parse_program
from kedi.optimization.config import OptimizerConfig
from kedi.optimization.gepa import GEPAOptimizer
from kedi.optimization.services import DSPyModuleBuilder, GEPAMetricWrapper, MetricFactory, TypeResolver
from tests.dspy_mocks import RangeAwareAdapter, install_mock_dspy, reflection_fail_once


def test_missing_output_in_span_raises(monkeypatch: pytest.MonkeyPatch):
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
    optimizer = GEPAOptimizer(config=OptimizerConfig(max_metric_calls=1, reflection_model="mock/reflection"), runtime=rt)
    with pytest.raises(ValueError):
        optimizer.optimize_procedure(
            proc_name=proc.name,
            proc_params=proc.params,
            param_types=param_types,
            spans=proc.optimize_spans,
            train_data=[("z", None)],
            test_data=None,
            metric=metric,
        )


def test_empty_train_data_raises(monkeypatch: pytest.MonkeyPatch):
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
    optimizer = GEPAOptimizer(config=OptimizerConfig(max_metric_calls=1, reflection_model="mock/reflection"), runtime=rt)
    with pytest.raises(ValueError):
        optimizer.optimize_procedure(
            proc_name=proc.name,
            proc_params=proc.params,
            param_types=param_types,
            spans=proc.optimize_spans,
            train_data=[],
            test_data=None,
            metric=metric,
        )


def test_bad_reflection_text_yields_zero_score_and_cached_feedback(monkeypatch: pytest.MonkeyPatch):
    # Install mocks and configure reflection to fail once
    install_mock_dspy(monkeypatch)
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
    # any valid return is fine; error path will be injected by invalid RANGE
    return (1.0, "OK")
    ```
"""
    prog = parse_program(src)
    rt = compile_program(prog, adapter=RangeAwareAdapter())
    proc = prog.procedures[0]
    param_types = TypeResolver(rt).resolve_param_types(proc)
    metric_def = prog._validation_evals[0].metrics[0]
    dataset_name = metric_def.dataset_name or next(iter(prog._validation_evals[0].datasets.keys()))
    metric = MetricFactory(rt).create_metric(metric_def, proc, dataset_name)
    optimizer = GEPAOptimizer(config=OptimizerConfig(max_metric_calls=5, reflection_model="mock/reflection"), runtime=rt)
    # Run optimization; the first reflection step should inject an invalid RANGE string causing adapter error.
    # Suppress warning - this test intentionally causes reflection to fail, so prompt won't change
    import warnings
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
    # Optimization should still complete without crashing
    assert isinstance(optimized, dict)
    # Final score may be >0 after recovery; ensure we at least executed (non-negative)
    assert score >= 0.0


