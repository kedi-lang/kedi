from __future__ import annotations

import pytest

from kedi.lang import compile_program, parse_program
from kedi.optimization.config import OptimizerConfig
from kedi.optimization.gepa import GEPAOptimizer
from kedi.optimization.services import MetricFactory, TypeResolver
from tests.dspy_mocks import Recorder, RangeAwareAdapter, install_mock_dspy, parse_first_range


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


def test_typed_outputs_and_lists(monkeypatch: pytest.MonkeyPatch):
    install_mock_dspy(monkeypatch)
    Recorder.reset()
    src = """
@proc(seed: str) -> float:
  > optimize: typed_span:
    Provide values:
    v\=[v: int], w\=[w: float], labels\=[tags: `list[str]`]
  = `float(w)`

@eval: proc:
  > data: ds:
    = ```
    return [("k", None)]
    ```
  > metric: typed_score(ds):
    = ```
    val = proc(ds)
    # Gradual metric for float values with large normalization for distant values
    if 5.5 <= val <= 6.5:
        return (1.0, "OK")
    elif val < 5.5:
        distance = 5.5 - val
        score = max(0.0, 1.0 - distance / 100.0)
        return (score, "TOO_LOW")
    else:
        distance = val - 6.5
        score = max(0.0, 1.0 - distance / 100.0)
        return (score, "TOO_HIGH")
    ```
"""
    prog, rt, proc, param_types, _, metric = _prepare(src)
    spans = proc.optimize_spans
    optimizer = GEPAOptimizer(
        config=OptimizerConfig(
            max_metric_calls=200,  # Even more iterations for this challenging case
            reflection_model="mock/reflection",
            reflection_minibatch_size=1,  # Smaller minibatch for faster iteration
        ),
        runtime=rt,
    )
    # Use multiple training examples to provide more signal
    train_data = [("k", None), ("j", None), ("m", None)]
    optimized, score = optimizer.optimize_procedure(
        proc_name=proc.name,
        proc_params=proc.params,
        param_types=param_types,
        spans=spans,
        train_data=train_data,
        test_data=None,
        metric=metric,
    )
    assert "typed_span" in optimized
    lo, hi = parse_first_range(optimized["typed_span"])
    # With gradual metrics and proper output override, GEPA should converge
    assert score >= 0.5, f"Expected score >= 0.5, got score={score}"
    assert isinstance(lo, float) and isinstance(hi, float)
    
    # Verify that the range has moved closer to the target [5.5, 6.5]
    # The final range should overlap with or be close to the target
    final_distance = min(abs(lo - 6.0), abs(hi - 6.0), abs((lo + hi) / 2 - 6.0))
    initial_distance = abs(-75.0 - 6.0)
    assert final_distance < initial_distance, f"Range should move closer to target: initial={initial_distance}, final={final_distance}"


def test_complex_span_inputs_forwarding(monkeypatch: pytest.MonkeyPatch):
    install_mock_dspy(monkeypatch)
    Recorder.reset()
    src = """
@proc(seed: str) -> int:
  [pretty] = `seed.upper()`
  > optimize: only_span:
    Text <pretty> -\> [y: int]
  = `y`

@eval: proc:
  > data: ds:
    = ```
    return [("foo", None)]
    ```
  > metric: noop(ds):
    = ```
    _ = proc(ds)
    return (1.0, None)
    ```
"""
    prog, rt, proc, param_types, _, metric = _prepare(src)
    spans = proc.optimize_spans
    optimizer = GEPAOptimizer(
        config=OptimizerConfig(max_metric_calls=5, reflection_model="mock/reflection"),
        runtime=rt,
    )
    # Suppress warning - this test uses a trivial metric (always returns 1.0) so GEPA can't improve
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*appears unchanged.*")
        _ = optimizer.optimize_procedure(
            proc_name=proc.name,
            proc_params=proc.params,
            param_types=param_types,
            spans=spans,
            train_data=[("foo", None)],
            test_data=None,
            metric=metric,
        )
    # Predict kwargs should use 'pretty' only (computed from seed), not 'seed'
    assert Recorder.calls
    first = Recorder.calls[0]
    assert set(first.kwargs.keys()) == {"pretty"}


