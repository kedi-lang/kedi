from __future__ import annotations

import pytest

from kedi.lang import compile_program, parse_program
from kedi.optimization.config import OptimizerConfig
from kedi.optimization.gepa import GEPAOptimizer
from kedi.optimization.services import MetricFactory, TypeResolver
from tests.dspy_mocks import Recorder, RangeAwareAdapter, install_mock_dspy


def test_span_inputs_only_forwarded(monkeypatch: pytest.MonkeyPatch):
    install_mock_dspy(monkeypatch)
    Recorder.reset()
    src = """
@proc(a: int, b: int, c: int) -> int:
  [p] = pre-<a>-<b>
  > optimize: only_span:
    Using <p> produce [y: int]
  = `y`

@eval: proc:
  > data: cases:
    = ```
    return [(((1,2,3)), None)]
    ```
  > metric: noop(cases):
    = ```
    # Minimal run to trigger GEPA; score doesn't matter
    _ = proc(cases)
    return (1.0, None)
    ```
"""
    prog = parse_program(src)
    rt = compile_program(prog, adapter=RangeAwareAdapter())
    proc = prog.procedures[0]
    param_types = TypeResolver(rt).resolve_param_types(proc)
    metric_def = prog._validation_evals[0].metrics[0]
    dataset = metric_def.dataset_name or next(iter(prog._validation_evals[0].datasets.keys()))
    metric = MetricFactory(rt).create_metric(metric_def, proc, dataset)
    optimizer = GEPAOptimizer(
        config=OptimizerConfig(max_metric_calls=5, reflection_model="mock/reflection"),
        runtime=rt,
    )
    spans = proc.optimize_spans
    train_data = [(((1, 2, 3)), None)]
    # Suppress warning - this test uses a trivial metric (always returns 1.0) so GEPA can't improve
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
    # Assert that only span inputs were forwarded to Predict (i.e., only 'p', not 'a','b','c')
    assert Recorder.calls, "Predict should be called"
    first = Recorder.calls[0]
    assert set(first.kwargs.keys()) == {"p"}


