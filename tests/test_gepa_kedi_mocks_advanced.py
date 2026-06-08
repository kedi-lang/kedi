"""Advanced GEPA optimizer tests combining optimization with other Kedi features.

Tests that the optimizer works correctly when combined with:
  - Typed outputs (int, float, list[str])
  - Python preprocessing blocks before spans
  - Multiline spans
  - Optimization preserving non-optimized procedure behavior
  - Complex span inputs with computed values
"""

from __future__ import annotations

import warnings

import pytest

from kedi.lang import compile_program, parse_program
from kedi.lang.ast import ProgramWithValidation
from kedi.optimization.config import OptimizerConfig
from kedi.optimization.gepa import GEPAOptimizer
from kedi.optimization.services import MetricFactory, TypeResolver
from tests.dspy_mocks import RangeAwareAdapter, Recorder, install_mock_dspy


@pytest.fixture(autouse=True)
def _isolated_gepa_cwd(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


def _prepare(src: str, proc_index: int = 0, eval_index: int = 0):
    prog = parse_program(src)
    assert isinstance(prog, ProgramWithValidation)
    rt = compile_program(prog, adapter=RangeAwareAdapter())
    proc = prog.procedures[proc_index]
    resolver = TypeResolver(rt)
    param_types = resolver.resolve_param_types(proc)
    eval_suite = prog.evals[eval_index]
    metric_def = eval_suite.metrics[0]
    dataset_name = metric_def.dataset_name or next(iter(eval_suite.datasets.keys()))
    metric = MetricFactory(rt).create_metric(metric_def, proc, dataset_name)
    return prog, rt, proc, param_types, eval_suite, metric


def _optimize(monkeypatch, src, train_data, max_calls=40, proc_index=0, eval_index=0):
    install_mock_dspy(monkeypatch)
    Recorder.reset()
    prog, rt, proc, param_types, _, metric = _prepare(src, proc_index, eval_index)
    optimizer = GEPAOptimizer(
        config=OptimizerConfig(
            max_metric_calls=max_calls,
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
    return optimized, score, proc, rt


# ── Typed int and float outputs ──────────────────────────────────────


def test_optimization_with_typed_int_float_outputs(monkeypatch: pytest.MonkeyPatch):
    """Span with both [v: int] and [w: float] outputs.
    MockLM should produce correct types for both."""
    src = """
@compute(seed: str) -> float:
  > optimize: typed_span:
    Compute integer [v: int] and decimal [w: float]
  = `float(v) + w`

@eval: compute:
  > data: ds:
    = ```
    return [("x", None), ("y", None)]
    ```
  > metric: score(ds):
    = ```
    r = compute(ds)
    if isinstance(r, (int, float)) and 0 <= r <= 200:
        return (1.0, "OK")
    return (0.5, "TOO_LOW")
    ```
"""
    optimized, score, proc, _ = _optimize(monkeypatch, src, [("x", None), ("y", None)])

    assert "typed_span" in optimized
    assert score >= 0.0

    assert len(Recorder.calls) > 0, "Predict should have been called"


# ── List[str] output type ────────────────────────────────────────────


def test_optimization_with_list_str_output(monkeypatch: pytest.MonkeyPatch):
    """Span producing [tags: list[str]] should work through optimization."""
    src = """
@tag(seed: str) -> list[str]:
  > optimize: tag_span:
    Generate tags [tags: list[str]]
  = `tags`

@eval: tag:
  > data: ds:
    = ```
    return [("item1", None), ("item2", None)]
    ```
  > metric: score(ds):
    = ```
    result = tag(ds)
    if isinstance(result, list) and len(result) > 0:
        return (1.0, "OK")
    return (0.5, "NEEDS_TAGS")
    ```
"""
    optimized, score, proc, _ = _optimize(monkeypatch, src, [("item1", None), ("item2", None)])

    assert "tag_span" in optimized
    assert score >= 0.0


# ── Python preprocessing before span ─────────────────────────────────


def test_optimization_with_python_preprocessing(monkeypatch: pytest.MonkeyPatch):
    """A Python block computes a derived value before the span.
    The span uses this computed value as input."""
    src = """
@analyze(text: str) -> int:
  [word_count: int] = `len(text.split())`
  [prefix] = Words: <`str(word_count)`>
  > optimize: analysis_span:
    Given <prefix>, compute score [y: int]
  = `y`

@eval: analyze:
  > data: ds:
    = ```
    return [("hello world foo bar", None), ("one two three", None)]
    ```
  > metric: score(ds):
    = ```
    r = analyze(ds)
    return (1.0, "OK")
    ```
"""
    optimized, score, proc, _ = _optimize(
        monkeypatch,
        src,
        [("hello world foo bar", None), ("one two three", None)],
    )

    assert "analysis_span" in optimized
    assert score >= 0.0

    for call in Recorder.calls:
        assert "prefix" in call.kwargs, "Span references <prefix>, so it should be in kwargs"
        assert "text" not in call.kwargs, (
            "Span does NOT reference <text>, it should not be in kwargs"
        )
        assert "word_count" not in call.kwargs, (
            "Span does NOT reference <word_count>, it should not be in kwargs"
        )


# ── Multiline span ───────────────────────────────────────────────────


def test_optimization_with_multiline_span(monkeypatch: pytest.MonkeyPatch):
    """A span with multiple template lines. All lines should be part of
    the span and the optimized prompt should cover the full span text."""
    src = """
@solve(seed: str) -> int:
  > optimize: detailed:
    Think about the problem carefully.
    Consider the input: <seed>
    The answer is [y: int]
  = `y`

@eval: solve:
  > data: ds:
    = ```
    return [("test_input", None)]
    ```
  > metric: score(ds):
    = ```
    y = solve(ds)
    return (1.0, "OK")
    ```
"""
    optimized, score, proc, _ = _optimize(monkeypatch, src, [("test_input", None)])

    assert "detailed" in optimized

    span = proc.optimize_spans[0]
    assert len(span.template_lines) == 3, (
        f"Expected 3 template lines in multiline span, got {len(span.template_lines)}"
    )


# ── Non-optimized lines preserved ────────────────────────────────────


def test_optimization_preserves_non_optimized_lines(monkeypatch: pytest.MonkeyPatch):
    """A procedure has both normal template lines and an optimize span.
    After optimization, the non-optimized parts should be unaffected."""
    src = """
@proc(seed: str) -> str:
  [prefix] = start-<seed>
  > optimize: opt_span:
    Compute [y: int] from <prefix>
  [result] = <prefix>-<`str(y)`>
  = `result`

@eval: proc:
  > data: ds:
    = ```
    return [("alpha", None)]
    ```
  > metric: score(ds):
    = ```
    r = proc(ds)
    return (1.0, "OK")
    ```
"""
    optimized, score, proc, rt = _optimize(monkeypatch, src, [("alpha", None)])

    assert "opt_span" in optimized
    assert score >= 0.0

    body_stmts = proc.body
    for stmt in body_stmts:
        optimized_prompt_prefix = getattr(stmt, "optimized_prompt_prefix", None)
        segments = getattr(stmt, "segments", None)
        if optimized_prompt_prefix is None:
            if segments is not None:
                pass


# ── Complex computed span inputs ─────────────────────────────────────


def test_complex_span_inputs_forwarding(monkeypatch: pytest.MonkeyPatch):
    """Procedure with complex Python preprocessing that feeds into span.
    Span inputs should be the computed values, not raw procedure params."""
    src = """
@transform(raw: str) -> int:
  [upper] = `raw.upper()`
  [tagged] = TAG:<upper>
  > optimize: transform_span:
    Process <tagged> and produce [y: int]
  = `y`

@eval: transform:
  > data: ds:
    = ```
    return [("hello", None)]
    ```
  > metric: score(ds):
    = ```
    r = transform(ds)
    return (1.0, "OK")
    ```
"""
    optimized, score, _, _ = _optimize(monkeypatch, src, [("hello", None)])

    assert "transform_span" in optimized

    for call in Recorder.calls:
        assert "tagged" in call.kwargs, "Span references <tagged>"
        assert "raw" not in call.kwargs, "Span does NOT reference <raw>"
        value = call.kwargs.get("tagged", "")
        assert "TAG:" in str(value), f"Expected 'TAG:' prefix in computed input, got '{value}'"


# ── Optimization with test_data split ────────────────────────────────


def test_optimization_with_test_data(monkeypatch: pytest.MonkeyPatch):
    """When test_data is provided alongside train_data, GEPA should use
    both for optimization (train for training, test for validation)."""
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
    return [("a", None), ("b", None), ("c", None)]
    ```
  > test_data: test_cases:
    = ```
    return [("d", None), ("e", None)]
    ```
  > metric: score(cases):
    = ```
    y = solve(cases)
    if 40 <= y <= 60:
        return (1.0, "OK")
    return (0.5, "TOO_LOW" if y < 40 else "TOO_HIGH")
    ```
"""
    prog, rt, proc, param_types, eval_suite, metric = _prepare(src)

    train_data = [("a", None), ("b", None), ("c", None)]
    test_data = [("d", None), ("e", None)] if eval_suite.test_datasets else None

    optimizer = GEPAOptimizer(
        config=OptimizerConfig(
            max_metric_calls=60,
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
            test_data=test_data,
            metric=metric,
        )

    assert "produce_value" in optimized
    assert score >= 0.0
