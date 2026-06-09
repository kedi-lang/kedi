">> Span isolation tests for the GEPA optimizer.\n\n>> The most critical bug-prevention tests in the suite. A span inside a procedure\nmust be treated as its own unit for optimization -- its input fields come from\nthe span's <references>, NOT the enclosing procedure's parameter list.\nSimilarly, its output fields come from the span's [outputs], not the\nprocedure's return type.\n"

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


def _run_optimization(monkeypatch, src, train_data, max_calls=10):
    install_mock_dspy(monkeypatch)
    Recorder.reset()
    prog, rt, proc, param_types, _, metric = _prepare(src)
    optimizer = GEPAOptimizer(
        config=OptimizerConfig(
            max_metric_calls=max_calls,
            reflection_model="mock/reflection",
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
    return optimized, score, proc


# ── Span inputs are span-level, not procedure-level ──────────────────


def test_span_inputs_keys_are_span_level(monkeypatch: pytest.MonkeyPatch):
    ">> Procedure (a, b, c) with [p] = pre-<a>-<b> and span using <p>.\n    >> Predict kwargs keys must be {'p'}, never {'a','b','c'}."
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
    _ = proc(cases)
    return (1.0, None)
    ```
"""
    _run_optimization(monkeypatch, src, [(((1, 2, 3)), None)])

    assert Recorder.calls, "Predict should be called"
    for call in Recorder.calls:
        assert set(call.kwargs.keys()) == {"p"}, (
            f"Span only references <p>; got kwargs keys {set(call.kwargs.keys())}"
        )
        assert "a" not in call.kwargs
        assert "b" not in call.kwargs
        assert "c" not in call.kwargs


def test_span_input_values_are_computed_correctly(monkeypatch: pytest.MonkeyPatch):
    ">> Same setup as above, but also verify the VALUE of the span input 'p'.\n    >> Given a=1, b=2, [p] = pre-<a>-<b> should produce p = 'pre-1-2'."
    src = """
@proc(a: int, b: int, c: int) -> int:
  [p] = pre-<a>-<b>
  > optimize: value_span:
    Using <p> produce [y: int]
  = `y`

@eval: proc:
  > data: cases:
    = ```
    return [(((1,2,3)), None)]
    ```
  > metric: noop(cases):
    = ```
    _ = proc(cases)
    return (1.0, None)
    ```
"""
    _run_optimization(monkeypatch, src, [(((1, 2, 3)), None)])

    assert Recorder.calls, "Predict should be called"
    first = Recorder.calls[0]
    assert "p" in first.kwargs
    assert first.kwargs["p"] == "pre-1-2", (
        f"Expected computed span input 'pre-1-2', got '{first.kwargs['p']}'"
    )


# ── Multiple training examples all use span inputs ───────────────────


def test_all_training_examples_use_span_inputs(monkeypatch: pytest.MonkeyPatch):
    """With multiple training examples, EVERY Predict call should use
    span-level inputs, not just the first one."""
    src = """
@proc(a: int, b: int) -> int:
  [tag] = val-<a>
  > optimize: tagged:
    Given <tag> produce [y: int]
  = `y`

@eval: proc:
  > data: cases:
    = ```
    return [((10, 20), None), ((30, 40), None), ((50, 60), None)]
    ```
  > metric: noop(cases):
    = ```
    _ = proc(cases)
    return (1.0, None)
    ```
"""
    _run_optimization(
        monkeypatch,
        src,
        [((10, 20), None), ((30, 40), None), ((50, 60), None)],
        max_calls=20,
    )

    assert len(Recorder.calls) >= 3, (
        f"Expected at least 3 Predict calls (one per training example), got {len(Recorder.calls)}"
    )
    for i, call in enumerate(Recorder.calls):
        assert set(call.kwargs.keys()) == {"tag"}, (
            f"Predict call {i} kwargs keys should be {{'tag'}}, got {set(call.kwargs.keys())}"
        )
        assert "a" not in call.kwargs
        assert "b" not in call.kwargs


# ── Span with direct parameter reference ─────────────────────────────


def test_span_uses_direct_param_still_isolated(monkeypatch: pytest.MonkeyPatch):
    """If a span references a procedure parameter directly via <param>,
    only that specific parameter should appear as a span input,
    not all procedure parameters."""
    src = """
@proc(question: str, context: str, mode: str) -> str:
  > optimize: answer_span:
    Given the question: <question>
    The answer is [answer: str]
  = `answer`

@eval: proc:
  > data: cases:
    = ```
    return [(("What is 2+2?", "math", "simple"), None)]
    ```
  > metric: noop(cases):
    = ```
    _ = proc(cases)
    return (1.0, None)
    ```
"""
    _run_optimization(
        monkeypatch,
        src,
        [(("What is 2+2?", "math", "simple"), None)],
    )

    assert Recorder.calls, "Predict should be called"
    for call in Recorder.calls:
        assert "question" in call.kwargs, "Span references <question>"
        assert "context" not in call.kwargs, "Span does NOT reference <context>"
        assert "mode" not in call.kwargs, "Span does NOT reference <mode>"


# ── Span outputs differ from procedure return ────────────────────────


def test_span_outputs_listed_in_instructions(monkeypatch: pytest.MonkeyPatch):
    ">> A span with output [x: float] and [label: str] inside a procedure\n    >> that returns int. The span's instructions should reference the span's\n    output fields, not the procedure's return type."
    src = """
@proc(seed: str) -> int:
  > optimize: detailed_span:
    Produce a value [x: float] and a label [label: str]
  [result: int] = `int(x)`
  = `result`

@eval: proc:
  > data: cases:
    = ```
    return [("alpha", None)]
    ```
  > metric: noop(cases):
    = ```
    _ = proc(cases)
    return (1.0, None)
    ```
"""
    _run_optimization(monkeypatch, src, [("alpha", None)])

    assert Recorder.calls, "Predict should be called"
    first = Recorder.calls[0]
    instructions = first.instructions.lower()
    assert "x" in instructions or "label" in instructions, (
        "Span instructions should reference span output fields x and/or label"
    )
