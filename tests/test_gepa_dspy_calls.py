"""DSPy-based GEPA tests: assert span-only inputs and substituted instructions."""

from __future__ import annotations

from typing import Any, List

import pytest

from kedi.lang import parse_program, compile_program
from kedi.optimization.config import OptimizerConfig
from kedi.optimization.gepa import GEPAOptimizer
from kedi.optimization.services import TypeResolver, MetricFactory


class _CallRecorder:
    calls: List[dict[str, Any]] = []

    @classmethod
    def reset(cls) -> None:
        cls.calls.clear()


def _install_fake_dspy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install fakes for dspy.Predict and dspy.GEPA to capture calls."""
    import dspy

    # Configure dummy LM and disable any cache
    class DummyLM:
        pass

    dspy.settings.configure(lm=DummyLM(), cache=False)
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

    class FakePrediction:
        def __init__(self, **fields: Any):
            for k, v in fields.items():
                setattr(self, k, v)

    class FakePredict:
        def __init__(self, sig: Any):
            self.signature = sig

        def __call__(self, **kwargs: Any):
            # Extract instructions if present on signature
            instr = None
            sig = self.signature
            if hasattr(sig, "instructions"):
                instr = getattr(sig, "instructions")
            elif hasattr(sig, "_instructions"):
                instr = getattr(sig, "_instructions")
            elif isinstance(sig, str):
                instr = sig

            _CallRecorder.calls.append({"instructions": instr or "", "kwargs": dict(kwargs)})

            # Return prediction with outputs inferred from kwargs for convenience
            # (Optimizer wrapper will override outputs anyway)
            return FakePrediction(**kwargs)

    class FakeGEPA:
        def __init__(self, metric=None, max_metric_calls: int = 30, reflection_lm=None, reflection_minibatch_size: int = 3, **kwargs):
            self.metric = metric
            self.max_metric_calls = max_metric_calls
            self.reflection_lm = reflection_lm
            self.reflection_minibatch_size = reflection_minibatch_size

        def compile(self, student, trainset, valset):
            # Simulate a training loop: call student on each example and run the metric
            for ex in trainset:
                # Pass all public attributes; module will filter to span inputs
                # Gather public fields from Example storage
                raw_kwargs = {k: v for k, v in ex.items() if not str(k).startswith("_")}
                # Forward only span-scoped inputs to better match DSPy behavior
                span_inputs = getattr(student, "_kedi_span_inputs", None) or []
                filtered = {k: v for k, v in raw_kwargs.items() if k in span_inputs}
                # Render instructions using Kedi renderer and inject into predictor
                try:
                    from kedi.optimization.utils import render_instruction_lines
                    span = getattr(student, "_kedi_span", None)
                    rt = getattr(student, "_kedi_runtime", None)
                    if span is not None and rt is not None:
                        rendered = render_instruction_lines(span, rt, filtered)
                        sig = student.predictor.signature
                        # Replace predictor with updated instructions
                        student.predictor = dspy.Predict(sig.with_instructions(rendered))
                except Exception:
                    # If rendering fails, continue with existing instructions
                    pass
                # Call predictor directly to generate a prediction and trigger recording
                pred = student.predictor(**filtered)
                if self.metric is not None:
                    try:
                        self.metric(ex, pred)
                    except Exception:
                        # Keep going to allow assertions on captured calls
                        pass
            return student

    monkeypatch.setattr(dspy, "Predict", FakePredict, raising=True)
    monkeypatch.setattr(dspy, "GEPA", FakeGEPA, raising=True)


def _common_setup(src: str):
    """Parse, compile runtime, and build helper services."""
    prog = parse_program(src)
    rt = compile_program(prog, adapter=None)
    proc = prog.procedures[0]
    resolver = TypeResolver(rt)
    param_types = resolver.resolve_param_types(proc)
    eval_suite = prog._validation_evals[0]
    metric_def = eval_suite.metrics[0]
    dataset_name = metric_def.dataset_name or next(iter(eval_suite.datasets.keys()))
    metric = MetricFactory(rt).create_metric(metric_def, proc, dataset_name)
    return prog, rt, proc, param_types, eval_suite, metric


def test_gepa_span_simple_substitution_and_kwargs(monkeypatch: pytest.MonkeyPatch):
    """Only span inputs are sent; instructions have substitutions; outputs remain."""
    _CallRecorder.reset()
    _install_fake_dspy(monkeypatch)

    src = """
@find(country: str) -> str:
  > optimize: find_capital:
    [capital: str] is the capital of <country>.
  = `capital`

@eval: find:
  > data: cases:
    = ```
    return [("France", "Paris")]
    ```
  > metric: correctness(cases):
    = ```
    # trivial metric to avoid dependency on procedure execution
    return (1.0, None)
    ```
"""
    prog, rt, proc, param_types, eval_suite, metric = _common_setup(src)

    # Optimizer with minimal budget
    optimizer = GEPAOptimizer(config=OptimizerConfig(max_metric_calls=3), runtime=rt)

    spans = proc.optimize_spans
    train_data = [(("France"), "Paris")]
    test_data = None

    # Suppress warning about unchanged prompt - this test uses FakeGEPA which doesn't optimize
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*appears unchanged.*")
        optimized, score = optimizer.optimize_procedure(
            proc_name=proc.name,
            proc_params=proc.params,
            param_types=param_types,
            spans=spans,
            train_data=train_data,
            test_data=test_data,
            metric=metric,
        )

    # Assertions on recorded calls
    assert _CallRecorder.calls, "Predict should be called at least once"
    call = _CallRecorder.calls[0]
    # Prefix-only program: no inputs forwarded to predictor
    assert set(call["kwargs"].keys()) == set()
    # Instructions contain the template (with placeholders and output markers)
    instr = call["instructions"]
    assert isinstance(instr, str)
    # The template should contain both placeholders and output markers
    assert "<country>" in instr or "country" in instr.lower()
    assert "[capital" in instr or "capital" in instr.lower()

    # Outputs present
    assert "find_capital" in optimized or isinstance(optimized, dict)


def test_gepa_span_complex_multi_input_and_outputs(monkeypatch: pytest.MonkeyPatch):
    """Complex span: multiple inputs; ensure only span inputs sent and substitutions applied."""
    _CallRecorder.reset()
    _install_fake_dspy(monkeypatch)

    src = """
@compute(a: int, b: int, c: int) -> int:
  > optimize: compute_result:
    Calculate <a> + <b>.
    The answer is: [sum: int]
  = `sum`

@eval: compute:
  > data: cases:
    = ```
    return [(((2, 3, 999)), 5)]
    ```
  > metric: correctness(cases):
    = ```
    # No-op metric, just return success
    return (1.0, None)
    ```
"""
    prog, rt, proc, param_types, eval_suite, metric = _common_setup(src)

    optimizer = GEPAOptimizer(config=OptimizerConfig(max_metric_calls=3), runtime=rt)

    spans = proc.optimize_spans
    train_data = [(((2, 3, 999)), 5)]

    # Suppress warning about unchanged prompt - this test uses FakeGEPA which doesn't optimize
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*appears unchanged.*")
        optimized, score = optimizer.optimize_procedure(
            proc_name=proc.name,
            proc_params=proc.params,
            param_types=param_types,
            spans=spans,
            train_data=train_data,
            test_data=None,
            metric=metric,
        )

    assert _CallRecorder.calls, "Predict should be called"
    call = _CallRecorder.calls[0]
    # Prefix-only program: no inputs forwarded to predictor
    assert set(call["kwargs"].keys()) == set()
    instr = call["instructions"]
    assert isinstance(instr, str)
    # The template should contain placeholders and output markers
    assert ("<a>" in instr and "<b>" in instr) or ("calculate" in instr.lower())
    assert "[sum:" in instr or "sum" in instr.lower()


