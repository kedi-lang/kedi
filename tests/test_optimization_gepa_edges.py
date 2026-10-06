from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from typing import Any, cast

import dspy
import pytest

from kedi.core import KediRuntime
from kedi.lang.ast import OptimizeSpan, OutputSegment
from kedi.optimization.config import OptimizerConfig
from kedi.optimization.gepa import GEPAOptimizer
from kedi.optimization.services.gepa_metric_wrapper import GEPAMetricWrapper
from tests.optimization_helpers import make_optimize_span


def test_dspy_introspection_finds_student_predictor() -> None:
    from kedi.optimization.dspy_introspection import find_primary_predictor

    class StudentModule:
        student = SimpleNamespace(predictor=dspy.Predict("a -> b"))

    assert find_primary_predictor(StudentModule()) is not None


def test_gepa_metric_wrapper_helper_edges() -> None:
    wrapper = GEPAMetricWrapper()
    span = make_optimize_span("s", [OutputSegment(name="answer")])

    with pytest.raises(ValueError):
        wrapper._extract_input(cast(Any, None), ["x"])
    assert wrapper._extract_input(cast(Any, SimpleNamespace(_proc_inputs={"x": 1})), ["x"]) == 1
    assert wrapper._extract_input(
        cast(Any, SimpleNamespace(_proc_inputs={"x": 1, "y": 2})), ["x", "y"]
    ) == (
        1,
        2,
    )
    assert wrapper._extract_input(cast(Any, SimpleNamespace(x=1, y=2)), ["x", "y"]) == (
        1,
        2,
    )

    assert wrapper._apply_prompt_prefix(None, "x") is None
    assert wrapper._apply_prompt_prefix(span, None) is None
    handle = wrapper._apply_prompt_prefix(span, "optimized\nAnswer [answer]")
    assert span.template_lines[0].optimized_prompt_prefix == "optimized\nAnswer"
    wrapper._restore_prompt_prefix(handle)
    assert span.template_lines[0].optimized_prompt_prefix is None

    assert wrapper._extract_prediction_outputs(None, span) is None
    assert wrapper._extract_prediction_outputs(SimpleNamespace(answer=1), None) is None
    assert wrapper._extract_prediction_outputs(SimpleNamespace(answer=1), span) == {"answer": 1}
    assert wrapper._extract_prediction_outputs(SimpleNamespace(other=1), span) is None
    assert (
        wrapper._extract_prediction_outputs(
            SimpleNamespace(answer=1), make_optimize_span("no_outputs", [])
        )
        is None
    )
    x_span = make_optimize_span("x_out", [OutputSegment(name="x")])
    assert wrapper._extract_prediction_outputs(SimpleNamespace(x=1), x_span) == {"x": 1}

    assert wrapper._apply_output_override(None, {"x": 1}) is None
    assert wrapper._apply_output_override(span, {}) is None
    assert wrapper._apply_output_override(SimpleNamespace(template_lines=[]), {"x": 1}) is None
    output_handle = wrapper._apply_output_override(span, {"answer": "ok"})
    assert span.template_lines[0].optimized_output_override == {"answer": "ok"}
    wrapper._restore_output_override(output_handle)
    assert span.template_lines[0].optimized_output_override is None

    assert wrapper._stringify_feedback(None) is None
    assert wrapper._stringify_feedback("ok") == "ok"
    assert wrapper._stringify_feedback(dspy.Prediction(score=0.5)) == "0.5"
    assert wrapper._stringify_feedback(123) == "123"

    prediction = object()
    assert wrapper._lookup_cached_feedback(prediction) is None
    wrapper._prediction_id_cache = cast(
        Any, SimpleNamespace(get=lambda _: (_ for _ in ()).throw(RuntimeError("bad")))
    )
    assert wrapper._lookup_cached_feedback(prediction) is None
    wrapper._prediction_id_cache = {}
    wrapper._store_cached_feedback(prediction, 0.25, "cached")
    assert wrapper._lookup_cached_feedback(prediction) == (0.25, "cached")

    class AttrPrediction:
        pass

    attr_prediction = AttrPrediction()
    wrapper._store_cached_feedback(attr_prediction, 0.75, None)
    assert wrapper._lookup_cached_feedback(attr_prediction) == (0.75, None)

    assert wrapper._extract_current_instructions(None) is None
    assert (
        wrapper._extract_current_instructions(SimpleNamespace(named_predictors=lambda: [])) is None
    )
    assert (
        wrapper._extract_current_instructions(
            SimpleNamespace(named_predictors=lambda: [(None, SimpleNamespace(signature=None))])
        )
        is None
    )
    assert (
        wrapper._extract_current_instructions(
            SimpleNamespace(
                named_predictors=lambda: [
                    SimpleNamespace(signature=SimpleNamespace(instructions="Given the fields"))
                ]
            )
        )
        is None
    )

    class BrokenModule:
        def named_predictors(self) -> object:
            raise RuntimeError("bad")

    assert wrapper._extract_current_instructions(BrokenModule()) is None

    long_default = "Given the fields " + ("x" * 80)
    assert (
        wrapper._extract_current_instructions(
            SimpleNamespace(
                predictor=SimpleNamespace(signature=SimpleNamespace(instructions=long_default))
            )
        )
        == long_default
    )
    named_only = SimpleNamespace(
        named_predictors=lambda: [
            (
                "main",
                SimpleNamespace(signature=SimpleNamespace(instructions="Optimized " + ("y" * 80))),
            )
        ]
    )
    assert wrapper._extract_current_instructions(named_only) == "Optimized " + ("y" * 80)

    student_instructions = "Student optimized " + ("z" * 80)
    student_predictor = dspy.Predict("x -> y")
    student_predictor.signature.instructions = student_instructions

    class StudentBackedModule:
        student = SimpleNamespace(predictor=student_predictor)

    assert wrapper._extract_current_instructions(StudentBackedModule()) == student_instructions
    assert (
        wrapper._extract_current_instructions(
            SimpleNamespace(predictor=SimpleNamespace(signature=SimpleNamespace(instructions="")))
        )
        is None
    )


def test_gepa_metric_wrapper_wrap_metric_success_error_and_cache() -> None:
    wrapper = GEPAMetricWrapper()
    span = make_optimize_span("s", [OutputSegment(name="answer")])

    calls: list[tuple[Any, Any]] = []

    def metric(input_val: Any, expected: Any) -> tuple[float, str]:
        calls.append((input_val, expected))
        return (0.8, "good")

    gepa_metric = wrapper.wrap_metric(metric, ["x"], span=span, module_ref=None)
    gold = dspy.Example(x=1, _expected=2).with_inputs("x")
    pred = dspy.Prediction(answer="ok")
    result = gepa_metric(gold, pred)
    assert result.score == 0.8
    assert result.feedback == "good"
    assert calls == [(1, 2)]
    assert gepa_metric(gold, pred).score == 0.8

    def broken_metric(_: Any, __: Any) -> tuple[float, str | None]:
        raise RuntimeError("bad")

    broken = wrapper.wrap_metric(broken_metric, ["x"])
    broken_result = broken(gold, dspy.Prediction())
    assert broken_result.score == 0.0
    assert "GEPA metric error" in broken_result.feedback

    module_ref = SimpleNamespace(
        predictor=SimpleNamespace(signature=SimpleNamespace(instructions="Use module."))
    )
    no_override = wrapper.wrap_metric(metric, ["x"], span=span, module_ref=module_ref)
    assert no_override(gold, dspy.Prediction(other="ignored")).score == 0.8


def test_gepa_metric_wrapper_coalesces_parallel_prediction_callbacks() -> None:
    class BarrierReporter:
        def emit(self, message: str, *, flush: bool = False, end: str = "\n") -> None:
            del flush, end
            if message.endswith("starting..."):
                rendezvous.wait(timeout=2)

        def progress_char(self, char: str = ".") -> None:
            del char

    rendezvous = Barrier(2)
    wrapper = GEPAMetricWrapper(reporter=cast(Any, BarrierReporter()))
    calls: list[tuple[Any, Any]] = []

    def metric(input_value: Any, expected: Any) -> tuple[float, str]:
        calls.append((input_value, expected))
        return 1.0, "ok"

    wrapped = wrapper.wrap_metric(metric, ["value"])
    gold = dspy.Example(value="input", _expected="expected").with_inputs("value")
    prediction = dspy.Prediction(answer="same")

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: wrapped(gold, prediction), range(2)))

    assert [result.score for result in results] == [1.0, 1.0]
    assert calls == [("input", "expected")]


def test_gepa_optimizer_edges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import kedi.optimization.gepa as gepa_module

    optimizer = gepa_module.GEPAOptimizer(
        OptimizerConfig(), KediRuntime(), tmp_path / "source.kedi"
    )
    assert optimizer._gepa_log_dir("proc name", "span/name") == (
        tmp_path / "source.kedi.gepa" / "proc_name__span_name"
    )

    assert optimizer.optimize_procedure(
        "p", ["x"], {}, [], [("x", "y")], None, lambda *_: (1, None)
    ) == (
        {},
        0.0,
    )
    with pytest.raises(ValueError, match="No training data"):
        optimizer.optimize_procedure(
            "p", ["x"], {}, [make_optimize_span()], [], None, lambda *_: (1, None)
        )

    optimizer._validate_types("p", ["x"], {}, [("x", "y")])
    optimizer._validate_types("p", ["x"], {"x": str}, [cast(Any, "bad")])

    runtime = KediRuntime()
    optimizer = GEPAOptimizer(OptimizerConfig(), runtime, tmp_path / "source.kedi")

    source_path = cast(Path, optimizer.source_path)
    optimized_path = source_path.with_suffix(source_path.suffix + ".optimized.json")
    optimized_path.write_text("{bad json", encoding="utf-8")
    optimizer._save_prompts_immediately("p", {"s": "prompt"})
    assert "prompt" in optimized_path.read_text(encoding="utf-8")

    optimizer.result_writer = None
    optimizer._save_prompts_immediately("p", {"s": "prompt"})

    cast(Any, optimizer).result_writer = SimpleNamespace(
        write_optimized_prompts=lambda prompts: (_ for _ in ()).throw(RuntimeError("disk"))
    )
    optimizer._save_prompts_immediately("p", {"s": "prompt"})


@pytest.mark.parametrize("reflection", [None, SimpleNamespace(model="test-reflector")])
def test_gepa_optimizer_fake_compile_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reflection
) -> None:
    import kedi.optimization.gepa as gepa_module
    import kedi.optimization.gepa_session as gepa_session_module

    class FakeGEPA:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def compile(self, **_: Any) -> object:
            return SimpleNamespace()

    fake_dspy = SimpleNamespace(
        LM=lambda **kwargs: SimpleNamespace(
            model=kwargs.get("model"), __call__=lambda prompt: ["ok"]
        ),
        GEPA=FakeGEPA,
        settings=SimpleNamespace(lm=None),
    )
    monkeypatch.setattr(gepa_session_module, "dspy", fake_dspy)
    monkeypatch.setattr(gepa_module, "build_reflection_lm", lambda *args: reflection)

    source = tmp_path / "source.kedi"
    source.write_text("= ok\n", encoding="utf-8")
    log_dir = tmp_path / "source.kedi.gepa" / "p__empty"
    log_dir.mkdir(parents=True)
    (log_dir / "gepa_state.bin").write_bytes(b"state")

    optimizer = gepa_module.GEPAOptimizer(
        OptimizerConfig(max_validation_examples=1),
        KediRuntime(),
        source,
    )
    cast(Any, optimizer).dspy_config_service = SimpleNamespace(verify_configured=lambda: None)
    cast(Any, optimizer).example_formatter = SimpleNamespace(
        format_examples=lambda data, params, span=None, proc_name=None: [SimpleNamespace()]
    )
    cast(Any, optimizer).module_builder = SimpleNamespace(
        build_module=lambda *args: SimpleNamespace()
    )
    cast(Any, optimizer).metric_wrapper = SimpleNamespace(
        wrap_metric=lambda *args, **kwargs: lambda *_: 1.0
    )
    cast(Any, optimizer).prompt_extractor = SimpleNamespace(
        extract_prompts=lambda module, spans: {spans[0].name: "optimized"}
    )
    cast(Any, optimizer).score_calculator = SimpleNamespace(calculate_score=lambda *args: 0.6)

    empty_span = make_optimize_span("empty")
    prompts, score = optimizer.optimize_procedure(
        "p",
        ["x"],
        {},
        [empty_span],
        [("x", "y"), ("x2", "y2"), ("x3", "y3"), ("x4", "y4")],
        [("test", "expected"), ("extra", "expected")],
        lambda *_: (1.0, None),
    )
    assert prompts == {"empty": "optimized"}
    assert score == 0.6

    cast(Any, optimizer).prompt_extractor = SimpleNamespace(
        extract_prompts=lambda module, spans: {}
    )
    monkeypatch.setattr(
        gepa_session_module,
        "normalize_optimized_prompt_prefix",
        lambda span, prompt: None,
    )
    with pytest.warns(UserWarning, match="No optimized prompt"):
        prompts, score = optimizer.optimize_procedure(
            "p",
            ["x"],
            {},
            [make_optimize_span("missing", [OutputSegment(name="answer")])],
            [("x", "y")],
            None,
            lambda *_: (1.0, None),
        )
    assert prompts == {}
    assert score == 0.0


def test_gepa_optimizer_reflection_and_proc_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import kedi.optimization.gepa as gepa_module
    import kedi.optimization.gepa_reflection as gepa_reflection_module
    import kedi.optimization.gepa_session as gepa_session_module

    fake_dspy = SimpleNamespace(
        LM=lambda **kwargs: SimpleNamespace(
            model=kwargs.get("model"),
            __call__=lambda prompt: ["ok"],
        ),
        GEPA=lambda **kwargs: SimpleNamespace(compile=lambda **_: SimpleNamespace()),
        settings=SimpleNamespace(lm=SimpleNamespace(model="main", kwargs={"x": 1})),
    )
    monkeypatch.setattr(gepa_reflection_module, "ThinkingStrippingLM", fake_dspy.LM)
    monkeypatch.setattr(gepa_session_module, "dspy", fake_dspy)

    runtime = KediRuntime()
    optimizer = gepa_module.GEPAOptimizer(
        OptimizerConfig(reflection_model="openai/gpt-4o-mini"),
        runtime,
        tmp_path / "source.kedi",
    )
    cast(Any, optimizer).dspy_config_service = SimpleNamespace(verify_configured=lambda: None)
    cast(Any, optimizer).example_formatter = SimpleNamespace(
        format_examples=lambda data, params, span=None, proc_name=None: [SimpleNamespace()]
    )
    cast(Any, optimizer).module_builder = SimpleNamespace(
        build_module=lambda *args: SimpleNamespace()
    )
    cast(Any, optimizer).metric_wrapper = SimpleNamespace(
        wrap_metric=lambda *args, **kwargs: SimpleNamespace(_progress={"count": 3})
    )
    cast(Any, optimizer).prompt_extractor = SimpleNamespace(
        extract_prompts=lambda module, spans: {spans[0].name: ""}
    )
    cast(Any, optimizer).score_calculator = SimpleNamespace(calculate_score=lambda *args: 1.0)

    with pytest.warns(UserWarning, match="appears unchanged"):
        optimizer.optimize_procedure(
            "p",
            ["x"],
            {},
            [OptimizeSpan("same", [])],
            [("x", "y")],
            None,
            lambda *_: (1.0, None),
        )
