from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import dspy
import pytest

from kedi.core import KediRuntime
from kedi.lang.ast import (
    InputSegment,
    OptimizeSpan,
    OutputSegment,
    ProcedureDef,
    Program,
    TextSegment,
)
from kedi.optimization.config import OptimizerConfig
from kedi.optimization.services import dspy_config as dspy_config_module
from kedi.optimization.services.dspy_config import AdaptiveJSONAdapter, DSPyConfigurationService
from kedi.optimization.services.dspy_example_formatter import DSPyExampleFormatter
from kedi.optimization.services.dspy_module_builder import DSPyModuleBuilder, _KediRuntimeRef
from kedi.optimization.services.dspy_prompt_extractor import DSPyPromptExtractor
from kedi.optimization.services.dspy_score_calculator import DSPyScoreCalculator
from kedi.optimization.services.prompt_loader import OptimizedPromptLoader


from tests.optimization_helpers import make_optimize_span


def test_dspy_config_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    config = OptimizerConfig(optimizer_model=None, reflection_model=None)
    service = DSPyConfigurationService(config)

    with pytest.raises(ValueError, match="No model"):
        service.configure_main_lm()
    with pytest.raises(ValueError, match="No model"):
        service.configure_reflection_lm()

    monkeypatch.setattr(dspy_config_module.dspy.settings, "lm", None, raising=False)
    with pytest.raises(RuntimeError, match="not configured"):
        service.verify_configured()

    class FakeLM:
        def __init__(
            self,
            model: str,
            model_type: str = "chat",
            temperature: float = 0.0,
            max_tokens: int = 4000,
            cache: bool = True,
            callbacks: Any = None,
            num_retries: int = 3,
            provider: Any = None,
            finetuning_model: Any = None,
            launch_kwargs: Any = None,
            train_kwargs: Any = None,
            **kwargs: Any,
        ) -> None:
            self.kwargs = {"max_tokens": max_tokens, **kwargs}
            self.model = model

    if hasattr(FakeLM, "_kedi_patched"):
        delattr(FakeLM, "_kedi_patched")
    monkeypatch.setattr(dspy_config_module, "LM", FakeLM)
    monkeypatch.setattr(dspy_config_module.dspy, "configure_cache", lambda **kwargs: None)
    service = DSPyConfigurationService(
        OptimizerConfig(optimizer_model="openai/gpt-4o", reflection_model="openai/gpt-4o-mini")
    )
    service.configure_main_lm()
    assert (
        dspy_config_module.LM("later").kwargs["max_tokens"]
        == service.config.get_optimizer_max_tokens()
    )
    reflection_lm = service.configure_reflection_lm()
    assert isinstance(reflection_lm, FakeLM)
    assert reflection_lm.kwargs["timeout"] == 300

    class PatchedLM(FakeLM):
        _kedi_patched = True

    class QuietSettings:
        lm = SimpleNamespace(model="quiet")

        def configure(self, **_: Any) -> None:
            return None

    monkeypatch.setattr(dspy_config_module, "LM", PatchedLM)
    monkeypatch.setattr(dspy_config_module.dspy, "settings", QuietSettings())
    DSPyConfigurationService(OptimizerConfig(optimizer_model="openai/gpt-4o")).configure_main_lm()

    class EmptySettings:
        def configure(self, **_: Any) -> None:
            return None

    monkeypatch.setattr(dspy_config_module.dspy, "settings", EmptySettings())
    DSPyConfigurationService(OptimizerConfig(optimizer_model="openai/gpt-4o")).configure_main_lm()


def test_adaptive_json_adapter_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = AdaptiveJSONAdapter()
    lm = SimpleNamespace(model="model-a")
    signature = object()
    calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        adapter,
        "_json_adapter_call_common",
        lambda *args, **kwargs: [{"ok": True}],
    )
    assert adapter(lm, {}, signature, [], {}) == [{"ok": True}]

    monkeypatch.setattr(adapter, "_json_adapter_call_common", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        dspy_config_module.ChatAdapter,
        "__call__",
        lambda self, lm, lm_kwargs, signature, demos, inputs: (
            calls.append(dict(lm_kwargs)) or [{"fallback": True}]
        ),
    )
    AdaptiveJSONAdapter._failed_models.add("model-a")
    assert adapter(lm, {}, signature, [], {}) == [{"fallback": True}]
    assert calls[-1]["response_format"] == {"type": "json_object"}

    monkeypatch.setattr(
        dspy_config_module,
        "_get_structured_outputs_response_format",
        lambda *args: {"type": "schema"},
    )
    lm2 = SimpleNamespace(model="model-b")
    assert adapter(lm2, {}, signature, [], {}) == [{"fallback": True}]
    assert calls[-1]["response_format"] == {"type": "schema"}

    def raise_structured(*_: Any) -> object:
        raise RuntimeError("no structured")

    monkeypatch.setattr(
        dspy_config_module, "_get_structured_outputs_response_format", raise_structured
    )
    lm3 = SimpleNamespace(model="model-c")
    assert adapter(lm3, {}, signature, [], {}) == [{"fallback": True}]
    assert "model-c" in AdaptiveJSONAdapter._failed_models
    assert calls[-1]["response_format"] == {"type": "json_object"}


def test_example_formatter_span_slow_paths() -> None:
    rt = KediRuntime()
    formatter = DSPyExampleFormatter(rt)
    direct_span = make_optimize_span(
        "direct", [InputSegment(name="x"), OutputSegment(name="answer")]
    )
    examples = formatter.format_examples(
        [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)], ["x"], direct_span, "p"
    )
    assert [example.x for example in examples] == ["1", "3", "5", "7", "9"]

    fallback_span = make_optimize_span("fallback", [InputSegment(name="computed")])
    assert formatter._compute_span_inputs(fallback_span, "missing", {"x": 1}) == {"x": "1"}

    def proc(x: int) -> None:
        current = rt._current_env.get()
        assert current is not None
        current["computed"] = x + 1

    rt._procedures["p"] = proc
    rt._main = cast(Any, SimpleNamespace(__name__="p"))
    assert formatter._compute_span_inputs(fallback_span, "p", {"x": 1}) == {"computed": "2"}

    def broken(_: int) -> None:
        raise RuntimeError("boom")

    rt._procedures["broken"] = broken
    assert formatter._compute_span_inputs(fallback_span, "broken", {"x": 1}) == {"x": "1"}
    assert formatter.format_examples([("single", "expected")], ["a", "b"])[0].a == "single"

    multi_span = make_optimize_span("multi", [InputSegment(name="a"), InputSegment(name="b")])

    def partial(_: int) -> None:
        current = rt._current_env.get()
        assert current is not None
        current["a"] = "A"

    rt._procedures["partial"] = partial
    assert formatter._compute_span_inputs(multi_span, "partial", {"x": 1}) == {"a": "A"}

    def none_env(_: int) -> None:
        rt._current_env.set(None)

    rt._procedures["none_env"] = none_env
    assert formatter._compute_span_inputs(fallback_span, "none_env", {"x": 1}) == {"x": "1"}


def test_module_builder_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    rt = KediRuntime()
    ref = _KediRuntimeRef(rt)
    assert ref.__deepcopy__({}) is ref

    builder = DSPyModuleBuilder(rt)
    with pytest.raises(ValueError, match="at least one output"):
        builder.build_module(make_optimize_span("empty", [InputSegment(name="x")]), ["x"], {})

    signature = builder._create_signature(["x"], [("answer", None)], {})
    assert "answer" in signature.model_fields

    monkeypatch.setattr(
        "kedi.lang.compiler.type_system.KediType.resolve",
        lambda self, runtime, env: None,
    )
    assert builder._resolve_type_annotation("Missing", {}) is str

    def fail_resolve(self: Any, runtime: Any, env: Any) -> object:
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "kedi.lang.compiler.type_system.KediType.resolve",
        fail_resolve,
    )
    assert builder._resolve_type_annotation("Missing", {}) is str


def test_prompt_extractor_edges() -> None:
    extractor = DSPyPromptExtractor()
    span = make_optimize_span("target", [TextSegment(text="Base")])

    assert (
        extractor._find_predictors(
            cast(
                Any,
                SimpleNamespace(named_predictors=lambda: (_ for _ in ()).throw(RuntimeError())),
            )
        )
        == {}
    )

    predictor = dspy.Predict(cast(Any, dspy).Signature("x -> y").with_instructions("Use this"))

    class AttrModule:
        good = predictor

        @property
        def bad(self) -> object:
            raise RuntimeError("bad attr")

        def __dir__(self) -> list[str]:
            return ["_private", "bad", "good"]

    assert extractor._find_predictors(cast(Any, AttrModule())) == {"good": predictor}
    assert extractor._find_predictors(
        cast(
            Any,
            SimpleNamespace(named_predictors=lambda: [("bad", object()), ("good", predictor)]),
        )
    ) == {"good": predictor}
    assert extractor._find_predictors(
        cast(Any, SimpleNamespace(named_predictors=lambda: {"target": predictor}))
    ) == {"target": predictor}
    assert (
        extractor._find_predictor_for_span({"target": predictor}, cast(Any, object()), span, 0)
        is predictor
    )
    assert (
        extractor._find_predictor_for_span({"other": predictor}, cast(Any, object()), span, 0)
        is predictor
    )
    assert (
        extractor._find_predictor_for_span(
            {}, cast(Any, SimpleNamespace(predictor=predictor)), span, 9
        )
        is predictor
    )
    assert (
        extractor._find_predictor_for_span(
            {},
            cast(Any, SimpleNamespace(student=SimpleNamespace(predictor=predictor))),
            span,
            9,
        )
        is predictor
    )
    assert extractor._find_predictor_for_span({}, cast(Any, object()), span, 9) is None

    assert extractor._extract_prompt_from_predictor(None, span) == ""
    assert extractor._extract_prompt_from_predictor(cast(Any, SimpleNamespace()), span) == ""
    assert (
        extractor._extract_prompt_from_predictor(
            cast(Any, SimpleNamespace(signature=SimpleNamespace(instructions=""))), span
        )
        == ""
    )
    default_sig = SimpleNamespace(instructions="Given the fields")
    assert (
        extractor._extract_prompt_from_predictor(
            cast(Any, SimpleNamespace(signature=default_sig)), span
        )
        == ""
    )
    custom_sig = SimpleNamespace(instructions="Use a concise answer.")
    assert (
        extractor._extract_prompt_from_predictor(
            cast(Any, SimpleNamespace(signature=custom_sig)), span
        )
        == "Use a concise answer."
    )


def test_score_calculator_edges() -> None:
    calculator = DSPyScoreCalculator()

    class Module:
        def __call__(self, **kwargs: Any) -> dspy.Prediction:
            if kwargs.get("x") == "boom":
                raise RuntimeError("boom")
            return dspy.Prediction(result=kwargs.get("x"))

    examples = [
        dspy.Example(x=1).with_inputs("x"),
        dspy.Example(x=2).with_inputs("x"),
        dspy.Example(x="boom").with_inputs("x"),
    ]
    values = iter([math.nan, -1.0])

    def metric(_: Any, __: Any) -> float:
        return next(values)

    assert calculator.calculate_score(cast(Any, Module()), examples, metric, sample_size=3) == 0.0

    class Score:
        score = 2.0

    class NoInputs:
        __slots__ = ()

        def inputs(self) -> dict[str, Any]:
            return {}

    assert (
        calculator.calculate_score(cast(Any, Module()), [cast(Any, NoInputs())], lambda *_: Score())
        == 1.0
    )
    assert calculator._extract_inputs(cast(Any, SimpleNamespace(a=1, _hidden=2))) == {"a": 1}
    assert calculator._extract_inputs(cast(Any, object())) == {}


def test_prompt_loader_edges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "sample.kedi"
    source.write_text("@p():\n  = ok\n", encoding="utf-8")
    loader = OptimizedPromptLoader(source)

    assert loader.load_optimized_prompts() == {}
    loader.optimized_path.write_text("{bad json", encoding="utf-8")
    assert loader.load_optimized_prompts() == {}
    loader.inject_optimized_prompts(Program(procedures=[ProcedureDef("p", [])]))

    loader.optimized_path.write_text('{"other": {"s": "optimized"}}', encoding="utf-8")
    program = Program(procedures=[ProcedureDef("p", [])])
    loader.inject_optimized_prompts(program)

    span = OptimizeSpan("s", [])
    program = Program(procedures=[ProcedureDef("p", [], optimize_spans=[span])])
    loader.optimized_path.write_text(json_text := '{"p": {"other": "optimized"}}', encoding="utf-8")
    assert json_text
    loader.inject_optimized_prompts(program)
    assert span.template_lines == []

    loader.optimized_path.write_text('{"p": {"s": "optimized"}}', encoding="utf-8")
    loader.inject_optimized_prompts(program)
    assert span.template_lines[0].optimized_prompt_prefix == "optimized"

    class FakePath:
        suffix = ".kedi"

        def __init__(self, value: object, parent: Path | None = None) -> None:
            self._parent = parent or Path("/source")

        def resolve(self) -> "FakePath":
            return self

        @property
        def parent(self) -> Path:
            return self._parent

        def with_suffix(self, suffix: str) -> "FakePath":
            assert suffix == ".kedi.optimized.json"
            return FakePath("optimized", Path("/different"))

    monkeypatch.setattr("kedi.optimization.services.prompt_loader.Path", FakePath)
    with pytest.raises(RuntimeError, match="Path resolution"):
        OptimizedPromptLoader(source)
