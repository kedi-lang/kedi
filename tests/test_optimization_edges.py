from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import dspy
import pytest

from kedi.core import KediRuntime
from kedi.lang import compile_program, parse_program
from kedi.lang.ast import (
    CallSegment,
    InputSegment,
    LiteralArg,
    OptimizeSpan,
    OutputSegment,
    ProcedureDef,
    Program,
    PythonExprNativeSegment,
    PythonExprSegment,
    TemplateLine,
    TextSegment,
)
from kedi.optimization import get_optimizer
from kedi.optimization.config import OptimizerConfig
from kedi.optimization.gepa import GEPAOptimizer
from kedi.optimization.mock import MockOptimizer
from kedi.optimization.services import dspy_config as dspy_config_module
from kedi.optimization.services import dspy_example_formatter as example_formatter_module
from kedi.optimization.services import dspy_module_builder as module_builder_module
from kedi.optimization.services import dspy_prompt_extractor as prompt_extractor_module
from kedi.optimization.services import dspy_score_calculator as score_calculator_module
from kedi.optimization.services import type_handler as type_handler_module
from kedi.optimization.services.dspy_config import AdaptiveJSONAdapter, DSPyConfigurationService
from kedi.optimization.services.dspy_example_formatter import DSPyExampleFormatter
from kedi.optimization.services.dspy_module_builder import DSPyModuleBuilder, _KediRuntimeRef
from kedi.optimization.services.dspy_prompt_extractor import DSPyPromptExtractor
from kedi.optimization.services.dspy_score_calculator import DSPyScoreCalculator
from kedi.optimization.services.gepa_metric_wrapper import GEPAMetricWrapper
from kedi.optimization.services.metric_factory import MetricFactory
from kedi.optimization.services.prompt_loader import OptimizedPromptLoader
from kedi.optimization.services.type_handler import StructuredTypeHandler
from kedi.optimization.utils import (
    extract_input_fields,
    extract_output_fields,
    normalize_optimized_prompt_prefix,
    render_instruction_lines,
    template_line_to_string,
)


def _span(name: str = "prompt", segments: list[Any] | None = None) -> OptimizeSpan:
    return OptimizeSpan(
        name=name,
        template_lines=[
            TemplateLine(
                segments=(
                    segments
                    if segments is not None
                    else [
                        TextSegment(text="Answer "),
                        OutputSegment(name="answer", type_annotation="str"),
                    ]
                )
            )
        ],
    )


def test_optimizer_registry_unknown_and_gepa_import_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    assert get_optimizer("missing") is None
    assert isinstance(get_optimizer("mock"), MockOptimizer)

    optimization_module = sys.modules["kedi.optimization"]
    real_import = __import__

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in {"gepa", "kedi.optimization.gepa"}:
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "kedi.optimization.gepa", raising=False)
    monkeypatch.setattr("builtins.__import__", blocked_import)
    reloaded = importlib.reload(optimization_module)

    assert reloaded.GEPA_AVAILABLE is False
    assert reloaded.GEPAOptimizer is None

    monkeypatch.undo()
    importlib.reload(optimization_module)


def test_mock_optimizer_error_and_prompt_branches() -> None:
    optimizer = MockOptimizer()
    spans = [
        _span("plain", [TextSegment(text="Plain prompt")]),
        _span("think", [TextSegment(text="Think step by step already")]),
        _span(
            "done",
            [TextSegment(text="Think step by step. Consider all aspects carefully.")],
        ),
    ]

    def broken_metric(_: Any, __: Any) -> tuple[float, str | None]:
        raise RuntimeError("metric failed")

    prompts, score = optimizer.optimize_procedure(
        "proc",
        ["x"],
        {},
        spans,
        [("input", "expected")],
        None,
        broken_metric,
    )

    assert score == 0.0
    assert prompts["plain"].startswith("Think step by step.")
    assert prompts["think"].endswith("Consider all aspects carefully.")
    assert prompts["done"].endswith("[optimized]")

    optimizer._validate_types("no_params", [], {}, [("x", "y")])
    with pytest.raises(TypeError, match="Expected tuple"):
        optimizer._validate_types("bad", ["x"], {}, [cast(Any, "not a tuple")])
    with pytest.raises(TypeError, match="Expected tuple of length"):
        optimizer._validate_types("bad", ["x", "y"], {}, [(("only-one",), "expected")])


def test_optimization_utils_render_and_normalize_edges() -> None:
    line = TemplateLine(
        segments=[
            TextSegment(text="A "),
            OutputSegment(name="raw"),
            PythonExprSegment(code="x + 1"),
            cast(Any, object()),
            PythonExprNativeSegment(),
        ]
    )
    assert template_line_to_string(line) == "A [raw]<`x + 1`><expr>"

    empty_span = _span("empty", [])
    assert normalize_optimized_prompt_prefix(empty_span, None) == ""
    assert normalize_optimized_prompt_prefix(empty_span, "custom") == "custom"

    base_span = _span("base", [TextSegment(text="Base")])
    assert normalize_optimized_prompt_prefix(base_span, "prefixBase") == "prefixBase"

    rt = KediRuntime()

    @rt.procedure
    def double(value: int) -> int:
        return value * 2

    rendered_span = _span(
        "render",
        [
            TextSegment(text="Name "),
            InputSegment(name="name"),
            TextSegment(text=" missing "),
            InputSegment(name="missing"),
            TextSegment(text=" call "),
            CallSegment(name="double", args=[LiteralArg(value=cast(Any, 2))]),
            TextSegment(text=" badcall "),
            CallSegment(name="missing", args=[]),
            TextSegment(text=" out "),
            OutputSegment(name="answer"),
            TextSegment(text=" typed "),
            OutputSegment(name="count", type_annotation="int"),
            TextSegment(text=" py "),
            PythonExprSegment(code="1 + 2"),
            TextSegment(text=" bad "),
            PythonExprNativeSegment(code="1 / 0"),
            cast(Any, object()),
        ],
    )

    assert render_instruction_lines(rendered_span, rt, {"name": "Ada"}) == (
        "Name Ada missing <missing> call 4 badcall <missing()> out [answer] typed [count: int] "
        "py 3 bad <`1 / 0`>"
    )

    repeated = _span(
        "repeated",
        [
            OutputSegment(name="x"),
            TextSegment(text="skip"),
            OutputSegment(name="x"),
            InputSegment(name="x"),
            TextSegment(text="skip"),
            InputSegment(name="x"),
        ],
    )
    assert extract_output_fields(repeated) == [("x", None)]
    assert extract_input_fields(repeated) == ["x"]


def test_metric_factory_inline_error_and_normalization_edges() -> None:
    rt = compile_program(parse_program("@p(a, b):\n  = ok\n"), adapter=None)
    factory = MetricFactory(rt)
    proc = ProcedureDef("p", ["a", "b"])

    inline_metric = factory.create_metric(
        cast(Any, SimpleNamespace(code="0.25", is_block=False)),
        proc,
        "case",
    )
    assert inline_metric("single", "expected") == (0.25, None)

    broken_metric = factory.create_metric(
        cast(Any, SimpleNamespace(code="1 / 0", is_block=False)),
        ProcedureDef("p", ["x"]),
        "case",
    )
    assert broken_metric("input", "expected")[0] == 0.0

    with pytest.raises(TypeError, match="Metric tuple result"):
        factory._normalize_metric_result((1, 2, 3))
    assert factory._normalize_metric_result("0.5") == (0.5, None)


def test_type_handler_schema_and_signature_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rt = compile_program(parse_program("= ok\n"), adapter=None)
    handler = StructuredTypeHandler(rt)

    schema = handler.build_output_schema(
        [
            ("plain", None),
            ("count", "int"),
            ("items", "list[int]"),
            ("missing", "MissingType"),
        ]
    )
    assert schema["plain"] is str
    assert schema["count"] is int
    assert getattr(schema["items"], "__origin__", None) is list
    assert schema["missing"] is str

    monkeypatch.setattr(type_handler_module, "DSPY_AVAILABLE", False)
    with pytest.raises(ImportError):
        handler.create_dspy_signature_sync(["x"], [("answer", "str")])
    monkeypatch.setattr(type_handler_module, "DSPY_AVAILABLE", True)

    with pytest.raises(ValueError, match="Output schema"):
        handler.create_dspy_signature_sync(["x"], [])

    async def typed_builder(output_schema: dict[str, type], **_: Any) -> str:
        assert output_schema
        return "user_prompt -> generated: Custom"

    monkeypatch.setattr(type_handler_module.DSPyAdapter, "type_builder", typed_builder)
    assert handler.create_dspy_signature_sync(["x"], [("answer", "str")]) == "x -> answer: Custom"

    def fail_resolve(self: Any, runtime: Any, env: Any) -> object:
        raise RuntimeError("bad type")

    with monkeypatch.context() as mp:
        mp.setattr("kedi.lang.compiler.type_system.KediType.resolve", fail_resolve)
        assert handler.build_output_schema([("broken", "int")]) == {"broken": str}

    async def no_arrow_builder(output_schema: dict[str, type], **_: Any) -> str:
        assert output_schema
        return "generated"

    monkeypatch.setattr(type_handler_module.DSPyAdapter, "type_builder", no_arrow_builder)
    assert handler.create_dspy_signature_sync(["x"], [("answer", "int")]) == "x -> answer: int"

    async def object_builder(output_schema: dict[str, type], **_: Any) -> object:
        assert output_schema
        return object()

    monkeypatch.setattr(type_handler_module.DSPyAdapter, "type_builder", object_builder)
    assert handler.create_dspy_signature_sync(["x"], [("answer", "int")]) == "x -> answer"

    handler = StructuredTypeHandler(rt)

    async def bare_output_builder(output_schema: dict[str, type], **_: Any) -> str:
        assert output_schema == {"different": int}
        return "user_prompt -> other"

    monkeypatch.setattr(type_handler_module.DSPyAdapter, "type_builder", bare_output_builder)
    monkeypatch.setattr(handler, "build_output_schema", lambda fields: {"different": int})
    assert handler.create_dspy_signature_sync(["x"], [("answer", "int")]) == "x -> answer"

    async def no_arrow_missing_builder(output_schema: dict[str, type], **_: Any) -> str:
        assert output_schema == {"different": int}
        return "other"

    monkeypatch.setattr(type_handler_module.DSPyAdapter, "type_builder", no_arrow_missing_builder)
    assert handler.create_dspy_signature_sync(["x"], [("answer", "int")]) == "x -> answer"

    unknown_type = object()

    async def matching_schema_builder(output_schema: dict[str, type], **_: Any) -> str:
        assert output_schema == {"other": cast(Any, unknown_type)}
        return "user_prompt -> other"

    monkeypatch.setattr(type_handler_module.DSPyAdapter, "type_builder", matching_schema_builder)
    monkeypatch.setattr(
        handler,
        "build_output_schema",
        lambda fields: {"other": cast(Any, unknown_type)},
    )
    assert handler.create_dspy_signature_sync(["x"], [("answer", "int")]).startswith(
        "x -> answer: <object object at"
    )


def test_dspy_availability_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    config = OptimizerConfig(optimizer_model="openai/gpt-4o")

    monkeypatch.setattr(dspy_config_module, "DSPY_AVAILABLE", False)
    with pytest.raises(ImportError):
        DSPyConfigurationService(config)
    service = object.__new__(DSPyConfigurationService)
    service.config = config
    with pytest.raises(ImportError):
        service.configure_main_lm()
    with pytest.raises(ImportError):
        service.configure_reflection_lm()
    with pytest.raises(ImportError):
        service.verify_configured()
    monkeypatch.setattr(dspy_config_module, "DSPY_AVAILABLE", True)

    monkeypatch.setattr(example_formatter_module, "DSPY_AVAILABLE", False)
    with pytest.raises(ImportError):
        DSPyExampleFormatter(KediRuntime()).format_examples([(1, 2)], ["x"])
    monkeypatch.setattr(example_formatter_module, "DSPY_AVAILABLE", True)

    monkeypatch.setattr(module_builder_module, "DSPY_AVAILABLE", False)
    with pytest.raises(ImportError):
        DSPyModuleBuilder(KediRuntime())
    builder = object.__new__(DSPyModuleBuilder)
    with pytest.raises(ImportError):
        builder.build_module(_span(), ["x"], {})
    monkeypatch.setattr(module_builder_module, "DSPY_AVAILABLE", True)

    monkeypatch.setattr(prompt_extractor_module, "DSPY_AVAILABLE", False)
    with pytest.raises(ImportError):
        DSPyPromptExtractor().extract_prompts(cast(Any, object()), [_span()])
    monkeypatch.setattr(prompt_extractor_module, "DSPY_AVAILABLE", True)

    monkeypatch.setattr(score_calculator_module, "DSPY_AVAILABLE", False)
    with pytest.raises(ImportError):
        DSPyScoreCalculator().calculate_score(cast(Any, object()), [], lambda *_: 1.0)
    monkeypatch.setattr(score_calculator_module, "DSPY_AVAILABLE", True)


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
    direct_span = _span("direct", [InputSegment(name="x"), OutputSegment(name="answer")])
    examples = formatter.format_examples(
        [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)], ["x"], direct_span, "p"
    )
    assert [example.x for example in examples] == ["1", "3", "5", "7", "9"]

    fallback_span = _span("fallback", [InputSegment(name="computed")])
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

    multi_span = _span("multi", [InputSegment(name="a"), InputSegment(name="b")])

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
        builder.build_module(_span("empty", [InputSegment(name="x")]), ["x"], {})

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
    span = _span("target", [TextSegment(text="Base")])

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


def test_gepa_metric_wrapper_helper_edges() -> None:
    wrapper = GEPAMetricWrapper()
    span = _span("s", [OutputSegment(name="answer")])

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
        wrapper._extract_prediction_outputs(SimpleNamespace(answer=1), _span("no_outputs", []))
        is None
    )
    assert wrapper._extract_prediction_outputs(SimpleNamespace(x=1), None, [("x", "int")]) == {
        "x": 1
    }

    assert wrapper._apply_output_override(None, {"x": 1}) is None
    assert wrapper._apply_output_override(span, {}) is None
    assert wrapper._apply_output_override(SimpleNamespace(template_lines=[]), {"x": 1}) is None
    output_handle = wrapper._apply_output_override(span, {"answer": "ok"})
    assert span.template_lines[0].optimized_output_override == {"answer": "ok"}
    wrapper._restore_output_override(output_handle)
    assert span.template_lines[0].optimized_output_override is None

    assert wrapper._line_output_names(span.template_lines[0]) == ["answer"]
    assert wrapper._apply_default_overrides(SimpleNamespace(optimize_spans=[]), span) is None
    no_output_sibling = _span("no_output", [TextSegment(text="plain")])
    assert (
        wrapper._apply_default_overrides(
            SimpleNamespace(optimize_spans=[span, no_output_sibling]), span
        )
        is None
    )
    sibling = _span("sibling", [OutputSegment(name="other")])
    default_handle = wrapper._apply_default_overrides(
        SimpleNamespace(optimize_spans=[span, sibling]), span
    )
    assert sibling.template_lines[0].optimized_output_override == {"other": ""}
    wrapper._restore_sibling_overrides(default_handle)
    assert sibling.template_lines[0].optimized_output_override is None
    wrapper._restore_sibling_overrides(None)

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
    assert (
        wrapper._extract_current_instructions(
            SimpleNamespace(predictor=SimpleNamespace(signature=SimpleNamespace(instructions="")))
        )
        is None
    )


def test_gepa_metric_wrapper_wrap_metric_success_error_and_cache() -> None:
    wrapper = GEPAMetricWrapper()
    span = _span("s", [OutputSegment(name="answer")])

    calls: list[tuple[Any, Any]] = []

    def metric(input_val: Any, expected: Any) -> tuple[float, str]:
        calls.append((input_val, expected))
        return (0.8, "good")

    gepa_metric = wrapper.wrap_metric(metric, ["x"], span, module_ref=None)
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
    no_override = wrapper.wrap_metric(metric, ["x"], span, module_ref=module_ref)
    assert no_override(gold, dspy.Prediction(other="ignored")).score == 0.8


def test_gepa_optimizer_edges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import kedi.optimization.gepa as gepa_module

    optimizer_obj = object.__new__(gepa_module.GEPAOptimizer)
    monkeypatch.setattr(gepa_module, "DSPY_AVAILABLE", False)
    with pytest.raises(ImportError):
        gepa_module.GEPAOptimizer(OptimizerConfig(), KediRuntime())
    with pytest.raises(ImportError):
        optimizer_obj.optimize_procedure(
            "p", ["x"], {}, [_span()], [("x", "y")], None, lambda *_: (1, None)
        )
    monkeypatch.setattr(gepa_module, "DSPY_AVAILABLE", True)

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
        optimizer.optimize_procedure("p", ["x"], {}, [_span()], [], None, lambda *_: (1, None))

    assert optimizer._get_proc_def("missing") is None
    optimizer.runtime = None
    assert optimizer._get_proc_def("missing") is None
    optimizer._validate_types("p", ["x"], {}, [("x", "y")])
    optimizer._validate_types("p", ["x"], {"x": str}, [cast(Any, "bad")])

    runtime = KediRuntime()
    optimizer = GEPAOptimizer(OptimizerConfig(), runtime, tmp_path / "source.kedi")
    runtime._procedures["plain"] = cast(Any, object())
    assert optimizer._get_proc_def("plain") is None

    source_path = cast(Path, optimizer.source_path)
    optimized_path = source_path.with_suffix(source_path.suffix + ".optimized.json")
    optimized_path.write_text("{bad json", encoding="utf-8")
    optimizer._save_prompts_immediately("p", {"s": "prompt"})
    assert "prompt" in optimized_path.read_text(encoding="utf-8")

    optimizer.result_writer = None
    optimizer._save_prompts_immediately("p", {"s": "prompt"})

    optimizer.result_writer = SimpleNamespace(
        write_optimized_prompts=lambda prompts: (_ for _ in ()).throw(RuntimeError("disk"))
    )
    optimizer._save_prompts_immediately("p", {"s": "prompt"})


def test_gepa_optimizer_fake_compile_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import kedi.optimization.gepa as gepa_module

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
    monkeypatch.setattr(gepa_module, "dspy", fake_dspy)

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
    optimizer.dspy_config_service = SimpleNamespace(verify_configured=lambda: None)
    optimizer.example_formatter = SimpleNamespace(
        format_examples=lambda data, params, span=None, proc_name=None: [SimpleNamespace()]
    )
    optimizer.module_builder = SimpleNamespace(build_module=lambda *args: SimpleNamespace())
    optimizer.metric_wrapper = SimpleNamespace(wrap_metric=lambda *args, **kwargs: lambda *_: 1.0)
    optimizer.prompt_extractor = SimpleNamespace(
        extract_prompts=lambda module, spans: {spans[0].name: "optimized"}
    )
    optimizer.score_calculator = SimpleNamespace(calculate_score=lambda *args: 0.6)

    empty_span = _span("empty")
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

    optimizer.prompt_extractor = SimpleNamespace(extract_prompts=lambda module, spans: {})
    monkeypatch.setattr(gepa_module, "normalize_optimized_prompt_prefix", lambda span, prompt: None)
    with pytest.warns(UserWarning, match="No optimized prompt"):
        prompts, score = optimizer.optimize_procedure(
            "p",
            ["x"],
            {},
            [_span("missing", [OutputSegment(name="answer")])],
            [("x", "y")],
            None,
            lambda *_: (1.0, None),
        )
    assert prompts == {}
    assert score == 0.0


def test_gepa_optimizer_reflection_and_proc_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import inspect

    import kedi.optimization.gepa as gepa_module

    fake_dspy = SimpleNamespace(
        LM=lambda **kwargs: lambda prompt: ["ok"],
        GEPA=lambda **kwargs: SimpleNamespace(compile=lambda **_: SimpleNamespace()),
        settings=SimpleNamespace(lm=SimpleNamespace(model="main", kwargs={"x": 1})),
    )
    monkeypatch.setattr(gepa_module, "dspy", fake_dspy)

    runtime = KediRuntime()
    optimizer = gepa_module.GEPAOptimizer(
        OptimizerConfig(reflection_model="openai/gpt-4o-mini"),
        runtime,
        tmp_path / "source.kedi",
    )
    optimizer.dspy_config_service = SimpleNamespace(verify_configured=lambda: None)
    optimizer.example_formatter = SimpleNamespace(
        format_examples=lambda data, params, span=None, proc_name=None: [SimpleNamespace()]
    )
    optimizer.module_builder = SimpleNamespace(build_module=lambda *args: SimpleNamespace())
    optimizer.metric_wrapper = SimpleNamespace(
        wrap_metric=lambda *args, **kwargs: SimpleNamespace(_progress={"count": 3})
    )
    optimizer.prompt_extractor = SimpleNamespace(
        extract_prompts=lambda module, spans: {spans[0].name: ""}
    )
    optimizer.score_calculator = SimpleNamespace(calculate_score=lambda *args: 1.0)

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

    def proc() -> None:
        return None

    runtime._procedures["p"] = proc
    monkeypatch.setattr(
        inspect, "getclosurevars", lambda fn: (_ for _ in ()).throw(RuntimeError("bad"))
    )
    assert optimizer._get_proc_def("p") is None
