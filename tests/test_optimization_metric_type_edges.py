from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from kedi.lang import compile_program, parse_program
from kedi.lang.ast import ProcedureDef
from kedi.optimization.services import type_handler as type_handler_module
from kedi.optimization.services.metric_factory import MetricFactory
from kedi.optimization.services.type_handler import StructuredTypeHandler


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

    from kedi.optimization.utils import normalize_metric_result

    with pytest.raises(TypeError, match="Metric tuple result"):
        normalize_metric_result((1, 2, 3))
    assert normalize_metric_result("0.5") == (0.5, None)
    assert normalize_metric_result(True) == (1.0, None)


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

    with pytest.raises(ValueError, match="Output schema"):
        handler.create_dspy_signature_sync(["x"], [])

    def typed_builder(output_schema: dict[str, type], **_: Any) -> str:
        assert output_schema
        return "user_prompt -> generated: Custom"

    monkeypatch.setattr(type_handler_module.DSPyAdapter, "type_builder", typed_builder)
    assert handler.create_dspy_signature_sync(["x"], [("answer", "str")]) == "x -> answer: Custom"

    def fail_resolve(self: Any, runtime: Any, env: Any) -> object:
        raise RuntimeError("bad type")

    with monkeypatch.context() as mp:
        mp.setattr("kedi.lang.compiler.type_system.KediType.resolve", fail_resolve)
        assert handler.build_output_schema([("broken", "int")]) == {"broken": str}

    def no_arrow_builder(output_schema: dict[str, type], **_: Any) -> str:
        assert output_schema
        return "generated"

    monkeypatch.setattr(type_handler_module.DSPyAdapter, "type_builder", no_arrow_builder)
    assert handler.create_dspy_signature_sync(["x"], [("answer", "int")]) == "x -> answer: int"

    def object_builder(output_schema: dict[str, type], **_: Any) -> object:
        assert output_schema
        return object()

    monkeypatch.setattr(type_handler_module.DSPyAdapter, "type_builder", object_builder)
    assert handler.create_dspy_signature_sync(["x"], [("answer", "int")]) == "x -> answer"

    handler = StructuredTypeHandler(rt)

    def bare_output_builder(output_schema: dict[str, type], **_: Any) -> str:
        assert output_schema == {"different": int}
        return "user_prompt -> other"

    monkeypatch.setattr(type_handler_module.DSPyAdapter, "type_builder", bare_output_builder)
    monkeypatch.setattr(handler, "build_output_schema", lambda fields: {"different": int})
    assert handler.create_dspy_signature_sync(["x"], [("answer", "int")]) == "x -> answer"

    def no_arrow_missing_builder(output_schema: dict[str, type], **_: Any) -> str:
        assert output_schema == {"different": int}
        return "other"

    monkeypatch.setattr(type_handler_module.DSPyAdapter, "type_builder", no_arrow_missing_builder)
    assert handler.create_dspy_signature_sync(["x"], [("answer", "int")]) == "x -> answer"

    unknown_type = object()

    def matching_schema_builder(output_schema: dict[str, type], **_: Any) -> str:
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
