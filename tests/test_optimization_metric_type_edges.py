from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from kedi.lang import compile_program, parse_program
from kedi.lang.ast import ProcedureDef
from kedi.optimization.services.metric_factory import MetricFactory


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


def test_gepa_signature_builder_uses_native_dspy_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dspy

    from kedi.optimization.services.dspy_module_builder import DSPyModuleBuilder

    runtime = compile_program(parse_program("= ok\n"), adapter=None)
    builder = DSPyModuleBuilder(runtime)
    signature = builder._create_signature(
        ["query"], [("plain", None), ("count", "int"), ("items", "list[int]")], {}
    )
    assert issubclass(signature, dspy.Signature)
    assert tuple(signature.input_fields) == ("query",)
    assert signature.output_fields["plain"].annotation is str
    assert signature.output_fields["count"].annotation is int
    assert signature.output_fields["items"].annotation == list[int]
    assert builder.runtime is runtime

    with pytest.raises(TypeError, match="Unknown type annotation MissingType"):
        builder._create_signature(["query"], [("missing", "MissingType")], {})

    def fail_resolve(self: Any, runtime: Any, env: Any) -> object:
        raise RuntimeError("bad type")

    monkeypatch.setattr("kedi.lang.compiler.type_system.KediType.resolve", fail_resolve)
    with pytest.raises(RuntimeError, match="bad type"):
        builder._create_signature(["query"], [("broken", "int")], {})
