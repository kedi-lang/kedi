from __future__ import annotations

import importlib
import sys
from typing import Any, cast

import pytest

from kedi.core import KediRuntime
from kedi.lang.ast import (
    CallSegment,
    InputReference,
    InputSegment,
    LiteralArg,
    OutputSegment,
    PythonExprNativeSegment,
    PythonExprSegment,
    TemplateLine,
    TextSegment,
)
from kedi.optimization import get_optimizer
from kedi.optimization.mock import MockOptimizer
from kedi.optimization.utils import (
    extract_input_fields,
    extract_output_fields,
    normalize_optimized_prompt_prefix,
    render_instruction_lines,
    template_line_to_string,
)
from tests.optimization_helpers import make_optimize_span


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
        make_optimize_span("plain", [TextSegment(text="Plain prompt")]),
        make_optimize_span("think", [TextSegment(text="Think step by step already")]),
        make_optimize_span(
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
            InputReference(name="photo"),
            OutputSegment(name="raw"),
            PythonExprSegment(code="x + 1"),
            cast(Any, object()),
            PythonExprNativeSegment(),
        ]
    )
    assert template_line_to_string(line) == "A <<photo>>[raw]<`x + 1`><expr>"

    empty_span = make_optimize_span("empty", [])
    assert normalize_optimized_prompt_prefix(empty_span, None) == ""
    assert normalize_optimized_prompt_prefix(empty_span, "custom") == "custom"

    base_span = make_optimize_span("base", [TextSegment(text="Base")])
    assert normalize_optimized_prompt_prefix(base_span, "prefixBase") == "prefixBase"

    rt = KediRuntime()

    @rt.procedure
    def double(value: int) -> int:
        return value * 2

    rendered_span = make_optimize_span(
        "render",
        [
            TextSegment(text="Name "),
            InputSegment(name="name"),
            TextSegment(text=" ref "),
            InputReference(name="name"),
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
        "Name Ada ref <<name>> missing <missing> call 4 badcall <missing()> out [answer] typed [count: int] "
        "py 3 bad <`1 / 0`>"
    )

    repeated = make_optimize_span(
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
