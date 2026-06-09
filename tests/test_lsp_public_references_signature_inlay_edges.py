"""LSP references, signature help, and inlay hint edge cases."""

from __future__ import annotations

from typing import Any, cast

from kedi.lang.ast import CallArg, CallSegment, Stmt
from kedi.lsp import features
from tests.lsp_helpers import (
    inlay_hints_for_lines,
    line_range,
    position_at_line,
    references_at,
    signature_help_at,
)


def test_references_returns_empty_without_symbol() -> None:
    refs = references_at("= ok\n", "file:///tmp/x.kedi", "/tmp/x.kedi", "ok", True)
    assert refs == []


def test_references_on_type_annotations() -> None:
    type_src = "~Person(name)\n@p(x: Person) -> list[Person]:\n  [y: Person] = <x>\n  >> Value [z: Person]\n  = <x>\n"
    refs = references_at(type_src, "file:///tmp/x.kedi", "/tmp/x.kedi", "Person", True)
    assert len(refs) >= 4
    local_refs = references_at(
        "@p(x):\n  >> Value [answer]\n  = <answer>\n",
        "file:///tmp/x.kedi",
        "/tmp/x.kedi",
        "answer",
        True,
    )
    assert local_refs


def test_signature_help_nested_and_missing() -> None:
    nested_sig_src = "@inner(x: int) -> int:\n  = `x`\n\n@outer():\n  = <inner(<inner(1)>)>\n"
    sig = signature_help_at(nested_sig_src, "/tmp/x.kedi", "inner", occurrence=2)
    assert sig is not None and sig.active_parameter == 0
    top_sig_src = "@inner(x: int) -> int:\n  = `x`\n\n[x] = <inner(1)>\n"
    top_sig = signature_help_at(top_sig_src, "/tmp/x.kedi", "inner", occurrence=2)
    assert top_sig is not None and top_sig.signatures
    no_return_sig_src = "@plain(x):\n  = <x>\n\n= <plain(ok)>\n"
    no_return_sig = signature_help_at(no_return_sig_src, "/tmp/x.kedi", "plain", occurrence=2)
    assert no_return_sig is not None and "->" not in no_return_sig.signatures[0].label
    assert signature_help_at("= <missing(1)>\n", "/tmp/x.kedi", "missing") is None


def test_inlay_hints_and_helper_branches() -> None:
    hints_src = "@inner(x, y):\n  = ok\n\n= <inner(<inner(1, 2)>, 3)>\n"
    hints = inlay_hints_for_lines(hints_src, "/tmp/x.kedi", 0, 10)
    assert len(hints) >= 3
    extra_hints = inlay_hints_for_lines("@one(x):\n  = ok\n\n= <one(1, 2)>\n", "/tmp/x.kedi", 0, 10)
    assert extra_hints
    assert inlay_hints_for_lines("@one(x):\n  = ok\n\n= <one(1)>\n", "/tmp/x.kedi", 10, 11) == []
    assert inlay_hints_for_lines(">> Value [answer]\n", "/tmp/x.kedi", 10, 11) == []
    assert inlay_hints_for_lines(">> Value [answer: int]\n= <answer>\n", "/tmp/x.kedi", 0, 3) == []

    assert (
        features._in_range(
            features.Span(0, 1, 1, 1, 1, 2),
            line_range(2, 3),
        )
        is False
    )
    assert (
        features._in_range(
            features.Span(0, 1, 5, 1, 5, 2),
            line_range(0, 3),
        )
        is False
    )
    assert (
        features._active_param_index(
            CallSegment(name="p", args=cast(Any, None)),
            position_at_line(0, 0),
        )
        == 0
    )
    assert features._innermost_call_at(Stmt(), position_at_line(0, 0)) is None
    nested_arg = CallArg(loc=features.Span(0, 10, 1, 1, 1, 20))
    setattr(
        nested_arg,
        "__call_segment__",
        CallSegment(name="deep", args=[], loc=features.Span(0, 10, 1, 5, 1, 12)),
    )
    innermost = features._innermost_call_in_segment(
        CallSegment(name="outer", args=[nested_arg], loc=features.Span(0, 20, 1, 1, 1, 21)),
        position_at_line(0, 6),
    )
    assert innermost is not None and innermost.name == "deep"

    rendered_arg = CallArg(loc=features.Span(0, 10, 1, 1, 1, 20))
    setattr(
        rendered_arg,
        "__render_segments__",
        [CallSegment(name="rendered", args=[], loc=features.Span(0, 10, 1, 5, 1, 12))],
    )
    rendered_innermost = features._innermost_call_in_segment(
        CallSegment(name="outer", args=[rendered_arg], loc=features.Span(0, 20, 1, 1, 1, 21)),
        position_at_line(0, 6),
    )
    assert rendered_innermost is not None and rendered_innermost.name == "rendered"
