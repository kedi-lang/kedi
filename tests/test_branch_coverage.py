"""Branch-coverage tests for previously hard-to-hit control-flow paths."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from lsprotocol import types as lsp

from kedi.lsp import features, module_surface
from kedi.resolved_types import (
    as_resolved_kedi_type,
    is_resolved_kedi_type,
    resolved_type_name,
    runtime_isinstance,
)
from tests.lsp_edges_fixtures import make_edge_span
from tests.lsp_helpers import position_of


def test_resolved_type_helpers_cover_typing_aliases() -> None:
    assert is_resolved_kedi_type(str)
    assert is_resolved_kedi_type(list[str])
    assert not is_resolved_kedi_type(1)
    assert as_resolved_kedi_type(1) is None
    assert as_resolved_kedi_type(str) is str
    assert resolved_type_name(str) == "str"
    assert resolved_type_name(list[str]) == "list"
    assert runtime_isinstance("x", str)
    assert runtime_isinstance("x", list[str]) is True


def test_type_registry_skips_private_and_missing_names(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = SimpleNamespace(
        __all__=("_private", "Visible", "Missing"),
        Visible=object(),
    )
    monkeypatch.setattr(features, "_typing", fake_module)
    monkeypatch.setattr(features, "_typing_extensions", None)
    registry = features._build_lsp_type_registry()
    assert "_private" not in registry
    assert "Missing" not in registry
    assert registry["Visible"] == "typing"


def test_exported_name_skips_position_outside_name_span(tmp_path) -> None:
    from kedi.lang.parser import parse_program

    source_path = str(tmp_path / "m.kedi")
    export_prog = parse_program("> export:\n  shared\n", source_path=source_path)
    assert (
        features._exported_name_at_position(export_prog, lsp.Position(line=1, character=1)) is None
    )
    assert (
        features._exported_name_at_position(export_prog, lsp.Position(line=1, character=3))
        is not None
    )


def test_exported_name_misses_narrow_name_span(tmp_path) -> None:
    from kedi.lang.ast import ModuleExport, Program, Span

    export_prog = Program(
        source_text="> export:\n  *\n  Name\n",
        exports=[ModuleExport(["Name"], loc=Span(0, 32, 1, 1, 4, 7))],
    )
    assert (
        features._exported_name_at_position(export_prog, lsp.Position(line=1, character=2)) is None
    )


def test_resolve_exported_imported_value(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from kedi.lang.ast import Program

    export_span = make_edge_span(tmp_path, line=2, start_col=3, end_col=9)
    export_prog = Program(source_text="> export:\n  shared\n")
    monkeypatch.setattr(
        features,
        "_exported_name_at_position",
        lambda prog, pos: ("shared", export_span),
    )
    monkeypatch.setattr(
        features,
        "imported_surface_for_program",
        lambda prog: module_surface.ImportedSurface(
            values={"shared": module_surface.ImportedValueInfo("shared", export_span)}
        ),
    )
    sym = features._resolve_symbol_at(export_prog, lsp.Position(line=1, character=3))
    assert sym is not None
    assert sym.kind == "imported_value"
    assert sym.name == "shared"


def test_resolve_output_segment_without_type_expr(tmp_path) -> None:
    from kedi.lang.ast import OutputSegment, ProcedureDef, ReturnStmt, TypeRef

    typed = OutputSegment(
        name="typed",
        type_expr=TypeRef(name="Person", loc=make_edge_span(tmp_path)),
        loc=make_edge_span(tmp_path, line=1, start_col=1, end_col=20),
    )
    plain = OutputSegment(
        name="plain",
        loc=make_edge_span(tmp_path, line=1, start_col=1, end_col=10),
    )
    proc = ProcedureDef(
        name="p",
        params=[],
        body=[ReturnStmt(segments=[typed, plain], loc=make_edge_span(tmp_path))],
        loc=make_edge_span(tmp_path),
    )
    pos = lsp.Position(line=0, character=8)
    from kedi.lang.ast import TypeDef

    assert (
        features._resolve_in_segment(
            plain,
            pos,
            proc,
            {"Person": TypeDef("Person", [], loc=make_edge_span(tmp_path))},
            {},
            {},
        )
        is None
    )


def test_prelude_defined_names_skips_non_name_assignment_targets() -> None:
    assert features._prelude_defined_names("obj.attr = 1\n") == set()


def test_prelude_defined_names_skips_unhandled_statement_kinds() -> None:
    assert features._prelude_defined_names("raise ValueError\n") == set()


def test_compute_definition_without_declaration_source_path(tmp_path, monkeypatch) -> None:
    from kedi.lang.ast import Span

    span = Span(0, 4, 1, 1, 1, 5, source_path=None)
    monkeypatch.setattr(
        features,
        "_resolve_symbol_at",
        lambda prog, pos: features.SymbolRef(
            kind="local",
            name="x",
            declaration_span=span,
            scope_proc=None,
            cursor_span=span,
        ),
    )
    loc = features.compute_definition(
        "= ok\n", "file:///tmp/x.kedi", "/tmp/x.kedi", lsp.Position(0, 0)
    )
    assert loc is not None
    assert loc.uri == "file:///tmp/x.kedi"


def test_compute_references_for_unhandled_symbol_kind(monkeypatch) -> None:
    span = features.Span(0, 4, 1, 1, 1, 5)
    monkeypatch.setattr(
        features,
        "_resolve_symbol_at",
        lambda prog, pos: features.SymbolRef(
            kind="module",
            name="profiles",
            declaration_span=span,
            scope_proc=None,
            cursor_span=span,
        ),
    )
    refs = features.compute_references(
        "> import: profiles\n= ok\n",
        "file:///tmp/x.kedi",
        "/tmp/x.kedi",
        lsp.Position(line=0, character=10),
        False,
    )
    assert refs == []


def test_signature_help_skips_decoy_procedure_without_call_at_cursor(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kedi.lang.ast import ProcedureDef, ReturnStmt, TextSegment
    from kedi.lang.parser import parse_program

    src = "@other(x):\n  = <x>\n\n= <other(1)>\n"
    prog = parse_program(src)
    pos = position_of(src, "other", occurrence=2)
    decoy = ProcedureDef(
        name="decoy",
        params=[],
        body=[ReturnStmt(segments=[TextSegment(text="plain", loc=make_edge_span(tmp_path))])],
        loc=make_edge_span(tmp_path, line=4, start_col=1, end_col=20),
    )
    monkeypatch.setattr(features, "_all_procedures", lambda _prog: [decoy, *prog.procedures])
    sig = features.compute_signature_help(src, str(tmp_path / "x.kedi"), pos)
    assert sig is not None
    assert "other" in sig.signatures[0].label


def test_signature_help_uses_toplevel_when_procedure_body_has_no_call(tmp_path) -> None:
    src = "@other(x):\n  = <x>\n\n@helper():\n  = plain text\n\n= <other(1)>\n"
    pos = position_of(src, "other", occurrence=2)
    sig = features.compute_signature_help(src, str(tmp_path / "x.kedi"), pos)
    assert sig is not None
    assert "other" in sig.signatures[0].label


def test_signature_help_skips_procedure_without_inner_call(tmp_path) -> None:
    from kedi.lang.ast import ProcedureDef, ReturnStmt, TextSegment

    prog_src = "@helper():\n  = ok\n\n= <helper()>\n"
    helper = ProcedureDef(
        name="helper",
        params=[],
        body=[ReturnStmt(segments=[TextSegment(text="ok", loc=make_edge_span(tmp_path))])],
        loc=make_edge_span(tmp_path),
    )
    from kedi.lang.parser import parse_program

    prog = parse_program(prog_src)
    prog.procedures.append(helper)
    pos = position_of(prog_src, "helper", occurrence=2)
    sig = features.compute_signature_help(prog_src, str(tmp_path / "x.kedi"), pos)
    assert sig is not None


def test_innermost_call_keeps_outer_when_nested_segment_misses(tmp_path) -> None:
    from kedi.lang.ast import CallArg, CallSegment

    inner = CallSegment(
        name="inner",
        args=[],
        loc=make_edge_span(tmp_path, line=2, start_col=1, end_col=10),
    )
    arg = CallArg(loc=make_edge_span(tmp_path, line=1, start_col=5, end_col=8))
    setattr(arg, "__call_segment__", inner)
    outer = CallSegment(
        name="outer",
        args=[arg],
        loc=make_edge_span(tmp_path, line=1, start_col=1, end_col=20),
    )
    pos = lsp.Position(line=0, character=5)
    found = features._innermost_call_in_segment(outer, pos)
    assert found is outer


def test_procedure_docstring_nodes_without_body(monkeypatch) -> None:
    class FakeNode:
        def __init__(self, type_: str, *, child_nodes: list[Any] | None = None) -> None:
            self.type = type_
            self.children = child_nodes or []
            self.is_named = True

        def child_by_field_name(self, name: str) -> Any:
            if name == "body":
                return None
            return None

    root = FakeNode("source_file", child_nodes=[FakeNode("procedure_def")])
    assert features._procedure_docstring_nodes(cast(Any, root)) == []


def test_semantic_tokens_without_docstring_capture(monkeypatch) -> None:
    src = (
        "@translate(text: str) -> str:\n  ###\n  Translate the provided text.\n  ###\n  = <text>\n"
    )
    monkeypatch.setattr(features, "_capture_to_token", lambda _name: None)
    tokens = features.compute_semantic_tokens_full(src, "/tmp/x.kedi")
    assert isinstance(tokens.data, list)


def test_python_code_lines_strips_trailing_empty_split_line() -> None:
    assert features._python_code_lines("  return 1\n") == ["  return 1"]
    assert features._python_code_lines("foo") == ["foo"]


def test_resolve_symbol_at_skips_unmatched_type_defs_before_match() -> None:
    from kedi.lang import parse_program
    from tests.lsp_helpers import position_of

    src = "~Alpha(x)\n~Beta(y)\n"
    prog = parse_program(src)
    sym = features._resolve_symbol_at(prog, position_of(src, "Beta"))
    assert sym is not None
    assert sym.name == "Beta"


def test_export_orphan_name_falls_through_to_procedure_walk(tmp_path, monkeypatch) -> None:
    from kedi.lang.ast import Program

    export_span = make_edge_span(tmp_path)
    export_prog = Program()
    monkeypatch.setattr(
        features,
        "_exported_name_at_position",
        lambda prog, pos: ("orphan", export_span),
    )
    monkeypatch.setattr(
        features,
        "imported_surface_for_program",
        lambda prog: module_surface.ImportedSurface(),
    )
    assert features._resolve_symbol_at(export_prog, lsp.Position(0, 0)) is None


def test_emit_token_spans_for_range_covers_multiline_edges() -> None:
    captured: list[tuple[int, int, int, int, int]] = []

    def capture(line: int, col: int, length: int, ttype: int, mods: int) -> None:
        captured.append((line, col, length, ttype, mods))

    features._emit_token_spans_for_range(
        src_lines=["only"],
        emit=capture,
        start_line=0,
        start_col=0,
        end_line=2,
        end_col=1,
        token_type=1,
        token_mods=0,
    )
    assert captured == [(0, 0, 4, 1, 0)]

    captured.clear()
    features._emit_token_spans_for_range(
        src_lines=["first", "second", "third"],
        emit=capture,
        start_line=0,
        start_col=0,
        end_line=2,
        end_col=3,
        token_type=2,
        token_mods=0,
    )
    assert captured == [(0, 0, 5, 2, 0), (1, 0, 6, 2, 0), (2, 0, 3, 2, 0)]

    captured.clear()
    features._emit_token_spans_for_range(
        src_lines=["only"],
        emit=capture,
        start_line=1,
        start_col=0,
        end_line=3,
        end_col=1,
        token_type=3,
        token_mods=0,
    )
    assert captured == []


def test_semantic_tokens_multiline_python_emit_edges() -> None:
    src = "= <`def f():\n    return 1\n`>\n"
    tokens = features.compute_semantic_tokens_full(src, "/tmp/x.kedi")
    assert len(tokens.data) % 5 == 0


def test_semantic_tokens_multiline_python_short_source() -> None:
    src = "= <`a\nb`>\n"
    tokens = features.compute_semantic_tokens_full(src, "/tmp/x.kedi")
    assert isinstance(tokens.data, list)
