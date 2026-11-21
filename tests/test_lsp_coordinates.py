"""Regression tests for shared LSP coordinate conversion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lsprotocol import types as lsp

from kedi.lang.ast import Span
from kedi.lsp import coordinates
from kedi.lsp.completion_source import parse_completion_source
from kedi.lsp.python_virtual.document import compute_python_virtual_document
from tests.lsp_helpers import (
    definition_at,
    diagnostics_for,
    inlay_hints_for_lines,
    references_at,
    rename_at,
    signature_help_at,
)


def test_coordinate_conversions_handle_utf16_and_tree_sitter_edges() -> None:
    assert coordinates.utf16_to_index("🐈a", 0) == 0
    assert coordinates.utf16_to_index("🐈a", 2) == 1
    assert coordinates.utf16_to_index("🐈a", 3) == 2
    assert coordinates.byte_column_to_utf16("🐈a", 4) == 2

    source = coordinates.CoordinateSource("🐈\nİstanbul", None)
    assert source.byte_offset_to_position(-1) == lsp.Position(line=0, character=0)
    assert source.byte_offset_to_position(999) == lsp.Position(line=1, character=8)
    assert source.tree_sitter_point_to_position(-1, 0) == lsp.Position(line=0, character=0)
    assert source.tree_sitter_point_to_position(999, 9) == lsp.Position(line=1, character=8)


def test_coordinate_decorator_normalizes_source_path_inputs() -> None:
    @coordinates.with_source_coordinates(1)
    def position(source: str, source_path: Any = None) -> lsp.Range:
        return coordinates.span_to_range(
            Span(
                start_byte=0,
                end_byte=len(source.encode("utf-8")),
                start_line=1,
                start_col=1,
                end_line=1,
                end_col=len(source) + 1,
                source_path=None,
            )
        )

    assert position("🐈", 123).end.character == 2
    assert position(source="🐈", source_path=123).end.character == 2
    assert position(source="🐈", source_path="/tmp/source.kedi").end.character == 2
    assert position("🐈", "/tmp/source.kedi").end.character == 2


def test_external_coordinate_source_and_fallback_paths(tmp_path: Path, monkeypatch: Any) -> None:
    external = tmp_path / "external.kedi"
    external.write_text("İstanbul", encoding="utf-8")
    span = Span(
        start_byte=0,
        end_byte=len("İstanbul".encode("utf-8")),
        start_line=1,
        start_col=1,
        end_line=1,
        end_col=9,
        source_path=str(external),
    )

    with coordinates.source_coordinates("main", str(tmp_path / "main.kedi")):
        assert coordinates.span_to_range(span).end.character == 8

    with coordinates.source_coordinates("main", None):
        assert coordinates.span_to_range(span).end.character == 8

    missing = Span(0, 1, 2, 3, 2, 4, source_path=str(tmp_path / "missing.kedi"))
    with coordinates.source_coordinates("main", str(tmp_path / "main.kedi")):
        assert coordinates.span_to_range(missing).start == lsp.Position(line=1, character=2)

    coordinates._clear_read_source_cache()
    monkeypatch.setattr(
        Path, "read_text", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError())
    )
    assert coordinates._read_source(str(external)) is None

    def fail_resolve(_path: Path) -> Path:
        raise OSError("resolve failed")

    monkeypatch.setattr(Path, "resolve", fail_resolve)
    same_path_span = Span(0, 1, 1, 1, 1, 2, source_path="same.kedi")
    with coordinates.source_coordinates("main", "same.kedi"):
        assert coordinates._span_source(same_path_span) is not None
    different_path_span = Span(0, 1, 1, 1, 1, 2, source_path="other.kedi")
    with coordinates.source_coordinates("main", "same.kedi"):
        assert coordinates._span_source(different_path_span) is None


def test_read_source_sees_edits_between_reads(tmp_path: Path) -> None:
    """#214 6.16: the file cache must not serve stale text after an edit."""
    target = tmp_path / "source.kedi"
    target.write_text("line one\n", encoding="utf-8")
    assert coordinates._read_source(str(target)) == "line one\n"
    target.write_text("line one CHANGED\nline two\n", encoding="utf-8")
    assert coordinates._read_source(str(target)) == "line one CHANGED\nline two\n"


def test_coordinate_source_caches_lines_and_line_starts() -> None:
    """#214 6.16: lines/_line_starts must not re-encode the source per call."""
    source = coordinates.CoordinateSource("a\nb", None)
    assert source.lines is source.lines
    assert source._line_starts is source._line_starts


# --- #214 LSP coordinate and coverage fixes ------------------------------------------------

_URI = "file:///tmp/audit.kedi"
_PATH = "/tmp/audit.kedi"


def _starts(locations: list[lsp.Location]) -> list[tuple[int, int]]:
    return [(item.range.start.line, item.range.start.character) for item in locations]


def test_references_after_non_ascii_text_resolve() -> None:
    # 6.6: candidates were resolved at their byte column, which drifts after `ş`.
    source = "@f(x: str) -> str:\n  = şş <x> şş <x>\n= <f(`'a'`)>\n"
    assert _starts(references_at(source, _URI, _PATH, "x", True)) == [(0, 3), (1, 8), (1, 15)]


def test_tab_indented_python_fence_keeps_whole_lines() -> None:
    # 6.8: the dedent counted a tab as four characters and cut into the line.
    source = "@f() -> int:\n  = ```\n\tvalue = 1\n\treturn value\n  ```\n= `f()`\n"
    virtual = compute_python_virtual_document(source, _PATH, focus_line=2)
    assert "    return value" in virtual["text"].splitlines()


def test_signature_help_and_inlay_hints_inside_control_blocks() -> None:
    # 6.12: calls inside `> if:` bodies were skipped.
    source = "@add(a: int, b: int) -> int:\n  = `a + b`\n> if: `True`:\n  [r: int] = <add(1, 2)>\n= `1`\n"
    help_ = signature_help_at(source, _PATH, "1, 2", offset=3)
    assert help_ is not None and help_.active_parameter == 1
    assert [item.label for item in inlay_hints_for_lines(source, _PATH, 0, 5)] == ["a: ", "b: "]


def test_signature_help_label_offsets_count_utf16_units() -> None:
    # 6.12: offsets index the label in UTF-16 code units.
    source = "@add(a: Annotated[str, '𝄞'], b: int) -> str:\n  = <a>\n= <add(x, 1)>\n"
    help_ = signature_help_at(source, _PATH, "x, 1", offset=3)
    assert help_ is not None
    (signature,) = help_.signatures
    start, end = signature.parameters[1].label
    units = signature.label.encode("utf-16-le")
    assert units[2 * start : 2 * end].decode("utf-16-le").startswith("b")


def test_parameter_declarations_skip_names_inside_types_and_defaults() -> None:
    # 6.14: the next parameter was searched from the end of the previous name.
    nested = (
        "@outer() -> str:\n  @f(a: str = `'(b)'`, b: str = `'c'`) -> str:\n    = <a> <b>\n"
        "  = <f()>\n= <outer()>\n"
    )
    found = definition_at(nested, _URI, _PATH, "<b>", offset=1)
    assert found is not None and (found.range.start.line, found.range.start.character) == (1, 23)
    typed = "@f(note: Annotated[str, 'a b'], b: str) -> str:\n  = <b>\n= <f(x, y)>\n"
    found = definition_at(typed, _URI, _PATH, "<b>", offset=1)
    assert found is not None and found.range.start.character == 32


def test_references_and_rename_cover_export_lists() -> None:
    # 6.17: `> export:` entries were neither references nor renamed.
    source = (
        "> export:\n  helper\n  Item\n@helper() -> str:\n  = x\n~Item(a: int)\n"
        "@main() -> str:\n  > use: helper\n  [r] << Call <helper()>\n  = <r>\n= <main()>\n"
    )
    assert (1, 2) in _starts(references_at(source, _URI, _PATH, "helper", True, occurrence=2))
    assert (2, 2) in _starts(references_at(source, _URI, _PATH, "Item", True, occurrence=2))
    edit = rename_at(source, _URI, _PATH, "helper", "assist", occurrence=2)
    assert edit is not None and edit.changes is not None
    assert (1, 2) in {(e.range.start.line, e.range.start.character) for e in edit.changes[_URI]}


def test_completion_recovery_keeps_later_byte_offsets() -> None:
    # 6.22: a placeholder longer than the unfinished line shifted every later span.
    source = "@f() -> str:\n  = x\n  >\n@g() -> str:\n  = y\n= <f()>\n"
    program = parse_completion_source(source, None, 2)
    (later,) = [proc for proc in program.procedures if proc.name == "g"]
    assert later.loc is not None and later.loc.start_byte == source.encode().index(b"@g")
    only_line = "@f() -> str:\n  >\n@g() -> str:\n  = y\n= <f()>\n"
    assert {proc.name for proc in parse_completion_source(only_line, None, 1).procedures} == {
        "f",
        "g",
    }


def test_diagnostics_check_procedures_defined_in_top_level_blocks() -> None:
    # 6.23: procedures inside top-level blocks were not visited.
    source = "> if: `True`:\n  @g() -> int:\n    [y] := `2`\n    = `1`\n= `1`\n"
    assert [d.code for d in diagnostics_for(source, _PATH)] == ["K-ASSIGN-UNKNOWN"]


def test_export_references_only_for_listed_top_level_symbols() -> None:
    # 6.17: an export of other names adds nothing; a nested namesake is not the exported one.
    listed = (
        "> export:\n  other\n> export:\n  helper\n@other() -> str:\n  = o\n"
        "@helper() -> str:\n  = x\n= <helper()>\n"
    )
    assert (3, 2) in _starts(references_at(listed, _URI, _PATH, "helper", True, occurrence=2))
    nested = (
        "> export:\n  helper\n@helper() -> str:\n  = x\n@outer() -> str:\n"
        "  @helper() -> str:\n    = inner\n  = <helper()>\n= <outer()> <helper()>\n"
    )
    assert (1, 2) not in _starts(references_at(nested, _URI, _PATH, "helper", True, occurrence=3))
    assert (1, 2) in _starts(references_at(nested, _URI, _PATH, "helper", True, occurrence=2))
    local_type = (
        "> export:\n  Item\n~Item(a: int)\n@f() -> int:\n  ~Item(b: int)\n"
        "  [i: Item] = `Item(b=1)`\n  = `i.b`\n= `f()`\n"
    )
    assert (1, 2) not in _starts(references_at(local_type, _URI, _PATH, "Item", True, occurrence=3))
    assert (1, 2) in _starts(references_at(local_type, _URI, _PATH, "Item", True, occurrence=2))


def test_parameter_split_handles_unclosed_headers_and_unknown_names() -> None:
    # 6.14: a header without its closing parenthesis still splits, and a segment that is
    # not a parameter is skipped.
    from kedi.lang.ast import ProcedureDef
    from kedi.lsp.features import scope

    assert list(scope._parameter_starts("@f(a, [b, c], `d,e`, 'f,g'", 3)) == [3, 5, 13, 20]
    source = "@f(extra, b) -> str:\n  = <b>\n"
    loc = Span(0, len(source), 1, 1, 2, 8)
    proc = ProcedureDef(name="f", params=["b"], loc=loc)
    with coordinates.source_coordinates(source, None):
        assert [name for name, _span in scope._param_decls(proc)] == ["b"]
