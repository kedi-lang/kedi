"""Kedi #214 (6.10, 6.15): unit coverage for imported-file span helpers.

Line text and line-start bytes for a span that lives in another file
come from that file: from the program's source map when it holds the
file, otherwise from disk. A missing file or a line past the end yields
``None`` instead of silently reading the current document.
"""

from __future__ import annotations

from kedi.lang.ast import Program, Span
from kedi.lsp.features.symbols import (
    _directive_name_span,
    _external_source_text,
    _line_start_byte,
    _program_line_text,
    _same_path,
)
from kedi.lsp.parsing import parse_program
from kedi.source_loading import merge_source_documents, single_file_source_map

HELPER_TEXT = "> profile: shared:\n    > model: opus\n\n> export:\n  shared\n"
MAIN_TEXT = "> import: helper\n\n> use: shared\n\n@main():\n  = ok\n"


def _merged_program(tmp_path):
    helper = tmp_path / "helper.kedi"
    helper.write_text(HELPER_TEXT, encoding="utf-8")
    main = tmp_path / "main.kedi"
    main.write_text(MAIN_TEXT, encoding="utf-8")
    source_map = merge_source_documents(
        [(main, MAIN_TEXT), (helper, HELPER_TEXT)],
    ).source_map
    prog = parse_program(MAIN_TEXT, source_path=str(main), source_map=source_map)
    return helper, main, prog


def test_same_path_handles_none_and_equal_paths() -> None:
    assert _same_path(None, "/tmp/x.kedi") is False
    assert _same_path("/tmp/x.kedi", None) is False
    assert _same_path("/tmp/x.kedi", "/tmp/x.kedi") is True
    assert _same_path("/tmp/x.kedi", "/tmp/y.kedi") is False


def test_external_text_prefers_the_source_map_document(tmp_path) -> None:
    helper, _main, prog = _merged_program(tmp_path)
    assert _external_source_text(prog, str(helper)) == HELPER_TEXT
    assert _program_line_text(prog, 1, str(helper)) == "> profile: shared:"
    assert _program_line_text(prog, 2, str(helper)) == "    > model: opus"


def test_line_start_byte_uses_the_source_map_offset(tmp_path) -> None:
    helper, _main, prog = _merged_program(tmp_path)
    assert _line_start_byte(prog, 1, str(helper)) == 0
    assert _line_start_byte(prog, 2, str(helper)) == len(HELPER_TEXT.splitlines()[0]) + 1


def test_external_text_falls_back_to_disk_without_a_source_map(tmp_path) -> None:
    helper = tmp_path / "helper.kedi"
    helper.write_text(HELPER_TEXT, encoding="utf-8")
    assert _external_source_text(Program(source_text=MAIN_TEXT), str(helper)) == HELPER_TEXT


def test_external_text_returns_none_for_a_missing_file(tmp_path) -> None:
    main = tmp_path / "main.kedi"
    main.write_text(MAIN_TEXT, encoding="utf-8")
    prog = parse_program(MAIN_TEXT, source_path=str(main))
    missing = str(tmp_path / "no-such-214.kedi")
    assert _external_source_text(prog, missing) is None
    assert _program_line_text(prog, 1, missing) is None
    assert _line_start_byte(prog, 1, missing) is None


def test_external_line_past_the_end_returns_none(tmp_path) -> None:
    helper, _main, prog = _merged_program(tmp_path)
    assert _program_line_text(prog, 99, str(helper)) is None
    assert _line_start_byte(prog, 99, str(helper)) is None


def test_program_without_a_path_uses_its_source_map_as_its_own_document(tmp_path) -> None:
    main = tmp_path / "main.kedi"
    prog = Program(
        source_text=MAIN_TEXT,
        source_map=single_file_source_map(str(main), MAIN_TEXT),
    )
    assert prog.loc is None
    assert _program_line_text(prog, 1, str(main)) == "> import: helper"
    assert _line_start_byte(prog, 1, str(main)) == 0


def test_line_start_byte_reads_disk_without_a_source_map(tmp_path) -> None:
    helper = tmp_path / "helper.kedi"
    helper.write_text(HELPER_TEXT, encoding="utf-8")
    main = tmp_path / "main.kedi"
    prog = Program(
        source_text=MAIN_TEXT,
        loc=Span(0, 1, 1, 1, 1, 2, source_path=str(main)),
    )
    assert _line_start_byte(prog, 1, str(helper)) == 0
    assert _line_start_byte(prog, 2, str(helper)) == len(HELPER_TEXT.splitlines()[0]) + 1


def test_exported_name_at_position_skips_a_name_the_cursor_is_not_on() -> None:
    from lsprotocol import types as lsp

    from kedi.lsp.features.scope import _exported_name_at_position

    src = "@foo():\n  = ok\n@bar():\n  = ok\n> export:\n  foo\n> export:\n  bar\n"
    prog = parse_program(src, source_path="/tmp/x.kedi")
    # On the `foo` export line but before the name: no hit, and the scan
    # continues past the first export instead of stopping there.
    assert _exported_name_at_position(prog, lsp.Position(line=5, character=0)) is None
    hit = _exported_name_at_position(prog, lsp.Position(line=5, character=3))
    assert hit is not None and hit[0] == "foo"


def test_directive_name_span_none_cases(tmp_path) -> None:
    _helper, main, prog = _merged_program(tmp_path)
    assert _directive_name_span(prog, None, "shared") is None
    out_of_range = Span(
        start_byte=0,
        end_byte=1,
        start_line=99,
        start_col=1,
        end_line=99,
        end_col=2,
        source_path=str(main),
    )
    assert _directive_name_span(prog, out_of_range, "shared") is None
