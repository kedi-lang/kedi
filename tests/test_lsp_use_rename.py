"""Procedure renames preserve native use references and exposed tool aliases."""

from __future__ import annotations

import pytest
from lsprotocol import types as lsp

from kedi.lang import compile_program, parse_program
from kedi.lsp.features.navigation import compute_references, compute_rename
from tests.mock_adapter import MockAdapter


@pytest.mark.parametrize("alias", ["public_lookup", "lookup"])
@pytest.mark.parametrize(
    "use_block",
    [
        "> use: lookup\n",
        "> use:\n  lookup\n",
        "> use: untouched\n> use: lookup\n",
        "> if: `True`:\n  > use: lookup\n",
        "@configure():\n  > use:\n    lookup\n  = ok\n",
    ],
)
def test_rename_updates_use_references_but_not_tool_alias(alias, use_block):
    source = (
        f"@lookup() -> str:\n  > tool:\n    name: {alias}\n  = ok\n"
        "@untouched() -> str:\n  = other\n" + use_block + "= <lookup()>\n"
    )
    uri = "file:///tmp/rename.kedi"
    position = lsp.Position(0, 2)
    refs = compute_references(source, uri, "/tmp/rename.kedi", position, False)
    assert len(refs) == 2
    edit = compute_rename(source, uri, "/tmp/rename.kedi", position, "renamed")
    assert edit is not None and edit.changes is not None
    lines = source.splitlines(keepends=True)
    for change in sorted(
        edit.changes[uri],
        key=lambda item: (item.range.start.line, item.range.start.character),
        reverse=True,
    ):
        start, end = change.range.start, change.range.end
        assert start.line == end.line
        line = lines[start.line]
        lines[start.line] = line[: start.character] + change.new_text + line[end.character :]
    renamed = "".join(lines)
    assert f"name: {alias}\n" in renamed
    assert renamed.count("renamed") == 3
    assert compile_program(parse_program(renamed), adapter=MockAdapter()).run_main() == "ok"


def test_rename_use_references_respects_shadowed_procedure():
    source = (
        "@lookup() -> str:\n  = outer\n"
        "@configure() -> str:\n"
        "  @lookup() -> str:\n    = inner\n"
        "  > use: lookup\n"
        "  = <lookup()>\n"
        "> use: lookup\n"
        "= <lookup()>\n"
    )
    uri = "file:///tmp/rename.kedi"
    edit = compute_rename(source, uri, "/tmp/rename.kedi", lsp.Position(0, 2), "renamed")
    assert edit is not None and edit.changes is not None
    assert {change.range.start.line for change in edit.changes[uri]} == {0, 7, 8}


def test_use_references_do_not_invent_locations_for_manual_ast():
    from kedi.lang.ast import Program, Span, UseDirectiveStmt
    from kedi.lsp.features.navigation import _find_all_references
    from kedi.lsp.features.symbols import SymbolRef

    program = Program(toplevel_body=[UseDirectiveStmt(target="lookup")])
    symbol = SymbolRef(
        kind="procedure",
        name="lookup",
        declaration_span=None,
        scope_proc=None,
        cursor_span=Span(0, 6, 1, 1, 1, 7),
    )
    assert _find_all_references(program, symbol, include_declaration=False) == []
