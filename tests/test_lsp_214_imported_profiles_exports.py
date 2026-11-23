"""Kedi #214 (6.10, 6.17): imported profiles and ``> export:`` references.

An imported declaration's spans belong to the imported file: line text
and byte offsets must never be read from the current document at a
coincident line number. ``> export:`` entries are genuine references to
the exported symbol.
"""

from __future__ import annotations

import pytest

from kedi.lsp.features.symbols import _directive_name_span
from kedi.lsp.module_surface import imported_surface_for_program
from kedi.lsp.parsing import parse_program
from tests.lsp_helpers import definition_at, references_at

URI = "file:///tmp/x.kedi"


def _imported_program(tmp_path, helper_text: str, main_text: str):
    helper = tmp_path / "helper.kedi"
    helper.write_text(helper_text, encoding="utf-8")
    main = tmp_path / "main.kedi"
    main.write_text(main_text, encoding="utf-8")
    return helper, main


def test_imported_profile_name_span_reads_the_imported_file(tmp_path) -> None:
    helper, main = _imported_program(
        tmp_path,
        "> profile: shared:\n    > model: opus\n\n> export:\n  shared\n",
        "> import: helper\n\n> use: shared\n\n@main():\n  = ok\n",
    )
    src = main.read_text(encoding="utf-8")
    prog = parse_program(src, source_path=str(main))
    profile = imported_surface_for_program(prog).profiles["shared"]
    assert profile.loc is not None
    span = _directive_name_span(prog, profile.loc, "shared")
    assert span is not None
    assert helper.read_bytes()[span.start_byte : span.end_byte] == b"shared"
    assert span.source_path == str(helper)


def test_definition_of_imported_profile_use_points_into_imported_file(tmp_path) -> None:
    helper, main = _imported_program(
        tmp_path,
        "> profile: shared:\n    > model: opus\n\n> export:\n  shared\n",
        "> import: helper\n\n> use: shared\n\n@main():\n  = ok\n",
    )
    src = main.read_text(encoding="utf-8")
    loc = definition_at(src, main.as_uri(), str(main), "shared", occurrence=1)
    assert loc is not None
    assert "helper.kedi" in loc.uri


def test_references_of_imported_profile_include_use_and_declaration(tmp_path) -> None:
    helper, main = _imported_program(
        tmp_path,
        "> profile: shared:\n    > model: opus\n\n> export:\n  shared\n",
        "> import: helper\n\n> use: shared\n\n@main():\n  = ok\n",
    )
    src = main.read_text(encoding="utf-8")
    refs = references_at(src, main.as_uri(), str(main), "shared", True, occurrence=1)
    uris = {ref.uri for ref in refs}
    assert main.as_uri() in uris
    assert helper.as_uri() in uris


def test_references_of_imported_profile_do_not_pick_up_local_text(tmp_path) -> None:
    # The imported profile is declared on line 1; the local document's
    # line 1 is the import line, which must not be mistaken for it.
    helper, main = _imported_program(
        tmp_path,
        "> profile: shared:\n    > model: opus\n\n> export:\n  shared\n",
        "> import: helper\n\n> use: shared\n\n@main():\n  = ok\n",
    )
    src = main.read_text(encoding="utf-8")
    refs = references_at(src, main.as_uri(), str(main), "shared", False, occurrence=1)
    assert len(refs) == 1
    assert refs[0].uri == main.as_uri()
    assert refs[0].range.start.line == 2


@pytest.mark.parametrize(
    "source,export_line",
    [
        ("@foo() -> str:\n  = ok\n\n> export:\n  foo\n", 4),
        ("~Thing(name: str)\n\n> export:\n  Thing\n", 3),
    ],
)
def test_export_entry_is_a_reference(source: str, export_line: int) -> None:
    name = "foo" if source.startswith("@foo") else "Thing"
    refs = references_at(source, URI, "/tmp/x.kedi", name, True, occurrence=2)
    assert sorted({ref.range.start.line for ref in refs}) == [0, export_line]


def test_export_entry_is_a_reference_for_top_level_value() -> None:
    src = "[answer] = `1`\n\n> export:\n  answer\n"
    refs = references_at(src, URI, "/tmp/x.kedi", "answer", True, occurrence=2)
    assert sorted({ref.range.start.line for ref in refs}) == [0, 3]


def test_export_all_adds_no_name_reference() -> None:
    src = "@foo() -> str:\n  = ok\n\n> export: *\n"
    refs = references_at(src, URI, "/tmp/x.kedi", "foo", True, occurrence=1)
    assert len(refs) == 1
