"""Editor calls respect lexical scope and export edits exclude comments."""

from __future__ import annotations

import pytest

from kedi.lang import parse_program
from kedi.lang.ast import CallSegment
from kedi.lsp.features import symbols
from kedi.lsp.features.signatures import _call_target
from tests.lsp_helpers import definition_at, inlay_hints_for_lines, rename_at, signature_help_at

SOURCE = """@f(a: str) -> str:
  = <a>

@outer() -> str:
  @f(b: str, c: str) -> str:
    = <b> <c>
  > if: `True`:
    = <f(one, two)>

> if: `True`:
  > show: <f(hello)>
"""


@pytest.mark.parametrize("occurrence,expected", [(3, "b: str, c: str"), (4, "a: str")])
def test_signature_help_resolves_the_procedure_at_the_call_site(occurrence, expected):
    signature = signature_help_at(SOURCE, "/tmp/scopes.kedi", "f(", occurrence=occurrence)
    assert signature is not None
    assert signature.signatures[0].label == f"@f({expected}) -> str"


def test_inlay_hints_resolve_the_procedure_at_the_call_site():
    hints = inlay_hints_for_lines(SOURCE, "/tmp/scopes.kedi", 0, 20)
    outer_call_line = SOURCE.splitlines().index("  > show: <f(hello)>")
    assert [hint.label for hint in hints if hint.position.line == outer_call_line] == ["a: "]
    inner_call_line = SOURCE.splitlines().index("    = <f(one, two)>")
    assert [hint.label for hint in hints if hint.position.line == inner_call_line] == [
        "b: ",
        "c: ",
    ]


def test_profile_export_rename_does_not_edit_matching_comments():
    source = (
        "> profile: shared:\n"
        "  > instructions: Review the supplied facts.\n"
        "\n"
        "> export:\n"
        "  # shared is public\n"
        "  shared # shared is exported\n"
    )
    uri = "file:///tmp/exports.kedi"
    edit = rename_at(source, uri, "/tmp/exports.kedi", "shared", "reviewer")
    assert edit is not None and edit.changes is not None
    assert sorted(
        (entry.range.start.line, entry.range.start.character) for entry in edit.changes[uri]
    ) == [
        (0, 11),
        (5, 2),
    ]


def test_signature_target_without_a_source_location_is_unavailable():
    program = parse_program("@f():\n  = ready\n")
    assert _call_target(program, CallSegment(name="f", args=[])) is None


def test_imported_values_do_not_offer_a_kedi_procedure_signature(tmp_path):
    (tmp_path / "helper.kedi").write_text(
        "[callback: Any] = `lambda value: value`\n> export:\n  callback\n"
    )
    source = "> import: helper\n= <callback(hello)>\n"
    path = str(tmp_path / "main.kedi")
    assert signature_help_at(source, path, "callback(") is None
    assert inlay_hints_for_lines(source, path, 0, 3) == []


def test_imported_profile_declaration_does_not_shadow_local_document_positions(tmp_path):
    (tmp_path / "helper.kedi").write_text(
        "# helper\n\n\n> profile: remote:\n  > instructions: Review.\n> export:\n  remote\n"
    )
    source = "> import: helper\n\n\n[local_message] = hello\n= <local_message>\n"
    main = tmp_path / "main.kedi"
    main.write_text(source)
    location = definition_at(source, main.as_uri(), str(main), "local_message", offset=10)
    assert location is not None
    assert location.uri == main.as_uri()
    assert (location.range.start.line, location.range.start.character) == (3, 1)


def test_export_scan_tolerates_an_unavailable_identifier_span(monkeypatch):
    program = parse_program("> profile: review:\n  > instructions: Review.\n> export:\n  review\n")
    monkeypatch.setattr(symbols, "_name_token_span_on_line", lambda *args, **kwargs: None)
    assert symbols._export_name_spans(program, "review") == []
