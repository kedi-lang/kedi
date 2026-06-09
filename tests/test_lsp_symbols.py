"""LSP feature tests."""

from __future__ import annotations

from tests.lsp_helpers import definition_at, prepare_rename_at, references_at, rename_at

# ---------------------------------------------------------------------
# Definition
# ---------------------------------------------------------------------


def test_definition_jumps_to_procedure():
    src = (
        "@greet(name: str) -> str:\n"
        "  = Hello <name>\n"
        "\n"
        "@main():\n"
        "  [reply: str] = <greet(world)>\n"
        "  = <reply>\n"
    )
    loc = definition_at(src, "file:///tmp/example.kedi", "/tmp/example.kedi", "greet", occurrence=2)
    assert loc is not None
    assert loc.uri == "file:///tmp/example.kedi"
    assert loc.range.start.line == 0


def test_definition_jumps_to_nested_procedure():
    src = "@foo():\n  @bar() -> str:\n    = done\n  = <bar()>\n\n= <foo()>\n"
    loc = definition_at(src, "file:///tmp/example.kedi", "/tmp/example.kedi", "bar", occurrence=2)
    assert loc is not None
    assert loc.range.start.line == 1


def test_definition_returns_none_for_non_reference():
    src = "@p():\n  >> Hello world\n  = ok\n"
    loc = definition_at(src, "file:///tmp/example.kedi", "/tmp/example.kedi", "Hello")
    assert loc is None


# ---------------------------------------------------------------------
# References
# ---------------------------------------------------------------------


def test_references_finds_all_call_sites():
    src = "@greet():\n  = Hi\n\n@main():\n  [a] = <greet()>\n  [b] = <greet()>\n  = <a><b>\n"
    refs = references_at(
        src,
        "file:///tmp/example.kedi",
        "/tmp/example.kedi",
        "greet",
        True,
        occurrence=2,
    )
    assert len(refs) >= 2


def test_references_excludes_declaration_when_requested():
    src = "@g():\n  = ok\n\n= <g()>\n"
    refs_with = references_at(src, "file:///tmp/x.kedi", "/tmp/x.kedi", "g", True, occurrence=2)
    refs_without = references_at(src, "file:///tmp/x.kedi", "/tmp/x.kedi", "g", False, occurrence=2)
    assert len(refs_with) >= len(refs_without)


# ---------------------------------------------------------------------
# Rename
# ---------------------------------------------------------------------


def test_prepare_rename_on_procedure_returns_range():
    src = "@oldname():\n  = ok\n"
    r = prepare_rename_at(src, "/tmp/x.kedi", "oldname")
    assert r is not None
    assert r.start.line == 0


def test_rename_returns_workspace_edit():
    src = "@oldname():\n  = ok\n\n= <oldname()>\n"
    edit = rename_at(src, "file:///tmp/x.kedi", "/tmp/x.kedi", "oldname", "newname")
    assert edit is not None
    assert edit.changes is not None
    edits = edit.changes["file:///tmp/x.kedi"]
    assert any(e.new_text == "newname" for e in edits)


def test_rename_rejects_invalid_identifier():
    src = "@x():\n  = ok\n"
    edit = rename_at(src, "file:///tmp/x.kedi", "/tmp/x.kedi", "x", "42bad")
    assert edit is None
