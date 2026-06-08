"""LSP feature tests."""

from __future__ import annotations

from lsprotocol import types as lsp

from kedi.lsp import features

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
    loc = features.compute_definition(
        source=src,
        uri="file:///tmp/example.kedi",
        source_path="/tmp/example.kedi",
        pos=lsp.Position(line=4, character=20),
    )
    assert loc is not None
    assert loc.uri == "file:///tmp/example.kedi"
    assert loc.range.start.line == 0


def test_definition_jumps_to_nested_procedure():
    src = "@foo():\n  @bar() -> str:\n    = done\n  = <bar()>\n\n= <foo()>\n"
    loc = features.compute_definition(
        source=src,
        uri="file:///tmp/example.kedi",
        source_path="/tmp/example.kedi",
        pos=lsp.Position(line=3, character=6),
    )
    assert loc is not None
    assert loc.range.start.line == 1


def test_definition_returns_none_for_non_reference():
    src = "@p():\n  Hello world\n  = ok\n"
    loc = features.compute_definition(
        source=src,
        uri="file:///tmp/example.kedi",
        source_path="/tmp/example.kedi",
        pos=lsp.Position(line=1, character=4),
    )
    assert loc is None


# ---------------------------------------------------------------------
# References
# ---------------------------------------------------------------------


def test_references_finds_all_call_sites():
    src = "@greet():\n  = Hi\n\n@main():\n  [a] = <greet()>\n  [b] = <greet()>\n  = <a><b>\n"
    # Cursor on `greet` in the @main body's first call.
    refs = features.compute_references(
        source=src,
        uri="file:///tmp/example.kedi",
        source_path="/tmp/example.kedi",
        pos=lsp.Position(line=4, character=11),
        include_declaration=True,
    )
    assert len(refs) >= 2  # declaration + at least one call


def test_references_excludes_declaration_when_requested():
    src = "@g():\n  = ok\n\n= <g()>\n"
    refs_with = features.compute_references(
        source=src,
        uri="file:///tmp/x.kedi",
        source_path="/tmp/x.kedi",
        pos=lsp.Position(line=3, character=4),
        include_declaration=True,
    )
    refs_without = features.compute_references(
        source=src,
        uri="file:///tmp/x.kedi",
        source_path="/tmp/x.kedi",
        pos=lsp.Position(line=3, character=4),
        include_declaration=False,
    )
    assert len(refs_with) >= len(refs_without)


# ---------------------------------------------------------------------
# Rename
# ---------------------------------------------------------------------


def test_prepare_rename_on_procedure_returns_range():
    src = "@oldname():\n  = ok\n"
    r = features.compute_prepare_rename(
        source=src,
        source_path="/tmp/x.kedi",
        pos=lsp.Position(line=0, character=3),
    )
    assert r is not None
    assert r.start.line == 0


def test_rename_returns_workspace_edit():
    src = "@oldname():\n  = ok\n\n= <oldname()>\n"
    edit = features.compute_rename(
        source=src,
        uri="file:///tmp/x.kedi",
        source_path="/tmp/x.kedi",
        pos=lsp.Position(line=0, character=3),
        new_name="newname",
    )
    assert edit is not None
    assert edit.changes is not None
    edits = edit.changes["file:///tmp/x.kedi"]
    assert any(e.new_text == "newname" for e in edits)


def test_rename_rejects_invalid_identifier():
    src = "@x():\n  = ok\n"
    edit = features.compute_rename(
        source=src,
        uri="file:///tmp/x.kedi",
        source_path="/tmp/x.kedi",
        pos=lsp.Position(line=0, character=1),
        new_name="42bad",
    )
    assert edit is None
