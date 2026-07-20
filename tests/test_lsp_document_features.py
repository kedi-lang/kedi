"""LSP feature tests."""

from __future__ import annotations

from pathlib import Path

from lsprotocol import types as lsp

from kedi.lsp import features
from tests.lsp_helpers import inlay_hints_for_lines, signature_help_at

# ---------------------------------------------------------------------
# Document symbols
# ---------------------------------------------------------------------


def test_document_symbols_lists_procedures_and_types():
    src = "~Person(name, age: int)\n\n@get(x: int) -> int:\n  = `x`\n"
    syms = features.compute_document_symbols(src, "/tmp/x.kedi")
    names = [s.name for s in syms]
    assert "Person" in names
    assert "get" in names


def test_document_symbols_lists_validation_blocks():
    src = '@p():\n  = ok\n\n@test: p:\n  > case: c1:\n    `assert p() == "ok"`\n'
    syms = features.compute_document_symbols(src, "/tmp/x.kedi")
    names = [s.name for s in syms]
    assert any("@test: p" in n for n in names)


def test_document_symbols_nest_procedure_local_types():
    src = "@outer() -> str:\n  ~Inner(name: str)\n  = ok\n"

    symbols = features.compute_document_symbols(src, "/tmp/x.kedi")

    outer = next(symbol for symbol in symbols if symbol.name == "outer")
    assert outer.children is not None
    assert [child.name for child in outer.children] == ["Inner"]


def test_imported_procedure_references_preserve_the_declaration_uri(tmp_path: Path) -> None:
    module = tmp_path / "module.kedi"
    module.write_text(
        "> export:\n    greet\n@greet(name: str) -> str:\n  = Hello <name>\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    source = "> import: module\n>> Call <greet(Ada)> [answer: str]\n= <answer>\n"
    main.write_text(source, encoding="utf-8")
    position = lsp.Position(line=1, character=source.splitlines()[1].index("greet"))

    references = features.compute_references(
        source,
        main.as_uri(),
        str(main),
        position,
        include_declaration=True,
    )

    assert [location.uri for location in references] == [module.resolve().as_uri(), main.as_uri()]


def test_references_include_nested_procedure_calls_and_type_annotations() -> None:
    source = (
        "~Person(name: str)\n"
        "\n"
        "@helper() -> str:\n"
        "  = ok\n"
        "\n"
        "@outer() -> Person:\n"
        "  @inner() -> Person:\n"
        "    [person: Person] = <`Person(name='Ada')`>\n"
        "    = <person>\n"
        "  [message: str] = <helper()>\n"
        "  = <inner()>\n"
    )
    uri = "file:///tmp/x.kedi"

    helper_references = features.compute_references(
        source,
        uri,
        "/tmp/x.kedi",
        lsp.Position(line=2, character=1),
        include_declaration=True,
    )
    assert [
        (location.range.start.line, location.range.start.character)
        for location in helper_references
    ] == [
        (2, 1),
        (9, 20),
    ]

    person_references = features.compute_references(
        source,
        uri,
        "/tmp/x.kedi",
        lsp.Position(line=0, character=1),
        include_declaration=True,
    )
    assert [
        (location.range.start.line, location.range.start.character)
        for location in person_references
    ] == [
        (0, 1),
        (5, 12),
        (6, 14),
        (7, 13),
    ]


def test_references_keep_shadowed_local_and_top_level_types_separate() -> None:
    source = (
        "~Inner(global_name: str)\n"
        "\n"
        "@outer() -> str:\n"
        "  ~Inner(name: str)\n"
        "  [item: Inner] = <value>\n"
        "  = <item>\n"
        "\n"
        "[item: Inner] = <value>\n"
    )
    uri = "file:///tmp/x.kedi"

    global_references = features.compute_references(
        source,
        uri,
        "/tmp/x.kedi",
        lsp.Position(line=0, character=1),
        include_declaration=True,
    )
    local_references = features.compute_references(
        source,
        uri,
        "/tmp/x.kedi",
        lsp.Position(line=3, character=3),
        include_declaration=True,
    )

    assert [
        (location.range.start.line, location.range.start.character)
        for location in global_references
    ] == [(0, 1), (7, 7)]
    assert [
        (location.range.start.line, location.range.start.character) for location in local_references
    ] == [(3, 3), (4, 9)]


def test_references_and_definitions_keep_shadowed_procedures_separate() -> None:
    source = (
        "@helper() -> str:\n"
        "  = global\n"
        "\n"
        "@outer() -> str:\n"
        "  @helper() -> str:\n"
        "    = local\n"
        "  = <helper()>\n"
        "\n"
        "@other() -> str:\n"
        "  = <helper()>\n"
    )
    uri = "file:///tmp/x.kedi"

    global_references = features.compute_references(
        source,
        uri,
        "/tmp/x.kedi",
        lsp.Position(line=0, character=1),
        include_declaration=True,
    )
    local_references = features.compute_references(
        source,
        uri,
        "/tmp/x.kedi",
        lsp.Position(line=4, character=3),
        include_declaration=True,
    )
    global_definition = features.compute_definition(
        source,
        uri,
        "/tmp/x.kedi",
        lsp.Position(line=9, character=7),
    )
    local_definition = features.compute_definition(
        source,
        uri,
        "/tmp/x.kedi",
        lsp.Position(line=6, character=7),
    )

    assert [
        (location.range.start.line, location.range.start.character)
        for location in global_references
    ] == [(0, 1), (9, 5)]
    assert [
        (location.range.start.line, location.range.start.character) for location in local_references
    ] == [(4, 3), (6, 5)]
    assert global_definition is not None
    assert global_definition.range.start.line == 0
    assert local_definition is not None
    assert local_definition.range.start.line == 4


def test_rename_only_edits_matching_type_identifier_tokens() -> None:
    source = (
        "~Inner(global_name: str)\n"
        "\n"
        "@outer() -> str:\n"
        "  ~Inner(name: str)\n"
        "  [item: Inner] = <value>\n"
        "  = <item>\n"
        "\n"
        "[item: Inner] = <value>\n"
    )
    uri = "file:///tmp/x.kedi"

    edit = features.compute_rename(
        source,
        uri,
        "/tmp/x.kedi",
        lsp.Position(line=3, character=3),
        "LocalInner",
    )

    assert edit is not None and edit.changes is not None
    assert [
        (item.range.start.line, item.range.start.character, item.range.end.character)
        for item in edit.changes[uri]
    ] == [(3, 3, 8), (4, 9, 14)]


def test_rename_type_field_edits_only_the_field_identifier() -> None:
    source = "~Person(name: str)\n"
    uri = "file:///tmp/x.kedi"

    edit = features.compute_rename(
        source,
        uri,
        "/tmp/x.kedi",
        lsp.Position(line=0, character=8),
        "display_name",
    )

    assert edit is not None and edit.changes is not None
    assert [
        (item.range.start.line, item.range.start.character, item.range.end.character)
        for item in edit.changes[uri]
    ] == [(0, 8, 12)]


def test_procedure_parameter_header_resolves_to_its_declaration() -> None:
    source = "@greet(user: str) -> str:\n  = Hello <user>\n"
    uri = "file:///tmp/x.kedi"

    hover = features.compute_hover(source, "/tmp/x.kedi", lsp.Position(line=0, character=7))
    definition = features.compute_definition(
        source,
        uri,
        "/tmp/x.kedi",
        lsp.Position(line=0, character=7),
    )
    references = features.compute_references(
        source,
        uri,
        "/tmp/x.kedi",
        lsp.Position(line=0, character=7),
        include_declaration=True,
    )

    assert hover is not None
    assert "Parameter" in hover.contents.value
    assert definition is not None
    assert (definition.range.start.line, definition.range.start.character) == (0, 7)
    assert [
        (location.range.start.line, location.range.start.character) for location in references
    ] == [
        (0, 7),
        (1, 11),
    ]


def test_nested_procedure_capture_resolves_to_the_outer_parameter() -> None:
    source = "@outer(value: str) -> str:\n  @inner() -> str:\n    = <value>\n  = <inner()>\n"
    uri = "file:///tmp/x.kedi"
    position = lsp.Position(line=2, character=7)

    hover = features.compute_hover(source, "/tmp/x.kedi", position)
    definition = features.compute_definition(source, uri, "/tmp/x.kedi", position)
    references = features.compute_references(
        source,
        uri,
        "/tmp/x.kedi",
        position,
        include_declaration=True,
    )

    assert hover is not None
    assert hover.contents.value == "`value: str`\n\nParameter of `@outer`."
    assert definition is not None
    assert (definition.range.start.line, definition.range.start.character) == (0, 7)
    assert [
        (location.range.start.line, location.range.start.character) for location in references
    ] == [
        (0, 7),
        (2, 7),
    ]


# ---------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------


def test_formatting_trims_trailing_whitespace():
    src = "@p():   \n  = ok   \n\n\n"
    edits = features.compute_formatting(src)
    assert edits
    new = edits[0].new_text
    assert "   \n" not in new
    assert new.endswith("\n")
    assert new.count("\n\n\n") == 0


def test_formatting_is_noop_for_clean_source():
    src = "@p():\n  = ok\n"
    edits = features.compute_formatting(src)
    assert edits == []


# ---------------------------------------------------------------------
# Signature help
# ---------------------------------------------------------------------


def test_signature_help_inside_call():
    src = "@add(x: int, y: int) -> int:\n  = `x + y`\n\n= <add(1, 2)>\n"
    sig = signature_help_at(src, "/tmp/x.kedi", "1")
    assert sig is not None
    assert sig.signatures
    label = sig.signatures[0].label
    assert "add" in label
    assert "x: int" in label and "y: int" in label


def test_signature_help_inside_procedure_body_call():
    src = "@outer() -> int:\n  = <add(1, 2)>\n\n@add(x: int, y: int) -> int:\n  = `x + y`\n"
    sig = signature_help_at(src, "/tmp/x.kedi", "1")
    assert sig is not None
    assert "add" in sig.signatures[0].label


def test_signature_help_outside_call_returns_none():
    src = "= ok\n"
    sig = signature_help_at(src, "/tmp/x.kedi", "ok")
    assert sig is None


def test_signature_help_inside_procedure_without_call_returns_none():
    src = "@outer() -> str:\n  = plain text\n"
    sig = signature_help_at(src, "/tmp/x.kedi", "plain")
    assert sig is None


# ---------------------------------------------------------------------
# Inlay hints
# ---------------------------------------------------------------------


def test_inlay_hint_parameter_names_for_call():
    src = "@add(x: int, y: int) -> int:\n  = `x + y`\n\n= <add(1, 2)>\n"
    hints = inlay_hints_for_lines(src, "/tmp/x.kedi", 0, 10)
    labels = [h.label for h in hints]
    assert any("x:" in str(lbl) for lbl in labels)


def test_inlay_hint_str_default_for_untyped_output():
    src = "@p():\n  >> Pick a [n].\n  = <n>\n"
    hints = inlay_hints_for_lines(src, "/tmp/x.kedi", 0, 10)
    labels = [str(h.label) for h in hints]
    assert any(": str" in lbl for lbl in labels)
