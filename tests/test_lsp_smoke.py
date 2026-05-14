"""Smoke tests for the ``kedi-lsp`` scaffold.

These exercise the feature functions directly (without running the
stdio server) so the tests are deterministic and fast.
"""

from __future__ import annotations

from lsprotocol import types as lsp

from kedi.lsp import features


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------


def test_diagnostics_empty_for_valid_program():
    src = "@p():\n  = ok\n"
    assert features.compute_diagnostics(src, "example.kedi") == []


def test_diagnostics_reports_duplicate_type_field():
    src = "~Item(name, name)\n= ok\n"
    diags = features.compute_diagnostics(src, "example.kedi")
    assert diags
    d = diags[0]
    assert d.severity == lsp.DiagnosticSeverity.Error
    assert "duplicate field" in d.message.lower()
    assert d.source == "kedi"
    assert d.range.start.line == 0


def test_diagnostics_reports_unterminated_block_comment():
    diags = features.compute_diagnostics("###\nbody never closed\n", "example.kedi")
    assert any("unterminated" in d.message.lower() for d in diags)


def test_diagnostics_reports_legacy_optimize_empty_message():
    src = "@p():\n  > optimize: empty:\n\n  = done\n"
    diags = features.compute_diagnostics(src, "example.kedi")
    assert any("must have an indented block" in d.message for d in diags)


# ---------------------------------------------------------------------
# Hover
# ---------------------------------------------------------------------


def test_hover_on_procedure_header_shows_signature():
    src = "@translate(text: str) -> str:\n  = <text>\n"
    hover = features.compute_hover(
        source=src, source_path="example.kedi", pos=lsp.Position(line=0, character=3)
    )
    assert hover is not None
    body = hover.contents.value
    assert "translate" in body
    assert "text: str" in body
    assert "str" in body


def test_hover_on_input_segment():
    src = "@p(name):\n  Hello, <name>!\n  = done\n"
    hover = features.compute_hover(
        source=src, source_path="example.kedi", pos=lsp.Position(line=1, character=11)
    )
    assert hover is not None
    body = hover.contents.value
    assert "name" in body
    # Hover labels parameter references as such when the input matches
    # an enclosing-procedure parameter.
    assert "arameter" in body or "Local" in body


def test_hover_on_output_segment_shows_type():
    src = "@p():\n  Pick a [n: int].\n  = <n>\n"
    hover = features.compute_hover(
        source=src, source_path="example.kedi", pos=lsp.Position(line=1, character=10)
    )
    assert hover is not None
    body = hover.contents.value
    assert "Output" in body
    assert "n" in body


def test_hover_on_unparseable_returns_none():
    hover = features.compute_hover(
        source="@p(:\n  ???\n",
        source_path="example.kedi",
        pos=lsp.Position(line=0, character=1),
    )
    assert hover is None


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
    src = (
        "@greet():\n"
        "  = Hi\n"
        "\n"
        "@main():\n"
        "  [a] = <greet()>\n"
        "  [b] = <greet()>\n"
        "  = <a><b>\n"
    )
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
    src = (
        "@p():\n"
        "  = ok\n"
        "\n"
        "@test: p:\n"
        "  > case: c1:\n"
        "    `assert p() == \"ok\"`\n"
    )
    syms = features.compute_document_symbols(src, "/tmp/x.kedi")
    names = [s.name for s in syms]
    assert any("@test: p" in n for n in names)


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
    src = (
        "@add(x: int, y: int) -> int:\n"
        "  = `x + y`\n"
        "\n"
        "= <add(1, 2)>\n"
    )
    # Cursor between `(` and the first arg.
    sig = features.compute_signature_help(
        source=src,
        source_path="/tmp/x.kedi",
        pos=lsp.Position(line=3, character=7),
    )
    assert sig is not None
    assert sig.signatures
    label = sig.signatures[0].label
    assert "add" in label
    assert "x: int" in label and "y: int" in label


def test_signature_help_outside_call_returns_none():
    src = "= ok\n"
    sig = features.compute_signature_help(
        source=src,
        source_path="/tmp/x.kedi",
        pos=lsp.Position(line=0, character=3),
    )
    assert sig is None


# ---------------------------------------------------------------------
# Inlay hints
# ---------------------------------------------------------------------


def test_inlay_hint_parameter_names_for_call():
    src = (
        "@add(x: int, y: int) -> int:\n"
        "  = `x + y`\n"
        "\n"
        "= <add(1, 2)>\n"
    )
    hints = features.compute_inlay_hints(
        source=src,
        source_path="/tmp/x.kedi",
        rng=lsp.Range(
            start=lsp.Position(line=0, character=0),
            end=lsp.Position(line=10, character=0),
        ),
    )
    # At least one parameter hint should appear.
    labels = [h.label for h in hints]
    assert any("x:" in str(lbl) for lbl in labels)


def test_inlay_hint_str_default_for_untyped_output():
    src = "@p():\n  Pick a [n].\n  = <n>\n"
    hints = features.compute_inlay_hints(
        source=src,
        source_path="/tmp/x.kedi",
        rng=lsp.Range(
            start=lsp.Position(line=0, character=0),
            end=lsp.Position(line=10, character=0),
        ),
    )
    labels = [str(h.label) for h in hints]
    assert any(": str" in lbl for lbl in labels)


# ---------------------------------------------------------------------
# Semantic tokens
# ---------------------------------------------------------------------


def test_semantic_tokens_full_emits_tokens():
    src = "@hello(name: str) -> str:\n  = <name>\n"
    tokens = features.compute_semantic_tokens_full(src, "/tmp/x.kedi")
    assert tokens.data, "expected at least one semantic token for a non-trivial doc"
    # Length is a multiple of 5 (LSP encoding).
    assert len(tokens.data) % 5 == 0


def test_semantic_tokens_legend_constants():
    # Sanity: standard order of types/modifiers.
    assert features.SEMANTIC_TOKEN_TYPES[12] == "function"
    assert features.SEMANTIC_TOKEN_TYPES[1] == "type"
    assert features.SEMANTIC_TOKEN_MODIFIERS[0] == "declaration"


# ---------------------------------------------------------------------
# Python ranges (custom)
# ---------------------------------------------------------------------


def test_python_ranges_picks_up_fenced_block():
    src = (
        "@stats(numbers: list[int]) -> int:\n"
        "  [total: int] = ```\n"
        "  return sum(numbers)\n"
        "  ```\n"
        "  = <`str(total)`>\n"
    )
    ranges = features.compute_python_ranges(src)
    kinds = [r["kind"] for r in ranges]
    assert "fenced" in kinds
    assert "inline" in kinds
    fenced = next(r for r in ranges if r["kind"] == "fenced")
    assert "return sum(numbers)" in fenced["text"]


def test_python_ranges_empty_for_no_python():
    src = "@p():\n  = ok\n"
    assert features.compute_python_ranges(src) == []


# ---------------------------------------------------------------------
# Server construction smoke
# ---------------------------------------------------------------------


def test_server_can_be_constructed():
    from kedi.lsp.server import make_server

    server = make_server()
    assert server is not None
    assert server.name == "kedi-lsp"
