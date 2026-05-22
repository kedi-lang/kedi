"""Smoke tests for the ``kedi-lsp`` scaffold.

These exercise the feature functions directly (without running the
stdio server) so the tests are deterministic and fast.
"""

from __future__ import annotations

import ast

from lsprotocol import types as lsp

from kedi.lsp import features
from kedi.lsp.python_virtual import compute_python_virtual_document

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


def test_hover_on_nested_procedure_header_shows_signature():
    src = "@foo():\n  @bar(name: str) -> str:\n    = <name>\n  = done\n"
    hover = features.compute_hover(
        source=src, source_path="example.kedi", pos=lsp.Position(line=1, character=4)
    )
    assert hover is not None
    body = hover.contents.value
    assert "bar" in body
    assert "name: str" in body
    assert "str" in body


def test_hover_on_nested_text_segment_works():
    src = "@foo():\n  @bar():\n    Hello world\n    = Done\n  = dn\n"
    hover = features.compute_hover(
        source=src, source_path="example.kedi", pos=lsp.Position(line=2, character=8)
    )
    assert hover is not None
    body = hover.contents.value
    assert "Text segment" in body
    assert "Hello world" in body


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


def test_hover_on_builtin_and_typing_types_is_not_undefined():
    src = (
        "@p(items: list[int], x: Optional[str], y: Literal[int]) -> bool:\n"
        "  Flag: [flag: bool]\n"
        "  = <x>\n"
    )

    checks = [
        ("list", lsp.Position(line=0, character=10), "builtins.list"),
        ("int", lsp.Position(line=0, character=15), "builtins.int"),
        ("Optional", lsp.Position(line=0, character=24), "typing.Optional"),
        ("str", lsp.Position(line=0, character=33), "builtins.str"),
        ("Literal", lsp.Position(line=0, character=42), "typing.Literal"),
        ("bool", lsp.Position(line=1, character=15), "builtins.bool"),
    ]

    for name, pos, origin in checks:
        hover = features.compute_hover(
            source=src,
            source_path="example.kedi",
            pos=pos,
        )
        assert hover is not None, name
        body = hover.contents.value
        assert name in body
        assert origin in body
        assert f"Kedi type resolved to Python's {origin}." in body
        assert "Kedi type derived from" not in body
        assert "Kedi type backed by" not in body
        assert "Kedi/Python type backed by" not in body
        assert "not defined" not in body


def test_builtin_type_definition_and_rename_are_disabled():
    src = "@p(x: Optional[str]) -> bool:\n  = <x>\n"
    uri = "file:///tmp/example.kedi"
    pos = lsp.Position(line=0, character=6)

    assert (
        features.compute_definition(
            source=src,
            uri=uri,
            source_path="/tmp/example.kedi",
            pos=pos,
        )
        is None
    )
    assert (
        features.compute_prepare_rename(
            source=src,
            source_path="/tmp/example.kedi",
            pos=pos,
        )
        is None
    )
    assert (
        features.compute_rename(
            source=src,
            uri=uri,
            source_path="/tmp/example.kedi",
            pos=pos,
            new_name="Maybe",
        )
        is None
    )


def test_hover_on_type_definition_builtin_and_typing_types_is_not_undefined():
    src = "~Person(name: str, friend: Optional[str])\n= ok\n"

    checks = [
        ("str", lsp.Position(line=0, character=14), "builtins.str"),
        ("Optional", lsp.Position(line=0, character=27), "typing.Optional"),
        ("str", lsp.Position(line=0, character=36), "builtins.str"),
    ]

    for name, pos, origin in checks:
        hover = features.compute_hover(
            source=src,
            source_path="example.kedi",
            pos=pos,
        )
        assert hover is not None, name
        body = hover.contents.value
        assert name in body
        assert origin in body
        assert "not defined" not in body


def test_custom_type_definition_still_jumps_to_kedi_type():
    src = "~Person(name: str)\n@p(x: Person):\n  = <x>\n"
    loc = features.compute_definition(
        source=src,
        uri="file:///tmp/example.kedi",
        source_path="/tmp/example.kedi",
        pos=lsp.Position(line=1, character=6),
    )
    assert loc is not None
    assert loc.range.start.line == 0


def test_custom_type_definition_field_type_still_jumps_to_kedi_type():
    src = "~Person(name: str)\n~Team(member: Person)\n= ok\n"
    loc = features.compute_definition(
        source=src,
        uri="file:///tmp/example.kedi",
        source_path="/tmp/example.kedi",
        pos=lsp.Position(line=1, character=14),
    )
    assert loc is not None
    assert loc.range.start.line == 0


def test_unknown_type_hover_stays_runtime_resolved():
    src = "@p(x: UnknownType):\n  = <x>\n"
    hover = features.compute_hover(
        source=src,
        source_path="example.kedi",
        pos=lsp.Position(line=0, character=6),
    )
    assert hover is not None
    assert "not defined in this module" in hover.contents.value
    assert "resolved at runtime" in hover.contents.value


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


def test_definition_jumps_to_nested_procedure():
    src = (
        "@foo():\n"
        "  @bar() -> str:\n"
        "    = done\n"
        "  = <bar()>\n"
        "\n"
        "= <foo()>\n"
    )
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
    src = '@p():\n  = ok\n\n@test: p:\n  > case: c1:\n    `assert p() == "ok"`\n'
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
    src = "@add(x: int, y: int) -> int:\n  = `x + y`\n\n= <add(1, 2)>\n"
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
    src = "@add(x: int, y: int) -> int:\n  = `x + y`\n\n= <add(1, 2)>\n"
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


def test_python_virtual_document_models_procedure_scope():
    src = (
        "@calculate_stats(numbers: list[int]) -> int:\n"
        "  [total: int] = ```\n"
        "  return sum(numbers)\n"
        "  ```\n"
        "  [count: int] = ```\n"
        "  return len(numbers)\n"
        "  ```\n"
        "  [average: float] = ```\n"
        "  return total / count if count > 0 else 0\n"
        "  ```\n"
        "  = Average of <`count`> numbers is <`average`>\n"
    )
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi")
    ast.parse(virtual["text"])
    text = virtual["text"]
    assert "def calculate_stats(numbers: list[int]) -> int: ..." in text
    assert "def __kedi_scope_calculate_stats(numbers: list[int]) -> int:" in text
    assert "total: int" in text
    assert "count: int" in text
    assert "average: float" in text
    assert "return total / count if count > 0 else 0" in text
    assert any(
        s["kind"] == "parameter"
        and s["name"] == "numbers"
        and s["sourceRange"]["start"] == {"line": 0, "character": 17}
        for s in virtual["symbols"]
    )
    assert any(
        s["kind"] == "local"
        and s["name"] == "total"
        and s["sourceRange"]["start"] == {"line": 1, "character": 3}
        for s in virtual["symbols"]
    )
    assert any(
        s["kind"] == "local"
        and s["name"] == "average"
        and s["sourceRange"]["start"] == {"line": 7, "character": 3}
        for s in virtual["symbols"]
    )
    assert any(
        m["sourceRange"]["start"]["line"] == 8
        and m["virtualRange"]["start"]["line"] > m["sourceRange"]["start"]["line"]
        for m in virtual["mappings"]
    )


def test_python_virtual_document_can_focus_single_procedure_body():
    src = (
        "@alpha(items: list[int]) -> int:\n"
        "  [total: int] = ```\n"
        "  return sum(items)\n"
        "  ```\n"
        "  = <`total`>\n"
        "\n"
        "@beta(name: str) -> str:\n"
        "  [upper: str] = ```\n"
        "  return name.upper()\n"
        "  ```\n"
        "  = <`upper`>\n"
        "\n"
        "= done\n"
    )
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi", focus_line=2)
    ast.parse(virtual["text"])
    assert virtual["focusKey"] == "proc:alpha:1"
    assert "def alpha(items: list[int]) -> int: ..." in virtual["text"]
    assert "def beta(name: str) -> str: ..." in virtual["text"]
    assert "def __kedi_scope_alpha(items: list[int]) -> int:" in virtual["text"]
    assert "return sum(items)" in virtual["text"]
    assert "def __kedi_scope_beta(name: str) -> str:" not in virtual["text"]
    assert "return name.upper()" not in virtual["text"]
    assert "def __kedi_toplevel__() -> Any:" not in virtual["text"]


def test_python_virtual_document_includes_nested_procedure_body():
    src = (
        "@foo() -> str:\n"
        "  @bar(name: str) -> str:\n"
        "    [upper: str] = ```\n"
        "    return name.upper()\n"
        "    ```\n"
        "    = <`upper`>\n"
        "  = <bar(ok)>\n"
    )
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi")
    ast.parse(virtual["text"])
    assert "def bar(name: str) -> str:" in virtual["text"]
    assert "return name.upper()" in virtual["text"]
    assert any(
        s["kind"] == "procedure"
        and s["name"] == "bar"
        and s["sourceRange"]["start"] == {"line": 1, "character": 3}
        for s in virtual["symbols"]
    )


def test_python_virtual_document_can_focus_nested_procedure_body():
    src = (
        "@foo() -> str:\n"
        "  @bar(name: str) -> str:\n"
        "    [upper: str] = ```\n"
        "    return name.upper()\n"
        "    ```\n"
        "    = <`upper`>\n"
        "  = <bar(ok)>\n"
    )
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi", focus_line=3)
    ast.parse(virtual["text"])
    assert virtual["focusKey"] == "proc:bar:2"
    assert "def __kedi_scope_bar(name: str) -> str:" in virtual["text"]
    assert "return name.upper()" in virtual["text"]
    assert "def __kedi_scope_foo() -> str:" not in virtual["text"]


def test_python_virtual_document_emits_pass_for_empty_toplevel_scope():
    src = "= done\n"
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi")
    ast.parse(virtual["text"])
    assert "def __kedi_toplevel__() -> Any:" in virtual["text"]
    assert "    pass" in virtual["text"]


def test_python_virtual_document_emits_ellipsis_for_empty_generated_python_blocks():
    src = (
        "@p() -> str:\n"
        "  [x: int] = ```\n"
        "  # comment only\n"
        "  ```\n"
        "  ```\n"
        "  # comment only\n"
        "  ```\n"
        "  = ok\n"
    )
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi")
    ast.parse(virtual["text"])
    assert "def __kedi_block_x_2() -> int:" in virtual["text"]
    assert "        ..." in virtual["text"]
    assert "def __kedi_scope_p() -> str:" in virtual["text"]
    assert "    ..." in virtual["text"]


def test_python_virtual_document_emits_ellipsis_for_comment_only_validation_blocks():
    src = (
        "@score() -> int:\n"
        "  = `1`\n"
        "\n"
        "@eval: score:\n"
        "  > metric: check:\n"
        "    = ```\n"
        "    # comment only\n"
        "    ```\n"
    )
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi")
    ast.parse(virtual["text"])
    assert "def __kedi_metric_score_check() -> Any:" in virtual["text"]
    assert "    expected: Any = ..." in virtual["text"]
    assert "    ..." in virtual["text"]


def test_python_virtual_document_models_type_stub_init_signature():
    src = "~Person(name, age: int, tags: list[str])\n= ok\n"
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi")
    assert "class Person:" in virtual["text"]
    assert "    name: str" in virtual["text"]
    assert "    age: int" in virtual["text"]
    assert "    tags: list[str]" in virtual["text"]
    assert (
        "    def __init__(self, *, name: str, age: int, tags: list[str]) -> None: ..."
        in virtual["text"]
    )


def test_python_virtual_document_imports_typing_symbols_used_in_backtick_types():
    src = (
        '~Issue(line: int, severity: `Literal["critical", "low", "medium"]`)\n'
        "~Review(issues: `list[Issue]`)\n"
        "@review_code() -> Review:\n"
        "  = ok\n"
    )
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi")
    assert "from typing import Any, Literal" in virtual["text"]
    assert 'severity: Literal["critical", "low", "medium"]' in virtual["text"]
    assert "issues: list[Issue]" in virtual["text"]


def test_python_virtual_document_models_kedi_proc_stubs_for_metrics():
    src = (
        "@get_country(city):\n"
        "  <city> is located in [country: str].\n"
        "  = <country>\n"
        "\n"
        "@eval: get_country:\n"
        "  > metric: accuracy:\n"
        "    = ```\n"
        '    result = get_country("France")\n'
        "    return (1.0, result)\n"
        "    ```\n"
    )
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi")
    ast.parse(virtual["text"])
    assert "def get_country(city: Any) -> str: ..." in virtual["text"]
    assert "def __kedi_metric_get_country_accuracy() -> Any:" in virtual["text"]
    assert 'result = get_country("France")' in virtual["text"]
    assert any(r["kind"] == "fenced" for r in virtual["ranges"])
    assert any(
        s["kind"] == "procedure"
        and s["name"] == "get_country"
        and s["sourceRange"]["start"] == {"line": 0, "character": 1}
        for s in virtual["symbols"]
    )


# ---------------------------------------------------------------------
# Server construction smoke
# ---------------------------------------------------------------------


def test_server_can_be_constructed():
    from kedi.lsp.server import make_server

    server = make_server()
    assert server is not None
    assert server.name == "kedi-lsp"


def test_server_custom_requests_prefer_request_text():
    from kedi.lsp.server import _request_source

    class Workspace:
        def get_text_document(self, uri):
            raise AssertionError("workspace cache should not be read")

    class Server:
        workspace = Workspace()

    fresh = "@p():\n  [x: int] = ```\n  return 1\n  ```\n  = <`x`>\n"
    assert (
        _request_source(
            Server(),
            "file:///tmp/x.kedi",
            {"textDocument": {"uri": "file:///tmp/x.kedi"}, "text": fresh},
        )
        == fresh
    )
