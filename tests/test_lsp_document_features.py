"""LSP feature tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from lsprotocol import types as lsp

import kedi.lang.ast as kedi_ast
import kedi.lsp.features.completion as lsp_completion_module
import kedi.lsp.features.diagnostics as lsp_diagnostics_module
import kedi.lsp.features.document as lsp_document_module
import kedi.lsp.features.hover as lsp_hover_module
import kedi.lsp.features.navigation as lsp_navigation_module
import kedi.lsp.features.semantic_tokens as lsp_semantic_tokens_module
import kedi.lsp.features.symbols as lsp_symbols_module
from kedi.lang.ast import ProfileSubagentEntry, Program, Span, UseDirectiveStmt
from tests.lsp_helpers import decode_semantic_tokens, inlay_hints_for_lines, signature_help_at

# ---------------------------------------------------------------------
# Document symbols
# ---------------------------------------------------------------------


def test_completion_exposes_directives_types_and_lexical_symbols() -> None:
    directive_items = lsp_completion_module.compute_completion(
        "> mo",
        "/tmp/x.kedi",
        lsp.Position(line=0, character=4),
    )
    assert any(
        item.label == "model" and item.text_edit.new_text == "> model: "
        for item in directive_items
        if isinstance(item.text_edit, lsp.TextEdit)
    )

    source = (
        "~Person(name: str)\n\n"
        "@render(person: Person) -> str:\n"
        "  [label: str] = <person>\n"
        "  = <label>\n"
    )
    items = lsp_completion_module.compute_completion(
        source,
        "/tmp/x.kedi",
        lsp.Position(line=3, character=5),
    )
    labels = {item.label for item in items}
    assert {"Person", "render", "person", "label", "str"} <= labels

    assert (
        lsp_completion_module.compute_completion(
            "```\nvalue = 1\n```",
            "/tmp/x.kedi",
            lsp.Position(line=1, character=3),
        )
        == []
    )
    invalid_labels = {
        item.label
        for item in lsp_completion_module.compute_completion(
            "[broken",
            "/tmp/x.kedi",
            lsp.Position(line=0, character=7),
        )
    }
    assert "int" in invalid_labels


def test_completion_exposes_codemode_settings_inside_expanded_block() -> None:
    source = "> codemode:\n    pre"
    items = lsp_completion_module.compute_completion(
        source,
        "/tmp/x.kedi",
        lsp.Position(line=1, character=len("    pre")),
    )
    labels = {item.label for item in items}
    assert {"preload_tools", "max_hydrated_tools", "enabled"} <= labels

    nested = "> codemode:\n    preload_tools:\n        lookup"
    nested_items = lsp_completion_module.compute_completion(
        nested,
        "/tmp/x.kedi",
        lsp.Position(line=2, character=len("        lookup")),
    )
    assert "preload_tools" not in {item.label for item in nested_items}


def test_completion_exposes_requirements_and_tool_metadata_fields() -> None:
    directive_items = lsp_completion_module.compute_completion(
        "> req",
        "/tmp/x.kedi",
        lsp.Position(line=0, character=len("> req")),
    )
    top_level_labels = {item.label for item in directive_items}
    assert "requires" in top_level_labels
    assert "tool" not in top_level_labels

    procedure_directives = lsp_completion_module.compute_completion(
        "@lookup() -> str:\n    > to",
        "/tmp/x.kedi",
        lsp.Position(line=1, character=len("    > to")),
    )
    assert "tool" in {item.label for item in procedure_directives}

    profile_directives = lsp_completion_module.compute_completion(
        "> profile: helper:\n    > to",
        "/tmp/x.kedi",
        lsp.Position(line=1, character=len("    > to")),
    )
    assert "tool" not in {item.label for item in profile_directives}

    source = "@lookup() -> str:\n    > tool:\n        ret"
    field_items = lsp_completion_module.compute_completion(
        source,
        "/tmp/x.kedi",
        lsp.Position(line=2, character=len("        ret")),
    )
    assert {"name", "description", "risk", "retries", "retry_on"} <= {
        item.label for item in field_items
    }

    nested = source + "ry_on:\n            Timeout"
    nested_items = lsp_completion_module.compute_completion(
        nested,
        "/tmp/x.kedi",
        lsp.Position(line=3, character=len("            Timeout")),
    )
    assert "retry_on" not in {item.label for item in nested_items}


def test_completion_enclosing_header_handles_incomplete_editor_sources() -> None:
    position = lsp.Position(line=1, character=len("    > to"))

    assert lsp_completion_module._enclosing_header("", position) is None
    assert (
        lsp_completion_module._enclosing_header(
            "> profile: helper:\n> to",
            lsp.Position(line=1, character=len("> to")),
        )
        is None
    )
    assert lsp_completion_module._enclosing_header("\n    > to", position) is None
    assert (
        lsp_completion_module._enclosing_header(
            "@lookup() -> str:\n    > to",
            position,
        )
        == "@lookup() -> str:"
    )
    assert (
        lsp_completion_module._enclosing_header(
            "@lookup() -> str:\n    # metadata follows\n    > to",
            lsp.Position(line=2, character=len("    > to")),
        )
        == "@lookup() -> str:"
    )


def test_completion_exposes_top_level_control_flow_and_imported_symbols(tmp_path: Path) -> None:
    module = tmp_path / "helpers.kedi"
    module.write_text(
        "~Record(name: str)\n"
        "[count: int] = `1`\n"
        "@render() -> str:\n"
        "  = ok\n"
        "> profile: analyst:\n"
        "  > model: test\n"
        "> export: *\n",
        encoding="utf-8",
    )
    source = (
        "> import: helpers\n"
        "> profile: local_profile:\n"
        "  > model: test\n"
        "[before: int] = `1`\n"
        "> loop [item]: `range(1)`:\n"
        "  [label: str] = <item>\n"
        "  >> Summarize <label> as [summary]\n"
        "\n"
        "[later: int] = `2`\n"
    )
    source_path = tmp_path / "main.kedi"
    items = lsp_completion_module.compute_completion(
        source,
        str(source_path),
        lsp.Position(line=7, character=0),
    )
    labels = {item.label for item in items}

    assert {
        "Record",
        "count",
        "render",
        "analyst",
        "local_profile",
        "before",
        "item",
        "label",
        "summary",
    } <= labels
    assert "later" not in labels


def test_document_symbols_lists_procedures_and_types():
    src = "~Person(name, age: int)\n\n@get(x: int) -> int:\n  = `x`\n"
    syms = lsp_document_module.compute_document_symbols(src, "/tmp/x.kedi")
    names = [s.name for s in syms]
    assert "Person" in names
    assert "get" in names


def test_package_manifest_hover_describes_installed_package_metadata() -> None:
    source = (
        "> package: kedi_http:\n"
        "  version: 1.0\n"
        "  source: src/kedi_http\n"
        "  python: python@3.11-3.14\n"
        "  python_dependencies:\n"
        "    httpx>=0.27\n"
    )

    hover = lsp_hover_module.compute_hover(
        source, "/tmp/package.kedi", lsp.Position(line=0, character=4)
    )

    assert hover is not None
    assert "Kedi package" in hover.contents.value
    assert "main.kedi" in hover.contents.value
    assert "Python dependencies" in hover.contents.value


def test_package_manifest_hover_handles_minimal_metadata() -> None:
    source = "> package: kedi_http:\n  author: Mert\n"

    hover = lsp_hover_module.compute_hover(
        source, "/tmp/package.kedi", lsp.Position(line=0, character=4)
    )

    assert hover is not None
    assert hover.contents.value == "**Kedi package** `kedi_http`"


def test_control_flow_hover_describes_kedi_headers_and_defers_python() -> None:
    source = (
        "> if: `ready`:\n  [status] = yes\n> else:\n  > loop [n]: `range(3)`:\n    [value] = <n>\n"
    )

    if_hover = lsp_hover_module.compute_hover(source, "/tmp/control.kedi", lsp.Position(0, 3))
    loop_hover = lsp_hover_module.compute_hover(source, "/tmp/control.kedi", lsp.Position(3, 3))
    python_hover = lsp_hover_module.compute_hover(source, "/tmp/control.kedi", lsp.Position(0, 8))

    assert if_hover is not None
    assert "Conditional branch" in if_hover.contents.value
    assert "> if: `condition`:" in if_hover.contents.value
    assert loop_hover is not None
    assert "Iterable loop" in loop_hover.contents.value
    assert python_hover is None


def test_template_condition_diagnostic_and_control_hover() -> None:
    invalid = "> if: [answer] is correct\n  [result] = yes\n"
    diagnostics = lsp_diagnostics_module.compute_diagnostics(invalid, "/tmp/control.kedi")

    assert len(diagnostics) == 1
    assert diagnostics[0].code == "K-CONDITION-OUTPUT"
    assert "cannot declare output fields" in diagnostics[0].message

    source = (
        "> if: <city> is in Turkey\n  [result] = yes\n"
        "> loop: `remaining > 0`:\n  `remaining -= 1`\n"
        "> loop: <city> remains available\n  [seen] = <city>\n"
    )
    if_hover = lsp_hover_module.compute_hover(source, "/tmp/control.kedi", lsp.Position(0, 3))
    loop_hover = lsp_hover_module.compute_hover(source, "/tmp/control.kedi", lsp.Position(2, 3))
    model_loop_hover = lsp_hover_module.compute_hover(
        source, "/tmp/control.kedi", lsp.Position(4, 3)
    )
    city_hover = lsp_hover_module.compute_hover(source, "/tmp/control.kedi", lsp.Position(4, 10))

    assert if_hover is not None
    assert "Model-classified condition" in if_hover.contents.value
    assert loop_hover is not None
    assert "Conditional loop" in loop_hover.contents.value
    assert model_loop_hover is not None
    assert "Model-classified loop" in model_loop_hover.contents.value
    assert city_hover is not None


@pytest.mark.parametrize(
    "source",
    [
        "> if: `is_ready`\n  > show: unreachable\n",
        "> if: Work remains while `pending`\n  > show: unreachable\n",
        "> loop: `is_ready`\n  > show: unreachable\n",
        "> loop: Work remains while `pending`\n  > show: unreachable\n",
    ],
)
def test_template_condition_bare_python_reports_wrapped_syntax(source: str) -> None:
    diagnostics = lsp_diagnostics_module.compute_diagnostics(source, "/tmp/control.kedi")

    assert diagnostics
    assert diagnostics[0].code == "K-PARSE"
    assert "must be wrapped as <`...`>" in diagnostics[0].message


def test_loop_binder_definition_and_references_respect_nested_shadowing() -> None:
    source = (
        "> loop [n]: `range(2)`:\n"
        "  [first] = <n>\n"
        "  > loop [n]: `range(1)`:\n"
        "    [inner] = <n>\n"
        "  > loop: `False`:\n"
        "    [conditional] = <n>\n"
        "  [last] = <n>\n"
    )
    uri = "file:///tmp/control.kedi"

    definition = lsp_navigation_module.compute_definition(
        source,
        uri,
        "/tmp/control.kedi",
        lsp.Position(line=1, character=13),
    )
    references = lsp_navigation_module.compute_references(
        source,
        uri,
        "/tmp/control.kedi",
        lsp.Position(line=0, character=8),
        include_declaration=True,
    )

    assert definition is not None
    assert definition.range.start == lsp.Position(line=0, character=8)
    assert {(item.range.start.line, item.range.start.character) for item in references} == {
        (0, 8),
        (1, 13),
        (5, 21),
        (6, 12),
    }


def test_control_flow_symbols_cover_nested_scopes_types_and_procedures() -> None:
    source = (
        "~Item(value: int)\n"
        "[outer: Item] = `Item(value=1)`\n"
        "> loop [n]: `range(2)`:\n"
        "  [copy: Item] = <outer>\n"
        "  > loop [m]: `range(1)`:\n"
        "    > if: `True`:\n"
        "      [text] = <n>-<m>\n"
        "\n"
        "@collect():\n"
        "  > loop [p]: `range(1)`:\n"
        "    [value] = <p>\n"
    )
    lines = source.splitlines()
    path = "/tmp/control-symbols.kedi"
    uri = "file:///tmp/control-symbols.kedi"

    binder_hover = lsp_hover_module.compute_hover(
        source, path, lsp.Position(2, lines[2].index("n"))
    )
    assert binder_hover is not None
    assert "Loop binder" in binder_hover.contents.value

    outer_hover = lsp_hover_module.compute_hover(
        source,
        path,
        lsp.Position(3, lines[3].index("outer")),
    )
    assert outer_hover is not None
    assert "Module value" in outer_hover.contents.value

    outer_binder_refs = lsp_navigation_module.compute_references(
        source,
        uri,
        path,
        lsp.Position(2, lines[2].index("n")),
        include_declaration=True,
    )
    assert {item.range.start.line for item in outer_binder_refs} == {2, 6}

    procedure_binder_refs = lsp_navigation_module.compute_references(
        source,
        uri,
        path,
        lsp.Position(9, lines[9].index("[p]") + 1),
        include_declaration=True,
    )
    assert {item.range.start.line for item in procedure_binder_refs} == {9, 10}

    type_refs = lsp_navigation_module.compute_references(
        source,
        uri,
        path,
        lsp.Position(0, lines[0].index("Item")),
        include_declaration=True,
    )
    assert {item.range.start.line for item in type_refs} >= {0, 1, 3}

    assert lsp_diagnostics_module.compute_diagnostics(source, path) == []


def test_generic_hover_visits_legacy_top_level_body() -> None:
    program = lsp_diagnostics_module.parse_program(
        "[value: str] = done\n", source_path="/tmp/x.kedi"
    )
    program.ordered_toplevel.clear()

    target = lsp_hover_module._smallest_enclosing(program, lsp.Position(line=0, character=2))

    assert target is not None


def test_generic_hover_returns_none_for_an_undocumented_statement() -> None:
    span = kedi_ast.Span(
        start_byte=0,
        end_byte=3,
        start_line=1,
        start_col=1,
        end_line=1,
        end_col=4,
    )
    program = kedi_ast.Program(
        ordered_toplevel=[
            kedi_ast.ProfileDefStmt(
                name="parent",
                subagent_members=[ProfileSubagentEntry(name="child", loc=span)],
                loc=span,
            )
        ]
    )

    assert lsp_hover_module._generic_hover(program, lsp.Position(line=0, character=1)) is None


def test_package_manifest_semantic_tokens_cover_metadata() -> None:
    source = (
        "> package: kedi_http:\n"
        "  version: 1.0\n"
        "  source: src/kedi_http\n"
        "  python_dependencies:\n"
        "    httpx>=0.27\n"
    )

    tokens = decode_semantic_tokens(
        lsp_semantic_tokens_module.compute_semantic_tokens_full(source, "/tmp/package.kedi").data
    )
    token_types = lsp_semantic_tokens_module.SEMANTIC_TOKEN_TYPES

    assert any(
        line == 0 and token_type == token_types.index("keyword")
        for line, _, _, token_type, _ in tokens
    )
    assert any(
        line == 0 and token_type == token_types.index("namespace")
        for line, _, _, token_type, _ in tokens
    )
    assert any(
        line == 1 and token_type == token_types.index("property")
        for line, _, _, token_type, _ in tokens
    )
    assert any(
        line == 4 and token_type == token_types.index("string")
        for line, _, _, token_type, _ in tokens
    )


def test_document_symbols_lists_validation_blocks():
    src = '@p():\n  = ok\n\n@test: p:\n  > case: c1:\n    `assert p() == "ok"`\n'
    syms = lsp_document_module.compute_document_symbols(src, "/tmp/x.kedi")
    names = [s.name for s in syms]
    assert any("@test: p" in n for n in names)


def test_document_symbols_nest_procedure_local_types():
    src = "@outer() -> str:\n  ~Inner(name: str)\n  = ok\n"

    symbols = lsp_document_module.compute_document_symbols(src, "/tmp/x.kedi")

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

    references = lsp_navigation_module.compute_references(
        source,
        main.as_uri(),
        str(main),
        position,
        include_declaration=True,
    )

    assert [location.uri for location in references] == [module.resolve().as_uri(), main.as_uri()]


def test_profile_references_include_composition_subagent_and_use() -> None:
    source = (
        "> profile: base:\n"
        "    > system: Base.\n\n"
        "> profile: child:\n"
        "    > use: base\n"
        "    > subagent: base\n\n"
        "> profile: other:\n"
        "    > use: child\n"
        "    > subagent: child\n\n"
        "> use: base\n"
    )
    position = lsp.Position(line=0, character=len("> profile: "))

    references = lsp_navigation_module.compute_references(
        source,
        "file:///main.kedi",
        "main.kedi",
        position,
        include_declaration=True,
    )

    assert [(ref.range.start.line, ref.range.start.character) for ref in references] == [
        (0, 0),
        (4, len("    > use: ")),
        (5, len("    > subagent: ")),
        (11, len("> use: ")),
    ]

    span = Span(0, 1, 1, 1, 1, 2)
    no_location = Program(ordered_toplevel=[UseDirectiveStmt(target="base")])
    symbol = lsp_symbols_module.SymbolRef(
        kind="agent_profile",
        name="base",
        declaration_span=None,
        scope_proc=None,
        cursor_span=span,
    )
    assert (
        lsp_navigation_module._find_all_references(
            no_location,
            symbol,
            include_declaration=False,
        )
        == []
    )


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

    helper_references = lsp_navigation_module.compute_references(
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

    person_references = lsp_navigation_module.compute_references(
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

    global_references = lsp_navigation_module.compute_references(
        source,
        uri,
        "/tmp/x.kedi",
        lsp.Position(line=0, character=1),
        include_declaration=True,
    )
    local_references = lsp_navigation_module.compute_references(
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

    global_references = lsp_navigation_module.compute_references(
        source,
        uri,
        "/tmp/x.kedi",
        lsp.Position(line=0, character=1),
        include_declaration=True,
    )
    local_references = lsp_navigation_module.compute_references(
        source,
        uri,
        "/tmp/x.kedi",
        lsp.Position(line=4, character=3),
        include_declaration=True,
    )
    global_definition = lsp_navigation_module.compute_definition(
        source,
        uri,
        "/tmp/x.kedi",
        lsp.Position(line=9, character=7),
    )
    local_definition = lsp_navigation_module.compute_definition(
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

    edit = lsp_navigation_module.compute_rename(
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

    edit = lsp_navigation_module.compute_rename(
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

    hover = lsp_hover_module.compute_hover(source, "/tmp/x.kedi", lsp.Position(line=0, character=7))
    definition = lsp_navigation_module.compute_definition(
        source,
        uri,
        "/tmp/x.kedi",
        lsp.Position(line=0, character=7),
    )
    references = lsp_navigation_module.compute_references(
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


def test_selective_import_name_resolves_hover_and_definition(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    module_path = package / "helpers.kedi"
    module_path.write_text(
        "@selected(name: str) -> str:\n  = <name>\n> export:\n  selected\n",
        encoding="utf-8",
    )
    source_path = tmp_path / "main.kedi"
    source = "> import: package/helpers:\n  selected\n"
    position = lsp.Position(line=1, character=3)

    hover = lsp_hover_module.compute_hover(source, str(source_path), position)
    definition = lsp_navigation_module.compute_definition(
        source,
        source_path.as_uri(),
        str(source_path),
        position,
    )

    assert hover is not None
    assert "`@selected(name: str)` → `str`" in hover.contents.value
    assert definition is not None
    assert definition.uri == module_path.as_uri()
    assert definition.range.start.line == 0


def test_selective_import_names_resolve_all_export_kinds(tmp_path: Path) -> None:
    module_path = tmp_path / "helpers.kedi"
    module_path.write_text(
        "~Record(name: str)\n"
        "[count: int] = `1`\n"
        "> profile: analyst:\n"
        "    > model: test\n"
        "> export:\n"
        "  Record\n"
        "  count\n"
        "  analyst\n",
        encoding="utf-8",
    )
    source_path = tmp_path / "main.kedi"
    source = "> import: helpers:\n  Record\n  count\n  analyst\n  missing\n"

    type_hover = lsp_hover_module.compute_hover(
        source, str(source_path), lsp.Position(line=1, character=3)
    )
    value_hover = lsp_hover_module.compute_hover(
        source, str(source_path), lsp.Position(line=2, character=3)
    )
    profile_hover = lsp_hover_module.compute_hover(
        source, str(source_path), lsp.Position(line=3, character=3)
    )
    missing_hover = lsp_hover_module.compute_hover(
        source, str(source_path), lsp.Position(line=4, character=3)
    )

    assert type_hover is not None and "`~Record(name: str)`" in type_hover.contents.value
    assert (
        value_hover is not None and "**Imported value** `count: int`" in value_hover.contents.value
    )
    assert (
        profile_hover is not None
        and "**Agent profile** `> profile: analyst:`" in profile_hover.contents.value
    )
    assert (
        missing_hover is not None and "**Imported value** `missing`" in missing_hover.contents.value
    )


def test_nested_procedure_capture_resolves_to_the_outer_parameter() -> None:
    source = "@outer(value: str) -> str:\n  @inner() -> str:\n    = <value>\n  = <inner()>\n"
    uri = "file:///tmp/x.kedi"
    position = lsp.Position(line=2, character=7)

    hover = lsp_hover_module.compute_hover(source, "/tmp/x.kedi", position)
    definition = lsp_navigation_module.compute_definition(source, uri, "/tmp/x.kedi", position)
    references = lsp_navigation_module.compute_references(
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


def test_assignment_and_map_references_resolve_to_iteration_owners() -> None:
    from kedi.lang import parse_program

    source = (
        "[count: int] = `0`\n"
        "> loop [item]: `[1, 2]`:\n"
        "  [label] = item-<item>\n"
        "> map:\n"
        "  [count] := <label>\n"
        "  = <item>\n"
        "= <count>\n"
    )
    uri = "file:///tmp/map.kedi"

    count_definition = lsp_navigation_module.compute_definition(
        source,
        uri,
        "/tmp/map.kedi",
        lsp.Position(line=4, character=3),
    )
    label_definition = lsp_navigation_module.compute_definition(
        source,
        uri,
        "/tmp/map.kedi",
        lsp.Position(line=4, character=14),
    )
    binder_references = lsp_navigation_module.compute_references(
        source,
        uri,
        "/tmp/map.kedi",
        lsp.Position(line=1, character=8),
        include_declaration=True,
    )
    count_references = lsp_navigation_module.compute_references(
        source,
        uri,
        "/tmp/map.kedi",
        lsp.Position(line=0, character=2),
        include_declaration=True,
    )
    map_hover = lsp_hover_module.compute_hover(
        source,
        "/tmp/map.kedi",
        lsp.Position(line=4, character=9),
    )

    assert count_definition is not None
    assert (count_definition.range.start.line, count_definition.range.start.character) == (0, 1)
    assert label_definition is not None
    assert (label_definition.range.start.line, label_definition.range.start.character) == (2, 3)
    assert [
        (location.range.start.line, location.range.start.character)
        for location in binder_references
    ] == [(1, 8), (2, 18), (5, 5)]
    assert [
        (location.range.start.line, location.range.start.character) for location in count_references
    ] == [(0, 1), (4, 3), (6, 3)]
    assert map_hover is not None
    assert (
        lsp_hover_module._generic_hover(
            parse_program(source),
            lsp.Position(line=4, character=9),
        )
        is not None
    )


def test_imported_assignment_target_resolves_to_exported_declaration(tmp_path: Path) -> None:
    module = tmp_path / "values.kedi"
    module.write_text("> export:\n  value\n\n[value] = original\n", encoding="utf-8")
    main = tmp_path / "main.kedi"
    source = "> import: values:\n  value\n\n[value] := replacement\n"
    main.write_text(source, encoding="utf-8")

    definition = lsp_navigation_module.compute_definition(
        source,
        main.as_uri(),
        str(main),
        lsp.Position(line=3, character=2),
    )

    assert definition is not None
    assert definition.uri == module.as_uri()
    assert (definition.range.start.line, definition.range.start.character) == (3, 1)


def test_control_flow_declaration_does_not_resolve_outside_its_child_scope() -> None:
    source = "> if: `True`:\n  [branch_only] = yes\n= <branch_only>\n"

    definition = lsp_navigation_module.compute_definition(
        source,
        "file:///tmp/scope.kedi",
        "/tmp/scope.kedi",
        lsp.Position(line=2, character=3),
    )

    assert definition is None


# ---------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------


def test_formatting_trims_trailing_whitespace():
    src = "@p():   \n  = ok   \n\n\n"
    edits = lsp_document_module.compute_formatting(src)
    assert edits
    new = edits[0].new_text
    assert "   \n" not in new
    assert new.endswith("\n")
    assert new.count("\n\n\n") == 0


def test_formatting_is_noop_for_clean_source():
    src = "@p():\n  = ok\n"
    edits = lsp_document_module.compute_formatting(src)
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
