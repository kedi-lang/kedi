"""Smoke tests for the ``kedi-lsp`` scaffold.

These exercise the feature functions directly (without running the
stdio server) so the tests are deterministic and fast.
"""

from __future__ import annotations

import ast
import io
import json
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, cast

from lsprotocol import types as lsp

from kedi.lsp import features
from kedi.lsp.python_virtual import compute_python_virtual_document


def _decode_semantic_tokens(data: Sequence[int]) -> list[tuple[int, int, int, int, int]]:
    out: list[tuple[int, int, int, int, int]] = []
    line = 0
    col = 0
    for d_line, d_col, length, token_type, token_mods in zip(
        data[0::5], data[1::5], data[2::5], data[3::5], data[4::5], strict=False
    ):
        line += d_line
        col = col + d_col if d_line == 0 else d_col
        out.append((line, col, length, token_type, token_mods))
    return out


def _hover_value(hover: lsp.Hover | None) -> str:
    assert hover is not None
    assert isinstance(hover.contents, lsp.MarkupContent)
    return hover.contents.value


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


def test_diagnostics_accept_module_import_export_syntax(tmp_path):
    profiles = tmp_path / "profiles.kedi"
    profiles.write_text("> export:\n  xd\n", encoding="utf-8")
    main = tmp_path / "main.kedi"
    src = "> import: profiles (file name without .kedi)\n\n> export:\n  xd\n  get_proc\n"
    main.write_text(src, encoding="utf-8")
    assert features.compute_diagnostics(src, str(main)) == []


def test_diagnostics_report_missing_imported_module(tmp_path):
    main = tmp_path / "main.kedi"
    src = "> import: profiles\n\n= ok\n"
    main.write_text(src, encoding="utf-8")

    diags = features.compute_diagnostics(src, str(main))

    assert len(diags) == 1
    diag = diags[0]
    assert diag.code == "K-IMPORT-NOT-FOUND"
    assert "profiles" in diag.message
    assert diag.range.start.line == 0


def test_diagnostics_accept_internal_registry_import(tmp_path):
    main = tmp_path / "main.kedi"
    src = "> import: this\n\n= ok\n"
    main.write_text(src, encoding="utf-8")

    assert features.compute_diagnostics(src, str(main)) == []


# ---------------------------------------------------------------------
# Hover
# ---------------------------------------------------------------------


def test_hover_on_procedure_header_shows_signature():
    src = "@translate(text: str) -> str:\n  = <text>\n"
    hover = features.compute_hover(
        source=src, source_path="example.kedi", pos=lsp.Position(line=0, character=3)
    )
    assert hover is not None
    body = _hover_value(hover)
    assert "translate" in body
    assert "text: str" in body
    assert "str" in body


def test_hover_on_defaulted_procedure_header_shows_defaults():
    src = '@translate(text: str, style = `"brief"`) -> str:\n  = <text>\n'
    hover = features.compute_hover(
        source=src, source_path="example.kedi", pos=lsp.Position(line=0, character=3)
    )
    assert hover is not None
    body = _hover_value(hover)
    assert "text: str" in body
    assert 'style: str = `"brief"`' in body


def test_hover_on_untyped_procedure_header_defaults_param_to_str():
    src = "@translate(text) -> str:\n  = <text>\n"
    hover = features.compute_hover(
        source=src, source_path="example.kedi", pos=lsp.Position(line=0, character=3)
    )
    assert hover is not None
    body = _hover_value(hover)
    assert "text: str" in body


def test_hover_on_procedure_header_includes_docstring():
    src = (
        "@translate(text: str) -> str:\n"
        "  ###\n"
        "  Translate the provided text.\n"
        "  Keep the answer concise.\n"
        "  ###\n"
        "  = <text>\n"
    )
    hover = features.compute_hover(
        source=src, source_path="example.kedi", pos=lsp.Position(line=0, character=3)
    )
    assert hover is not None
    body = _hover_value(hover)
    assert "translate" in body
    assert "Translate the provided text." in body
    assert "Keep the answer concise." in body


def test_hover_on_docstring_comment_uses_procedure_docstring():
    src = (
        "@translate(text: str) -> str:\n  ###\n  Translate the provided text.\n  ###\n  = <text>\n"
    )
    hover = features.compute_hover(
        source=src, source_path="example.kedi", pos=lsp.Position(line=2, character=6)
    )
    assert hover is not None
    assert "translate" in _hover_value(hover)
    assert "Translate the provided text." in _hover_value(hover)


def test_hover_on_regular_block_comment_stays_none():
    src = "@translate(text: str) -> str:\n  = <text>\n  ###\n  Not a docstring.\n  ###\n"
    hover = features.compute_hover(
        source=src, source_path="example.kedi", pos=lsp.Position(line=3, character=6)
    )
    assert hover is None


def test_python_ranges_exclude_header_inline_defaults():
    src = '@say_hi(name: str = `get_name()`) -> str:\n  = `f"Hello, {name}!"`\n'
    ranges = features.compute_python_ranges(src)
    assert ranges == [
        {
            "kind": "inline",
            "range": {
                "start": {"line": 1, "character": 5},
                "end": {"line": 1, "character": 22},
            },
            "text": 'f"Hello, {name}!"',
        }
    ]


def test_hover_on_header_default_expression_defers_to_python():
    src = '@say_hi(name: str = `get_name()`) -> str:\n  = `f"Hello, {name}!"`\n'
    hover = features.compute_hover(
        source=src,
        source_path="example.kedi",
        pos=lsp.Position(line=0, character=24),
    )
    assert hover is None


def test_hover_on_nested_procedure_header_shows_signature():
    src = "@foo():\n  @bar(name: str) -> str:\n    = <name>\n  = done\n"
    hover = features.compute_hover(
        source=src, source_path="example.kedi", pos=lsp.Position(line=1, character=4)
    )
    assert hover is not None
    body = _hover_value(hover)
    assert "bar" in body
    assert "name: str" in body
    assert "str" in body


def test_hover_on_nested_text_segment_works():
    src = "@foo():\n  @bar():\n    Hello world\n    = Done\n  = dn\n"
    hover = features.compute_hover(
        source=src, source_path="example.kedi", pos=lsp.Position(line=2, character=8)
    )
    assert hover is not None
    body = _hover_value(hover)
    assert "Text segment" in body
    assert "Hello world" in body


def test_hover_and_definition_on_imported_procedure(tmp_path):
    profiles = tmp_path / "profiles.kedi"
    profiles.write_text(
        "@make_profile(name: str) -> str:\n  = <name>\n\n> export:\n  make_profile\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    src = '> import: profiles\n\n= <make_profile(`"Ada"`)>\n'
    main.write_text(src, encoding="utf-8")

    hover = features.compute_hover(
        source=src,
        source_path=str(main),
        pos=lsp.Position(line=2, character=4),
    )
    assert hover is not None
    assert "make_profile" in _hover_value(hover)
    assert "name: str" in _hover_value(hover)

    location = features.compute_definition(
        source=src,
        uri=main.resolve().as_uri(),
        source_path=str(main),
        pos=lsp.Position(line=2, character=4),
    )
    assert location is not None
    assert location.uri == profiles.resolve().as_uri()
    assert location.range.start.line == 0


def test_hover_on_imported_type_in_annotation(tmp_path):
    profiles = tmp_path / "profiles.kedi"
    profiles.write_text(
        "~Profile(name: str)\n\n> export:\n  Profile\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    src = '> import: profiles\n\n[p: Profile] = `Profile(name="Ada")`\n'
    main.write_text(src, encoding="utf-8")

    hover = features.compute_hover(
        source=src,
        source_path=str(main),
        pos=lsp.Position(line=2, character=5),
    )
    assert hover is not None
    assert "Profile" in _hover_value(hover)
    assert "not defined in this module" not in _hover_value(hover)

    location = features.compute_definition(
        source=src,
        uri=main.resolve().as_uri(),
        source_path=str(main),
        pos=lsp.Position(line=2, character=5),
    )
    assert location is not None
    assert location.uri == profiles.resolve().as_uri()
    assert location.range.start.line == 0


def test_hover_on_imported_value(tmp_path):
    profiles = tmp_path / "profiles.kedi"
    profiles.write_text(
        "[xd: int] = `1`\n\n> export:\n  xd\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    src = "> import: profiles\n\n= <xd>\n"
    main.write_text(src, encoding="utf-8")

    hover = features.compute_hover(
        source=src,
        source_path=str(main),
        pos=lsp.Position(line=2, character=3),
    )
    assert hover is not None
    assert "Imported value" in _hover_value(hover)
    assert "`xd: int`" in _hover_value(hover)

    location = features.compute_definition(
        source=src,
        uri=main.resolve().as_uri(),
        source_path=str(main),
        pos=lsp.Position(line=2, character=3),
    )
    assert location is not None
    assert location.uri == profiles.resolve().as_uri()
    assert location.range.start.line == 0


def test_hover_and_definition_on_imported_module_name_uses_module_docstring(tmp_path):
    profiles = tmp_path / "profiles.kedi"
    profiles.write_text(
        "# Profile utilities\n"
        "# Used across the workspace.\n\n"
        "@make_profile(name: str) -> str:\n"
        "  = <name>\n\n"
        "> export:\n"
        "  make_profile\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    src = '> import: profiles\n\n= <make_profile(`"Ada"`)>\n'
    main.write_text(src, encoding="utf-8")

    hover = features.compute_hover(
        source=src,
        source_path=str(main),
        pos=lsp.Position(line=0, character=12),
    )
    assert hover is not None
    assert "**Module** `profiles`" in _hover_value(hover)
    assert "Profile utilities" in _hover_value(hover)
    assert "Used across the workspace." in _hover_value(hover)

    location = features.compute_definition(
        source=src,
        uri=main.resolve().as_uri(),
        source_path=str(main),
        pos=lsp.Position(line=0, character=12),
    )
    assert location is not None
    assert location.uri == profiles.resolve().as_uri()
    assert location.range.start.line == 0


def test_hover_and_definition_on_internal_registry_module_name():
    from kedi.module_registry import internal_module_path

    internal_path = internal_module_path("this")
    assert internal_path is not None

    src = "> import: this\n\n= ok\n"
    hover = features.compute_hover(
        source=src,
        source_path="example.kedi",
        pos=lsp.Position(line=0, character=11),
    )
    assert hover is not None
    assert "**Module** `this`" in _hover_value(hover)

    location = features.compute_definition(
        source=src,
        uri="file:///tmp/example.kedi",
        source_path="example.kedi",
        pos=lsp.Position(line=0, character=11),
    )
    assert location is not None
    assert location.uri == internal_path.as_uri()
    assert location.range.start.line == 0


def test_hover_on_imported_module_name_ignores_nonleading_comment_as_docstring(tmp_path):
    profiles = tmp_path / "profiles.kedi"
    profiles.write_text(
        "@make_profile(name: str) -> str:\n"
        "  = <name>\n\n"
        "# Not a module docstring.\n"
        "> export:\n"
        "  make_profile\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    src = '> import: profiles\n\n= <make_profile(`"Ada"`)>\n'
    main.write_text(src, encoding="utf-8")

    hover = features.compute_hover(
        source=src,
        source_path=str(main),
        pos=lsp.Position(line=0, character=12),
    )
    assert hover is not None
    assert "**Module** `profiles`" in _hover_value(hover)
    assert "Not a module docstring." not in _hover_value(hover)


def test_hover_and_definition_on_exported_procedure_name():
    src = "@make_profile(name: str) -> str:\n  = <name>\n\n> export:\n  make_profile\n"

    hover = features.compute_hover(
        source=src,
        source_path="example.kedi",
        pos=lsp.Position(line=4, character=4),
    )
    assert hover is not None
    assert "make_profile" in _hover_value(hover)
    assert "name: str" in _hover_value(hover)

    location = features.compute_definition(
        source=src,
        uri="file:///tmp/example.kedi",
        source_path="example.kedi",
        pos=lsp.Position(line=4, character=4),
    )
    assert location is not None
    assert location.range.start.line == 0


def test_hover_and_definition_on_exported_value_name():
    src = "[profile_id: int] = `1`\n\n> export:\n  profile_id\n"

    hover = features.compute_hover(
        source=src,
        source_path="example.kedi",
        pos=lsp.Position(line=3, character=4),
    )
    assert hover is not None
    assert "Module value" in _hover_value(hover)
    assert "profile_id" in _hover_value(hover)

    location = features.compute_definition(
        source=src,
        uri="file:///tmp/example.kedi",
        source_path="example.kedi",
        pos=lsp.Position(line=3, character=4),
    )
    assert location is not None
    assert location.range.start.line == 0


def test_hover_on_input_segment():
    src = "@p(name):\n  Hello, <name>!\n  = done\n"
    hover = features.compute_hover(
        source=src, source_path="example.kedi", pos=lsp.Position(line=1, character=11)
    )
    assert hover is not None
    body = _hover_value(hover)
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
    body = _hover_value(hover)
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
        body = _hover_value(hover)
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
        body = _hover_value(hover)
        assert name in body
        assert origin in body
        assert "not defined" not in body


def test_hover_on_type_definition_shows_field_defaults():
    src = "~Person(name: str, salary: int = `0`)\n= ok\n"
    hover = features.compute_hover(
        source=src,
        source_path="example.kedi",
        pos=lsp.Position(line=0, character=1),
    )
    assert hover is not None
    body = _hover_value(hover)
    assert "salary: int = `0`" in body


def test_hover_on_type_definition_defaults_untyped_field_to_str():
    src = "~Person(name)\n= ok\n"
    hover = features.compute_hover(
        source=src,
        source_path="example.kedi",
        pos=lsp.Position(line=0, character=1),
    )
    assert hover is not None
    assert "name: str" in _hover_value(hover)


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
    assert "not defined in this module" in _hover_value(hover)
    assert "resolved at runtime" in _hover_value(hover)


def test_hover_on_unparseable_returns_none():
    hover = features.compute_hover(
        source="@p(:\n  ???\n",
        source_path="example.kedi",
        pos=lsp.Position(line=0, character=1),
    )
    assert hover is None


def test_hover_on_prelude_defined_python_type_is_not_runtime_unknown():
    src = (
        "```\n"
        "from pydantic import BaseModel, Field\n"
        "\n"
        "class Person(BaseModel):\n"
        '    name: str = Field(examples=["Ayşe", "Ali", "Berk"])\n'
        "    age: int = Field(gt=50)\n"
        "```\n\n"
        "Fake me a [person: Person]\n"
        "\n"
        "= <`person`>\n"
    )
    hover = features.compute_hover(
        source=src,
        source_path="example.kedi",
        pos=lsp.Position(line=8, character=19),
    )
    assert hover is not None
    body = _hover_value(hover)
    assert "Person" in body
    assert "defined in the Python prelude" in body
    assert "not defined in this module" not in body


def test_hover_on_top_level_python_block_type_is_not_runtime_unknown():
    src = (
        "```\n"
        "import this\n"
        "```\n\n"
        "```\n"
        "from pydantic import BaseModel, Field\n"
        "\n"
        "class Person(BaseModel):\n"
        '    name: str = Field(examples=["Ayşe", "Ali", "Berk"])\n'
        "    age: int = Field(gt=50)\n"
        "```\n\n"
        "Fake me a [person: Person]\n"
        "\n"
        "= <`person`>\n"
    )
    hover = features.compute_hover(
        source=src,
        source_path="example.kedi",
        pos=lsp.Position(line=12, character=19),
    )
    assert hover is not None
    body = _hover_value(hover)
    assert "Person" in body
    assert "defined in a top-level Python block" in body
    assert "not defined in this module" not in body


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


def test_semantic_tokens_docstring_comment_uses_variable_token():
    src = (
        "@translate(text: str) -> str:\n  ###\n  Translate the provided text.\n  ###\n  = <text>\n"
    )
    tokens = features.compute_semantic_tokens_full(src, "/tmp/x.kedi")
    decoded = _decode_semantic_tokens(tokens.data)
    variable_type = features.SEMANTIC_TOKEN_TYPES.index("variable")
    assert any(line == 1 and token_type == variable_type for line, _, _, token_type, _ in decoded)
    assert any(line == 2 and token_type == variable_type for line, _, _, token_type, _ in decoded)


def test_semantic_tokens_regular_block_comment_stays_comment_token():
    src = "@translate(text: str) -> str:\n  = <text>\n  ###\n  Not a docstring.\n  ###\n"
    tokens = features.compute_semantic_tokens_full(src, "/tmp/x.kedi")
    decoded = _decode_semantic_tokens(tokens.data)
    comment_type = features.SEMANTIC_TOKEN_TYPES.index("comment")
    assert any(line == 2 and token_type == comment_type for line, _, _, token_type, _ in decoded)


def test_semantic_tokens_cover_module_directives():
    src = "> import: profiles\n\n> export:\n  xd\n  get_proc\n"
    tokens = features.compute_semantic_tokens_full(src, "/tmp/x.kedi")
    decoded = _decode_semantic_tokens(tokens.data)

    operator_type = features.SEMANTIC_TOKEN_TYPES.index("operator")
    namespace_type = features.SEMANTIC_TOKEN_TYPES.index("namespace")
    variable_type = features.SEMANTIC_TOKEN_TYPES.index("variable")

    assert any(line == 0 and token_type == operator_type for line, _, _, token_type, _ in decoded)
    assert any(line == 0 and token_type == namespace_type for line, _, _, token_type, _ in decoded)
    assert any(line == 2 and token_type == operator_type for line, _, _, token_type, _ in decoded)
    assert any(line == 3 and token_type == variable_type for line, _, _, token_type, _ in decoded)
    assert any(line == 4 and token_type == variable_type for line, _, _, token_type, _ in decoded)


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


def test_lsp_feature_helper_edges(tmp_path, monkeypatch):
    from kedi.lang.ast import (
        AssignBlockStmt,
        AssignStmt,
        AutoSpec,
        BacktickLineStmt,
        CallArg,
        CallSegment,
        EvalDataset,
        EvalMetric,
        EvalSuite,
        InputSegment,
        ModuleExport,
        ModuleImport,
        OptimizeSpan,
        OutputSegment,
        ProcDefStmt,
        ProcedureDef,
        Program,
        ProgramWithValidation,
        PythonBlock,
        PythonExprSegment,
        ReturnBlockStmt,
        ReturnStmt,
        Span,
        Stmt,
        TemplateLine,
        TestCase,
        TestSuite,
        TextSegment,
        TypeApply,
        TypeDef,
        TypeExpr,
        TypeField,
        TypeRef,
        TypeUnion,
    )
    from kedi.lang.parser import parse_program

    def span(line: int = 1, start_col: int = 1, end_col: int = 80) -> Span:
        return Span(
            start_byte=0,
            end_byte=max(end_col - start_col, 0),
            start_line=line,
            start_col=start_col,
            end_line=line,
            end_col=end_col,
            source_path=str(tmp_path / "x.kedi"),
        )

    pos = lsp.Position(line=0, character=4)
    assert "builtins" in features._build_lsp_type_registry()["str"]
    monkeypatch.setattr(features, "_typing_extensions", None)
    assert "str" in features._build_lsp_type_registry()
    monkeypatch.undo()
    assert features._program_path(Program(source_map=SimpleNamespace(documents=[]))) is None
    mapped_prog = Program(
        source_map=SimpleNamespace(documents=[SimpleNamespace(path=str(tmp_path / "m.kedi"))])
    )
    assert features._program_path(mapped_prog) == (tmp_path / "m.kedi").resolve()
    assert features._assignment_name_span(AssignStmt(name="x")) is None
    assert features._is_public_export_name("public")
    assert not features._is_public_export_name("")
    assert not features._is_public_export_name("_private")

    source_map = SimpleNamespace(get_line=lambda path, line: f"{path}:{line}")
    mapped_line_prog = Program(loc=span(), source_map=source_map)
    assert features._program_line_text(mapped_line_prog, 7) == f"{tmp_path / 'x.kedi'}:7"
    assert features._program_line_text(Program(source_text="one"), 2) is None
    assert features._program_line_text(Program(source_map=source_map), 1) is None
    assert features._program_line_text(Program(), 1) is None
    assert features._extract_module_docstring("") is None
    assert features._extract_module_docstring("###\n###\n") is None
    assert features._extract_module_docstring("# one\n# two\n= ok\n") == "one\ntwo"
    assert features._extract_module_docstring("# one\n# two\n") == "one\ntwo"

    class FakeNode:
        def __init__(self, type_: str, start: int = 0, end: int = 3):
            self.type = type_
            self.start_byte = start
            self.end_byte = end
            self.is_named = True

    monkeypatch.setattr(
        features,
        "_ts_parse",
        lambda source: SimpleNamespace(
            root_node=SimpleNamespace(children=[FakeNode("block_comment")])
        ),
    )
    assert features._extract_module_docstring("###") is None
    monkeypatch.undo()

    assert features._import_name_span(Program(), ModuleImport("profiles")) is None
    no_line_prog = Program(imports=[ModuleImport("profiles", loc=span())])
    assert features._import_name_span(no_line_prog, no_line_prog.imports[0]) is None
    wrong_line_prog = Program(
        source_text="> import: other\n",
        imports=[ModuleImport("profiles", loc=span())],
    )
    assert features._import_name_span(wrong_line_prog, wrong_line_prog.imports[0]) is None

    export_prog = Program(
        source_text="> export:\n  *\n  !bad\n  Name\n",
        exports=[ModuleExport(["Name"], loc=Span(0, 32, 1, 1, 4, 7))],
    )
    assert (
        features._exported_name_at_position(
            Program(exports=[ModuleExport(["Name"], loc=None)]), lsp.Position(line=0, character=0)
        )
        is None
    )
    assert (
        features._exported_name_at_position(export_prog, lsp.Position(line=10, character=0)) is None
    )
    assert (
        features._exported_name_at_position(export_prog, lsp.Position(line=0, character=3)) is None
    )
    assert (
        features._exported_name_at_position(export_prog, lsp.Position(line=1, character=3)) is None
    )
    assert (
        features._exported_name_at_position(export_prog, lsp.Position(line=2, character=3)) is None
    )
    hit = features._exported_name_at_position(export_prog, lsp.Position(line=3, character=3))
    assert hit is not None and hit[0] == "Name"
    no_source_export = Program(exports=[ModuleExport(["Name"], loc=Span(0, 32, 1, 1, 4, 7))])
    assert (
        features._exported_name_at_position(no_source_export, lsp.Position(line=3, character=3))
        is None
    )

    class FindlessLine(str):
        def find(self, sub, start=0, end=None):  # noqa: ANN001
            return (
                -1 if sub == "Name" else super().find(sub, start, len(self) if end is None else end)
            )

    monkeypatch.setattr(
        features, "_program_line_text", lambda *args, **kwargs: FindlessLine("  Name")
    )
    assert (
        features._exported_name_at_position(export_prog, lsp.Position(line=3, character=3)) is None
    )
    monkeypatch.undo()

    assert features._load_imported_surface(tmp_path / "missing.kedi", set()).modules == {}
    assert (
        features._load_imported_surface(
            tmp_path / "missing.kedi", {tmp_path / "missing.kedi"}
        ).modules
        == {}
    )
    bad_module = tmp_path / "bad.kedi"
    bad_module.write_text("[bad\n", encoding="utf-8")
    assert features._load_imported_surface(bad_module, set()).modules == {}
    exported_module = tmp_path / "exported.kedi"
    exported_module.write_text(
        "> import: missing_nested\n"
        "~Public(name)\n"
        "~_Private(name)\n"
        "[value: int] = `1`\n"
        "[_hidden: int] = `2`\n"
        "@proc() -> str:\n"
        "  = ok\n"
        "> export: *\n",
        encoding="utf-8",
    )
    surface = features._load_imported_surface(exported_module, set(), "exported")
    assert "exported" in surface.modules
    assert "Public" in surface.types
    assert "_Private" not in surface.types
    assert "value" in surface.values
    assert "_hidden" not in surface.values
    assert "proc" in surface.procedures
    no_module_surface = features._load_imported_surface(exported_module, set(), None)
    assert no_module_surface.modules == {}
    nested_module = tmp_path / "nested.kedi"
    nested_module.write_text(
        "@nested_proc() -> str:\n  = ok\n> export:\n  nested_proc\n", encoding="utf-8"
    )
    parent_module = tmp_path / "parent.kedi"
    parent_module.write_text("> import: nested\n> export:\n  nested_proc\n", encoding="utf-8")
    nested_surface = features._load_imported_surface(parent_module, set(), "parent")
    assert "nested_proc" in nested_surface.procedures
    missing_export_module = tmp_path / "missing_export.kedi"
    missing_export_module.write_text("> export:\n  absent\n", encoding="utf-8")
    assert features._load_imported_surface(missing_export_module, set()).procedures == {}

    missing_import_prog = Program(imports=[ModuleImport("missing", loc=None)])
    assert features._missing_import_diagnostics(missing_import_prog) == []
    unresolved_prog = parse_program(
        "> import: no_such_module\n= ok\n", source_path=str(tmp_path / "main.kedi")
    )
    assert features._imported_surface_for_program(unresolved_prog).modules == {}

    type_ref = TypeRef(name="Person", loc=Span(0, 6, 1, 1, 1, 7))
    type_apply = TypeApply(name="list", args=[type_ref], loc=Span(0, 12, 1, 1, 1, 13))
    type_union = TypeUnion(left=type_apply, right=TypeRef(name="str", loc=span()), loc=span())
    assert features._type_expr_to_str(None) == "None"
    assert features._type_expr_to_str(type_ref) == "Person"
    assert features._type_expr_to_str(type_apply) == "list[Person]"
    assert features._type_expr_to_str(type_union) == "list[Person] | str"
    assert "TypeExpr" in features._type_expr_to_str(TypeExpr())

    assert features._prelude_defined_names(None) == set()
    assert features._prelude_defined_names("class Broken(:\n") == set()
    assert {
        "Model",
        "fn",
        "async_fn",
        "path",
        "alias",
        "x",
        "y",
    } <= features._prelude_defined_names(
        "import os.path as path\n"
        "from typing import Any as alias\n"
        "class Model: pass\n"
        "def fn(): pass\n"
        "async def async_fn(): pass\n"
        "x = y = 1\n"
    )

    proc = ProcedureDef(
        name="p",
        params=["x"],
        body=[],
        param_type_annotations={"x": "int"},
        return_type_annotation="str",
        loc=span(),
    )
    assert "@p(x: int)" in features._proc_signature_markdown(proc)
    assert "AI-generated" in features._proc_signature_markdown(
        ProcedureDef(name="auto", params=[], auto_spec=AutoSpec("make it", loc=span()), loc=span())
    )
    assert "optimization spans" in features._proc_signature_markdown(
        ProcedureDef(
            name="opt", params=[], optimize_spans=[OptimizeSpan("prompt", loc=span())], loc=span()
        )
    )
    assert "doc" in features._proc_signature_markdown(
        ProcedureDef(name="doc", params=[], docstring="doc", loc=span())
    )

    described_nodes = [
        proc,
        TypeDef("Person", [TypeField("name", loc=span())], loc=span()),
        TypeField("name", default_expr="'Ada'", loc=span()),
        TemplateLine(segments=[TextSegment(text="hello", loc=span())], loc=span()),
        AssignStmt(name="x", type_annotation="int", loc=span()),
        AssignBlockStmt(name="x", type_annotation="int", loc=span()),
        ReturnStmt(loc=span()),
        ReturnBlockStmt(loc=span()),
        BacktickLineStmt(code="x = 1", loc=span()),
        PythonBlock(code="x = 1", loc=span()),
        AutoSpec("make it", loc=span()),
        OptimizeSpan("prompt", loc=span()),
        TextSegment(text="x" * 50, loc=span()),
        InputSegment(name="x", loc=span()),
        OutputSegment(name="x", type_annotation="int", loc=span()),
        CallSegment(name="p", loc=span()),
        PythonExprSegment(code="x", loc=span()),
        EvalSuite("p", loc=span()),
        TestSuite("p", loc=span()),
        EvalDataset("train", "[]", loc=span()),
        EvalMetric("score", "return 1", loc=span()),
    ]
    assert all(features._describe_ast_node(node) for node in described_nodes)
    assert features._describe_ast_node(Stmt(loc=span())) is None

    assert features._generic_hover(Program(), pos) is None
    assert (
        features._generic_hover(
            SimpleNamespace(procedures=[], type_defs=[], top_level_statements=[]), pos
        )
        is None
    )
    assert (
        features._generic_hover(
            SimpleNamespace(procedures=[], type_defs=[], top_level_statements=[Stmt(loc=span())]),
            pos,
        )
        is None
    )

    nested_arg = CallArg(loc=span())
    setattr(nested_arg, "__call_segment__", CallSegment(name="inner", args=[], loc=span()))
    rendered_arg = CallArg(loc=span())
    setattr(
        rendered_arg,
        "__render_segments__",
        [CallSegment(name="rendered", args=[], loc=span())],
    )
    outer_call = CallSegment(name="outer", args=[nested_arg, rendered_arg], loc=span())
    calls: list[Span] = []
    features._collect_call_refs(outer_call, "inner", calls)
    features._collect_call_refs(outer_call, "rendered", calls)
    assert len(calls) == 2
    inputs: list[Span] = []
    nested_input_arg = CallArg(loc=span())
    setattr(nested_input_arg, "__python_segment__", InputSegment(name="needle", loc=span()))
    rendered_input_arg = CallArg(loc=span())
    setattr(
        rendered_input_arg,
        "__render_segments__",
        [InputSegment(name="needle", loc=span())],
    )
    features._collect_input_refs(
        CallSegment(name="outer", args=[nested_input_arg, rendered_input_arg], loc=span()),
        "needle",
        inputs,
    )
    assert len(inputs) == 2

    refs: list[Span] = []
    features._collect_type_refs_in_expr(type_union, "Person", refs)
    features._collect_type_refs_in_expr(type_apply, "list", refs)
    features._collect_type_refs_in_stmt(
        OutputSegment(name="x", type_expr=type_ref, loc=span()), "Person", refs
    )
    features._collect_type_refs_in_type_def(
        TypeDef("Box", [TypeField("item", type_expr=type_ref)]), "Person", refs
    )
    assert refs

    assert features._position_in_proc_name(proc, lsp.Position(line=1, character=1)) is False
    assert features._find_local_declaration(None, "x") is None
    local_proc = ProcedureDef(
        "local",
        [],
        body=[
            AssignStmt(name="x", loc=span()),
            TemplateLine(segments=[OutputSegment(name="y", loc=span())], loc=span()),
        ],
    )
    assert features._find_local_declaration(local_proc, "x") is not None
    assert features._find_local_declaration(local_proc, "y") is not None
    assert features._find_local_declaration(local_proc, "z") is None

    monkeypatch.setattr(
        features, "_param_decls", lambda proc: [("miss", span(1, 20, 25)), ("x", span())]
    )
    assert features._resolve_in_procedure(proc, pos, {}, {}, {}) is not None
    monkeypatch.undo()

    assert (
        features._resolve_in_type_expr(
            TypeUnion(left=type_ref, right=TypeRef(name="Other", loc=span()), loc=span()),
            pos,
            {"Person": TypeDef("Person", [], loc=span())},
        )
        is not None
    )
    assert (
        features._resolve_in_type_expr(
            TypeApply(name="list", args=[type_ref], loc=span()),
            lsp.Position(line=0, character=4),
            {"Person": TypeDef("Person", [], loc=span())},
        )
        is not None
    )
    assert (
        features._resolve_in_stmt(
            AssignBlockStmt(name="x", type_expr=type_ref, loc=span()),
            pos,
            None,
            {},
            {"Person": TypeDef("Person", [], loc=span())},
            {},
        )
        is not None
    )
    assert (
        features._resolve_in_segment(
            OutputSegment(name="x", type_expr=type_ref, loc=span()),
            pos,
            None,
            {},
            {"Person": TypeDef("Person", [], loc=span())},
            {},
        )
        is not None
    )
    field_ref = features._resolve_in_type_def(
        TypeDef("Person", [TypeField("name", loc=span(1, 9, 13))], loc=span()),
        lsp.Position(line=0, character=8),
        {},
    )
    assert field_ref is not None and field_ref.kind == "type_field"
    assert (
        features._resolve_in_type_def(
            TypeDef("Person", [TypeField("name", loc=span(1, 9, 13))], loc=span()),
            lsp.Position(line=0, character=20),
            {},
        )
        is None
    )
    assert (
        features._resolve_in_type_expr(
            TypeUnion(
                left=TypeRef(name="Nope", loc=span(1, 20, 24)),
                right=TypeRef(name="Other", loc=span(1, 30, 35)),
                loc=span(),
            ),
            pos,
            {},
        )
        is None
    )
    assert (
        features._resolve_in_type_expr(
            TypeApply(name="list", args=[TypeRef(name="Other", loc=span(1, 20, 25))], loc=span()),
            pos,
            {},
        )
        is None
    )
    assert (
        features._resolve_in_stmt(
            AssignBlockStmt(
                name="x", type_expr=TypeRef(name="Other", loc=span(1, 20, 25)), loc=span()
            ),
            pos,
            None,
            {},
            {},
            {},
        )
        is None
    )

    resolve_arg = CallArg(loc=span())
    setattr(resolve_arg, "__call_segment__", InputSegment(name="resolved", loc=span()))
    resolved = features._resolve_in_segment(
        CallSegment(name="outer", args=[resolve_arg], loc=span()),
        lsp.Position(line=0, character=10),
        None,
        {},
        {},
        {},
    )
    assert resolved is not None and resolved.name == "resolved"
    rendered_resolve_arg = CallArg(loc=span())
    setattr(
        rendered_resolve_arg,
        "__render_segments__",
        [InputSegment(name="rendered", loc=span())],
    )
    rendered_resolved = features._resolve_in_segment(
        CallSegment(name="outer", args=[rendered_resolve_arg], loc=span()),
        lsp.Position(line=0, character=10),
        None,
        {},
        {},
        {},
    )
    assert rendered_resolved is not None and rendered_resolved.name == "rendered"
    missing_inner_arg = CallArg(loc=span())
    setattr(missing_inner_arg, "__call_segment__", InputSegment(name="miss", loc=span(1, 20, 25)))
    missing_render_arg = CallArg(loc=span())
    setattr(
        missing_render_arg,
        "__render_segments__",
        [InputSegment(name="miss", loc=span(1, 20, 25))],
    )
    assert (
        features._resolve_in_segment(
            CallSegment(name="outer", args=[missing_inner_arg, missing_render_arg], loc=span()),
            lsp.Position(line=0, character=10),
            None,
            {},
            {},
            {},
        )
        is None
    )
    assert (
        features._resolve_in_segment(
            OutputSegment(
                name="x", type_expr=TypeRef(name="Other", loc=span(1, 20, 25)), loc=span()
            ),
            pos,
            None,
            {},
            {},
            {},
        )
        is None
    )

    program = ProgramWithValidation(
        procedures=[
            ProcedureDef(
                "outer",
                [],
                body=cast(
                    Any,
                    [
                        None,
                        ProcDefStmt(
                            proc=ProcedureDef("inner", [], body=[], loc=span()), loc=span()
                        ),
                        TemplateLine(segments=[outer_call], loc=span()),
                    ],
                ),
                auto_spec=AutoSpec("auto", loc=span()),
                optimize_spans=[OptimizeSpan("prompt", [TemplateLine(loc=span())], loc=span())],
                loc=span(),
            )
        ],
        type_defs=[TypeDef("Box", [TypeField("item", loc=span())], loc=span())],
        tests=[TestSuite("outer", [TestCase("basic", "assert True", loc=span())], loc=span())],
        evals=[
            EvalSuite(
                "outer",
                datasets={"train": EvalDataset("train", "[]", loc=span())},
                test_datasets={"holdout": EvalDataset("holdout", "[]", loc=span())},
                metrics=[EvalMetric("score", "return 1", loc=span())],
                loc=span(),
            )
        ],
    )
    setattr(
        program,
        "top_level_statements",
        [
            TemplateLine(segments=[InputSegment(name="x", loc=span())], loc=span()),
            AssignStmt(name="x", expr=[TextSegment(text="value", loc=span())], loc=span()),
        ],
    )
    assert features._smallest_enclosing(program, pos) is not None

    ref_prog = Program(
        procedures=[
            ProcedureDef(
                name="target",
                params=[],
                body=[TemplateLine(segments=[CallSegment(name="target", loc=span())], loc=span())],
                loc=span(),
            ),
            ProcedureDef(
                name="typed",
                params=["x"],
                body=[AssignStmt(name="v", type_expr=type_ref, loc=span())],
                param_type_exprs={"x": type_ref, "skip": None},
                return_type_expr=type_ref,
                loc=span(),
            ),
        ],
        toplevel_return=ReturnStmt(segments=[CallSegment(name="target", loc=span())], loc=span()),
        toplevel_body=[
            TemplateLine(segments=[CallSegment(name="target", loc=span())], loc=span()),
            AssignStmt(name="v", type_expr=type_ref, loc=span()),
        ],
        type_defs=[TypeDef("Person", [TypeField("item", type_expr=type_ref)], loc=span())],
    )
    duplicate_span = span()
    proc_refs = features._find_all_references(
        ref_prog,
        features.SymbolRef("procedure", "target", duplicate_span, None, duplicate_span),
        include_declaration=True,
    )
    type_refs = features._find_all_references(
        ref_prog,
        features.SymbolRef("type", "Person", duplicate_span, None, duplicate_span),
        include_declaration=True,
    )
    assert len(proc_refs) >= 2
    assert len({(s.start_line, s.start_col) for s in type_refs}) == 1
    assert (
        features._find_all_references(
            ref_prog,
            features.SymbolRef("local", "x", None, None, duplicate_span),
            include_declaration=False,
        )
        == []
    )

    proc_symbol = features._proc_symbol(program.procedures[0])
    assert proc_symbol.children
    assert any(child.name == "inner" for child in proc_symbol.children)
    assert any(child.name == "> optimize: prompt" for child in proc_symbol.children)
    no_loc_opt_symbol = features._proc_symbol(
        ProcedureDef(
            name="no_loc_opt", params=[], optimize_spans=[OptimizeSpan("prompt")], loc=span()
        )
    )
    assert no_loc_opt_symbol.children == []
    eval_symbol = features._eval_suite_symbol(program.evals[0])
    assert eval_symbol.children is not None
    assert [child.name for child in eval_symbol.children] == [
        "> data: train",
        "> test_data: holdout",
        "> metric: score",
    ]
    test_symbol = features._test_suite_symbol(program.tests[0])
    assert test_symbol.children is not None
    assert test_symbol.children[0].name == "basic"

    assert features._empty_range().start.line == 0
    assert features._resolve_overlaps([]) == []
    assert features._resolve_overlaps([(0, -2, 1, 1, 0), (0, 0, 1, 2, 0)])
    assert (
        features._position_in_range(
            lsp.Position(line=2, character=10),
            {"start": {"line": 1, "character": 0}, "end": {"line": 2, "character": 5}},
        )
        is False
    )


def test_lsp_feature_public_edge_paths(tmp_path, monkeypatch):
    from kedi.lang.ast import CallArg, CallSegment, EvalDataset, EvalMetric, EvalSuite, Stmt

    bad = "@broken(:\n"
    assert features.compute_hover(bad, "/tmp/bad.kedi", lsp.Position(line=0, character=1)) is None
    assert (
        features.compute_definition(
            bad, "file:///tmp/bad.kedi", "/tmp/bad.kedi", lsp.Position(line=0, character=1)
        )
        is None
    )
    assert (
        features.compute_references(
            bad, "file:///tmp/bad.kedi", "/tmp/bad.kedi", lsp.Position(line=0, character=1), True
        )
        == []
    )
    assert (
        features.compute_prepare_rename(bad, "/tmp/bad.kedi", lsp.Position(line=0, character=1))
        is None
    )
    assert (
        features.compute_rename(
            bad, "file:///tmp/bad.kedi", "/tmp/bad.kedi", lsp.Position(line=0, character=1), "x"
        )
        is None
    )
    assert (
        features.compute_signature_help(bad, "/tmp/bad.kedi", lsp.Position(line=0, character=1))
        is None
    )
    assert (
        features.compute_inlay_hints(
            bad,
            "/tmp/bad.kedi",
            lsp.Range(
                start=lsp.Position(line=0, character=0), end=lsp.Position(line=1, character=0)
            ),
        )
        == []
    )
    assert features.compute_document_symbols(bad, "/tmp/bad.kedi") == []

    main = tmp_path / "main.kedi"
    missing_src = "> import: missing\n= ok\n"
    main.write_text(missing_src, encoding="utf-8")
    hover = features.compute_hover(missing_src, str(main), lsp.Position(line=0, character=11))
    assert hover is not None and "import target not found" in _hover_value(hover)

    module_src = "~Person(name)\n[x: int] = `1`\n> export:\n  Person\n  x\n"
    assert "Kedi type definition" in _hover_value(
        features.compute_hover(module_src, "/tmp/mod.kedi", lsp.Position(line=3, character=3))
    )
    assert "Module value" in _hover_value(
        features.compute_hover(module_src, "/tmp/mod.kedi", lsp.Position(line=4, character=2))
    )

    producer = tmp_path / "producer.kedi"
    producer.write_text("[value] = ok\n> export:\n  value\n", encoding="utf-8")
    consumer = tmp_path / "consumer.kedi"
    consumer_src = "> import: producer\n> export:\n  value\n= <`value`>\n"
    consumer.write_text(consumer_src, encoding="utf-8")
    hover = features.compute_hover(consumer_src, str(consumer), lsp.Position(line=2, character=3))
    assert hover is not None and "Imported value" in _hover_value(hover)
    same_file_definition = features.compute_definition(
        module_src,
        "file:///tmp/mod.kedi",
        "/tmp/mod.kedi",
        lsp.Position(line=3, character=3),
    )
    assert same_file_definition is not None and same_file_definition.uri == "file:///tmp/mod.kedi"

    unknown_call_src = "= <missing()>\n"
    assert "procedure not found" in _hover_value(
        features.compute_hover(unknown_call_src, "/tmp/x.kedi", lsp.Position(line=0, character=4))
    )
    assert "Local" in _hover_value(
        features.compute_hover(
            "@p():\n  = <unknown>\n", "/tmp/x.kedi", lsp.Position(line=1, character=6)
        )
    )
    assert "Field" in _hover_value(
        features.compute_hover("~Person(name)\n", "/tmp/x.kedi", lsp.Position(line=0, character=9))
    )

    generic = features.compute_hover(
        "@p():\n  plain prompt text\n", "/tmp/x.kedi", lsp.Position(line=1, character=4)
    )
    assert generic is not None and "Text segment" in _hover_value(generic)

    monkeypatch.setattr(
        features,
        "_resolve_symbol_at",
        lambda prog, pos: features.SymbolRef(
            kind="parameter",
            name="x",
            declaration_span=None,
            scope_proc=None,
            cursor_span=features.Span(0, 1, 1, 1, 1, 2),
        ),
    )
    assert "Parameter" in _hover_value(
        features.compute_hover("= ok\n", "/tmp/x.kedi", lsp.Position(line=0, character=0))
    )
    monkeypatch.setattr(
        features,
        "_resolve_symbol_at",
        lambda prog, pos: features.SymbolRef(
            kind="type_field",
            name="field",
            declaration_span=None,
            scope_proc=None,
            cursor_span=features.Span(0, 5, 1, 1, 1, 6),
        ),
    )
    assert "Field" in _hover_value(
        features.compute_hover("= ok\n", "/tmp/x.kedi", lsp.Position(line=0, character=0))
    )
    monkeypatch.setattr(
        features,
        "_resolve_symbol_at",
        lambda prog, pos: features.SymbolRef(
            kind="custom",
            name="thing",
            declaration_span=None,
            scope_proc=None,
            cursor_span=features.Span(0, 5, 1, 1, 1, 6),
        ),
    )
    assert "custom" in _hover_value(
        features.compute_hover("= ok\n", "/tmp/x.kedi", lsp.Position(line=0, character=0))
    )
    monkeypatch.undo()

    value_span = features.Span(0, 5, 1, 1, 1, 6)
    monkeypatch.setattr(
        features,
        "_resolve_symbol_at",
        lambda prog, pos: features.SymbolRef(
            kind="imported_value",
            name="loose",
            declaration_span=value_span,
            scope_proc=None,
            cursor_span=value_span,
        ),
    )
    monkeypatch.setattr(
        features,
        "_imported_surface_for_program",
        lambda prog: features.ImportedSurface(
            values={"loose": features.ImportedValueInfo("loose", value_span)}
        ),
    )
    assert "another Kedi module" in _hover_value(
        features.compute_hover("= ok\n", "/tmp/x.kedi", lsp.Position(line=0, character=0))
    )
    monkeypatch.undo()

    assert (
        features.compute_prepare_rename("= ok\n", "/tmp/x.kedi", lsp.Position(line=0, character=1))
        is None
    )
    assert (
        features.compute_rename(
            "= ok\n", "file:///tmp/x.kedi", "/tmp/x.kedi", lsp.Position(line=0, character=1), "x"
        )
        is None
    )
    assert (
        features.compute_prepare_rename(missing_src, str(main), lsp.Position(line=0, character=11))
        is None
    )
    assert (
        features.compute_prepare_rename(
            "@p(x: int):\n  = <x>\n", "/tmp/x.kedi", lsp.Position(line=0, character=6)
        )
        is None
    )
    assert (
        features.compute_rename(
            "@p(x: int):\n  = <x>\n",
            "file:///tmp/x.kedi",
            "/tmp/x.kedi",
            lsp.Position(line=0, character=6),
            "y",
        )
        is None
    )

    external_span = features.Span(0, 1, 1, 1, 1, 2, source_path=str(tmp_path / "other.kedi"))
    monkeypatch.setattr(
        features,
        "_resolve_symbol_at",
        lambda prog, pos: features.SymbolRef(
            "imported_value", "v", external_span, None, external_span
        ),
    )
    assert (
        features.compute_prepare_rename("= ok\n", "/tmp/x.kedi", lsp.Position(line=0, character=0))
        is None
    )
    assert (
        features.compute_rename(
            "= ok\n", "file:///tmp/x.kedi", "/tmp/x.kedi", lsp.Position(line=0, character=0), "v2"
        )
        is None
    )
    monkeypatch.undo()

    rename_span = features.Span(0, 1, 1, 1, 1, 2)
    monkeypatch.setattr(
        features,
        "_resolve_symbol_at",
        lambda prog, pos: features.SymbolRef("local", "x", rename_span, None, rename_span),
    )
    monkeypatch.setattr(
        features,
        "_find_all_references",
        lambda prog, sym, *, include_declaration: [rename_span],
    )
    rename_edit = features.compute_rename(
        "= ok\n", "file:///tmp/x.kedi", "/tmp/x.kedi", lsp.Position(line=0, character=0), "y"
    )
    assert rename_edit is not None and rename_edit.changes is not None
    assert len(rename_edit.changes["file:///tmp/x.kedi"]) == 1
    monkeypatch.setattr(
        features,
        "_resolve_symbol_at",
        lambda prog, pos: features.SymbolRef("local", "x", None, None, cast(Any, None)),
    )
    rename_without_cursor = features.compute_rename(
        "= ok\n", "file:///tmp/x.kedi", "/tmp/x.kedi", lsp.Position(line=0, character=0), "y"
    )
    assert rename_without_cursor is not None and rename_without_cursor.changes is not None
    assert len(rename_without_cursor.changes["file:///tmp/x.kedi"]) == 1
    monkeypatch.undo()

    type_src = "~Person(name)\n@p(x: Person) -> list[Person]:\n  [y: Person] = <x>\n  Value [z: Person]\n  = <x>\n"
    refs = features.compute_references(
        type_src,
        "file:///tmp/x.kedi",
        "/tmp/x.kedi",
        lsp.Position(line=0, character=2),
        True,
    )
    assert len(refs) >= 4
    local_refs = features.compute_references(
        "@p(x):\n  Value [answer]\n  = <answer>\n",
        "file:///tmp/x.kedi",
        "/tmp/x.kedi",
        lsp.Position(line=2, character=6),
        True,
    )
    assert local_refs

    nested_sig_src = "@inner(x: int) -> int:\n  = `x`\n\n@outer():\n  = <inner(<inner(1)>)>\n"
    sig = features.compute_signature_help(
        nested_sig_src, "/tmp/x.kedi", lsp.Position(line=4, character=15)
    )
    assert sig is not None and sig.active_parameter == 0
    top_sig_src = "@inner(x: int) -> int:\n  = `x`\n\n[x] = <inner(1)>\n"
    top_sig = features.compute_signature_help(
        top_sig_src, "/tmp/x.kedi", lsp.Position(line=3, character=13)
    )
    assert top_sig is not None and top_sig.signatures
    no_return_sig_src = "@plain(x):\n  = <x>\n\n= <plain(ok)>\n"
    no_return_sig = features.compute_signature_help(
        no_return_sig_src, "/tmp/x.kedi", lsp.Position(line=3, character=8)
    )
    assert no_return_sig is not None and "->" not in no_return_sig.signatures[0].label
    assert (
        features.compute_signature_help(
            "= <missing(1)>\n", "/tmp/x.kedi", lsp.Position(line=0, character=10)
        )
        is None
    )

    hints_src = "@inner(x, y):\n  = ok\n\n= <inner(<inner(1, 2)>, 3)>\n"
    hints = features.compute_inlay_hints(
        hints_src,
        "/tmp/x.kedi",
        lsp.Range(start=lsp.Position(line=0, character=0), end=lsp.Position(line=10, character=0)),
    )
    assert len(hints) >= 3
    extra_hints = features.compute_inlay_hints(
        "@one(x):\n  = ok\n\n= <one(1, 2)>\n",
        "/tmp/x.kedi",
        lsp.Range(start=lsp.Position(line=0, character=0), end=lsp.Position(line=10, character=0)),
    )
    assert extra_hints
    assert (
        features.compute_inlay_hints(
            "@one(x):\n  = ok\n\n= <one(1)>\n",
            "/tmp/x.kedi",
            lsp.Range(
                start=lsp.Position(line=10, character=0), end=lsp.Position(line=11, character=0)
            ),
        )
        == []
    )
    assert (
        features.compute_inlay_hints(
            "Value [answer]\n",
            "/tmp/x.kedi",
            lsp.Range(
                start=lsp.Position(line=10, character=0), end=lsp.Position(line=11, character=0)
            ),
        )
        == []
    )
    assert (
        features.compute_inlay_hints(
            "Value [answer: int]\n= <answer>\n",
            "/tmp/x.kedi",
            lsp.Range(
                start=lsp.Position(line=0, character=0), end=lsp.Position(line=3, character=0)
            ),
        )
        == []
    )

    assert (
        features._in_range(
            features.Span(0, 1, 1, 1, 1, 2),
            lsp.Range(
                start=lsp.Position(line=2, character=0), end=lsp.Position(line=3, character=0)
            ),
        )
        is False
    )
    assert (
        features._in_range(
            features.Span(0, 1, 5, 1, 5, 2),
            lsp.Range(
                start=lsp.Position(line=0, character=0), end=lsp.Position(line=3, character=0)
            ),
        )
        is False
    )
    assert (
        features._active_param_index(
            CallSegment(name="p", args=cast(Any, None)), lsp.Position(line=0, character=0)
        )
        == 0
    )
    assert features._innermost_call_at(Stmt(), lsp.Position(line=0, character=0)) is None
    rendered_arg = CallArg(loc=features.Span(0, 10, 1, 1, 1, 20))
    setattr(
        rendered_arg,
        "__render_segments__",
        [CallSegment(name="deep", args=[], loc=features.Span(0, 10, 1, 5, 1, 12))],
    )
    innermost = features._innermost_call_in_segment(
        CallSegment(name="outer", args=[rendered_arg], loc=features.Span(0, 20, 1, 1, 1, 21)),
        lsp.Position(line=0, character=6),
    )
    assert innermost is not None and innermost.name == "deep"

    eval_symbols = features.compute_document_symbols(
        "@eval: p:\n"
        "  > data: train:\n"
        "    = `[]`\n"
        "  > test_data: holdout:\n"
        "    = `[]`\n"
        "  > metric: score(train):\n"
        "    = `1.0`\n",
        "/tmp/x.kedi",
    )
    assert any(symbol.name == "@eval: p" for symbol in eval_symbols)
    direct_eval = features._eval_suite_symbol(
        EvalSuite(
            "p",
            datasets={"train": EvalDataset("train", "[]")},
            test_datasets={"holdout": EvalDataset("holdout", "[]")},
            metrics=[EvalMetric("score", "return 1")],
        )
    )
    assert direct_eval.range.start.line == 0

    monkeypatch.setitem(features._CAPTURE_TO_LSP, "custom.deep", ("variable", 0))
    assert features._capture_to_token("custom.deep.extra") == (
        features.SEMANTIC_TOKEN_TYPES.index("variable"),
        0,
    )
    assert features._capture_to_token("missing.capture") is None

    original_py_parser = features._PY_PARSER
    original_py_query = features._PY_HIGHLIGHTS_QUERY
    features._PY_PARSER = original_py_parser
    features._PY_HIGHLIGHTS_QUERY = original_py_query
    py_tokens = features.compute_semantic_tokens_full("= <`len([1, 2])`>\n", "/tmp/x.kedi")
    assert len(py_tokens.data) % 5 == 0
    monkeypatch.setattr(features, "_KEDI_HIGHLIGHTS_QUERY", None)
    monkeypatch.setattr(features, "_KEDI_INJECTIONS_QUERY", None)
    assert features.compute_semantic_tokens_full("@p():\n  = ok\n", "/tmp/x.kedi").data == []
    monkeypatch.undo()

    all_ranges = features.compute_python_ranges("@p(x: int = `1`):\n  = <`x`>\n")
    header_ranges = features.compute_header_python_ranges("@p(x: int = `1`):\n  = <`x`>\n")
    assert any(r.get("ownerKind") == "param" for r in header_ranges)
    assert all(r["text"] == "1" for r in header_ranges)
    assert any(r["text"] == "x" for r in all_ranges)


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


def test_python_virtual_document_includes_procedure_docstring():
    src = (
        "@calculate_stats(numbers: list[int]) -> int:\n"
        "  ###\n"
        "  Compute aggregate metrics.\n"
        "  Reuses Python locals below.\n"
        "  ###\n"
        "  [total: int] = ```\n"
        "  return sum(numbers)\n"
        "  ```\n"
        "  = <`str(total)`>\n"
    )
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi")
    text = virtual["text"]
    assert '"""Compute aggregate metrics.' in text
    assert "Reuses Python locals below." in text


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


def test_python_virtual_document_includes_prelude_symbols_for_toplevel_scope():
    src = (
        "```\n"
        "from pydantic import BaseModel, Field\n"
        "\n"
        "class Person(BaseModel):\n"
        '    name: str = Field(examples=["Ayşe", "Ali", "Berk"])\n'
        "    age: int = Field(gt=50)\n"
        "```\n\n"
        "Fake me a [person: Person]\n"
        "\n"
        "= <`person`>\n"
    )
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi")
    ast.parse(virtual["text"])
    assert "class Person(BaseModel):" in virtual["text"]
    assert "person: Person = ..." in virtual["text"]
    assert "    person" in virtual["text"]


def test_focused_virtual_document_stubs_unfocused_module_python():
    src = (
        "```\n"
        "from pydantic import BaseModel, Field\n"
        "\n"
        "class Person(BaseModel):\n"
        '    name: str = Field(examples=["Ayşe", "Ali", "Berk"])\n'
        "    age: int = Field(gt=50)\n"
        "```\n\n"
        "Fake me a [person: Person]\n"
        "\n"
        "= <`person`>\n"
    )
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi", focus_line=9)
    ast.parse(virtual["text"])
    assert "from pydantic import BaseModel, Field" not in virtual["text"]
    assert "class BaseModel: ..." in virtual["text"]
    assert "def Field(*args: Any, **kwargs: Any) -> Any: ..." in virtual["text"]
    assert "class Person:" in virtual["text"]
    assert "def __init__(self, name: str, age: int) -> None: ..." in virtual["text"]
    assert "person: Person = ..." in virtual["text"]


def test_focused_virtual_document_keeps_focused_module_python_mapped():
    src = (
        "```\n"
        "from pydantic import BaseModel\n"
        "\n"
        "class Person(BaseModel):\n"
        "    name: str\n"
        "```\n\n"
        "= <`Person(name='Ada')`>\n"
    )
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi", focus_line=3)
    ast.parse(virtual["text"])
    assert "from pydantic import BaseModel" in virtual["text"]
    assert "class Person(BaseModel):" in virtual["text"]
    assert any(region["range"]["start"]["line"] == 1 for region in virtual["ranges"])


def test_python_virtual_document_hoists_leading_toplevel_python_blocks():
    src = (
        "```\n"
        "import this\n"
        "```\n\n"
        "```\n"
        "from pydantic import BaseModel, Field\n"
        "\n"
        "class Person(BaseModel):\n"
        '    name: str = Field(examples=["Ayşe", "Ali", "Berk"])\n'
        "    age: int = Field(gt=50)\n"
        "```\n\n"
        "Fake me a [person: Person]\n"
        "\n"
        "= <`person`>\n"
    )
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi")
    ast.parse(virtual["text"])
    assert "import this" in virtual["text"]
    assert "class Person(BaseModel):" in virtual["text"]
    assert "def __kedi_toplevel__() -> Any:" in virtual["text"]
    assert "    person: Person = ..." in virtual["text"]


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
        "    def __init__(self, name: str, age: int, tags: list[str]) -> None: ..."
        in virtual["text"]
    )


def test_python_virtual_document_models_defaulted_signatures():
    src = (
        "~Person(name: str, salary: int = `0`)\n"
        "@make(name: str, salary: int = `0`) -> Person:\n"
        "  = `Person(name, salary)`\n"
    )
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi")
    ast.parse(virtual["text"])
    assert "    salary: int = 0" in virtual["text"]
    assert "    def __init__(self, name: str, salary: int = 0) -> None: ..." in virtual["text"]
    assert "def make(name: str, salary: int = 0) -> Person: ..." in virtual["text"]
    assert "def __kedi_scope_make(name: str, salary: int = 0) -> Person:" in virtual["text"]
    assert any(region["text"] == "0" for region in virtual["ranges"])


def test_python_virtual_document_defaults_untyped_proc_params_to_str():
    src = "@greet(name):\n  = ```\n  return name.upper()\n  ```\n"
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi")
    ast.parse(virtual["text"])
    assert "def greet(name: str) -> str: ..." in virtual["text"]
    assert "def __kedi_scope_greet(name: str) -> str:" in virtual["text"]


def test_python_virtual_document_maps_header_default_expression():
    src = (
        "@get_name() -> str:\n"
        "  = Mert\n"
        "@say_hi(name: str = `get_name()`) -> str:\n"
        '  = `f"Hello, {name}!"`\n'
    )
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi")
    ast.parse(virtual["text"])
    mapped = [
        region
        for region in virtual["ranges"]
        if region["text"] == "get_name()"
        and region["range"]["start"] == {"line": 2, "character": 21}
        and region["range"]["end"] == {"line": 2, "character": 31}
    ]
    assert len(mapped) == 1
    assert mapped[0]["virtualRange"]["start"]["line"] >= 0


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
    assert "def get_country(city: str) -> str: ..." in virtual["text"]
    assert "def __kedi_metric_get_country_accuracy() -> Any:" in virtual["text"]
    assert 'result = get_country("France")' in virtual["text"]
    assert any(r["kind"] == "fenced" for r in virtual["ranges"])
    assert any(
        s["kind"] == "procedure"
        and s["name"] == "get_country"
        and s["sourceRange"]["start"] == {"line": 0, "character": 1}
        for s in virtual["symbols"]
    )
    assert "    city: str = ..." in virtual["text"]
    assert 'result = get_country("France")' in virtual["text"]


def test_python_virtual_document_models_metric_scope_params_as_any():
    src = (
        "@solve_knapsack(problems, capacity):\n"
        "  = ok\n"
        "\n"
        "@eval: solve_knapsack:\n"
        "  > metric: accuracy(problems):\n"
        "    = ```\n"
        "    items_str, capacity = problems\n"
        "    pred = solve_knapsack(items_str, capacity)\n"
        "    return pred\n"
        "    ```\n"
    )
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi")
    ast.parse(virtual["text"])
    assert "def __kedi_metric_solve_knapsack_accuracy() -> Any:" in virtual["text"]
    assert "    problems: Any = ..." in virtual["text"]
    assert "    capacity: str = ..." in virtual["text"]
    assert "    items_str, capacity = problems" in virtual["text"]
    assert "    pred = solve_knapsack(items_str, capacity)" in virtual["text"]


def test_python_virtual_helper_edges(monkeypatch):
    from kedi.lang.ast import (
        AssignStmt,
        BacktickLineStmt,
        EvalMetric,
        EvalSuite,
        OutputSegment,
        PythonBlock,
        PythonExprNativeSegment,
        ProcedureDef,
        ProgramWithValidation,
        Span,
        TestCase,
        TestSuite,
        TextSegment,
        TypeApply,
        TypeDef,
        TypeField,
        TypePython,
        TypeRef,
        TypeUnion,
    )
    from kedi.lsp import python_virtual as pv

    source = "```\n    # comment\n    value = 1\n```\n"
    builder = pv.VirtualBuilder(source, "/tmp/x.kedi")
    builder.add_symbol(
        kind="missing",
        name="x",
        source_range=None,
        virtual_range={"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 1}},
    )
    assert builder.symbols == []

    empty_region = pv.PythonRegion(
        kind="inline",
        range={"start": {"line": 99, "character": 0}, "end": {"line": 99, "character": 0}},
        text="",
    )
    assert builder.add_mapped_region(empty_region)["start"]["line"] == 0
    assert "..." in builder.finish()

    comment_region = pv.PythonRegion(
        kind="fenced",
        range={"start": {"line": 1, "character": 0}, "end": {"line": 1, "character": 13}},
        text="# comment",
    )
    builder.add_mapped_region(comment_region, "    ")
    assert builder.lines[-1] == "    ..."
    assert builder._common_indent(99, 100) == 0

    reversed_region = pv.PythonRegion(
        kind="inline",
        range={"start": {"line": 2, "character": 9}, "end": {"line": 2, "character": 2}},
        text="",
    )
    builder.add_mapped_region(reversed_region)

    region_cursor = pv.RegionCursor(
        [
            {
                "kind": "inline",
                "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 1}},
                "text": "x",
            }
        ]
    )
    assert region_cursor.next("fenced") is None
    assert (
        region_cursor.next(
            "inline",
            Span(0, 1, 2, 1, 2, 2),
        )
        is None
    )

    stub_builder = pv.VirtualBuilder("", None)
    pv._emit_python_module_stub(stub_builder, "if")
    pv._emit_python_module_stub(stub_builder, "1 + 1")
    pv._emit_python_module_stub(
        stub_builder,
        "import os, sys as system\n"
        "from typing import *\n"
        "from pydantic import Field\n"
        "from models import User\n"
        "from package import value as alias\n"
        "class Empty:\n"
        "    pass\n"
        "class OnlyMethod:\n"
        "    def method(self):\n"
        "        return None\n"
        "class Model:\n"
        "    name: str\n"
        "    a = b = 1\n"
        "    count = 1\n"
        "    enabled: bool = Field(default=True)\n"
        "    required: int = Field(...)\n"
        "    async def run(self) -> str:\n"
        "        return 'ok'\n"
        "def fn(x: int) -> str:\n"
        "    return str(x)\n"
        "a = b = 1\n"
        "(c, [d, obj.attr]) = (1, [2, 3])\n",
    )
    text = stub_builder.finish()
    assert "os: Any = ..." in text
    assert "def Field(*args: Any, **kwargs: Any) -> Any: ..." in text
    assert "class User: ..." in text
    assert "class Empty:" in text and "    pass" in text
    assert "async def run" in text
    assert "c: Any = ..." in text and "d: Any = ..." in text
    field_call = cast(ast.Call, ast.parse("obj.Field(...)", mode="eval").body)
    factory_call = cast(ast.Call, ast.parse("factory()", mode="eval").body)
    assert pv._python_call_name(field_call.func) == "Field"
    assert pv._python_call_name(factory_call.func) == "factory"
    assert pv._unparse_or_any(None) == "Any"

    class BadNode(ast.AST):
        _fields = ("bad",)

        @property
        def bad(self):
            raise RuntimeError("bad")

    assert pv._unparse_or_any(BadNode()) == "Any"
    required_field = cast(ast.AnnAssign, ast.parse("x: int = Field(...)\n").body[0])
    defaulted_field = cast(ast.AnnAssign, ast.parse("x: int = Field(1)\n").body[0])
    assert pv._python_field_has_default(required_field.value) is False
    assert pv._python_field_has_default(defaulted_field.value) is True
    assert pv._python_field_has_default(ast.Constant(value=1)) is True
    assert pv._assigned_python_names(ast.parse("x: int\n").body[0]) == ["x"]
    multi_assign_class = cast(ast.ClassDef, ast.parse("class C:\n    a = b = 1\n").body[0])
    mixed_assign_class = cast(ast.ClassDef, ast.parse("class C:\n    self.a = b = 1\n").body[0])
    assert pv._python_class_fields(multi_assign_class) == [
        ("a", "Any", True),
        ("b", "Any", True),
    ]
    assert pv._python_class_fields(mixed_assign_class) == [("b", "Any", True)]
    assert (
        pv._single_python_segment([TextSegment(text="x"), PythonExprNativeSegment(code="1")])
        is None
    )
    assert pv._single_python_segment([TextSegment(text="x")]) is None
    assert pv._type_init_params(cast(Any, SimpleNamespace(fields=[]))) == "self"
    assert pv._indent_width("\t  x") == 6
    assert pv._python_call_name(ast.Constant(value=1)) == ""

    typing_symbols = pv._typing_symbols_in_type_expr(
        TypeUnion(
            left=TypeApply(name="Optional", args=[TypeRef(name="Literal")]),
            right=TypePython(code="Annotated[str, 'x']"),
        )
    )
    assert {
        ("typing", "Optional"),
        ("typing", "Literal"),
        ("typing", "Annotated"),
    } <= typing_symbols
    assert ("builtins", "str") in pv._typing_symbols_in_type_expr(
        TypeApply(name="Custom", args=[TypeRef(name="str")])
    )
    assert pv._typing_symbols_in_type_expr(cast(Any, TypeField(name="not-type"))) == set()
    assert pv._path_from_uri(None) is None
    assert pv._path_from_uri("untitled:test") is None
    assert pv._path_from_uri("file:///tmp/a%20b.kedi") == "/tmp/a b.kedi"

    empty_type_builder = pv.VirtualBuilder("", None)
    pv._emit_type_stub(empty_type_builder, TypeDef(name="Empty", fields=[]), {})
    assert "    pass" in empty_type_builder.finish()

    nested_builder = pv.VirtualBuilder("@outer():\n  @inner():\n    ###\n    doc\n    ###\n", None)
    inner = ProcedureDef(
        name="inner",
        params=[],
        body=[],
        docstring="doc",
        loc=Span(0, 1, 2, 3, 5, 8),
    )
    pv._emit_nested_procedure(nested_builder, pv.RegionCursor([]), inner, "    ", {})
    empty_inner = ProcedureDef(name="empty", params=[], body=[], loc=Span(0, 1, 1, 1, 1, 10))
    pv._emit_nested_procedure(nested_builder, pv.RegionCursor([]), empty_inner, "    ", {})
    assert '"""doc"""' in nested_builder.finish()
    assert "        pass" in nested_builder.finish()
    assert pv._docstring_lines("one line", "  ") == ['  """one line"""']

    assign = AssignStmt(
        name="x",
        expr=[PythonExprNativeSegment(code="1", loc=Span(0, 1, 1, 12, 1, 13))],
        loc=Span(0, 1, 1, 2, 1, 3),
    )
    emit_builder = pv.VirtualBuilder("[x: int] = `1`\n", None)
    pv._emit_assign_stmt(emit_builder, pv.RegionCursor([]), assign, {}, "    ")
    assert "x: str = (" in emit_builder.finish()
    plain_assign = AssignStmt(
        name="y",
        expr=[TextSegment(text="hello")],
        loc=Span(0, 1, 1, 2, 1, 3),
    )
    pv._emit_assign_stmt(emit_builder, pv.RegionCursor([]), plain_assign, {}, "    ")
    assert "y: str = ..." in emit_builder.finish()
    region_for_assign = {
        "kind": "inline",
        "range": {
            "start": {"line": 0, "character": 11},
            "end": {"line": 0, "character": 12},
        },
        "text": "1",
    }
    pv._emit_assign_stmt(emit_builder, pv.RegionCursor([region_for_assign]), assign, {}, "    ")

    stmt_builder = pv.VirtualBuilder("[a] = ok\n`x = 1`\n", None)
    pv._emit_statements(
        stmt_builder,
        pv.RegionCursor(
            [
                {
                    "kind": "inline",
                    "range": {
                        "start": {"line": 1, "character": 1},
                        "end": {"line": 1, "character": 6},
                    },
                    "text": "x = 1",
                }
            ]
        ),
        [
            AssignStmt(name="a", expr=[TextSegment(text="ok")], loc=Span(0, 1, 1, 2, 1, 3)),
            PythonBlock(code="x = 1", loc=Span(0, 1, 2, 1, 2, 7)),
            BacktickLineStmt(code="x = 1", loc=Span(0, 1, 2, 1, 2, 7)),
            BacktickLineStmt(code="y = 2", loc=Span(0, 1, 2, 1, 2, 7)),
        ],
        {},
        "    ",
        {},
    )
    assert "a: str = ..." in stmt_builder.finish()
    assert "x = 1" in stmt_builder.finish()
    pv._emit_statements(
        stmt_builder,
        pv.RegionCursor([]),
        [
            BacktickLineStmt(code="z = 3", loc=Span(0, 1, 2, 1, 2, 7)),
            pv.ReturnBlockStmt(code="return 1", loc=Span(0, 1, 2, 1, 2, 8)),
            pv.Stmt(),
        ],
        {},
        "    ",
        {},
    )

    validation_builder = pv.VirtualBuilder("`assert True`\n```\nDATA = []\n```\n", None)
    program = ProgramWithValidation(
        tests=[
            TestSuite(
                procedure_name="solve",
                cases=[
                    TestCase(
                        name="basic",
                        code="assert True",
                        is_block=False,
                        loc=Span(0, 1, 1, 1, 1, 12),
                    )
                ],
            )
        ],
        evals=[
            EvalSuite(
                procedure_name="solve",
                datasets={
                    "train": pv.EvalDataset(
                        name="train",
                        code="DATA = []",
                        is_block=True,
                        loc=Span(0, 1, 3, 1, 3, 10),
                    )
                },
                metrics=[EvalMetric(name="score", code="return 1.0")],
            )
        ],
    )
    pv._emit_validation(
        validation_builder,
        pv.RegionCursor(
            [
                {
                    "kind": "inline",
                    "range": {
                        "start": {"line": 0, "character": 1},
                        "end": {"line": 0, "character": 12},
                    },
                    "text": "assert True",
                },
                {
                    "kind": "fenced",
                    "range": {
                        "start": {"line": 2, "character": 0},
                        "end": {"line": 2, "character": 9},
                    },
                    "text": "DATA = []",
                },
            ]
        ),
        program,
        {},
    )
    validation_text = validation_builder.finish()
    assert "__kedi_test_solve_basic" in validation_text
    assert "__kedi_metric_solve_score" in validation_text
    assert "    pass" in validation_text
    metric_builder = pv.VirtualBuilder("", None)
    pv._emit_metric(
        metric_builder,
        pv.RegionCursor([]),
        EvalSuite(procedure_name="solve"),
        EvalMetric(name="score", code="return 1.0"),
        None,
    )
    assert "    pass" in metric_builder.finish()

    symbol_builder = pv.VirtualBuilder("", None)
    pv._add_scope_param_symbols(symbol_builder, {"x": pv.ScopeName("str", None)}, 0, "no params")
    pv._add_scope_param_symbols(symbol_builder, {"missing": pv.ScopeName("str", None)}, 0, "fn(x)")
    assert symbol_builder.symbols == []
    assert not pv._range_within_span(
        {"start": {"line": 0, "character": 0}, "end": {"line": 3, "character": 0}},
        Span(0, 1, 2, 1, 2, 5),
    )
    assert not pv._range_within_span(
        {"start": {"line": 1, "character": 0}, "end": {"line": 1, "character": 99}},
        Span(0, 1, 2, 2, 2, 5),
    )
    assert pv._proc_name_range(ProcedureDef(name="p", params=[])) is None
    assert pv._type_name_range(TypeDef(name="T", fields=[])) is None
    assert pv._assign_name_range(AssignStmt(name="x")) is None
    assert pv._output_name_range(OutputSegment(name="x")) is None
    assert (
        pv._param_name_range(
            ["@p(x):"],
            ProcedureDef(name="p", params=["x"]),
            "x",
        )
        is None
    )
    assert (
        pv._param_name_range(
            [],
            ProcedureDef(name="p", params=["x"], loc=Span(0, 1, 1, 1, 1, 5)),
            "x",
        )
        is None
    )
    assert (
        pv._param_name_range(
            ["@p x):"],
            ProcedureDef(name="p", params=["x"], loc=Span(0, 1, 1, 1, 1, 5)),
            "x",
        )
        is None
    )
    assert (
        pv._param_name_range(
            ["@p(x"],
            ProcedureDef(name="p", params=["y"], loc=Span(0, 1, 1, 1, 1, 5)),
            "y",
        )
        is None
    )
    assert (
        pv._param_name_range(
            ["@p(x):"],
            ProcedureDef(name="p", params=["y"], loc=Span(0, 1, 1, 1, 1, 6)),
            "y",
        )
        is None
    )
    assert pv._container_line(ProcedureDef(name="p", params=[])) == -1
    pv._add_proc_default_regions(
        symbol_builder,
        ProcedureDef(
            name="p",
            params=["x"],
            param_defaults={"x": "1"},
            loc=Span(0, 1, 1, 1, 1, 6),
        ),
        signature_line=0,
        signature="def p(y: str = 1):",
        header_defaults={
            "param": {
                0: [
                    {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 1},
                    }
                ]
            }
        },
    )
    monkeypatch.setattr(pv, "_proc_param_to_python", lambda proc, name: "x: str")
    pv._add_proc_default_regions(
        symbol_builder,
        ProcedureDef(
            name="p",
            params=["x"],
            param_defaults={"x": "1"},
            loc=Span(0, 1, 1, 1, 1, 6),
        ),
        signature_line=0,
        signature="def p(x: str):",
        header_defaults={
            "param": {
                0: [
                    {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 1},
                    }
                ]
            }
        },
    )
    monkeypatch.undo()
    pv._add_default_expr_region(
        symbol_builder,
        source_range={"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 1}},
        line=0,
        line_text="x: int = 1",
        expr="missing",
    )

    assert pv._find_deepest_procedure_at_line([ProcedureDef(name="p", params=[])], 1) is None
    assert pv._should_emit_full_module_python(None, 1) is False
    assert pv._indent_width("") == 0
    pv._fallback_virtual_document(
        "```\nvalue = 1\n```\n",
        [
            {
                "kind": "fenced",
                "range": {
                    "start": {"line": 1, "character": 0},
                    "end": {"line": 1, "character": 9},
                },
                "text": "value = 1",
            }
        ],
    )

    monkeypatch.setattr(pv, "compute_python_ranges", lambda src: [])
    outside_focus = pv.compute_python_virtual_document(
        "@p():\n  = ok\n\n```\nSIDE = 2\n```\n",
        focus_line=99,
    )
    assert "SIDE: Any = ..." in outside_focus["text"]
    fallback = pv.compute_python_virtual_document("```\nvalue = 1\n```\n@bad(:\n")
    assert fallback["focusKey"] is None

    monkeypatch.setattr(pv, "compute_python_ranges", lambda src: [])
    no_region = pv.compute_python_virtual_document("```\nPRELUDE = 1\n```\n```\nSIDE = 2\n```\n")
    assert "PRELUDE = 1" in no_region["text"]
    assert "SIDE = 2" in no_region["text"]

    stdin = io.StringIO('{"text":"= ok\\n","uri":"file:///tmp/x.kedi","focusLine":0}')
    stdout = io.StringIO()
    monkeypatch.setattr(pv.sys, "stdin", stdin)
    monkeypatch.setattr(pv.sys, "stdout", stdout)
    pv.main()
    assert json.loads(stdout.getvalue())["focusKey"] is None

    stdin = io.StringIO('\n{"id":1,"text":"= ok\\n"}\nnot-json\n')
    stdout = io.StringIO()
    monkeypatch.setattr(pv.sys, "stdin", stdin)
    monkeypatch.setattr(pv.sys, "stdout", stdout)
    pv.main_loop()
    responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert responses[0]["ok"] is True
    assert responses[1]["ok"] is False


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
            cast(Any, Server()),
            "file:///tmp/x.kedi",
            {"textDocument": {"uri": "file:///tmp/x.kedi"}, "text": fresh},
        )
        == fresh
    )


def test_server_helpers_and_registered_handlers(monkeypatch):
    from kedi.lsp import server as server_module

    uri = "file:///tmp/example.kedi"
    source = "@hello(name: str) -> str:\n  = <name>\n"

    class Document:
        def __init__(self, text):
            self.source = text

    class Workspace:
        def get_text_document(self, requested_uri):
            assert requested_uri == uri
            return Document(source)

    server = server_module.make_server()
    server.protocol._workspace = Workspace()
    handlers = server.protocol.fm.features
    published = []
    monkeypatch.setattr(
        server, "text_document_publish_diagnostics", lambda params: published.append(params)
    )

    calls = []

    def record(name, value):
        def inner(**kwargs):
            calls.append((name, kwargs))
            return value

        return inner

    monkeypatch.setattr(server_module.features, "compute_diagnostics", record("diagnostics", []))
    monkeypatch.setattr(server_module.features, "compute_hover", record("hover", "hovered"))
    monkeypatch.setattr(
        server_module.features, "compute_definition", record("definition", "definition")
    )
    monkeypatch.setattr(server_module.features, "compute_references", record("references", ["ref"]))
    monkeypatch.setattr(
        server_module.features, "compute_document_symbols", record("symbols", ["sym"])
    )
    monkeypatch.setattr(
        server_module.features, "compute_prepare_rename", record("prepare", "range")
    )
    monkeypatch.setattr(server_module.features, "compute_rename", record("rename", "edit"))
    monkeypatch.setattr(server_module.features, "compute_formatting", lambda src: ["format"])
    monkeypatch.setattr(
        server_module.features, "compute_signature_help", record("signature", "sig")
    )
    monkeypatch.setattr(server_module.features, "compute_inlay_hints", record("inlay", ["hint"]))
    monkeypatch.setattr(
        server_module.features, "compute_semantic_tokens_full", record("semantic", "tokens")
    )
    monkeypatch.setattr(server_module.features, "compute_python_ranges", lambda src: ["range"])
    monkeypatch.setattr(
        server_module,
        "compute_python_virtual_document",
        lambda src, source_path=None, focus_line=None: {
            "text": "virtual",
            "ranges": [],
            "mappings": [],
            "symbols": [],
            "focusKey": focus_line,
        },
    )

    text_document = SimpleNamespace(uri=uri)
    position = lsp.Position(line=0, character=1)

    handlers[lsp.TEXT_DOCUMENT_DID_OPEN](SimpleNamespace(text_document=text_document))
    handlers[lsp.TEXT_DOCUMENT_DID_CHANGE](SimpleNamespace(text_document=text_document))
    handlers[lsp.TEXT_DOCUMENT_DID_CLOSE](SimpleNamespace(text_document=text_document))
    assert handlers["workspace/didChangeWatchedFiles"](SimpleNamespace()) is None
    assert len(published) == 3
    assert published[-1].diagnostics == []

    assert (
        handlers[lsp.TEXT_DOCUMENT_HOVER](
            SimpleNamespace(text_document=text_document, position=position)
        )
        == "hovered"
    )
    assert (
        handlers[lsp.TEXT_DOCUMENT_DEFINITION](
            SimpleNamespace(text_document=text_document, position=position)
        )
        == "definition"
    )
    assert handlers[lsp.TEXT_DOCUMENT_REFERENCES](
        SimpleNamespace(
            text_document=text_document,
            position=position,
            context=SimpleNamespace(include_declaration=False),
        )
    ) == ["ref"]
    assert handlers[lsp.TEXT_DOCUMENT_DOCUMENT_SYMBOL](
        SimpleNamespace(text_document=text_document)
    ) == ["sym"]
    assert (
        handlers[lsp.TEXT_DOCUMENT_PREPARE_RENAME](
            SimpleNamespace(text_document=text_document, position=position)
        )
        == "range"
    )
    assert (
        handlers[lsp.TEXT_DOCUMENT_RENAME](
            SimpleNamespace(text_document=text_document, position=position, new_name="new")
        )
        == "edit"
    )
    assert handlers[lsp.TEXT_DOCUMENT_FORMATTING](SimpleNamespace(text_document=text_document)) == [
        "format"
    ]
    assert (
        handlers[lsp.TEXT_DOCUMENT_SIGNATURE_HELP](
            SimpleNamespace(text_document=text_document, position=position)
        )
        == "sig"
    )
    assert handlers[lsp.TEXT_DOCUMENT_INLAY_HINT](
        SimpleNamespace(text_document=text_document, range=lsp.Range(start=position, end=position))
    ) == ["hint"]
    assert (
        handlers[lsp.TEXT_DOCUMENT_SEMANTIC_TOKENS_FULL](
            SimpleNamespace(text_document=text_document)
        )
        == "tokens"
    )

    assert handlers["kedi/pythonRanges"]({}) == {"uri": "", "ranges": []}
    assert handlers["kedi/pythonRanges"]({"textDocument": {"uri": uri}, "text": source}) == {
        "uri": uri,
        "ranges": ["range"],
    }
    assert handlers["kedi/pythonVirtualDocument"]({}) == {
        "uri": "",
        "text": "",
        "ranges": [],
        "mappings": [],
        "symbols": [],
        "focusKey": None,
    }
    assert (
        handlers["kedi/pythonVirtualDocument"](
            {"textDocument": {"uri": uri}, "text": source, "position": {"line": 3}}
        )["focusKey"]
        == 3
    )

    assert server_module._uri_to_path("untitled:Untitled-1") is None
    assert server_module._uri_to_path("file:///tmp/a%20b.kedi") == "/tmp/a b.kedi"
    assert server_module._text_document_uri({"textDocument": "bad"}) is None
    assert (
        server_module._text_document_uri(SimpleNamespace(textDocument=SimpleNamespace(uri=uri)))
        == uri
    )
    assert server_module._text_document_uri(SimpleNamespace()) is None
    assert server_module._request_source(server, uri, {"text": 1}) == source
    assert server_module._request_source(server, uri, SimpleNamespace(text="fresh")) == "fresh"
    assert server_module._request_source(server, uri, SimpleNamespace()) == source
    assert server_module._request_focus_line({"focusLine": 4}) == 4
    assert server_module._request_focus_line({"position": {"line": "bad"}}) is None
    assert server_module._request_focus_line({"position": "bad"}) is None
    assert server_module._request_focus_line(SimpleNamespace(focus_line=5)) == 5
    assert server_module._request_focus_line(SimpleNamespace(focusLine=6)) == 6
    assert server_module._request_focus_line(SimpleNamespace(position=SimpleNamespace(line=7))) == 7
    assert (
        server_module._request_focus_line(SimpleNamespace(position=SimpleNamespace(line="bad")))
        is None
    )
    assert server_module._request_focus_line(SimpleNamespace()) is None


def test_server_main_starts_io(monkeypatch):
    from kedi.lsp import server as server_module

    started = []
    monkeypatch.setattr(
        server_module,
        "make_server",
        lambda: SimpleNamespace(start_io=lambda: started.append(True)),
    )

    server_module.main()

    assert started == [True]
