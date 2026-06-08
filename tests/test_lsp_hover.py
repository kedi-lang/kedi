"""LSP feature tests."""

from __future__ import annotations

import ast

from lsprotocol import types as lsp

from kedi.lsp import features
from kedi.lsp.python_virtual import compute_python_virtual_document
from tests.lsp_helpers import decode_semantic_tokens, hover_value, position_of

# ---------------------------------------------------------------------
# Hover
# ---------------------------------------------------------------------


def test_hover_on_procedure_header_shows_signature():
    src = "@translate(text: str) -> str:\n  = <text>\n"
    hover = features.compute_hover(
        source=src, source_path="example.kedi", pos=lsp.Position(line=0, character=3)
    )
    assert hover is not None
    body = hover_value(hover)
    assert "translate" in body
    assert "text: str" in body
    assert "str" in body


def test_hover_on_defaulted_procedure_header_shows_defaults():
    src = '@translate(text: str, style = `"brief"`) -> str:\n  = <text>\n'
    hover = features.compute_hover(
        source=src, source_path="example.kedi", pos=lsp.Position(line=0, character=3)
    )
    assert hover is not None
    body = hover_value(hover)
    assert "text: str" in body
    assert 'style: str = `"brief"`' in body


def test_hover_on_untyped_procedure_header_defaults_param_to_str():
    src = "@translate(text) -> str:\n  = <text>\n"
    hover = features.compute_hover(
        source=src, source_path="example.kedi", pos=lsp.Position(line=0, character=3)
    )
    assert hover is not None
    body = hover_value(hover)
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
    body = hover_value(hover)
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
    assert "translate" in hover_value(hover)
    assert "Translate the provided text." in hover_value(hover)


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
    body = hover_value(hover)
    assert "bar" in body
    assert "name: str" in body
    assert "str" in body


def test_hover_on_nested_text_segment_works():
    src = "@foo():\n  @bar():\n    Hello world\n    = Done\n  = dn\n"
    hover = features.compute_hover(
        source=src, source_path="example.kedi", pos=lsp.Position(line=2, character=8)
    )
    assert hover is not None
    body = hover_value(hover)
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
    assert "make_profile" in hover_value(hover)
    assert "name: str" in hover_value(hover)

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
    assert "Profile" in hover_value(hover)
    assert "not defined in this module" not in hover_value(hover)

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
    assert "Imported value" in hover_value(hover)
    assert "`xd: int`" in hover_value(hover)

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
    assert "**Module** `profiles`" in hover_value(hover)
    assert "Profile utilities" in hover_value(hover)
    assert "Used across the workspace." in hover_value(hover)

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
    assert "**Module** `this`" in hover_value(hover)

    location = features.compute_definition(
        source=src,
        uri="file:///tmp/example.kedi",
        source_path="example.kedi",
        pos=lsp.Position(line=0, character=11),
    )
    assert location is not None
    assert location.uri == internal_path.as_uri()
    assert location.range.start.line == 0


def test_hover_on_imported_module_name_ignores_nonleading_comment_as_docstring(
    tmp_path,
):
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
    assert "**Module** `profiles`" in hover_value(hover)
    assert "Not a module docstring." not in hover_value(hover)


def test_hover_and_definition_on_exported_procedure_name():
    src = "@make_profile(name: str) -> str:\n  = <name>\n\n> export:\n  make_profile\n"

    hover = features.compute_hover(
        source=src,
        source_path="example.kedi",
        pos=lsp.Position(line=4, character=4),
    )
    assert hover is not None
    assert "make_profile" in hover_value(hover)
    assert "name: str" in hover_value(hover)

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
    assert "Module value" in hover_value(hover)
    assert "profile_id" in hover_value(hover)

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
    body = hover_value(hover)
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
    body = hover_value(hover)
    assert "Output" in body
    assert "n" in body


def test_hover_on_assignment_target_shows_local_variable():
    src = "@p():\n  [x: str] = ok\n  = <x>\n"
    hover = features.compute_hover(
        source=src, source_path="example.kedi", pos=lsp.Position(line=1, character=3)
    )

    assert hover is not None
    body = hover_value(hover)
    assert "Local" in body
    assert "`x`" in body


def test_hover_on_runtime_args_shows_attrdict_type():
    hover = features.compute_hover(
        source="= <args>\n",
        source_path="example.kedi",
        pos=lsp.Position(line=0, character=3),
    )

    assert hover is not None
    body = hover_value(hover)
    assert "Runtime value" in body
    assert "AttrDict[str, ArgType]" in body


def test_lsp_features_cover_kedi_nodes_and_python_interop_relations():
    src = (
        "# Module doc\n"
        "> import: this\n"
        "\n"
        "```\n"
        "class Helper:\n"
        "    pass\n"
        "```\n"
        "\n"
        "~Review(decision: str, issues: list[str], line_count: int = `0`)\n"
        "\n"
        "[global_note: str] = `args.note`\n"
        "\n"
        '@review_code(language: str = `args.language`, code: str = `"print(1)"`) -> Review:\n'
        "  ###\n"
        "  Review code with CLI args and Python interop.\n"
        "  ###\n"
        "  @line_count() -> int:\n"
        "    [count: int] = ```\n"
        "    return len([line for line in code.splitlines() if line.strip()])\n"
        "    ```\n"
        "    = <count>\n"
        "\n"
        "  > optimize: review_prompt:\n"
        "    Review this <language> snippet with <line_count()> non-empty lines.\n"
        "    Runtime args <args>.\n"
        "    Return [review: Review]\n"
        "\n"
        "  `print(args.debug)`\n"
        "  [review: Review] = ```\n"
        '  return Review(decision="approve", issues=[], line_count=line_count())\n'
        "  ```\n"
        "  = <`review`>\n"
        "\n"
        "@test: review_code:\n"
        "  > case: basic:\n"
        '    `assert review_code("python", "print(1)").decision == "approve"`\n'
        "\n"
        "@eval: review_code:\n"
        "  > data: examples:\n"
        "    = ```\n"
        '    return [("python", {"decision": "approve"})]\n'
        "    ```\n"
        "  > metric: accuracy(examples):\n"
        "    = ```\n"
        "    return 1.0\n"
        "    ```\n"
        "\n"
        "> export:\n"
        "  Review\n"
        "  review_code\n"
        "  global_note\n"
        "\n"
        "= <review_code(`args.language`, `args.code`)>\n"
    )
    source_path = "/tmp/all_nodes.kedi"
    uri = "file:///tmp/all_nodes.kedi"

    assert features.compute_diagnostics(src, source_path) == []

    hover_checks = [
        ("this", 1, 1, "**Module** `this`"),
        ("~Review", 1, 2, "Kedi type definition"),
        ("decision", 1, 0, "**Field** `decision: str`"),
        ("global_note", 1, 1, "**Local** `global_note`"),
        ("@review_code", 1, 2, "Review code with CLI args and Python interop."),
        ("@line_count", 1, 2, "`@line_count()`"),
        ("<language>", 1, 2, "**Parameter** `language: str`"),
        ("<line_count()>", 1, 2, "`@line_count()`"),
        ("<args>", 1, 2, "**Runtime value** `args: AttrDict[str, ArgType]`"),
        ("[review: Review]", 1, 1, "**Output** `review`"),
        ("[review: Review]", 2, 2, "**Local** `review`"),
        ("review_code\n  global_note", 1, 1, "Review code with CLI args"),
        ("review_code(`args", 1, 1, "Review code with CLI args"),
    ]
    for needle, occurrence, offset, expected in hover_checks:
        hover = features.compute_hover(
            src,
            source_path,
            position_of(src, needle, occurrence=occurrence, offset=offset),
        )
        assert expected in hover_value(hover)

    definitions = [
        ("this", 1, 1, "src/kedi/modules/this.kedi"),
        ("Review:", 1, 1, source_path),
        ("<line_count()>", 1, 2, source_path),
        ("review_code(`args", 1, 1, source_path),
        ("review_code\n  global_note", 1, 1, source_path),
    ]
    for needle, occurrence, offset, expected_uri in definitions:
        loc = features.compute_definition(
            src,
            uri,
            source_path,
            position_of(src, needle, occurrence=occurrence, offset=offset),
        )
        assert loc is not None
        assert expected_uri in loc.uri

    refs = features.compute_references(
        src,
        uri,
        source_path,
        position_of(src, "@review_code", offset=2),
        include_declaration=True,
    )
    assert len(refs) >= 2
    assert any(ref.range.start.line == 12 for ref in refs)
    assert any(ref.range.start.line == 52 for ref in refs)

    prepare = features.compute_prepare_rename(
        src,
        source_path,
        position_of(src, "[review: Review]", occurrence=2, offset=2),
    )
    assert prepare is not None
    assert prepare.start.line == 28
    assert prepare.start.character == 3
    assert prepare.end.character == 9

    signature = features.compute_signature_help(
        src,
        source_path,
        position_of(src, "review_code(`args", offset=len("review_code(`args.language")),
    )
    assert signature is not None
    assert signature.signatures[0].label.startswith("@review_code(")
    assert signature.active_parameter == 0

    hints = features.compute_inlay_hints(
        src,
        source_path,
        lsp.Range(
            start=lsp.Position(line=0, character=0),
            end=lsp.Position(line=80, character=0),
        ),
    )
    assert {str(hint.label) for hint in hints} >= {"language: ", "code: "}

    decoded = decode_semantic_tokens(features.compute_semantic_tokens_full(src, source_path).data)
    variable_type = features.SEMANTIC_TOKEN_TYPES.index("variable")
    function_type = features.SEMANTIC_TOKEN_TYPES.index("function")
    type_type = features.SEMANTIC_TOKEN_TYPES.index("type")
    assert any(
        line == 10 and length == len("global_note") and t == variable_type
        for line, _, length, t, _ in decoded
    )
    assert any(
        line == 12 and length == len("review_code") and t == function_type
        for line, _, length, t, _ in decoded
    )
    assert any(
        line == 8 and length == len("Review") and t == type_type
        for line, _, length, t, _ in decoded
    )
    assert any(
        line == 24 and length == len("args") and t == variable_type
        for line, _, length, t, _ in decoded
    )

    virtual = compute_python_virtual_document(src, source_path)
    ast.parse(virtual["text"])
    text = virtual["text"]
    assert "args: AttrDict[str, ArgType] = ..." in text
    assert "global_note: str = (" in text
    assert "def review_code(language: str = args.language" in text
    assert "print(args.debug)" in text
    assert 'return Review(decision="approve", issues=[], line_count=line_count())' in text


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
        body = hover_value(hover)
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
        body = hover_value(hover)
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
    body = hover_value(hover)
    assert "salary: int = `0`" in body


def test_hover_on_type_definition_defaults_untyped_field_to_str():
    src = "~Person(name)\n= ok\n"
    hover = features.compute_hover(
        source=src,
        source_path="example.kedi",
        pos=lsp.Position(line=0, character=1),
    )
    assert hover is not None
    assert "name: str" in hover_value(hover)


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
    assert "not defined in this module" in hover_value(hover)
    assert "resolved at runtime" in hover_value(hover)


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
    body = hover_value(hover)
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
    body = hover_value(hover)
    assert "Person" in body
    assert "defined in a top-level Python block" in body
    assert "not defined in this module" not in body
