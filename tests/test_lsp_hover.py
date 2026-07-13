"""LSP feature tests."""

from __future__ import annotations

import ast

from kedi.lsp import features
from kedi.lsp.python_virtual import compute_python_virtual_document
from tests.lsp_helpers import (
    decode_semantic_tokens,
    definition_at,
    hover_at,
    hover_text_at,
    inlay_hints_for_lines,
    prepare_rename_at,
    references_at,
    rename_at,
    signature_help_at,
)

# ---------------------------------------------------------------------
# Hover
# ---------------------------------------------------------------------


def test_hover_on_procedure_header_shows_signature():
    src = "@translate(text: str) -> str:\n  = <text>\n"
    body = hover_text_at(src, "example.kedi", "translate")
    assert "translate" in body
    assert "text: str" in body
    assert "str" in body


def test_hover_on_defaulted_procedure_header_shows_defaults():
    src = '@translate(text: str, style = `"brief"`) -> str:\n  = <text>\n'
    body = hover_text_at(src, "example.kedi", "translate")
    assert "text: str" in body
    assert 'style: str = `"brief"`' in body


def test_hover_on_untyped_procedure_header_defaults_param_to_str():
    src = "@translate(text) -> str:\n  = <text>\n"
    body = hover_text_at(src, "example.kedi", "translate")
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
    body = hover_text_at(src, "example.kedi", "translate")
    assert "translate" in body
    assert "Translate the provided text." in body
    assert "Keep the answer concise." in body


def test_hover_on_docstring_comment_uses_procedure_docstring():
    src = "@translate(text: str) -> str:\n  ###\n  >> Translate the provided text.\n  ###\n  = <text>\n"
    body = hover_text_at(src, "example.kedi", "Translate the provided text.")
    assert "translate" in body
    assert "Translate the provided text." in body


def test_hover_on_regular_block_comment_stays_none():
    src = "@translate(text: str) -> str:\n  = <text>\n  ###\n  >> Not a docstring.\n  ###\n"
    assert hover_at(src, "example.kedi", "Not a docstring.") is None


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
    assert hover_at(src, "example.kedi", "get_name") is None


def test_hover_on_nested_procedure_header_shows_signature():
    src = "@foo():\n  @bar(name: str) -> str:\n    = <name>\n  = done\n"
    body = hover_text_at(src, "example.kedi", "@bar")
    assert "bar" in body
    assert "name: str" in body
    assert "str" in body


def test_hover_on_nested_text_segment_works():
    src = "@foo():\n  @bar():\n    >> Hello world\n    = Done\n  = dn\n"
    body = hover_text_at(src, "example.kedi", "world")
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

    body = hover_text_at(src, str(main), "make_profile")
    assert "make_profile" in body
    assert "name: str" in body

    location = definition_at(
        src,
        main.resolve().as_uri(),
        str(main),
        "make_profile",
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

    body = hover_text_at(src, str(main), "Profile")
    assert "Profile" in body
    assert "not defined in this module" not in body

    location = definition_at(
        src,
        main.resolve().as_uri(),
        str(main),
        "Profile",
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

    body = hover_text_at(src, str(main), "xd")
    assert "Imported value" in body
    assert "`xd: int`" in body

    location = definition_at(
        src,
        main.resolve().as_uri(),
        str(main),
        "xd",
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

    body = hover_text_at(src, str(main), "profiles")
    assert "**Module** `profiles`" in body
    assert "Profile utilities" in body
    assert "Used across the workspace." in body

    location = definition_at(
        src,
        main.resolve().as_uri(),
        str(main),
        "profiles",
    )
    assert location is not None
    assert location.uri == profiles.resolve().as_uri()
    assert location.range.start.line == 0


def test_hover_and_definition_on_internal_registry_module_name():
    from kedi.module_registry import internal_module_path

    internal_path = internal_module_path("this")
    assert internal_path is not None

    src = "> import: this\n\n= ok\n"
    body = hover_text_at(src, "example.kedi", "this")
    assert "**Module** `this`" in body

    location = definition_at(
        src,
        "file:///tmp/example.kedi",
        "example.kedi",
        "this",
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

    body = hover_text_at(src, str(main), "profiles")
    assert "**Module** `profiles`" in body
    assert "Not a module docstring." not in body


def test_hover_and_definition_on_exported_procedure_name():
    src = "@make_profile(name: str) -> str:\n  = <name>\n\n> export:\n  make_profile\n"

    body = hover_text_at(src, "example.kedi", "make_profile", occurrence=2)
    assert "make_profile" in body
    assert "name: str" in body

    location = definition_at(
        src,
        "file:///tmp/example.kedi",
        "example.kedi",
        "make_profile",
        occurrence=2,
    )
    assert location is not None
    assert location.range.start.line == 0


def test_hover_and_definition_on_exported_value_name():
    src = "[profile_id: int] = `1`\n\n> export:\n  profile_id\n"

    body = hover_text_at(src, "example.kedi", "profile_id", occurrence=2)
    assert "Module value" in body
    assert "profile_id" in body

    location = definition_at(
        src,
        "file:///tmp/example.kedi",
        "example.kedi",
        "profile_id",
        occurrence=2,
    )
    assert location is not None
    assert location.range.start.line == 0


def test_hover_on_input_segment():
    src = "@p(name):\n  >> Hello, <name>!\n  = done\n"
    body = hover_text_at(src, "example.kedi", "name", occurrence=2)
    assert "name" in body
    # Hover labels parameter references as such when the input matches
    # an enclosing-procedure parameter.
    assert "arameter" in body or "Local" in body


def test_hover_on_output_segment_shows_type():
    src = "@p():\n  >> Pick a [n: int].\n  = <n>\n"
    body = hover_text_at(src, "example.kedi", "n")
    assert "Output" in body
    assert "n" in body


def test_hover_on_assignment_target_shows_local_variable():
    src = "@p():\n  [x: str] = ok\n  = <x>\n"
    body = hover_text_at(src, "example.kedi", "x")
    assert "Local" in body
    assert "`x`" in body


def test_hover_on_runtime_args_shows_attrdict_type():
    body = hover_text_at("= <args>\n", "example.kedi", "args")
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
        assert expected in hover_text_at(
            src,
            source_path,
            needle,
            occurrence=occurrence,
            offset=offset,
        )

    definitions = [
        ("this", 1, 1, "src/kedi/modules/this.kedi"),
        ("Review:", 1, 1, source_path),
        ("<line_count()>", 1, 2, source_path),
        ("review_code(`args", 1, 1, source_path),
        ("review_code\n  global_note", 1, 1, source_path),
    ]
    for needle, occurrence, offset, expected_uri in definitions:
        loc = definition_at(
            src,
            uri,
            source_path,
            needle,
            occurrence=occurrence,
            offset=offset,
        )
        assert loc is not None
        assert expected_uri in loc.uri

    refs = references_at(
        src,
        uri,
        source_path,
        "@review_code",
        True,
        offset=2,
    )
    assert len(refs) >= 2
    assert any(ref.range.start.line == 12 for ref in refs)
    assert any(ref.range.start.line == 52 for ref in refs)

    prepare = prepare_rename_at(
        src,
        source_path,
        "[review: Review]",
        occurrence=2,
        offset=2,
    )
    assert prepare is not None
    assert prepare.start.line == 28
    assert prepare.start.character == 3
    assert prepare.end.character == 9

    signature = signature_help_at(
        src,
        source_path,
        "review_code(`args",
        offset=len("review_code(`args.language"),
    )
    assert signature is not None
    assert signature.signatures[0].label.startswith("@review_code(")
    assert signature.active_parameter == 0

    hints = inlay_hints_for_lines(src, source_path, 0, 80)
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
        "  >> Flag: [flag: bool]\n"
        "  = <x>\n"
    )

    checks = [
        ("list", "builtins.list"),
        ("int", "builtins.int"),
        ("Optional", "typing.Optional"),
        ("str", "builtins.str"),
        ("Literal", "typing.Literal"),
        ("bool", "builtins.bool"),
    ]

    for name, origin in checks:
        body = hover_text_at(src, "example.kedi", name)
        assert name in body
        assert origin in body
        assert f"Kedi type resolved to Python's {origin}." in body
        assert "Kedi type derived from" not in body
        assert "Kedi type backed by" not in body
        assert "Kedi/Python type backed by" not in body
        assert "not defined" not in body


def test_hover_on_annotated_explains_kedi_schema_metadata():
    src = '@p():\n  >> Capital is [capital: Annotated[str, "Canonical city name"]].\n'

    body = hover_text_at(src, "example.kedi", "Annotated")

    assert 'Annotated[type, "description"]' in body
    assert "schema metadata" in body
    assert "runtime type" in body
    assert "LLM adapter field descriptions" in body


def test_hover_on_datetime_type_uses_datetime_namespace():
    src = "@p(started_at: datetime) -> timedelta:\n  = <started_at>\n"

    assert "datetime.datetime" in hover_text_at(src, "example.kedi", "datetime")
    assert "datetime.timedelta" in hover_text_at(src, "example.kedi", "timedelta")


def test_builtin_type_definition_and_rename_are_disabled():
    src = "@p(x: Optional[str]) -> bool:\n  = <x>\n"
    uri = "file:///tmp/example.kedi"
    assert definition_at(src, uri, "/tmp/example.kedi", "Optional") is None
    assert prepare_rename_at(src, "/tmp/example.kedi", "Optional") is None
    assert rename_at(src, uri, "/tmp/example.kedi", "Optional", "Maybe") is None


def test_hover_on_type_definition_builtin_and_typing_types_is_not_undefined():
    src = "~Person(name: str, friend: Optional[str])\n= ok\n"

    checks = [
        ("str", "builtins.str"),
        ("Optional", "typing.Optional"),
        ("str", "builtins.str"),
    ]

    for name, origin in checks:
        body = hover_text_at(src, "example.kedi", name)
        assert name in body
        assert origin in body
        assert "not defined" not in body


def test_hover_on_type_definition_shows_field_defaults():
    src = "~Person(name: str, salary: int = `0`)\n= ok\n"
    body = hover_text_at(src, "example.kedi", "Person")
    assert "salary: int = `0`" in body


def test_hover_on_type_definition_defaults_untyped_field_to_str():
    src = "~Person(name)\n= ok\n"
    assert "name: str" in hover_text_at(src, "example.kedi", "Person")


def test_custom_type_definition_still_jumps_to_kedi_type():
    src = "~Person(name: str)\n@p(x: Person):\n  = <x>\n"
    loc = definition_at(
        src, "file:///tmp/example.kedi", "/tmp/example.kedi", "Person", occurrence=2
    )
    assert loc is not None
    assert loc.range.start.line == 0


def test_custom_type_definition_field_type_still_jumps_to_kedi_type():
    src = "~Person(name: str)\n~Team(member: Person)\n= ok\n"
    loc = definition_at(src, "file:///tmp/example.kedi", "/tmp/example.kedi", "Person")
    assert loc is not None
    assert loc.range.start.line == 0


def test_unknown_type_hover_stays_runtime_resolved():
    src = "@p(x: UnknownType):\n  = <x>\n"
    body = hover_text_at(src, "example.kedi", "UnknownType")
    assert "not defined in this module" in body
    assert "resolved at runtime" in body


def test_hover_on_unparseable_returns_none():
    assert hover_at("@p(:\n  ???\n", "example.kedi", "p", offset=1) is None


def test_hover_on_prelude_defined_python_type_is_not_runtime_unknown():
    src = (
        "```\n"
        "from pydantic import BaseModel, Field\n"
        "\n"
        "class Person(BaseModel):\n"
        '    name: str = Field(examples=["Ayşe", "Ali", "Berk"])\n'
        "    age: int = Field(gt=50)\n"
        "```\n\n"
        ">> Fake me a [person: Person]\n"
        "\n"
        "= <`person`>\n"
    )
    body = hover_text_at(src, "example.kedi", "Person", occurrence=2)
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
        ">> Fake me a [person: Person]\n"
        "\n"
        "= <`person`>\n"
    )
    body = hover_text_at(src, "example.kedi", "Person", occurrence=2)
    assert "Person" in body
    assert "defined in a top-level Python block" in body
    assert "not defined in this module" not in body
