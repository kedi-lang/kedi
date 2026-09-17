"""LSP feature tests."""

from __future__ import annotations

from pathlib import Path

import tree_sitter_kedi
from tree_sitter import Language, Parser, Query, QueryCursor

from kedi.lsp import features
from tests.lsp_helpers import decode_semantic_tokens

# ---------------------------------------------------------------------
# Semantic tokens
# ---------------------------------------------------------------------


def test_semantic_tokens_full_emits_tokens():
    src = "@hello(name: str) -> str:\n  = <name>\n"
    tokens = features.compute_semantic_tokens_full(src, "/tmp/x.kedi")
    assert tokens.data, "expected at least one semantic token for a non-trivial doc"
    # Length is a multiple of 5 (LSP encoding).
    assert len(tokens.data) % 5 == 0


def test_semantic_tokens_cover_template_condition_text_and_inputs() -> None:
    src = "> if: <city> is in Turkey\n  [result] = yes\n"
    decoded = decode_semantic_tokens(
        features.compute_semantic_tokens_full(src, "/tmp/control.kedi").data
    )
    string_type = features.SEMANTIC_TOKEN_TYPES.index("string")
    variable_type = features.SEMANTIC_TOKEN_TYPES.index("variable")

    assert any(line == 0 and token_type == string_type for line, _, _, token_type, _ in decoded)
    assert any(
        line == 0 and col == 7 and length == 4 and token_type == variable_type
        for line, col, length, token_type, _ in decoded
    )


def test_semantic_tokens_legend_constants():
    assert "function" in features.SEMANTIC_TOKEN_TYPES
    assert "type" in features.SEMANTIC_TOKEN_TYPES
    assert "declaration" in features.SEMANTIC_TOKEN_MODIFIERS
    assert features.SEMANTIC_TOKEN_TYPES.index("function") == 12
    assert features.SEMANTIC_TOKEN_TYPES.index("type") == 1
    assert features.SEMANTIC_TOKEN_MODIFIERS.index("declaration") == 0


def test_semantic_tokens_docstring_comment_uses_variable_token():
    src = "@translate(text: str) -> str:\n  ###\n  >> Translate the provided text.\n  ###\n  = <text>\n"
    tokens = features.compute_semantic_tokens_full(src, "/tmp/x.kedi")
    decoded = decode_semantic_tokens(tokens.data)
    variable_type = features.SEMANTIC_TOKEN_TYPES.index("variable")
    assert any(line == 1 and token_type == variable_type for line, _, _, token_type, _ in decoded)
    assert any(line == 2 and token_type == variable_type for line, _, _, token_type, _ in decoded)


def test_semantic_tokens_regular_block_comment_stays_comment_token():
    src = "@translate(text: str) -> str:\n  = <text>\n  ###\n  >> Not a docstring.\n  ###\n"
    tokens = features.compute_semantic_tokens_full(src, "/tmp/x.kedi")
    decoded = decode_semantic_tokens(tokens.data)
    comment_type = features.SEMANTIC_TOKEN_TYPES.index("comment")
    assert any(line == 2 and token_type == comment_type for line, _, _, token_type, _ in decoded)


def test_semantic_tokens_cover_module_directives():
    src = "> import: profiles\n\n> export:\n  xd\n  get_proc\n"
    tokens = features.compute_semantic_tokens_full(src, "/tmp/x.kedi")
    decoded = decode_semantic_tokens(tokens.data)

    operator_type = features.SEMANTIC_TOKEN_TYPES.index("operator")
    namespace_type = features.SEMANTIC_TOKEN_TYPES.index("namespace")
    variable_type = features.SEMANTIC_TOKEN_TYPES.index("variable")

    assert any(line == 0 and token_type == operator_type for line, _, _, token_type, _ in decoded)
    assert any(line == 0 and token_type == namespace_type for line, _, _, token_type, _ in decoded)
    assert any(line == 2 and token_type == operator_type for line, _, _, token_type, _ in decoded)
    assert any(line == 3 and token_type == variable_type for line, _, _, token_type, _ in decoded)
    assert any(line == 4 and token_type == variable_type for line, _, _, token_type, _ in decoded)


def test_semantic_tokens_cover_submodule_and_selective_imports():
    src = "> import: package/helpers:\n  selected\n"
    decoded = decode_semantic_tokens(features.compute_semantic_tokens_full(src, "/tmp/x.kedi").data)
    namespace_type = features.SEMANTIC_TOKEN_TYPES.index("namespace")
    variable_type = features.SEMANTIC_TOKEN_TYPES.index("variable")

    assert any(
        line == 0
        and col == 10
        and length == len("package/helpers")
        and token_type == namespace_type
        for line, col, length, token_type, _ in decoded
    )
    assert any(
        line == 1 and col == 2 and length == len("selected") and token_type == variable_type
        for line, col, length, token_type, _ in decoded
    )


def test_semantic_tokens_cover_assignment_targets_and_args():
    src = "[x: str] = `args.country`\n= <args>\n"
    tokens = features.compute_semantic_tokens_full(src, "/tmp/x.kedi")
    decoded = decode_semantic_tokens(tokens.data)

    variable_type = features.SEMANTIC_TOKEN_TYPES.index("variable")

    assert any(
        line == 0 and col == 1 and length == 1 and token_type == variable_type
        for line, col, length, token_type, _ in decoded
    )
    assert any(
        line == 0 and col == 12 and length == 4 and token_type == variable_type
        for line, col, length, token_type, _ in decoded
    )
    assert any(
        line == 1 and col == 3 and length == 4 and token_type == variable_type
        for line, col, length, token_type, _ in decoded
    )


def test_semantic_tokens_cover_assignment_and_map_syntax() -> None:
    src = (
        "[value: int] = `1`\n"
        "[value] := `2`\n"
        "> loop [item]: `[1]`:\n"
        "  [copy: int] = `item`\n"
        "> map:\n"
        "  [value] := `copy`\n"
    )
    decoded = decode_semantic_tokens(
        features.compute_semantic_tokens_full(src, "/tmp/control.kedi").data
    )
    keyword_type = features.SEMANTIC_TOKEN_TYPES.index("keyword")
    operator_type = features.SEMANTIC_TOKEN_TYPES.index("operator")
    variable_type = features.SEMANTIC_TOKEN_TYPES.index("variable")

    assert any(
        line == 1 and col == 1 and length == len("value") and token_type == variable_type
        for line, col, length, token_type, _ in decoded
    )
    assert any(
        line == 1 and col == len("[value] ") and length == 2 and token_type == operator_type
        for line, col, length, token_type, _ in decoded
    )
    assert any(
        line == 4 and col == len("> ") and length == len("map") and token_type == keyword_type
        for line, col, length, token_type, _ in decoded
    )


def test_semantic_tokens_cover_annotated_description_string():
    line = '  >> Capital is [capital: Annotated[str, "City name"]].'
    src = f"@p():\n{line}\n"
    tokens = features.compute_semantic_tokens_full(src, "/tmp/x.kedi")
    decoded = decode_semantic_tokens(tokens.data)

    string_type = features.SEMANTIC_TOKEN_TYPES.index("string")
    description_col = line.index('"City name"')

    assert any(
        line_no == 1
        and col == description_col
        and length == len('"City name"')
        and token_type == string_type
        for line_no, col, length, token_type, _ in decoded
    )


def test_semantic_tokens_cover_approval_directive_values():
    src = "> approval: allow\n> approval: `approve_tool`\n"
    tokens = decode_semantic_tokens(features.compute_semantic_tokens_full(src, "/tmp/x.kedi").data)
    keyword_type = features.SEMANTIC_TOKEN_TYPES.index("keyword")
    string_type = features.SEMANTIC_TOKEN_TYPES.index("string")
    variable_type = features.SEMANTIC_TOKEN_TYPES.index("variable")

    assert any(line == 0 and token_type == keyword_type for line, _, _, token_type, _ in tokens)
    assert any(
        line == 0 and col == 12 and token_type == string_type
        for line, col, _, token_type, _ in tokens
    )
    assert any(
        line == 1 and col == 13 and token_type == variable_type
        for line, col, _, token_type, _ in tokens
    )


def test_semantic_tokens_cover_artifact_policy_fields_and_values() -> None:
    src = (
        "> artifacts:\n"
        "  query_artifacts: enabled\n"
        "  threshold: 100kb\n"
        "  cleanup_interval: `timedelta(seconds=5)`\n"
    )
    tokens = decode_semantic_tokens(
        features.compute_semantic_tokens_full(src, "/tmp/artifacts.kedi").data
    )
    keyword_type = features.SEMANTIC_TOKEN_TYPES.index("keyword")
    property_type = features.SEMANTIC_TOKEN_TYPES.index("property")
    string_type = features.SEMANTIC_TOKEN_TYPES.index("string")
    variable_type = features.SEMANTIC_TOKEN_TYPES.index("variable")

    assert any(
        line == 0 and col == len("> ") and length == len("artifacts") and token_type == keyword_type
        for line, col, length, token_type, _ in tokens
    )
    assert any(
        line == 1 and col == 2 and token_type == property_type
        for line, col, _, token_type, _ in tokens
    )
    assert any(
        line == 1 and col == len("  query_artifacts: ") and token_type == keyword_type
        for line, col, _, token_type, _ in tokens
    )
    assert any(
        line == 2 and col == len("  threshold: ") and token_type == string_type
        for line, col, _, token_type, _ in tokens
    )
    assert any(
        line == 3 and col > len("  cleanup_interval: `") and token_type == variable_type
        for line, col, _, token_type, _ in tokens
    )


def test_semantic_tokens_cover_codemode_directive_fields_and_values() -> None:
    src = "> codemode:\n  enabled: true\n  max_nested_calls: `8`\n"
    tokens = decode_semantic_tokens(
        features.compute_semantic_tokens_full(src, "/tmp/codemode.kedi").data
    )
    keyword_type = features.SEMANTIC_TOKEN_TYPES.index("keyword")
    property_type = features.SEMANTIC_TOKEN_TYPES.index("property")
    string_type = features.SEMANTIC_TOKEN_TYPES.index("string")

    assert any(
        line == 0 and col == len("> ") and length == len("codemode") and token_type == keyword_type
        for line, col, length, token_type, _ in tokens
    )
    assert any(
        line == 1 and col == 2 and length == len("enabled") and token_type == property_type
        for line, col, length, token_type, _ in tokens
    )
    assert any(
        line == 1
        and col == len("  enabled: ")
        and length == len("true")
        and token_type == string_type
        for line, col, length, token_type, _ in tokens
    )
    assert any(
        line == 2 and col == 2 and length == len("max_nested_calls") and token_type == property_type
        for line, col, length, token_type, _ in tokens
    )


def test_semantic_tokens_cover_hooks_directive_fields_and_python_values() -> None:
    src = "> hooks:\n  pre_tool_use: `[validate_path, audit_call]`\n"
    tokens = decode_semantic_tokens(
        features.compute_semantic_tokens_full(src, "/tmp/hooks.kedi").data
    )
    keyword_type = features.SEMANTIC_TOKEN_TYPES.index("keyword")
    property_type = features.SEMANTIC_TOKEN_TYPES.index("property")
    variable_type = features.SEMANTIC_TOKEN_TYPES.index("variable")

    assert (0, 2, len("hooks"), keyword_type) in [token[:4] for token in tokens]
    assert (1, 2, len("pre_tool_use"), property_type) in [token[:4] for token in tokens]
    assert any(
        line == 1 and col > len("  pre_tool_use: `[") and token_type == variable_type
        for line, col, _, token_type, _ in tokens
    )


def test_semantic_tokens_keep_unicode_prefixes_on_source_boundaries():
    src = "@p(name: str) -> str:\n  >> İstanbul için [answer: str] from <name>.\n"
    tokens = decode_semantic_tokens(features.compute_semantic_tokens_full(src, "/tmp/x.kedi").data)
    variable_type = features.SEMANTIC_TOKEN_TYPES.index("variable")

    assert any(
        line == 1 and col == 20 and length == len("answer") and token_type == variable_type
        for line, col, length, token_type, _ in tokens
    )


def test_semantic_tokens_keep_template_substitution_delimiters_consistent():
    src = ">> Compare <left> with <right> and <`value`>.\n"
    tokens = decode_semantic_tokens(features.compute_semantic_tokens_full(src, "/tmp/x.kedi").data)
    operator_type = features.SEMANTIC_TOKEN_TYPES.index("operator")

    for column in (11, 16, 23, 29):
        assert any(
            line == 0 and col == column and length == 1 and token_type == operator_type
            for line, col, length, token_type, _ in tokens
        )


def test_tree_sitter_highlights_mark_both_template_angle_delimiters_special():
    root = Path(__file__).resolve().parents[1]
    canonical = (root / "tree-sitter-kedi/queries/highlights.scm").read_text(encoding="utf-8")
    binding = (
        root / "tree-sitter-kedi/bindings/python/tree_sitter_kedi/queries/highlights.scm"
    ).read_text(encoding="utf-8")
    zed = (root / "kedi-zed/languages/kedi/highlights.scm").read_text(encoding="utf-8")

    assert canonical == binding
    assert '["@" "~" "<" ">"] @punctuation.special' in canonical
    assert '  "<"\n  ">"\n  "```"\n  "`"\n] @punctuation.special' in zed


def test_tree_sitter_highlights_capture_both_template_angle_delimiters_special():
    source = b">> Text <topic> <compose(`topic`)> [answer: str] <`value`>\n"
    root = Path(__file__).resolve().parents[1]
    language = Language(tree_sitter_kedi.language())
    query = Query(
        language,
        (root / "tree-sitter-kedi/queries/highlights.scm").read_text(encoding="utf-8"),
    )
    captures = QueryCursor(query).captures(Parser(language).parse(source).root_node)
    special = {
        source[node.start_byte : node.end_byte].decode("utf-8")
        for node in captures["punctuation.special"]
    }

    assert "<" in special
    assert ">" in special


def test_tree_sitter_highlights_capture_artifacts_syntax():
    source = b"> artifacts:\n  query_artifacts: enabled\n"
    root = Path(__file__).resolve().parents[1]
    language = Language(tree_sitter_kedi.language())

    for query_path in (
        root / "tree-sitter-kedi/queries/highlights.scm",
        root / "kedi-zed/languages/kedi/highlights.scm",
    ):
        query = Query(language, query_path.read_text(encoding="utf-8"))
        captures = QueryCursor(query).captures(Parser(language).parse(source).root_node)
        keywords = {
            source[node.start_byte : node.end_byte].decode("utf-8") for node in captures["keyword"]
        }
        properties = {
            source[node.start_byte : node.end_byte].decode("utf-8") for node in captures["property"]
        }
        constants = {
            source[node.start_byte : node.end_byte].decode("utf-8")
            for capture in ("constant.builtin", "constant")
            for node in captures.get(capture, [])
        }

        assert "artifacts" in keywords
        assert "query_artifacts" in properties
        assert "enabled" in constants


def test_tree_sitter_highlights_capture_workflow_and_compaction_settings() -> None:
    source = (
        b"> profile: coordinator:\n"
        b"  > workflow: dynamic\n"
        b"  > history:\n"
        b"    enabled: true\n"
        b"    compaction_mode: native\n"
        b"    compaction_threshold: `100_000`\n"
    )
    root = Path(__file__).resolve().parents[1]
    language = Language(tree_sitter_kedi.language())

    for query_path in (
        root / "tree-sitter-kedi/queries/highlights.scm",
        root / "kedi-zed/languages/kedi/highlights.scm",
    ):
        query = Query(language, query_path.read_text(encoding="utf-8"))
        captures = QueryCursor(query).captures(Parser(language).parse(source).root_node)

        def captured(name: str) -> set[str]:
            return {
                source[node.start_byte : node.end_byte].decode("utf-8")
                for node in captures.get(name, [])
            }

        assert {"workflow", "history"} <= captured("keyword")
        assert {"enabled", "compaction_mode", "compaction_threshold"} <= captured("property")
        assert "dynamic" in captured("constant")
        assert "native" in captured("string") | captured("text.literal")


def test_tree_sitter_and_zed_highlight_control_flow_keywords_and_binder() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        b"> if: `ready`:\n  [status] = yes\n> else:\n  > loop [n]: `range(3)`:\n    [value] = <n>\n"
    )
    language = Language(tree_sitter_kedi.language())

    for query_path in (
        root / "tree-sitter-kedi/queries/highlights.scm",
        root / "kedi-zed/languages/kedi/highlights.scm",
    ):
        query = Query(language, query_path.read_text(encoding="utf-8"))
        captures = QueryCursor(query).captures(Parser(language).parse(source).root_node)

        def captured(name: str) -> set[str]:
            return {
                source[node.start_byte : node.end_byte].decode("utf-8")
                for node in captures.get(name, [])
            }

        assert {"if", "else", "loop"} <= captured("keyword")
        assert "n" in captured("variable")


def test_lsp_semantic_tokens_cover_workflow_and_compaction_settings() -> None:
    source = (
        "> profile: coordinator:\n"
        "  > workflow: dynamic\n"
        "  > history:\n"
        "    enabled: true\n"
        "    compaction_mode: native\n"
        "    compaction_threshold: `100_000`\n"
    )
    tokens = decode_semantic_tokens(
        features.compute_semantic_tokens_full(source, "/tmp/directives.kedi").data
    )
    keyword_type = features.SEMANTIC_TOKEN_TYPES.index("keyword")
    variable_type = features.SEMANTIC_TOKEN_TYPES.index("variable")
    property_type = features.SEMANTIC_TOKEN_TYPES.index("property")
    string_type = features.SEMANTIC_TOKEN_TYPES.index("string")

    expected = {
        (1, 4, len("workflow"), keyword_type),
        (1, 14, len("dynamic"), variable_type),
        (2, 4, len("history"), keyword_type),
        (3, 4, len("enabled"), property_type),
        (4, 4, len("compaction_mode"), property_type),
        (4, 21, len("native"), string_type),
        (5, 4, len("compaction_threshold"), property_type),
    }
    actual = {(line, col, length, token_type) for line, col, length, token_type, _ in tokens}
    assert expected <= actual


def test_tree_sitter_and_lsp_highlight_expanded_skills_policy() -> None:
    source = (
        b"> skills:\n"
        b"  enabled: true\n"
        b"  include_registry: false\n"
        b"  include_all: true\n"
        b'  exclude_paths: `["~/.agents/skills"]`\n'
    )
    root = Path(__file__).resolve().parents[1]
    language = Language(tree_sitter_kedi.language())

    for query_path in (
        root / "tree-sitter-kedi/queries/highlights.scm",
        root / "kedi-zed/languages/kedi/highlights.scm",
    ):
        query = Query(language, query_path.read_text(encoding="utf-8"))
        captures = QueryCursor(query).captures(Parser(language).parse(source).root_node)

        def captured(name: str) -> set[str]:
            return {
                source[node.start_byte : node.end_byte].decode("utf-8")
                for node in captures.get(name, [])
            }

        assert "skills" in captured("keyword")
        assert {"enabled", "include_registry", "include_all", "exclude_paths"} <= captured(
            "property"
        )

    decoded = decode_semantic_tokens(
        features.compute_semantic_tokens_full(source.decode(), "/tmp/skills.kedi").data
    )
    keyword_type = features.SEMANTIC_TOKEN_TYPES.index("keyword")
    property_type = features.SEMANTIC_TOKEN_TYPES.index("property")
    actual = {(line, col, length, token_type) for line, col, length, token_type, _ in decoded}
    assert (0, 2, len("skills"), keyword_type) in actual
    assert (1, 2, len("enabled"), property_type) in actual


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
