"""LSP feature tests."""

from __future__ import annotations

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
