from __future__ import annotations

from pathlib import Path

from kedi.lsp import features
from tests.lsp_helpers import decode_semantic_tokens, hover_text_at


def test_semantic_tokens_cover_agent_profile_directives() -> None:
    src = "> model: haiku\n> profile: light:\n    > model: opus\n> use: light\n"
    tokens = features.compute_semantic_tokens_full(src, "/tmp/profile.kedi")
    decoded = decode_semantic_tokens(tokens.data)

    keyword_type = features.SEMANTIC_TOKEN_TYPES.index("keyword")
    label_type = features.SEMANTIC_TOKEN_TYPES.index("enumMember")
    string_type = features.SEMANTIC_TOKEN_TYPES.index("string")

    assert any(line == 0 and token_type == keyword_type for line, _, _, token_type, _ in decoded)
    assert any(line == 0 and token_type == string_type for line, _, _, token_type, _ in decoded)
    assert any(line == 1 and token_type == keyword_type for line, _, _, token_type, _ in decoded)
    assert any(line == 1 and token_type == label_type for line, _, _, token_type, _ in decoded)
    assert any(line == 3 and token_type == keyword_type for line, _, _, token_type, _ in decoded)
    assert any(line == 3 and token_type == label_type for line, _, _, token_type, _ in decoded)


def test_hover_on_model_profile_and_use_directives() -> None:
    src = """
> model: haiku
> profile: light:
    > model: opus
> use: light
@foo():
    >> answer [x]
"""
    assert "> model:" in hover_text_at(src, "profile.kedi", "model", occurrence=1)
    assert "haiku" in hover_text_at(src, "profile.kedi", "haiku")
    assert "Agent profile" in hover_text_at(src, "profile.kedi", "light", occurrence=1)
    assert "> use:" in hover_text_at(src, "profile.kedi", "light", occurrence=2)


def test_use_hover_when_module_has_no_exported_profiles(tmp_path: Path) -> None:
    (tmp_path / "utils.kedi").write_text(
        "@foo() -> str:\n  = ok\n\n> export:\n  foo\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    src = "> import: utils\n\n> use: utils\n"
    body = hover_text_at(src, str(main), "utils", occurrence=2)
    assert "no exported agent profiles" in body


def test_use_hover_hints_when_module_name_used_as_profile(tmp_path: Path) -> None:
    (tmp_path / "model_profile.kedi").write_text(
        "> profile: advanced:\n    > model: opus\n\n> export:\n  advanced\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    src = "> import: model_profile\n\n> use: model_profile\n"
    body = hover_text_at(src, str(main), "model_profile", occurrence=2)
    assert "imported **module**" in body
    assert "> use: advanced" in body


def test_document_symbols_include_profile_definition() -> None:
    src = """
> profile: light:
    > model: opus
@foo():
    > model: haiku
    >> x [y]
"""
    symbols = features.compute_document_symbols(src, "profile.kedi")
    names = [symbol.name for symbol in symbols]
    assert "> profile: light" in names
    proc = next(symbol for symbol in symbols if symbol.name == "foo")
    child_names = [child.name for child in proc.children or ()]
    assert any(name.startswith("> model:") for name in child_names)
