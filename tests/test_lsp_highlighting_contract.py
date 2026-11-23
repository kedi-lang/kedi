"""Keep shipped editor queries aligned with the language-server token contract."""

import subprocess
import tomllib
from pathlib import Path

import pytest
import tree_sitter_kedi
from tree_sitter import Language, Parser, Query, QueryCursor

from kedi.lsp.features.semantic_tokens import SEMANTIC_TOKEN_TYPES, compute_semantic_tokens_full
from tests.lsp_helpers import decode_semantic_tokens


@pytest.mark.parametrize(
    ("source", "word", "kind"),
    [
        ("> profile: review:\n  > output: list[str]\n", "output", "keyword"),
        ("> profile: review:\n  > output: list[str]\n", "list", "type"),
        ("> profile: review:\n  > workflow: autonomous\n", "workflow", "keyword"),
        ("> profile: review:\n  > max_agents: 2\n", "max_agents", "keyword"),
        ("> profile: review:\n  > subagent: helper\n", "subagent", "keyword"),
        ("> budget:\n  requests: 5\n", "budget", "keyword"),
        ("> instructions: Be precise.\n", "instructions", "keyword"),
        ("> task [job]: reviewer:\n  >> Review the changes.\n", "task", "keyword"),
        ("> await [review]: job\n", "await", "keyword"),
        ("> show: <result>\n", "show", "keyword"),
        ("= <result>\n", "=", "keyword"),
        ("= ```python\nreturn 1\n```\n", "=", "keyword"),
    ],
)
def test_statement_and_profile_keywords_have_semantic_tokens(source, word, kind):
    tokens = compute_semantic_tokens_full(source, "/tmp/highlighting.kedi")
    assert any(
        source.splitlines()[line][col : col + length] == word
        and SEMANTIC_TOKEN_TYPES[token_type] == kind
        for line, col, length, token_type, _ in decode_semantic_tokens(tokens.data)
    )


def test_notebook_and_python_wheel_ship_the_canonical_highlights():
    root = Path(__file__).resolve().parents[1]
    canonical = (root / "tree-sitter-kedi/queries/highlights.scm").read_text()
    for relative in (
        "tree-sitter-kedi/bindings/python/tree_sitter_kedi/queries/highlights.scm",
        "notebook/src/kedi_notebook/static/grammars/kedi/highlights.scm",
    ):
        assert (root / relative).read_text() == canonical


def test_zed_pins_the_vendored_grammar_revision():
    root = Path(__file__).resolve().parents[1]
    manifest = tomllib.loads((root / "kedi-zed/extension.toml").read_text())
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root / "tree-sitter-kedi", text=True
    ).strip()
    assert manifest["grammars"]["kedi"]["rev"] == revision


def test_zed_highlights_compile_and_color_fallback_like_model():
    root = Path(__file__).resolve().parents[1]
    language = Language(tree_sitter_kedi.language())
    tree = Parser(language).parse(b"> model: openai:primary\n> fallback: openai:backup\n")
    assert not tree.root_node.has_error
    query = Query(language, (root / "kedi-zed/languages/kedi/highlights.scm").read_text())
    captures = QueryCursor(query).captures(tree.root_node)
    model_scopes = {
        name for name, nodes in captures.items() if any(n.text == b"model" for n in nodes)
    }
    fallback_scopes = {
        name for name, nodes in captures.items() if any(n.text == b"fallback" for n in nodes)
    }
    assert model_scopes == fallback_scopes == {"keyword"}
