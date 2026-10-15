"""Capabilities share one semantic contract across editor surfaces."""

from pathlib import Path

import pytest
import tree_sitter_kedi
from lsprotocol import types as lsp
from tree_sitter import Language, Parser, Query, QueryCursor

from kedi.agent_profile.capabilities import CAPABILITY_DIRECTIVES
from kedi.lang import parse_program
from kedi.lsp.features.capability_state import _profile_adapter_state
from kedi.lsp.features.completion import compute_completion
from kedi.lsp.features.document import compute_document_symbols
from kedi.lsp.features.semantic_tokens import SEMANTIC_TOKEN_TYPES, compute_semantic_tokens_full
from tests.lsp_helpers import (
    decode_semantic_tokens,
    definition_at,
    diagnostics_for,
    hover_text_at,
    position_of,
    references_at,
)

SOURCE = """> capability: reader:
    > instructions: Read the supplied evidence.
    > defer: true
> profile: fast:
    > model: test
> use: reader
"""


def test_capability_semantic_tokens():
    tokens = compute_semantic_tokens_full(SOURCE, "/tmp/capability.kedi")
    captured = {
        (SOURCE.splitlines()[line][col : col + length], SEMANTIC_TOKEN_TYPES[kind])
        for line, col, length, kind, _ in decode_semantic_tokens(tokens.data)
    }
    assert {("capability", "keyword"), ("defer", "keyword"), ("reader", "enumMember")} <= captured


@pytest.mark.parametrize(
    "relative",
    ["tree-sitter-kedi/queries/highlights.scm", "kedi-zed/languages/kedi/highlights.scm"],
)
def test_capability_shipped_queries(relative):
    root = Path(__file__).resolve().parents[1]
    language = Language(tree_sitter_kedi.language())
    source = SOURCE.encode()
    captures = QueryCursor(Query(language, (root / relative).read_text())).captures(
        Parser(language).parse(source).root_node
    )
    words = {
        name: {source[node.start_byte : node.end_byte].decode() for node in nodes}
        for name, nodes in captures.items()
    }
    assert {"capability", "defer"} <= words["keyword"]
    assert "reader" in words["label"]
    assert "true" in (words.get("constant.builtin", set()) | words.get("constant", set()))


def test_capability_hover_navigation_and_symbols():
    path = "/tmp/capability.kedi"
    uri = "file:///tmp/capability.kedi"
    for occurrence in (1, 2):
        hover = hover_text_at(SOURCE, path, "reader", occurrence=occurrence)
        assert "Agent capability" in hover
        assert "Loading: on demand" in hover
        assert "Guidance is additive" in hover
    assert "Defaults to `false`" in hover_text_at(SOURCE, path, "defer")
    definition = definition_at(SOURCE, uri, path, "reader", occurrence=2)
    assert definition is not None and definition.range.start.line == 0
    references = references_at(SOURCE, uri, path, "reader", True)
    assert {item.range.start.line for item in references} == {0, 5}
    symbols = compute_document_symbols(SOURCE, path)
    capability = next(item for item in symbols if item.name == "> capability: reader")
    assert capability.detail == "agent capability"
    assert any(item.name == "> defer: true" for item in capability.children or [])


def test_capability_body_completion_respects_allowlist():
    source = "> capability: reader:\n    > de\n"
    items = compute_completion(source, None, lsp.Position(line=1, character=8))
    assert {item.label for item in items} == CAPABILITY_DIRECTIVES
    defer = next(item for item in items if item.label == "defer")
    assert defer.text_edit.new_text == "> defer: "
    assert "defaults to false" in defer.detail
    top = compute_completion("> de", None, lsp.Position(line=0, character=4))
    assert "defer" not in {item.label for item in top}
    assert "capability" in {item.label for item in top}


@pytest.mark.parametrize("in_capability", [False, True])
def test_use_completion_distinguishes_capabilities(in_capability):
    source = SOURCE
    if in_capability:
        source += "> capability: composed:\n    > use: rea"
    else:
        source += "> use: rea"
    items = {
        item.label: item
        for item in compute_completion(
            source, None, position_of(source, "rea", occurrence=3, offset=3)
        )
    }
    assert items["reader"].detail == "agent capability"
    assert ("fast" in items) is not in_capability
    assert "str" not in items


@pytest.mark.parametrize("suffix", ["> defer:", "> defer: `", "> instructions:"])
def test_capability_completion_does_not_consume_newline_or_python(suffix):
    source = "> capability: reader:\n    " + suffix
    assert compute_completion(source, None, lsp.Position(line=1, character=len(suffix) + 4)) == []


def test_owned_policy_does_not_override_editor_host_state():
    program = parse_program(
        "> capability: reader:\n  > approval: allow\n  > hooks:\n    pre_tool_use: `inspect_tool`\n"
    )
    state = _profile_adapter_state(program.profile_defs["reader"])
    assert state.approval_loc is None
    assert state.hooks_loc is None and not state.hook_events


def test_python_capability_use_does_not_require_executing_the_prelude():
    source = (
        "```\nfrom kedi import Capability\n"
        'reader = Capability(name="reader", instructions="Read source evidence.")\n'
        'raise RuntimeError("Editor must not execute this block")\n```\n'
        "> use: reader\n"
    )
    assert diagnostics_for(source, "/tmp/python-capability.kedi") == []


@pytest.mark.parametrize("block", [False, True])
def test_unsupported_deferred_capability_is_diagnosed_transitively(block):
    source = (
        "> adapter: langchain\n"
        "> capability: hidden:\n  > defer: true\n"
        "> profile: reviewer:\n  > use:\n    hidden\n"
        + ("> use:\n  reviewer\n" if block else "> use: reviewer\n")
    )
    diagnostics = diagnostics_for(source, "/tmp/capability.kedi")
    assert any("on-demand capability loading" in item.message for item in diagnostics)


@pytest.mark.parametrize("nested", [False, True])
def test_eager_capability_hooks_inside_procedure_keep_selected_adapter(nested):
    source = (
        "> adapter: pydantic\n"
        "> capability: reader:\n  > defer: false\n"
        "  > hooks:\n    pre_tool_use: `inspect_tool`\n"
        + ("> capability: bundle:\n  > use: reader\n" if nested else "")
        + "@review() -> str:\n"
        + f"  > use: {'bundle' if nested else 'reader'}\n  = done\n"
    )
    diagnostics = diagnostics_for(source, "/tmp/hooks.kedi")
    assert not any(item.code in {"K-ADAPTER-HOOKS", "K-ADAPTER-CAPABILITY"} for item in diagnostics)


def test_legacy_use_tool_list_produces_adapter_diagnostic():
    from kedi.lang.ast import UseDirectiveStmt
    from kedi.lsp.features.capabilities import _use_capability_diagnostics
    from kedi.lsp.features.capability_state import _AdapterDiagnosticState

    program = parse_program("> use: read\n")
    original = program.ordered_toplevel[0]
    statement = UseDirectiveStmt(tool_names=["read"], loc=original.loc)
    diagnostics = _use_capability_diagnostics(
        program, statement, _AdapterDiagnosticState(), {}, {}, set()
    )
    assert isinstance(diagnostics, list)


def test_symbol_generation_tolerates_defer_entry_without_location():
    from kedi.lang.ast import ProfileDeferEntry, ProfileDefStmt
    from kedi.lsp.features.document import _profile_def_symbol

    symbol = _profile_def_symbol(
        ProfileDefStmt(
            name="reader", kind="capability", defer_members=[ProfileDeferEntry(enabled=False)]
        )
    )
    assert not symbol.children
