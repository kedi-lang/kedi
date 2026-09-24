from __future__ import annotations

import pytest
from lsprotocol import types as lsp

from kedi.lang import parse_program
from kedi.lsp.features.completion import compute_completion
from kedi.lsp.features.diagnostics import compute_diagnostics
from tests.lsp_helpers import hover_text_at


def test_history_processor_completion_and_hover():
    source = "> history:\n    pro"
    items = compute_completion(source, "history.kedi", lsp.Position(line=1, character=7))
    assert "processor" in {item.label for item in items}
    source = "> history:\n    enabled: true\n    processor: `keep`\n"
    hover = hover_text_at(source, "history.kedi", "processor", offset=2)
    assert "processor" in hover
    assert "keep_recent_groups" in hover
    assert "scoped native history processor" in hover_text_at(
        source, "history.kedi", "history", offset=2
    )


@pytest.mark.parametrize("adapter", ["pydantic", "langchain"])
def test_supported_history_processor_has_no_capability_error(adapter):
    source = f"> adapter: {adapter}\n> history:\n    enabled: false\n    processor: `keep`\n"
    diagnostics = compute_diagnostics(source, "history.kedi")
    assert [item.code for item in diagnostics] == ["K-HISTORY-PREFIX-CACHE"]
    assert diagnostics[0].severity == lsp.DiagnosticSeverity.Warning


@pytest.mark.parametrize("processor", ["keep", "get_processor()"])
def test_unsupported_history_processor_is_reported_without_evaluating_python(processor):
    source = f"> agent: codex\n> history:\n    enabled: false\n    processor: `{processor}`\n"
    diagnostics = compute_diagnostics(source, "history.kedi")
    assert any("native history processing" in item.message for item in diagnostics)


def test_explicit_none_disables_unsupported_processor_in_profile():
    source = """
> profile: remote:
    > agent: codex
    > history:
        enabled: false
        processor: `keep`
    > history:
        enabled: false
        processor: `None`
> use: remote
"""
    assert [item.code for item in compute_diagnostics(source, "history.kedi")] == [
        "K-HISTORY-PREFIX-CACHE"
    ]


def test_history_shorthand_does_not_erase_inherited_processor_diagnostic():
    source = """
> profile: remote:
    > agent: codex
    > history:
        enabled: false
        processor: `keep`
    > history: disabled
> use: remote
"""
    assert any(
        "native history processing" in item.message
        for item in compute_diagnostics(source, "history.kedi")
    )


def test_history_processor_parses_without_enabling_other_unknown_fields():
    with pytest.raises(Exception, match="history setting must be one of"):
        parse_program("> history:\n    enabled: true\n    arbitrary_callback: `keep`\n")
