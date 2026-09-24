from __future__ import annotations

import pytest
from lsprotocol import types as lsp
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.messages import ModelResponse, TextPart

from kedi.agent_adapter.adapters.pydantic import PydanticAdapter
from kedi.lang import compile_program, parse_program
from kedi.lsp.features.capabilities import _budget_capability_diagnostics
from kedi.lsp.features.capability_state import _AdapterDiagnosticState
from kedi.lsp.features.completion import compute_completion
from kedi.lsp.features.diagnostics import compute_diagnostics
from kedi.lang.ast import BudgetDirectiveStmt
from kedi.run_budget import RunBudgetExceeded


def test_partial_budget_nodes_are_safe_for_editor_parsing() -> None:
    from tests.test_parser_grammar import _FakeNode, _fake_walker

    walker = _fake_walker()
    assert walker._walk_budget_directive(_FakeNode("budget_directive")).request_limit is None
    node = _FakeNode(
        "budget_directive",
        fields={"body": _FakeNode("budget_body", children=[_FakeNode("line_comment")])},
    )
    assert walker._walk_budget_directive(node).tool_attempt_limit is None


def test_budget_declarations_preserve_profile_when_no_ambient_ledger(monkeypatch) -> None:
    import kedi.run_budget as module
    import kedi.lang.compiler.top_level as top_level

    monkeypatch.setattr(module, "current_run_budget", lambda: None)
    monkeypatch.setattr(top_level, "current_run_budget", lambda: None)
    runtime = compile_program(
        parse_program("""
> budget:
  request_limit: 1
@worker() -> str:
  > budget:
    request_limit: 0
  = done
= <worker()>
""")
    )
    assert runtime.run_main() == "done"


def test_native_procedure_budget_stops_generation_and_root_remains_fresh() -> None:
    calls = 0

    async def respond(messages, info):
        nonlocal calls
        calls += 1
        return ModelResponse(parts=[TextPart("ok")])

    runtime = compile_program(
        parse_program("""
> budget:
  request_limit: 2
@worker() -> str:
  > budget:
    request_limit: 0
  [answer] << Answer briefly.
  = <answer>
= <worker()>
"""),
        adapter=PydanticAdapter(FunctionModel(respond)),
    )
    for _ in range(2):
        with pytest.raises(RunBudgetExceeded):
            runtime.run_main()
    assert calls == 0


@pytest.mark.parametrize("body", ["unknown: 1", "request_limit: 1\n  request_limit: 2"])
def test_invalid_budget_fields_are_parser_errors(body) -> None:
    with pytest.raises(Exception, match="budget field"):
        parse_program(f"> budget:\n  {body}\n")


def test_budget_lsp_completion_and_capability_diagnostics() -> None:
    source = "> budget:\n  req"
    items = compute_completion(source, "/tmp/budget.kedi", lsp.Position(line=1, character=5))
    assert {item.label for item in items} == {"request_limit", "tool_attempt_limit"}
    source = "> agent: acp\n> budget:\n  request_limit: 1\n  tool_attempt_limit: 2\n"
    diagnostics = compute_diagnostics(source, "budget.kedi")
    assert len([item for item in diagnostics if item.code == "K-ADAPTER-BUDGET"]) == 2
    profile = "> profile: remote:\n  > agent: acp\n  > budget:\n    request_limit: 1\n"
    assert any(
        item.code == "K-ADAPTER-BUDGET" for item in compute_diagnostics(profile, "profile.kedi")
    )
    assert (
        _budget_capability_diagnostics(
            _AdapterDiagnosticState(), BudgetDirectiveStmt(request_limit=1)
        )
        == []
    )
