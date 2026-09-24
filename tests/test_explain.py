from __future__ import annotations

import kedi
import pytest
from click.testing import CliRunner

from kedi.cli import main
from kedi.explain import explain_program, render_explain
from kedi.lang import parse_program


def test_explain_preserves_source_order_and_lexical_procedure_capture() -> None:
    program = parse_program(
        "> adapter: pydantic\n"
        "> model: first-model\n"
        "@first() -> str:\n"
        "  >> Answer is [answer: str].\n"
        "  = `answer`\n"
        "> model: second-model\n"
        "@second() -> str:\n"
        "  >> Answer is [answer: str].\n"
        "  = `answer`\n"
    )
    report = explain_program(program)
    assert [(call.scope, call.model.value) for call in report.calls] == [
        ("procedure first", "first-model"),
        ("procedure second", "second-model"),
    ]
    assert not report.diagnostics


def test_explain_does_not_evaluate_dynamic_model_or_invoke_query() -> None:
    calls = 0

    def select_model() -> str:
        nonlocal calls
        calls += 1
        raise AssertionError("explain must not evaluate model expressions")

    @kedi.query
    def answer() -> str:
        """kedi
        > model: `select_model()`
        >> The answer is [value: str].
        = `value`
        """

    report = kedi.explain(answer)
    assert calls == 0
    assert report.calls[0].model.value is None
    assert "dynamic expression" in report.calls[0].model.unresolved
    assert select_model is not None


def test_explain_profile_tool_metadata_and_requirements() -> None:
    program = parse_program(
        "> adapter: pydantic\n"
        "@lookup(value: str) -> str:\n"
        "  > tool:\n"
        "    name: find_value\n"
        "    risk: read_only\n"
        "    retries: 2\n"
        "  = `value`\n"
        "> profile: local:\n"
        "  > model: local-model\n"
        "  > requires: tool_registration\n"
        "  > use: lookup\n"
        "> use: local\n"
        ">> Result is [result: str].\n"
    )
    report = explain_program(program)
    assert len(report.calls) == 1
    call = report.calls[0]
    assert call.model.value == "local-model"
    assert [(tool.name, tool.risk, tool.retries) for tool in call.tools] == [
        ("find_value", "read_only", 2)
    ]
    assert any(
        req.name == "tool_registration" and req.status == "satisfied" for req in call.requirements
    )
    rendered = render_explain(report)
    assert "tool: find_value" in rendered
    assert "risk=read_only, retries=2, effects=unknown" in rendered


def test_explain_cli_parses_without_executing_python_or_leaking_prompt(tmp_path) -> None:
    source = tmp_path / "example.kedi"
    source.write_text(
        "> model: `side_effect()`\n>> Secret prompt payload 12345 [result: str].\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(main, ["explain", str(source)])
    assert result.exit_code == 0, result.output
    assert "dynamic expression" in result.output
    assert "Secret prompt payload" not in result.output
    assert "side_effect()" not in result.output


def test_explain_cli_fails_for_known_missing_requirement(tmp_path) -> None:
    source = tmp_path / "unsupported.kedi"
    source.write_text(
        "> adapter: a2a\n> requires: codemode\n>> Answer is [answer: str].\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(main, ["explain", str(source)])
    assert result.exit_code == 1
    assert "codemode missing" in result.output


def test_render_explain_distinguishes_unknown_effects_from_empty_set() -> None:
    from dataclasses import replace
    from kedi.explain import ExplainedTool

    report = explain_program(parse_program(">> Answer is [answer: str].\n"))
    call = replace(
        report.calls[0],
        tools=(
            ExplainedTool("unknown_tool", "external"),
            ExplainedTool("pure_tool", "local", "read_only", 0, frozenset()),
            ExplainedTool("writer", "local", "mutating", 1, frozenset({"write", "read"})),
        ),
    )
    rendered = render_explain(replace(report, calls=(call,)))
    assert "Answer is" not in rendered
    assert "unknown_tool (external, risk=unknown, retries=unknown, effects=unknown)" in rendered
    assert "pure_tool (local, risk=read_only, retries=0, effects=)" in rendered
    assert "writer (local, risk=mutating, retries=1, effects=read,write)" in rendered


def test_unknown_profile_invalidates_effective_values_until_overridden() -> None:
    report = explain_program(
        parse_program(
            "> adapter: a2a\n> model: old-model\n> use: imported_profile\n"
            ">> Answer is [answer: str].\n"
            "> adapter: pydantic\n> model: new-model\n>> Next answer is [next: str].\n"
        )
    )
    first, second = report.calls
    assert first.model.value is None and first.model.unresolved
    assert first.adapter.value is None and first.adapter.unresolved
    assert second.model.value == "new-model" and second.adapter.value == "pydantic"


def test_profile_resolution_uses_runtime_member_order_and_redacts_settings() -> None:
    report = explain_program(
        parse_program(
            "> profile: base:\n  > model: base-model\n"
            "> profile: combined:\n  > use: base\n  > model: local-model\n"
            "  > settings:\n    api_key: secret-never-render\n"
            "> use: combined\n>> Answer is [answer: str].\n"
        )
    )
    # Runtime profile definitions apply use-members after their own settings.
    assert report.calls[0].model.value == "base-model"
    assert dict(report.calls[0].configuration)["settings"].value == "configured (value withheld)"
    assert "secret-never-render" not in render_explain(report)


def test_disabling_codemode_clears_only_implicit_requirement() -> None:
    report = explain_program(
        parse_program(
            "> adapter: a2a\n> codemode: enabled\n> codemode: disabled\n"
            ">> Answer is [answer: str].\n"
            "> requires: codemode\n>> Another answer is [another: str].\n"
        )
    )
    assert not any(req.name == "codemode" for req in report.calls[0].requirements)
    assert any(
        req.name == "codemode" and req.status == "missing" for req in report.calls[1].requirements
    )


def test_explain_does_not_load_imports_or_execute_branch_loop_and_map() -> None:
    report = explain_program(
        parse_program("""
> import: unavailable_module
> adapter: pydantic
@tool() -> str:
  = ok
> use: tool
> use:
  tool
  external
  external
> budget:
  request_limit: 2
  tool_attempt_limit: 4
> if: `unknown_condition()`:
  > model: branch
  >> Answer is [answer: str].
> else:
  > model: alternative
  >> Answer is [answer: str].
> loop [item]: `unknown_items()`:
  >> Answer is [answer: str].
> map:
  >> Mapped answer is [mapped: str].
> loop: `unknown_condition()`:
  >> Loop answer is [loop_answer: str].
""")
    )
    assert any("conditional" in call.scope for call in report.calls)
    assert any("alternative" in call.scope for call in report.calls)
    assert any("map" in call.scope for call in report.calls)
    assert all(
        any("imported symbols" in message for message in call.unresolved) for call in report.calls
    )


def test_explain_python_tools_profile_and_bind_reload(tmp_path) -> None:
    @kedi.tool(name="read_value", retries=1, risk="read_only")
    def read() -> str:
        raise AssertionError("must not run")

    @kedi.query(tools=[read], model="example-model")
    def answer() -> str:
        """kedi
        >> The answer is [value: str].
        = <value>
        """

    report = kedi.explain(answer)
    assert report.calls[0].tools[0].retries == 1
    path = tmp_path / "example.kedi"
    path.write_text("> model: file-model\n>> Answer is [answer: str].\n= <answer>\n")

    @kedi.bind(file=path, reload=True)
    def bound() -> str:
        raise AssertionError("must not run")

    assert kedi.explain(bound).calls[0].model.value == "file-model"
    path.write_text("> model: changed-model\n>> Answer is [answer: str].\n= <answer>\n")
    assert kedi.explain(bound).calls[0].model.value == "changed-model"


def test_explain_invalid_targets_empty_report_and_cli_errors(tmp_path) -> None:
    for target in (object(), lambda: None):
        with pytest.raises(TypeError):
            kedi.explain(target)
    assert (
        render_explain(explain_program(parse_program("= done\n"))) == "No model call sites found."
    )
    runner = CliRunner()
    assert "Inspect model calls" in runner.invoke(main, ["explain", "--help"]).output
    assert runner.invoke(main, ["explain"]).exit_code != 0
    assert runner.invoke(main, ["explain", str(tmp_path / "missing.kedi")]).exit_code != 0


def test_explain_profile_cycles_unknown_effects_and_capability_lookup_failure(monkeypatch) -> None:
    import importlib

    module = importlib.import_module("kedi.explain")

    report = explain_program(
        parse_program(
            "> profile: recursive:\n  > use: recursive\n> use: recursive\n"
            ">> Answer is [answer: str].\n"
        )
    )
    assert report.calls[0].model.value is None
    assert any("cyclic" in reason for reason in report.calls[0].unresolved)

    def unavailable(name):
        raise ValueError("metadata unavailable")

    monkeypatch.setattr(module, "adapter_capabilities", unavailable)
    report = explain_program(
        parse_program(
            "> adapter: pydantic\n> requires: codemode\n> requires: codemode\n"
            ">> Answer is [answer: str].\n"
        )
    )
    assert report.calls[0].requirements[0].status == "unresolved"


def test_explain_handles_legacy_partial_asts_without_evaluating_unknowns() -> None:
    from kedi.lang.ast import (
        Program,
        ModelDirectiveStmt,
        ProcDefStmt,
        UseDirectiveStmt,
        TemplateLine,
        IfStmt,
        LoopStmt,
        BudgetDirectiveStmt,
        ProfileDefStmt,
    )

    program = Program(
        toplevel_body=[
            ModelDirectiveStmt(),
            ProcDefStmt(),
            UseDirectiveStmt(),
            BudgetDirectiveStmt(tool_attempt_limit=0),
            BudgetDirectiveStmt(request_limit=0),
            IfStmt(condition_code="unknown()", body=[TemplateLine()]),
            LoopStmt(binder="item", iterable_code="unknown()", body=[TemplateLine()]),
            ProfileDefStmt(name="tools", tool_names=["external"]),
            UseDirectiveStmt(target="tools"),
            TemplateLine(),
        ]
    )
    report = explain_program(program)
    assert len(report.calls) == 3
    assert report.calls[-1].tools[0].name == "external"
    assert report.calls[0].source_location.path is None
    assert report.calls[0].model.unresolved


@pytest.mark.parametrize("enabled", [None, "true", "false", "`must_not_run()`"])
def test_explain_artifact_mcp_and_budget_configuration(enabled) -> None:
    source = "> adapter: pydantic\n> artifacts:\n"
    source += f"  enabled: {enabled}\n" if enabled is not None else "  threshold: 2048\n"
    source += "> artifacts:\n  threshold: 4096\n> mcp:\n  command: unavailable\n"
    source += "> codemode: enabled\n>> Answer is [answer: str].\n"
    report = explain_program(parse_program(source))
    names = {requirement.name for requirement in report.calls[0].requirements}
    assert "codemode" in names and "mcp" in names
    assert ("artifacts" in names) == (enabled in (None, "true"))
    rendered = render_explain(report)
    assert "artifacts:" in rendered and "requires: mcp" in rendered
    assert "must_not_run()" not in rendered


def test_python_profile_inspection_distinguishes_unknown_and_known_tools() -> None:
    from dataclasses import replace
    from kedi.agent_profile import AgentProfile

    @kedi.query(requires=["tool_registration"])
    def answer() -> str:
        """kedi
        >> The answer is [value: str].
        = <value>
        """

    source = answer.__kedi_inspection_source__
    answer.__kedi_inspection_source__ = replace(
        source,
        profile=AgentProfile(requirements=("tool_registration",), tools=("external", "external")),
    )
    call = kedi.explain(answer).calls[0]
    assert [tool.name for tool in call.tools] == ["external"]
    assert any(requirement.name == "tool_registration" for requirement in call.requirements)

    from kedi.api.state import _config_var

    token = _config_var.set(replace(_config_var.get(), adapter=object(), adapter_shortname=None))
    try:
        assert kedi.explain(answer).calls[0].adapter.unresolved
    finally:
        _config_var.reset(token)
