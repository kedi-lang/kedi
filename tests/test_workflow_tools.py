"""`> workflow:` block form: `> mode:` selects the orchestration, `> tools:` the control tools."""

import asyncio

import pytest
from lsprotocol import types as lsp
from pydantic_ai.models.test import TestModel

from kedi.agent_adapter import PydanticAdapter
from kedi.agent_profile import AgentProfile
from kedi.lang import compile_program, parse_program
from kedi.lang.parser.diagnostics import KediParseError
from kedi.lsp.features.completion import compute_completion
from kedi.lsp.features.diagnostics import compute_diagnostics
from tests.lsp_helpers import hover_text_at
from tests.test_subagents import RecordingPydanticAdapter

CHILD = "> profile: worker:\n  > adapter: pydantic\n"
CONTROL = {
    "continue_subagent",
    "wait_subagent",
    "cancel_subagent",
    "subagent_status",
    "send_subagent_input",
}


def _manager(block: str) -> str:
    indented = "".join(f"  {line}\n" for line in block.splitlines())
    return (
        CHILD
        + "> profile: manager:\n  > adapter: pydantic\n  > subagent: worker\n"
        + indented
        + "> use: manager\n"
    )


def _tools(block: str) -> dict:
    adapter = PydanticAdapter(TestModel())
    runtime = compile_program(parse_program(_manager(block)), adapter=adapter)
    profile = runtime._evaluate_profile_definition(runtime._profile_registry["manager"])
    return {spec.name: spec for spec in runtime._resolve_agent_tool_specs(profile, adapter)}


def _control(block: str) -> set[str]:
    return set(_tools(block)) & CONTROL


def test_a_mode_only_block_is_the_shorthand():
    assert _control("> workflow:\n  > mode: delegate") == _control("> workflow: delegate")
    assert set(_tools("> workflow:\n  > mode: blocking")) == set(_tools("> workflow: blocking"))


def test_tools_default_to_every_control_tool_and_all_says_so_explicitly():
    assert _control("> workflow:\n  > tools: all") == CONTROL
    assert _control("") == CONTROL


def test_a_tools_block_without_mode_selects_among_the_delegate_tools():
    block = "> workflow:\n  > tools:\n    wait_subagent\n    cancel_subagent"
    tools = _tools(block)
    assert _control(block) == {"wait_subagent", "cancel_subagent"}
    assert "delegate_task" in tools
    assert "background" in tools["delegate_task"].json_schema["properties"]


def test_background_is_offered_only_together_with_wait_subagent():
    tools = _tools("> workflow:\n  > tools: continue_subagent")
    assert _control("> workflow:\n  > tools: continue_subagent") == {"continue_subagent"}
    assert "background" not in tools["delegate_task"].json_schema["properties"]
    assert "background" not in tools["continue_subagent"].json_schema["properties"]
    assert "wait_subagent" not in tools["delegate_task"].description


def test_live_input_is_one_selectable_control_tool():
    assert _control("> workflow:\n  > tools: send_subagent_input") == {"send_subagent_input"}
    assert "send_subagent_input" not in _control("> workflow:\n  > tools: wait_subagent")


def test_dynamic_exposes_neither_delegation_nor_live_input_tools():
    tools = _tools("> workflow:\n  > mode: dynamic")
    assert "run_workflow" in tools
    assert not CONTROL & set(tools)


@pytest.mark.parametrize("mode", ["blocking", "dynamic"])
@pytest.mark.parametrize("tools", ["> tools: wait_subagent", "> tools: all"])
def test_tools_with_blocking_or_dynamic_is_one_error_that_names_both_and_the_fix(mode, tools):
    with pytest.raises(KediParseError) as caught:
        parse_program(_manager(f"> workflow:\n  > mode: {mode}\n  {tools}"))
    [diagnostic] = [d for d in caught.value.diagnostics if d.code == "K-WORKFLOW-TOOLS"]
    for needle in ("mode: blocking", "mode: dynamic", "mode: delegate", "Remove `tools`"):
        assert needle in diagnostic.message
    assert diagnostic.span.start_line == 8  # 1-based `> tools:` line


def test_the_editor_reports_the_same_error_on_the_tools_line():
    source = _manager("> workflow:\n  > mode: dynamic\n  > tools: wait_subagent")
    [diagnostic] = [d for d in compute_diagnostics(source, None) if d.code == "K-WORKFLOW-TOOLS"]
    assert "`mode: delegate`" in diagnostic.message
    assert diagnostic.range.start.line == 7  # 0-based `> tools:` line


@pytest.mark.parametrize(
    ("tools", "message"),
    [
        ("> tools: wait", "unknown subagent control tool 'wait'"),
        ("> tools:\n    wait_subagent\n    wait_subagent", "more than once"),
        ("> tools: cancel_subagent", "only apply to background runs"),
        ("> tools:\n    all\n    wait_subagent", "use `all` alone or list the tools"),
        ("> tools:\n    subagent_status\n    continue_subagent", "add `wait_subagent`"),
    ],
)
def test_invalid_tool_selections_say_what_to_change(tools, message):
    with pytest.raises(KediParseError, match=message):
        parse_program(_manager(f"> workflow:\n  {tools}"))


def test_mode_and_tools_are_each_defined_once():
    with pytest.raises(KediParseError, match="defines `mode` twice"):
        parse_program(_manager("> workflow:\n  > mode: delegate\n  > mode: delegate"))
    with pytest.raises(KediParseError, match="defines `tools` twice"):
        parse_program(_manager("> workflow:\n  > tools: all\n  > tools: all"))


def test_an_unknown_mode_in_a_block_is_rejected():
    with pytest.raises(KediParseError, match="'delegate', 'dynamic' or 'blocking'"):
        parse_program(_manager("> workflow:\n  > mode: sequential"))


def test_a_later_workflow_replaces_mode_and_tools_together():
    base = AgentProfile(workflow_mode="delegate", workflow_tools=("wait_subagent",))
    replaced = base.merge(AgentProfile(workflow_mode="delegate"))
    assert replaced.workflow_tools is None
    assert base.merge(AgentProfile.empty()).workflow_tools == ("wait_subagent",)


@pytest.mark.parametrize(
    ("block", "can_report"),
    [
        ("> workflow: delegate", True),
        ("> workflow:\n  > tools: send_subagent_input", True),
        ("> workflow:\n  > tools: wait_subagent", False),
        ("> workflow: blocking", False),
        ("> workflow: dynamic", False),
    ],
)
def test_children_can_message_the_parent_only_when_it_can_answer(block, can_report):
    adapter = RecordingPydanticAdapter()
    runtime = compile_program(parse_program(_manager(block)), adapter=adapter)

    async def scenario():
        async with runtime.subagents(parent="manager") as agents:
            job = await agents.start("worker", task="work")
            await job.wait()

    asyncio.run(scenario())
    assert ("send_parent_message" in adapter.invocations[-1][2]) is can_report


def _complete(block: str, *, explicit: bool = False) -> set[str]:
    """Completions at the end of ``block``, written as a profile body."""
    source = CHILD + "> profile: manager:\n  > subagent: worker\n" + block
    last = source.split("\n")[-1]
    position = lsp.Position(line=source.count("\n"), character=len(last))
    return {item.label for item in compute_completion(source, None, position, explicit=explicit)}


def test_the_editor_completes_workflow_sub_directives_and_their_values():
    assert _complete("  > workflow:\n    > ") == {"mode", "tools"}
    assert _complete("  > workflow:\n    > tools:\n      > ") == set()
    assert _complete("  > workflow:\n    > mode: ", explicit=True) == {
        "delegate",
        "dynamic",
        "blocking",
    }
    assert _complete("  > workflow:\n    > tools:\n      wait_subagent\n      ", explicit=True) == {
        *CONTROL,
        "all",
    }


def test_hover_lists_the_selected_control_tools():
    source = _manager("> workflow:\n  > tools: wait_subagent")
    body = hover_text_at(source, "example.kedi", "workflow")
    assert "`wait_subagent`" in body
    assert "besides `delegate_task`" in body
    assert "Control tools: `wait_subagent`" in hover_text_at(source, "example.kedi", "manager")
