"""The rendered request: what a custom adapter forwards (#232, "Kedi renders, the adapter
transports")."""

from __future__ import annotations

from collections.abc import Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import pytest

from kedi.agent_adapter import (
    AdapterCapabilities,
    AgentAdapterKind,
    RenderedRequest,
    ToolSpec,
    current_request,
    has_current_request,
)
from kedi.agent_adapter.request import (
    RawPromptBuilder,
    RunRequestAssembler,
    TextOutput,
    TypedOutput,
    request_protocol,
)
from kedi.agent_adapter.request.context import subagent_entry_scope
from kedi.agent_adapter.request.prompts import PROTOCOL_VERSION
from kedi.agent_adapter.request.protocol import protocol_parts, tool_catalog
from kedi.agent_profile import AgentProfile
from kedi.lang import compile_program, parse_program


class RenderingAdapter:
    """Forwards ``current_request().render()``, the way a custom adapter should."""

    kind: AgentAdapterKind = "agent-framework"
    shortname = "rendering"

    def __init__(self, capabilities: AdapterCapabilities, reply: Any) -> None:
        self.capabilities = capabilities
        self._reply = reply
        self.rendered: list[RenderedRequest] = []
        self.instructions: list[str | None] = []
        self._tools: ContextVar[tuple[ToolSpec, ...]] = ContextVar(
            f"rendering_tools_{id(self)}", default=()
        )

    @contextmanager
    def tool_scope(self, tools: Sequence[object] = ()) -> Generator[None, None, None]:
        token = self._tools.set(tuple(tool for tool in tools if isinstance(tool, ToolSpec)))
        try:
            yield
        finally:
            self._tools.reset(token)

    def register_tool(self, tool: ToolSpec) -> None:
        self._tools.set((*self._tools.get(), tool))

    def invoke_sync(self, *, prompt: str, instructions: str | None = None, **_: Any) -> str:
        self._record(instructions)
        return str(self._reply)

    def produce_sync(self, *, template: str, instructions: str | None = None, **_: Any) -> Any:
        self._record(instructions)
        return self._reply

    def _record(self, instructions: str | None) -> None:
        self.instructions.append(instructions)
        self.rendered.append(current_request().render())


NATIVE = AdapterCapabilities(
    supports_structured_output=True, supports_tool_registration=True, supports_artifacts=True
)
PROMPTED = AdapterCapabilities(
    supports_structured_output=True,
    supports_tool_registration=True,
    supports_artifacts=True,
    supports_native_structured_output=False,
    supports_native_tool_calling=False,
)

PROGRAM = """> instructions: Be brief.

@lookup(city: str) -> str:
  ###
  Look up a city.
  ###
  = sunny

> use: lookup
>> The weather in Ankara is [weather].
= <weather>
"""


def _run(source: str, adapter: Any) -> Any:
    return compile_program(parse_program(source), adapter=adapter).run_main()


def test_a_native_backend_gets_the_request_as_kedi_rendered_it() -> None:
    adapter = RenderingAdapter(NATIVE, {"weather": "sunny"})
    assert _run(PROGRAM, adapter) == "sunny"
    (rendered,) = adapter.rendered
    # Without protocol parts the system text is exactly what invoke/produce receive.
    assert rendered.system == adapter.instructions[0]
    assert [part.identity for part in rendered.instruction_parts][:1] == ["template.instructions"]
    assert "The weather in Ankara is [weather]." in rendered.user
    assert rendered.media == ()
    assert rendered.output_json_schema is not None
    assert "weather" in rendered.output_json_schema["properties"]
    assert rendered.output_tool_name == "final_result"
    # Kedi's tools, then its artifact tools: everything the model may call.
    names = [tool.name for tool in rendered.tools]
    assert names[0] == "lookup" and "read_artifact" in names
    assert (rendered.prompt_kind, rendered.call_kind) == ("template", "produce")


def test_a_backend_without_native_output_or_tools_gets_kedis_protocol_parts() -> None:
    adapter = RenderingAdapter(PROMPTED, {"weather": "sunny"})
    _run(PROGRAM, adapter)
    (rendered,) = adapter.rendered
    identities = [part.identity for part in rendered.instruction_parts]
    assert identities[-2:] == ["protocol.output", "protocol.tools"]
    assert rendered.system is not None and rendered.system.endswith("</ToolProtocol>")
    # The adapter's own instructions argument and the request's parts stay as they were.
    assert "<OutputProtocol>" not in (adapter.instructions[0] or "")
    assert '<Tool name="lookup">\n<Description>Look up a city.' in rendered.system


def test_protocol_parts_follow_the_capabilities() -> None:
    assert request_protocol(NATIVE) == frozenset()
    assert request_protocol(PROMPTED) == frozenset({"output", "tools"})
    assert request_protocol(AdapterCapabilities(supports_native_tool_calling=False)) == frozenset(
        {"tools"}
    )


def test_protocol_text_is_versioned() -> None:
    """Any wording change bumps PROTOCOL_VERSION and this snapshot together."""

    assert PROTOCOL_VERSION == 1
    request = RunRequestAssembler().assemble(
        prompt=RawPromptBuilder(content="List two.").build(),
        instructions=(),
        conversation=None,
        tool_specs=(ToolSpec(fn=lambda: "x", name="pick", description="Pick one."),),
        output=TypedOutput(output_type=list[str]),
        run_id="r",
        turn_sequence=0,
        call_kind="produce",
        protocol=("output", "tools"),
    )
    assert request.render().system == (
        "<OutputProtocol>\n"
        "Answer with a JSON value that matches the output schema below, and nothing else: no "
        "prose and no code fences.\n"
        '<OutputSchema>{"items": {"type": "string"}, "type": "array"}</OutputSchema>\n'
        "</OutputProtocol>\n\n"
        "<ToolProtocol>\n"
        "You can call the tools below. Answer with exactly one JSON object and nothing else. "
        'To call a tool: {"action": "call_tool", "callId": "<a new id>", "name": "<tool name>", '
        '"arguments": {<arguments matching its input schema>}}. Call one tool per answer; its '
        "result comes back as a tool message. "
        'To finish: {"action": "final", "data": <a value matching the output schema>}.\n'
        "<Tools>\n"
        '<Tool name="pick">\n<Description>Pick one.</Description>\n'
        '<InputSchema>{"type": "object"}</InputSchema>\n</Tool>\n'
        "</Tools>\n"
        "</ToolProtocol>"
    )


def test_text_answers_finish_with_text_and_empty_protocols_add_nothing() -> None:
    tool = ToolSpec(fn=lambda: "x", name="pick").name
    assert protocol_parts(frozenset({"output"}), output_json_schema=None, tools=()) == ()
    request = RunRequestAssembler().assemble(
        prompt=RawPromptBuilder(content="Hi.").build(),
        instructions=(),
        conversation=None,
        tool_specs=(ToolSpec(fn=lambda: "x", name=tool),),
        output=TextOutput(),
        run_id="r",
        turn_sequence=0,
        call_kind="invoke",
        protocol=("tools",),
    )
    rendered = request.render()
    assert rendered.output_json_schema is None and rendered.output_tool_name is None
    assert '{"action": "final", "text": "<your answer>"}' in (rendered.system or "")
    assert tool_catalog(()) == "<Tools>\n\n</Tools>"


def test_current_request_is_refused_outside_a_call_and_while_running_a_child() -> None:
    assert not has_current_request()
    with pytest.raises(RuntimeError, match="only available inside an adapter call"):
        current_request()
    with subagent_entry_scope():
        assert not has_current_request()
        with pytest.raises(RuntimeError, match="runs a subagent child"):
            current_request()


class ChildAwareAdapter(RenderingAdapter):
    """Renders Kedi's request when there is one and falls back to its arguments otherwise."""

    def __init__(self) -> None:
        super().__init__(
            AdapterCapabilities(
                supports_structured_output=True,
                supports_tool_registration=True,
                supports_artifacts=True,
                supports_subagents=True,
            ),
            "child answer",
        )
        self.fallbacks: list[str] = []

    def invoke_sync(self, *, prompt: str, instructions: str | None = None, **_: Any) -> str:
        if has_current_request():
            self.rendered.append(current_request().render())
        else:
            self.fallbacks.append(prompt)
        return "child answer"


def test_a_text_subagent_child_falls_back_to_its_arguments() -> None:
    adapter = ChildAwareAdapter()
    # Kedi's delegation calls the child's invoke inside this scope.
    with subagent_entry_scope():
        assert adapter.invoke_sync(prompt="Summarize the report.") == "child answer"
    assert adapter.fallbacks == ["Summarize the report."] and adapter.rendered == []


def test_a_json_schema_output_reaches_the_model_as_given_and_is_checked() -> None:
    from pydantic import ValidationError

    from kedi.agent_adapter.request import JsonSchemaOutput, call_request, output_validator
    from kedi.agent_adapter.request.diagnostics import request_diagnostics

    schema = {
        "type": "object",
        "title": "KediSubagentResult",
        "properties": {"ocean": {"type": "string", "description": "An ocean's name."}},
        "required": ["ocean"],
        "additionalProperties": False,
    }
    output = JsonSchemaOutput(schema=schema)
    request = call_request(
        prompt="Name an ocean.",
        output=output,
        instructions=None,
        profile=AgentProfile.empty(),
        capabilities=AdapterCapabilities(supports_native_structured_output=False),
        tools=(),
    )
    rendered = request.render()
    assert rendered.call_kind == "produce"
    assert rendered.output_json_schema == schema
    # A backend without native structured output gets the schema in Kedi's protocol part.
    assert "An ocean's name." in (rendered.system or "")
    validator = output_validator(output)
    assert validator is not None
    assert validator.validate_python({"ocean": "Pacific"}) == {"ocean": "Pacific"}
    with pytest.raises(ValidationError, match="'ocean' is a required property"):
        validator.validate_python({"sea": "Baltic"})
    diagnostics = request_diagnostics(request)
    assert diagnostics.output_kind == "json_schema"
