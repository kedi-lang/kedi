from __future__ import annotations

from dataclasses import replace
import re

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
import pytest

from kedi.agent_adapter.adapters import LangChainAdapter, PydanticAdapter
from kedi.agent_profile.telemetry import capability_trace_id
from kedi.lang import compile_program, parse_program
from kedi.telemetry import span
from tests.telemetry_helpers import RecordingTelemetryBackend, installed_recording_backend


class PydanticCapabilities(PydanticAdapter):
    capabilities = replace(
        PydanticAdapter.capabilities, supports_capabilities=True, supports_deferred_tools=True
    )


class LangChainCapabilities(LangChainAdapter):
    capabilities = replace(LangChainAdapter.capabilities, supports_capabilities=True)


class ToolChatModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def make_adapter(kind, *, tool_name=None, arguments=None):
    if kind == "langchain":
        responses = []
        if tool_name is not None:
            responses.append(
                AIMessage(
                    content="",
                    tool_calls=[{"name": tool_name, "args": arguments or {}, "id": "call"}],
                )
            )
        responses.append(AIMessage(content="done"))
        return LangChainCapabilities(ToolChatModel(responses=responses))

    def respond(messages, info):
        if tool_name is not None and not any(
            isinstance(message, ModelResponse) for message in messages
        ):
            return ModelResponse([ToolCallPart(tool_name, arguments or {}, "call")])
        return ModelResponse([TextPart("done")])

    return PydanticCapabilities(FunctionModel(respond))


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
@pytest.mark.parametrize("codemode", [False, True])
def test_owned_tool_trace_records_actual_inner_call_and_artifact_policy(kind, codemode):
    def read_document() -> str:
        return "source evidence " * 100

    source = (
        "> approval: deny\n"
        '> artifacts:\n  threshold: 64b\n  exclude_tools: `["read_document"]`\n'
        "> capability: reader:\n  > approval: allow\n  > use: read_document\n"
        "> use: reader\n"
        + ("> codemode:\n  preload_tools: read_document\n" if codemode else "")
        + "[reply] << Read the source document.\n= <reply>\n"
    )
    adapter = make_adapter(
        kind,
        tool_name="execute_code" if codemode else "read_document",
        arguments={"code": "await read_document()"} if codemode else {},
    )
    runtime = compile_program(
        parse_program(source), adapter=adapter, runtime_globals={"read_document": read_document}
    )
    try:
        with installed_recording_backend() as backend:
            assert runtime.run_main() == "done"
    finally:
        runtime.artifacts.close()
    calls = [
        item
        for item in backend.finished_spans
        if item.operation == "call_tool"
        and item.attributes.get("gen_ai.tool.name") == "read_document"
    ]
    assert len(calls) == 1
    call = calls[0]
    assert call.attributes["kedi.tool.capability.owners"] == (
        capability_trace_id("<program>::reader"),
    )
    assert call.attributes["kedi.tool.approval.source"] == "capability"
    approvals = [
        event
        for item in backend.finished_spans
        for event in item.events
        if event.name == "tool approval"
        and event.attributes.get("gen_ai.tool.name") == "read_document"
    ]
    assert len(approvals) == 1
    assert approvals[0].attributes["kedi.approval.decision"] == "allow"
    assert approvals[0].attributes["gen_ai.tool.call.id"] == call.attributes["gen_ai.tool.call.id"]
    assert any(event.attributes.get("kedi.tool.artifact.excluded") is True for event in call.events)
    if codemode:
        ancestors = []
        parent = call.parent
        while parent is not None:
            ancestors.append(parent)
            parent = parent.parent
        assert any(item.operation == "execute_code" for item in ancestors)
        assert any(event.name == "tool artifact admission deferred" for event in call.events)


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
@pytest.mark.parametrize("policy", ["allow", "deny"])
def test_repeated_tool_calls_correlate_approval_execution_and_artifacts(kind, policy):
    executed = []

    def read_document(document_id: int) -> str:
        executed.append(document_id)
        return f"Document {document_id}: " + "evidence " * 100

    calls = [
        {"name": "read_document", "args": {"document_id": number}, "id": f"read-{number}"}
        for number in (1, 2)
    ]
    if kind == "langchain":
        adapter = LangChainCapabilities(
            ToolChatModel(
                responses=[AIMessage(content="", tool_calls=calls), AIMessage(content="done")]
            )
        )
    else:

        def respond(messages, info):
            if not any(isinstance(message, ModelResponse) for message in messages):
                return ModelResponse(
                    [ToolCallPart(call["name"], call["args"], call["id"]) for call in calls]
                )
            return ModelResponse([TextPart("done")])

        adapter = PydanticCapabilities(FunctionModel(respond))

    runtime = compile_program(
        parse_program(
            '> artifacts:\n  threshold: 64b\n  exclude_tools: `["read_document"]`\n'
            f"> capability: reader:\n  > approval: {policy}\n  > use: read_document\n"
            "> use: reader\n[reply] << Read both documents.\n= <reply>\n"
        ),
        adapter=adapter,
        runtime_globals={"read_document": read_document},
    )
    try:
        with installed_recording_backend() as backend:
            assert runtime.run_main() == "done"
    finally:
        runtime.artifacts.close()

    events = [event for item in backend.finished_spans for event in item.events]
    approvals = [event for event in events if event.name == "tool approval"]
    assert {
        (event.attributes["gen_ai.tool.call.id"], event.attributes["kedi.approval.decision"])
        for event in approvals
    } == {("read-1", policy), ("read-2", policy)}
    tool_spans = [item for item in backend.finished_spans if item.operation == "call_tool"]
    expected = {"read-1", "read-2"} if policy == "allow" else set()
    assert {item.attributes["gen_ai.tool.call.id"] for item in tool_spans} == expected
    assert {
        event.attributes["gen_ai.tool.call.id"]
        for event in events
        if event.name == "tool artifact policy"
    } == expected
    assert sorted(executed) == ([1, 2] if policy == "allow" else [])


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
@pytest.mark.parametrize("capture_source", [False, True])
def test_executed_reset_reuse_and_request_snapshots_are_observable(kind, capture_source, tmp_path):
    source_path = str(tmp_path / "private-source.kedi")
    runtime = compile_program(
        parse_program(
            "> capability: concise:\n  > instructions: Keep the reply brief.\n"
            "> instructions: Base guidance.\n> use: concise\n> use: concise\n"
            "> instructions: Base guidance.\n[first] << Report the status.\n"
            "> show: <first>\n> use: concise\n[second] << Report the status again.\n= <second>\n",
            source_path=source_path,
        ),
        adapter=make_adapter(kind),
    )
    backend = RecordingTelemetryBackend(
        capture=frozenset({"source_path"}) if capture_source else frozenset()
    )
    try:
        with installed_recording_backend(backend), span("runtime", "fixture", "fixture"):
            assert runtime.run_main() == "done"
    finally:
        runtime.artifacts.close()
    events = [event for item in backend.finished_spans for event in item.events]
    resets = [event for event in events if event.name == "instructions replaced"]
    identity = capability_trace_id(f"{source_path}::concise")
    assert any(
        event.attributes["kedi.instructions.suppressed_capabilities"] == (identity,)
        for event in resets
    )
    applied = [event.attributes for event in events if event.name == "capability applied"]
    assert [item["kedi.capability.reused"] for item in applied] == [False, True, True]
    assert applied[-1]["kedi.capability.instructions.restored"] is True
    snapshots = [event.attributes for event in events if event.name == "capability snapshot"]
    assert sorted(item["kedi.capability.instructions.enabled"] for item in snapshots) == [
        False,
        True,
    ]
    for item in snapshots:
        assert item["kedi.capability.id"] == identity
        assert (item.get("kedi.source.path") == source_path) is capture_source
        assert ("kedi.capability.identity" in item) is capture_source
        assert "Keep the reply brief." not in str(item)


def test_native_load_attempts_and_confirmed_activation_remain_run_local():
    requests = []

    def read_document() -> str:
        return "revision 7"

    def respond(messages, info):
        step = len(requests) % 4
        requests.append(info)
        if step == 0:
            return ModelResponse([ToolCallPart("load_capability", {"id": "unknown"}, "invalid")])
        if step == 1:
            identity = re.search(r"kedi_reader_[a-f0-9]{12}", info.instructions).group()
            return ModelResponse(
                [
                    ToolCallPart("load_capability", {"id": identity}, "load"),
                    ToolCallPart("load_capability", {"id": identity}, "duplicate"),
                ]
            )
        if step == 2:
            return ModelResponse([ToolCallPart("read_document", {}, "read")])
        return ModelResponse([TextPart("done")])

    runtime = compile_program(
        parse_program(
            "> capability: reader:\n  > defer: true\n  > approval: allow\n"
            "  > use: read_document\n> use: reader\n[reply] << Read the revision.\n= <reply>\n"
        ),
        adapter=PydanticCapabilities(FunctionModel(respond), retries=3),
        runtime_globals={"read_document": read_document},
    )
    try:
        with installed_recording_backend() as backend:
            for _ in range(2):
                assert runtime.run_main() == "done"
    finally:
        runtime.artifacts.close()
    loads = [item for item in backend.finished_spans if item.operation == "load_capability"]
    assert len(loads) == 6
    assert [item.attributes["kedi.capability.load.outcome"] for item in loads].count(
        "delivered"
    ) == 2
    activations = [
        event
        for item in backend.finished_spans
        for event in item.events
        if event.name == "capability activated"
    ]
    assert len(activations) == 2
    assert all(event.attributes["kedi.capability.resumed"] is False for event in activations)
    assert all(
        event.attributes["kedi.capability.id"] == capability_trace_id("<program>::reader")
        for event in activations
    )
