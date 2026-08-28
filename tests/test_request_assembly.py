from __future__ import annotations

import asyncio
from dataclasses import replace
from types import MappingProxyType
from typing import Any, cast

import pytest

from kedi.agent_adapter import conversation as conversation_module
from kedi.agent_adapter.conversation import ConversationState
from kedi.agent_adapter.request import (
    AdapterCallOptions,
    AdapterRunOutcome,
    AgentRunRequest,
    ClaimPromptBuilder,
    ConversationCoordinates,
    InstructionPart,
    KediTemplatePromptBuilder,
    OutputContract,
    RawPromptBuilder,
    RequestDiagnostics,
    RunMetadata,
    RunRequestAssembler,
    SchemaOutput,
    TextOutput,
    ToolDefinition,
    TypedOutput,
    conversation_lane_identity,
    lower_request_sync,
    request_diagnostics,
)
from kedi.agent_adapter.tool_override import ToolSpec
from kedi.agent_profile.mcp import McpServerSpec
from kedi.core import _type_display_name


def test_prompt_builders_preserve_existing_wire_text() -> None:
    template = KediTemplatePromptBuilder(
        template="Summarize Kedi. Return [summary].",
        output_fields=("summary",),
    ).build()
    assert template.content == (
        "Fill each unescaped [name] or [name: type] output field with only that field's "
        "replacement value; never return the completed template as a field value. "
        "Escaped square brackets are literal text, not output fields; for example, "
        "\\[ok\\] represents the literal text [ok]. "
        "A replacement value may itself contain delimiters; include them when the template "
        "explicitly requests them. Preserve literal values already present in the template "
        "exactly; do not replace or invent them. The template must remain grammatically "
        "and semantically coherent after substitution.\n"
        "Fields to be substituted into the template string for final construction: (summary)."
        "\n\nThe template string:\nSummarize Kedi. Return [summary]."
    )
    assert ClaimPromptBuilder(claim="Kedi is typed").build().content == (
        "Return true only if the following claim can be established as true from the available "
        "context; otherwise return false.\nClaim: Kedi is typed"
    )
    assert RawPromptBuilder(content="hello").build().content == "hello"


def test_output_type_display_name_supports_plain_and_annotated_types() -> None:
    from typing import Annotated

    assert _type_display_name(str) == "str"
    assert _type_display_name(Annotated[bool, "decision"]) == "bool"


def test_request_assembler_snapshots_resolved_collections() -> None:
    schema: dict[str, Any] = {"answer": str}
    instructions = [InstructionPart(identity="profile.system", source="profile", content="system")]
    request = RunRequestAssembler().assemble(
        prompt=RawPromptBuilder(content="hello").build(),
        instructions=instructions,
        conversation=ConversationCoordinates(
            session_id="session",
            lane_identity="pydantic:lane",
            cache_epoch=2,
            cache_key="cache",
        ),
        tool_specs=(),
        output=SchemaOutput(schema=schema),
        run_id="run",
        turn_sequence=3,
        call_kind="produce",
    )

    schema["later"] = int
    instructions.clear()

    assert isinstance(request.output, SchemaOutput)
    assert tuple(request.output.schema) == ("answer",)
    assert request.instruction_text == "system"
    assert request.metadata.prompt_kind == "raw"
    assert request.metadata.turn_sequence == 3


def test_agent_run_request_snapshots_direct_collection_inputs() -> None:
    instructions = [InstructionPart(identity="system", source="profile", content="stable")]
    tools = [
        ToolDefinition(
            name="lookup",
            description=None,
            input_schema=None,
            output_schema=None,
            risk="read_only",
            required_before_output=False,
            artifact_result="auto",
            sequential=False,
        )
    ]
    request = AgentRunRequest(
        prompt=RawPromptBuilder(content="hello").build(),
        instructions=cast(tuple[InstructionPart, ...], instructions),
        tool_definitions=cast(tuple[ToolDefinition, ...], tools),
        metadata=RunMetadata(
            run_id="run",
            turn_sequence=0,
            prompt_kind="raw",
            call_kind="invoke",
        ),
    )

    instructions.clear()
    tools.clear()

    assert tuple(part.identity for part in request.instructions) == ("system",)
    assert tuple(tool.name for tool in request.tool_definitions) == ("lookup",)


def test_request_assembler_projects_live_tools_into_immutable_definitions() -> None:
    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "examples": {"one", "two"},
    }
    tool = ToolSpec(
        fn=lambda path: path,
        name="read_file",
        json_schema=schema,
        return_schema={"type": "string"},
    )
    request = RunRequestAssembler().assemble(
        prompt=RawPromptBuilder(content="secret prompt").build(),
        instructions=(
            InstructionPart(
                identity="profile.system",
                source="profile",
                content="secret instruction",
            ),
        ),
        conversation=None,
        tool_specs=(tool,),
        output=TextOutput(),
        run_id="run",
        turn_sequence=0,
        call_kind="invoke",
    )

    schema["properties"]["path"]["type"] = "integer"
    definition = request.tool_definitions[0]
    assert definition.input_schema is not None
    assert definition.input_schema["properties"]["path"]["type"] == "string"
    assert definition.input_schema["examples"] in (("one", "two"), ("two", "one"))
    assert definition.output_schema == {"type": "string"}
    assert not hasattr(definition, "fn")

    diagnostics = request_diagnostics(request)
    assert diagnostics.tool_names == ("read_file",)
    assert diagnostics.instruction_identities == ("profile.system",)
    assert "secret prompt" not in repr(diagnostics)
    assert "secret instruction" not in repr(diagnostics)


def test_request_diagnostics_canonicalizes_nested_mapping_order() -> None:
    first_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path"},
            "limit": {"type": "integer"},
        },
    }
    second_schema = {
        "properties": {
            "limit": {"type": "integer"},
            "path": {"description": "Path", "type": "string"},
        },
        "type": "object",
    }

    def diagnostics_for(schema: dict[str, Any]) -> RequestDiagnostics:
        request = RunRequestAssembler().assemble(
            prompt=RawPromptBuilder(content="hello").build(),
            instructions=(),
            conversation=None,
            tool_specs=(ToolSpec(fn=lambda: None, name="lookup", json_schema=schema),),
            output=TextOutput(),
            run_id="run",
            turn_sequence=0,
            call_kind="invoke",
        )
        return request_diagnostics(request)

    assert (
        diagnostics_for(first_schema).tool_contract_digest
        == diagnostics_for(second_schema).tool_contract_digest
    )

    def output_diagnostics(schema: dict[str, Any]) -> RequestDiagnostics:
        request = RunRequestAssembler().assemble(
            prompt=RawPromptBuilder(content="hello").build(),
            instructions=(),
            conversation=None,
            tool_specs=(),
            output=SchemaOutput(schema=schema),
            run_id="run",
            turn_sequence=0,
            call_kind="produce",
        )
        return request_diagnostics(request)

    first_output = output_diagnostics(first_schema)
    second_output = output_diagnostics(second_schema)
    assert first_output.output_field_count == 2
    assert first_output.output_contract_digest == second_output.output_contract_digest


def test_live_call_options_and_native_continuation_only_snapshot_the_outer_mapping() -> None:
    native_messages = [{"role": "assistant", "parts": ["answer"]}]
    options = AdapterCallOptions(native_kwargs={"messages": native_messages})
    outcome = AdapterRunOutcome(
        value="answer",
        portable_result="answer",
        continuation_state={"messages": native_messages},
    )

    assert options.native_kwargs["messages"] is native_messages
    assert outcome.continuation_state["messages"] is native_messages


def test_request_assembler_rejects_duplicate_instruction_identities() -> None:
    duplicate = InstructionPart(identity="system", source="profile", content="one")
    with pytest.raises(ValueError, match="instruction identities must be unique"):
        RunRequestAssembler().assemble(
            prompt=RawPromptBuilder(content="hello").build(),
            instructions=(duplicate, duplicate),
            conversation=None,
            tool_specs=(),
            output=TextOutput(),
            run_id="run",
            turn_sequence=0,
            call_kind="invoke",
        )


@pytest.mark.parametrize(
    ("run_id", "turn_sequence", "message"),
    [
        ("", 0, "run_id must not be empty"),
        ("run", -1, "turn_sequence must be non-negative"),
    ],
)
def test_request_assembler_rejects_invalid_run_metadata(
    run_id: str,
    turn_sequence: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RunRequestAssembler().assemble(
            prompt=RawPromptBuilder(content="hello").build(),
            instructions=(),
            conversation=None,
            tool_specs=(),
            output=TextOutput(),
            run_id=run_id,
            turn_sequence=turn_sequence,
            call_kind="invoke",
        )


def test_conversation_lane_identity_canonicalizes_nested_tool_contracts() -> None:
    class Adapter:
        shortname = "test"

        @staticmethod
        def default_model() -> str:
            return "model"

    profile = type(
        "Profile",
        (),
        {"model": None, "effort": None, "settings": {}, "mcp_servers": ()},
    )()
    tool = ToolSpec(
        fn=lambda: None,
        name="nested",
        json_schema={"choices": ["a", "b"]},
    )

    first = conversation_lane_identity(Adapter(), profile, (tool,))
    second = conversation_lane_identity(Adapter(), profile, (tool,))
    assert first == second
    assert first.startswith("test:")


def test_conversation_lane_identity_canonicalizes_mcp_mapping_order() -> None:
    class Adapter:
        shortname = "test"

        @staticmethod
        def default_model() -> str:
            return "model"

    first_server = McpServerSpec(
        transport="http",
        url="https://mcp.example.test",
        headers={"Authorization": "Bearer secret", "X-Mode": "read"},
    )
    second_server = McpServerSpec(
        transport="http",
        url="https://mcp.example.test",
        headers={"X-Mode": "read", "Authorization": "Bearer secret"},
    )

    def profile(server: McpServerSpec) -> Any:
        return type(
            "Profile",
            (),
            {
                "model": None,
                "effort": None,
                "settings": {},
                "mcp_servers": (server,),
            },
        )()

    assert conversation_lane_identity(Adapter(), profile(first_server), ()) == (
        conversation_lane_identity(Adapter(), profile(second_server), ())
    )
    opaque_profile = profile(first_server)
    opaque_profile.mcp_servers = ("opaque-server",)
    assert conversation_lane_identity(Adapter(), opaque_profile, ()) == (
        conversation_lane_identity(Adapter(), opaque_profile, ())
    )


def test_request_assembler_rejects_mismatched_call_and_output_contract() -> None:
    with pytest.raises(ValueError, match="invoke requires text output"):
        RunRequestAssembler().assemble(
            prompt=RawPromptBuilder(content="hello").build(),
            instructions=(),
            conversation=None,
            tool_specs=(),
            output=SchemaOutput(schema={"answer": str}),
            run_id="run",
            turn_sequence=0,
            call_kind="invoke",
        )

    with pytest.raises(ValueError, match="unsupported agent call kind"):
        RunRequestAssembler().assemble(
            prompt=RawPromptBuilder(content="hello").build(),
            instructions=(),
            conversation=None,
            tool_specs=(),
            output=SchemaOutput(schema={"answer": str}),
            run_id="run",
            turn_sequence=0,
            call_kind=cast(Any, "unknown"),
        )

    with pytest.raises(TypeError, match="unsupported Kedi output contract"):
        RunRequestAssembler().assemble(
            prompt=RawPromptBuilder(content="hello").build(),
            instructions=(),
            conversation=None,
            tool_specs=(),
            output=cast(Any, object()),
            run_id="run",
            turn_sequence=0,
            call_kind="produce",
        )


def test_failed_turn_atomically_discards_history_and_continuation() -> None:
    state = ConversationState(session_id="atomic")

    with pytest.raises(RuntimeError, match="failed"):
        with state.adapter_turn("test") as turn:
            turn.result_state["continuation"] = "must-not-commit"
            turn.record_exchange("prompt", "result")
            raise RuntimeError("failed")

    assert state.snapshot() == ()
    with state.adapter_turn("test") as retry:
        assert retry.resume_state == {}
    state.close()


def test_async_turns_follow_reserved_source_order() -> None:
    async def scenario() -> tuple[list[tuple[int, dict[str, int]]], tuple[str, ...]]:
        state = ConversationState(session_id="ordered")
        first = state.reserve_turn()
        second = state.reserve_turn()
        observed: list[tuple[int, dict[str, int]]] = []

        async def first_run() -> None:
            async with state.adapter_turn_async("test", ticket=first) as turn:
                observed.append((1, dict(turn.resume_state)))
                await asyncio.sleep(0.02)
                turn.result_state["turn"] = 1
                turn.record_exchange("first", "one")

        async def second_run() -> None:
            async with state.adapter_turn_async("test", ticket=second) as turn:
                observed.append((2, dict(turn.resume_state)))
                turn.result_state["turn"] = 2
                turn.record_exchange("second", "two")

        await asyncio.gather(second_run(), first_run())
        prompts = tuple(turn.content for turn in state.snapshot() if turn.role == "user")
        state.close()
        return observed, prompts

    observed, prompts = asyncio.run(scenario())
    assert observed == [(1, {}), (2, {"turn": 1})]
    assert prompts == ("first", "second")


def test_lane_identity_isolates_native_continuation() -> None:
    state = ConversationState(session_id="lanes")
    with state.adapter_turn("pydantic", lane_identity="pydantic:model-a") as first:
        first.result_state["message"] = "a"
    with state.adapter_turn("pydantic", lane_identity="pydantic:model-b") as second:
        assert second.resume_state == {}
        assert second.cache_key != first.cache_key
    state.close()


def test_legacy_shortname_lane_is_upgraded_without_losing_continuation() -> None:
    state = ConversationState(session_id="legacy-lane")
    with state.adapter_turn("pydantic") as seeded:
        seeded.result_state["message"] = "legacy"

    with state.adapter_turn("pydantic", lane_identity="pydantic:fingerprint") as upgraded:
        assert upgraded.resume_state == {"message": "legacy"}
        upgraded.result_state["message"] = "upgraded"

    snapshot = state.adapter_state_snapshot("pydantic")
    assert snapshot.lane_identity == "pydantic:fingerprint"
    assert snapshot.state == {"message": "upgraded"}
    state.close()


def test_cancelled_reserved_ticket_does_not_strand_later_turn() -> None:
    state = ConversationState(session_id="cancelled-ticket")
    cancelled = state.reserve_turn()
    later = state.reserve_turn()

    state.cancel_turn(cancelled)
    with state.adapter_turn("test", ticket=later) as turn:
        assert turn.turn_sequence == 1
        assert turn.resume_state == {}
    state.close()


def test_cancelled_async_waiter_does_not_claim_or_strand_its_ticket() -> None:
    async def scenario() -> tuple[str, ...]:
        state = ConversationState(session_id="cancelled-async-waiter")
        first = state.reserve_turn()
        cancelled = state.reserve_turn()
        later = state.reserve_turn()
        first_entered = asyncio.Event()
        release_first = asyncio.Event()

        async def hold_first() -> None:
            async with state.adapter_turn_async("test", ticket=first) as turn:
                first_entered.set()
                await release_first.wait()
                turn.record_exchange("first", "done")

        async def wait_for_cancelled_turn() -> None:
            async with state.adapter_turn_async("test", ticket=cancelled):
                raise AssertionError("cancelled turn must not enter adapter execution")

        first_task = asyncio.create_task(hold_first())
        await first_entered.wait()
        cancelled_task = asyncio.create_task(wait_for_cancelled_turn())
        await asyncio.sleep(0)
        cancelled_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled_task

        release_first.set()
        await first_task
        async with state.adapter_turn_async("test", ticket=later) as turn:
            turn.record_exchange("later", "done")

        prompts = tuple(turn.content for turn in state.snapshot() if turn.role == "user")
        state.close()
        return prompts

    assert asyncio.run(scenario()) == ("first", "later")


def test_cancelled_active_async_turn_rolls_back_before_next_turn() -> None:
    async def scenario() -> tuple[tuple[str, ...], dict[str, object]]:
        state = ConversationState(session_id="cancelled-active-turn")
        entered = asyncio.Event()
        blocked = asyncio.Event()

        async def active_turn() -> None:
            async with state.adapter_turn_async("test") as turn:
                turn.result_state["partial"] = True
                turn.record_exchange("cancelled", "partial")
                entered.set()
                await blocked.wait()

        task = asyncio.create_task(active_turn())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        async with state.adapter_turn_async("test") as retry:
            resume_state = dict(retry.resume_state)
            retry.record_exchange("retry", "done")
        prompts = tuple(turn.content for turn in state.snapshot() if turn.role == "user")
        state.close()
        return prompts, resume_state

    prompts, resume_state = asyncio.run(scenario())
    assert prompts == ("retry",)
    assert resume_state == {}


def test_cancellation_after_async_acquire_rolls_back_the_undelivered_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_shield = asyncio.shield
    shield_calls = 0

    async def cancel_first_delivery(awaitable: Any) -> Any:
        nonlocal shield_calls
        result = await original_shield(awaitable)
        shield_calls += 1
        if shield_calls == 1:
            raise asyncio.CancelledError
        return result

    monkeypatch.setattr(conversation_module.asyncio, "shield", cancel_first_delivery)

    async def scenario() -> tuple[str, ...]:
        state = ConversationState(session_id="cancelled-after-acquire")
        with pytest.raises(asyncio.CancelledError):
            async with state.adapter_turn_async("test"):
                raise AssertionError("cancelled turn must not be delivered")
        async with state.adapter_turn_async("test") as retry:
            retry.record_exchange("retry", "done")
        prompts = tuple(turn.content for turn in state.snapshot() if turn.role == "user")
        state.close()
        return prompts

    assert asyncio.run(scenario()) == ("retry",)


def test_active_turn_rejects_external_state_mutation_and_consumed_ticket_reuse() -> None:
    state = ConversationState(session_id="active-invariants")
    ticket = state.reserve_turn()
    with state.adapter_turn("test", ticket=ticket):
        state.cancel_turn(ticket)
        with pytest.raises(RuntimeError, match="advance cache epoch"):
            state.advance_cache_epoch()
        with pytest.raises(RuntimeError, match="record an active conversation exchange"):
            state.record_exchange("external", "write")
        with pytest.raises(RuntimeError, match="close conversation state"):
            state.close()

    with pytest.raises(RuntimeError, match="already been retired"):
        with state.adapter_turn("test", ticket=ticket):
            raise AssertionError("consumed ticket must not be reusable")
    state.close()


def test_rejected_nested_sync_turn_does_not_consume_a_ticket() -> None:
    state = ConversationState(session_id="nested")
    with state.adapter_turn("test"):
        with pytest.raises(RuntimeError, match="nested turns"):
            with state.adapter_turn("test"):
                raise AssertionError("unreachable")
    with state.adapter_turn("test") as later:
        assert later.turn_sequence == 1
    state.close()


def test_rejected_nested_async_turn_does_not_consume_a_ticket() -> None:
    async def scenario() -> int:
        state = ConversationState(session_id="nested-async")
        async with state.adapter_turn_async("test"):
            with pytest.raises(RuntimeError, match="nested turns"):
                async with state.adapter_turn_async("test"):
                    raise AssertionError("unreachable")
        async with state.adapter_turn_async("test") as later:
            turn_sequence = later.turn_sequence
        state.close()
        return turn_sequence

    assert asyncio.run(scenario()) == 1


def test_lowering_uses_existing_adapter_surfaces_for_each_output_contract() -> None:
    class Adapter:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, dict[str, object]]] = []

        def invoke_sync(self, *, prompt: str, **kwargs: object) -> str:
            self.calls.append(("invoke", prompt, kwargs))
            return "text"

        def produce_sync(self, *, template: str, **kwargs: object) -> str:
            self.calls.append(("produce", template, kwargs))
            return "structured"

    adapter = Adapter()
    assembler = RunRequestAssembler()

    def request(output: OutputContract):
        return assembler.assemble(
            prompt=RawPromptBuilder(content="prompt").build(),
            instructions=(
                InstructionPart(identity="system", source="profile", content="instruction"),
            ),
            conversation=None,
            tool_specs=(),
            output=output,
            run_id="run",
            turn_sequence=0,
            call_kind="invoke" if isinstance(output, TextOutput) else "produce",
        )

    assert lower_request_sync(adapter, request(TextOutput())) == "text"
    assert lower_request_sync(adapter, request(SchemaOutput(schema={"answer": str}))) == (
        "structured"
    )
    assert lower_request_sync(adapter, request(TypedOutput(output_type=bool))) == "structured"
    assert adapter.calls == [
        ("invoke", "prompt", {"instructions": "instruction"}),
        (
            "produce",
            "prompt",
            {"instructions": "instruction", "output_schema": {"answer": str}},
        ),
        (
            "produce",
            "prompt",
            {"instructions": "instruction", "output_type": bool},
        ),
    ]

    no_instructions = request(TextOutput())
    no_instructions = replace(no_instructions, instructions=())
    assert (
        lower_request_sync(
            adapter,
            no_instructions,
            options=AdapterCallOptions(native_kwargs={"temperature": 0}),
        )
        == "text"
    )
    assert adapter.calls[-1] == ("invoke", "prompt", {"temperature": 0})
    assert isinstance(AdapterCallOptions().native_kwargs, MappingProxyType)

    invalid = replace(no_instructions, output=cast(Any, object()))
    with pytest.raises(TypeError, match="Unsupported Kedi output contract"):
        lower_request_sync(adapter, invalid)
