"""Execute a resolved request within budget, telemetry and tool scopes."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from kedi.agent_adapter.history_provenance import history_source_scope
from kedi.agent_adapter.invocation import observe_invocations
from kedi.agent_adapter.request import SchemaOutput, TypedOutput
from kedi.agent_adapter.request_budget import SupportsHistoryTokenEstimate
from kedi.agent_adapter.telemetry import (
    agent_call_span,
    model_call_span,
    record_conversation_assets,
    resolved_adapter_model,
)
from kedi.agent_adapter.tools.scope import (
    artifact_admission_scope,
    required_tool_call_scope,
    validate_required_tool_calls,
)
from kedi.core.expressions import _type_display_name
from kedi.debugging import execution_observed
from kedi.groups import current_group_member
from kedi.telemetry import capture_enabled

if TYPE_CHECKING:
    from kedi.agent_adapter.conversation import AdapterConversationContext
    from kedi.agent_adapter.invocation import InvocationUsage
    from kedi.agent_adapter.request import AgentRunRequest
    from kedi.agent_adapter.request_budget import ModelRequestBudget
    from kedi.agent_adapter.subagents import SubagentConversationContext
    from kedi.agent_adapter.tools.specs import ToolSpec
    from kedi.agent_profile import AgentProfile
    from kedi.artifacts import ArtifactManager
    from kedi.errors import KediTraceFrame


def execute_request(
    request: AgentRunRequest,
    *,
    adapter: Any,
    call: Callable[[AgentRunRequest], Any],
    profile: AgentProfile,
    tools: Sequence[ToolSpec],
    budget: ModelRequestBudget,
    artifacts: ArtifactManager,
    source_frame: KediTraceFrame | None,
    checkpoint_event: Callable[..., None],
    resume_state: Mapping[str, Any] | None = None,
    conversation_scope: AbstractContextManager[Any] | None = None,
    live_conversation: AdapterConversationContext | SubagentConversationContext | None = None,
) -> Any:
    call_kind = request.metadata.call_kind
    configured_model = profile.model
    frame = source_frame
    source_span = None if frame is None else frame.span
    source_path = (
        source_span.source_path
        if source_span is not None
        and source_span.source_path is not None
        and capture_enabled("source_path")
        else None
    )
    source_line = None if source_span is None else source_span.start_line
    source_snippet = (
        frame.source_snippet if frame is not None and capture_enabled("source_snippet") else None
    )
    output_schema = (
        {key: repr(value) for key, value in request.output.schema.items()}
        if isinstance(request.output, SchemaOutput)
        else {"result": _type_display_name(request.output.output_type)}
        if isinstance(request.output, TypedOutput)
        else None
    )
    output_field_count = 0 if output_schema is None else len(output_schema)

    from kedi.run_budget import assert_run_budget_supported

    if execution_observed():
        checkpoint_event(
            "request",
            safe=False,
            payload={
                "prompt": request.prompt.content,
                "instructions": request.instruction_text,
                "model": configured_model,
                "tools": [tool.name for tool in tools],
            },
        )

    assert_run_budget_supported(
        adapter,
        opaque_tools=bool(profile.mcp_servers)
        or any((tool.metadata or {}).get("source") == "external" for tool in tools),
    )
    record_conversation_assets(
        prompt=request.prompt.content,
        instructions=request.instruction_text,
        profile=profile,
        tools=tools,
        output_schema=output_schema,
        call_kind=call_kind,
        source_path=source_path,
        source_line=source_line,
        source_snippet=source_snippet,
        group_member=current_group_member(),
    )
    with agent_call_span(
        adapter,
        profile_name=profile.name,
        call_kind=call_kind,
        prompt_chars=len(request.prompt.content),
        output_field_count=output_field_count,
        tool_count=len(request.tool_definitions),
        mcp_server_count=len(profile.mcp_servers),
        artifacts_enabled=bool(profile.artifacts and profile.artifacts.enabled),
        configured_model=configured_model,
        prompt_kind=request.prompt.kind,
        instruction_count=len(request.instructions),
        history_count=(0 if request.conversation is None else request.conversation.history_count),
        cache_epoch=(None if request.conversation is None else request.conversation.cache_epoch),
        turn_sequence=request.metadata.turn_sequence,
    ) as agent_telemetry:
        from kedi.agent_profile.telemetry import record_capability_snapshot

        record_capability_snapshot(profile)
        history_estimator = adapter
        context: list[Any] = []
        if request.instruction_text is not None:
            context.append(request.instruction_text)
        context.append(
            [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.json_schema,
                    "returns": tool.return_schema,
                }
                for tool in tools
            ]
        )
        if output_schema is not None:
            context.append(output_schema)
        reservation = budget.reserve(
            request.prompt.content,
            context=tuple(context),
            resume_state=resume_state,
            history_tokens=(
                history_estimator.estimate_history_tokens(resume_state)
                if resume_state is not None
                and isinstance(history_estimator, SupportsHistoryTokenEstimate)
                else None
            ),
        )
        try:
            artifact_scope = (
                artifact_admission_scope(
                    manager=artifacts,
                    policy=profile.artifacts,
                )
                if profile.artifacts is not None
                else nullcontext()
            )
            with (
                history_source_scope(request.metadata.run_id, source_span),
                required_tool_call_scope(tools),
                artifact_scope,
                conversation_scope or nullcontext(),
                model_call_span(
                    adapter,
                    call_kind=call_kind,
                    configured_model=configured_model,
                ) as model_telemetry,
            ):

                def observe(usage: InvocationUsage) -> None:
                    reservation.observe(usage)
                    agent_telemetry.observe(usage)
                    model_telemetry.observe(usage)
                    if execution_observed():
                        checkpoint_event(
                            "usage",
                            safe=False,
                            payload={
                                "requests": usage.requests,
                                "tool_calls": usage.tool_calls,
                                "input_tokens": usage.input_tokens,
                                "cache_read_tokens": usage.cache_read_tokens,
                                "output_tokens": usage.output_tokens,
                            },
                        )

                with observe_invocations(observe):
                    result = _call_with_live_input(
                        adapter,
                        request,
                        call,
                        profile=profile,
                        conversation_scope=conversation_scope,
                        live_conversation=live_conversation,
                    )
                validate_required_tool_calls()
                model = resolved_adapter_model(adapter, configured_model)
                model_telemetry.complete(result, model=model)
                agent_telemetry.complete(result, model=model)
        except BaseException:
            reservation.finish(raise_on_exceeded=False)
            raise
        reservation.finish()
        return result


def _call_with_live_input(
    adapter: Any,
    request: AgentRunRequest,
    call: Callable[[AgentRunRequest], Any],
    *,
    profile: AgentProfile,
    conversation_scope: AbstractContextManager[Any] | None,
    live_conversation: AdapterConversationContext | SubagentConversationContext | None,
) -> Any:
    from kedi.agent_adapter.conversation import AdapterConversationContext
    from kedi.agent_adapter.subagents.control import LiveInput, current_live_input, live_input_scope

    control = current_live_input()
    owns_control = (
        control is None
        and bool(profile.subagents)
        and adapter.capabilities.supports("live_subagent_input")
    )
    if owns_control:
        control = LiveInput(request.metadata.run_id)
        control.configure(input=adapter.capabilities.supports("live_subagent_input"))
    ephemeral: AdapterConversationContext | None = None
    if owns_control and conversation_scope is None:
        ephemeral = AdapterConversationContext(
            session_id=request.metadata.run_id,
            cache_key=request.metadata.run_id,
        )
    try:
        with (
            live_input_scope(control),
            adapter.conversation_scope(ephemeral) if ephemeral is not None else nullcontext(),
        ):
            active_request = request
            while True:
                result = call(active_request)
                if not owns_control or control is None or control.try_finish(0, lambda: None):
                    break
                continuation = live_conversation or ephemeral
                assert continuation is not None
                continuation.resume_state.update(continuation.result_state)
                control.begin_attempt()
                inputs = control.take_pending(revision=0)
                active_request = replace(
                    request,
                    prompt=request.prompt.with_content(
                        "\n\n".join(item.render() for item in inputs)
                    ),
                )
    finally:
        if owns_control and control is not None:
            control.close()
        if ephemeral is not None and ephemeral.cleanup is not None:
            ephemeral.cleanup()
    return result
