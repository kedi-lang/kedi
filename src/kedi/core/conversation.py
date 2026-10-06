from __future__ import annotations

from contextlib import nullcontext
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Mapping,
    Sequence,
)

from kedi.core.expressions import _type_display_name
from kedi.core.model_calls import _tool_request_context
from kedi.debugging import checkpoint, execution_frame, execution_observed, observation_only
from kedi.errors import KediExecutionError
from kedi.telemetry import capture_enabled

if TYPE_CHECKING:
    from kedi.agent_adapter.conversation import (
        AdapterConversationContext,
        ConversationState,
        ConversationTurnTicket,
    )
    from kedi.agent_adapter.request import (
        AgentRunRequest,
        InstructionPart,
        OutputContract,
        PromptDraft,
    )
    from kedi.agent_adapter.subagents import SubagentConversationContext
    from kedi.agent_adapter.tools.specs import ToolSpec
    from kedi.agent_profile import AgentProfile
    from kedi.core.runtime import KediRuntime


def _run_conversation_call(
    self: KediRuntime,
    adapter: Any,
    prompt: PromptDraft,
    call: Callable[[AgentRunRequest], Any],
    **kwargs: Any,
) -> Any:
    """Model-lane safe points surround the entire conversation transaction."""
    env = self._current_env.get()
    if env is None:
        env = {}
    with execution_frame(self, "<model>", env):
        if execution_observed():
            checkpoint(self, "model_input", payload={"rendered_input": prompt.content})
        try:
            with observation_only():
                result = _execute_conversation_call(self, adapter, prompt, call, **kwargs)
        except KediExecutionError as exc:
            # Nested tool failures were observed inside the transaction. Stop
            # only after its lock and other request-owned resources are released.
            checkpoint(self, "exception", payload=exc.original if exc.original is not None else exc)
            raise
        if execution_observed():
            checkpoint(self, "model_result", payload={"result": result})
        return result


def _execute_conversation_call(
    self: KediRuntime,
    adapter: Any,
    prompt: PromptDraft,
    call: Callable[[AgentRunRequest], Any],
    *,
    instructions: Sequence[InstructionPart] = (),
    tools: Sequence[ToolSpec] = (),
    profile: AgentProfile,
    call_kind: Literal["produce", "invoke"],
    output: OutputContract,
    conversation_state: ConversationState | None = None,
    conversation_ticket: ConversationTurnTicket | None = None,
    lane_identity: str | None = None,
) -> Any:
    """Assemble and execute one adapter-neutral request transaction."""
    from uuid import uuid4

    from kedi.agent_adapter.history_provenance import history_source_scope
    from kedi.agent_adapter.hooks import HookRegistrationMixin, prepared_agent_hook_run
    from kedi.agent_adapter.invocation import observe_invocations
    from kedi.agent_adapter.request import (
        AdapterRunOutcome,
        ConversationCoordinates,
        RunRequestAssembler,
        SchemaOutput,
        TypedOutput,
    )
    from kedi.agent_adapter.request_budget import ModelRequestBudget, SupportsHistoryTokenEstimate
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

    budget = self._request_budget.get() or ModelRequestBudget()
    configured_model = profile.model
    frames = self.current_trace_frames()
    frame = frames[-1] if frames else None
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
        {key: repr(value) for key, value in output.schema.items()}
        if isinstance(output, SchemaOutput)
        else {"result": _type_display_name(output.output_type)}
        if isinstance(output, TypedOutput)
        else None
    )
    output_field_count = 0 if output_schema is None else len(output_schema)
    assembler = RunRequestAssembler()

    def request_context(request: AgentRunRequest) -> tuple[Any, ...]:
        values: list[Any] = []
        if request.instruction_text is not None:
            values.append(request.instruction_text)
        values.append(_tool_request_context(tools))
        if output_schema is not None:
            values.append(output_schema)
        return tuple(values)

    def execute(
        request: AgentRunRequest,
        *,
        resume_state: Mapping[str, Any] | None = None,
        conversation_scope: Any = None,
        live_conversation: AdapterConversationContext | SubagentConversationContext | None = None,
    ) -> Any:
        from kedi.run_budget import assert_run_budget_supported

        if execution_observed():
            checkpoint(
                self,
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
            history_count=(
                0 if request.conversation is None else request.conversation.history_count
            ),
            cache_epoch=(
                None if request.conversation is None else request.conversation.cache_epoch
            ),
            turn_sequence=request.metadata.turn_sequence,
        ) as agent_telemetry:
            history_estimator = adapter
            reservation = budget.reserve(
                request.prompt.content,
                context=request_context(request),
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
                        manager=self.artifacts,
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

                    def observe(usage: Any) -> None:
                        reservation.observe(usage)
                        agent_telemetry.observe(usage)
                        model_telemetry.observe(usage)
                        if execution_observed():
                            checkpoint(
                                self,
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
                        from kedi.agent_adapter.subagents.control import (
                            LiveInput,
                            current_live_input,
                            live_input_scope,
                        )

                        control = current_live_input()
                        owns_control = (
                            control is None
                            and bool(profile.subagents)
                            and adapter.capabilities.supports("live_subagent_input")
                        )
                        if owns_control:
                            control = LiveInput(request.metadata.run_id)
                            control.configure(
                                input=adapter.capabilities.supports("live_subagent_input")
                            )
                        ephemeral = None
                        if owns_control and conversation_scope is None:
                            from kedi.agent_adapter.conversation import AdapterConversationContext

                            ephemeral = AdapterConversationContext(
                                session_id=request.metadata.run_id,
                                cache_key=request.metadata.run_id,
                            )
                        try:
                            with (
                                live_input_scope(control),
                                (
                                    adapter.conversation_scope(ephemeral)
                                    if ephemeral is not None
                                    else nullcontext()
                                ),
                            ):
                                active_request = request
                                while True:
                                    result = call(active_request)
                                    if (
                                        not owns_control
                                        or control is None
                                        or control.try_finish(0, lambda: None)
                                    ):
                                        break
                                    from dataclasses import replace

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
                    validate_required_tool_calls()
                    model = resolved_adapter_model(adapter, configured_model)
                    model_telemetry.complete(result, model=model)
                    agent_telemetry.complete(result, model=model)
            except BaseException:
                reservation.finish(raise_on_exceeded=False)
                raise
            reservation.finish()
            return result

    runtime_adapter: Any = adapter
    if isinstance(runtime_adapter, HookRegistrationMixin):
        hook_context = prepared_agent_hook_run(runtime_adapter, prompt.content)
    else:
        hook_context = nullcontext(None)

    state = conversation_state
    try:
        with hook_context as prepared:
            if prepared is None:
                active_prompt = prompt
                run_id = f"agent_{uuid4().hex}"
            else:
                active_prompt = prompt.with_content(prepared.content)
                run_id = prepared.identity.run_id

            if state is None:
                request = assembler.assemble(
                    prompt=active_prompt,
                    instructions=instructions,
                    conversation=None,
                    tool_specs=tools,
                    output=output,
                    run_id=run_id,
                    turn_sequence=0,
                    call_kind=call_kind,
                )
                return execute(request)

            from kedi.agent_adapter.conversation import SupportsConversationState
            from kedi.lang.compiler.environment import model_value

            capabilities = adapter.capabilities
            adapter_shortname = adapter.shortname
            if profile.history is True and not (
                capabilities.supports("stateful_history")
                and isinstance(adapter, SupportsConversationState)
            ):
                raise RuntimeError(
                    f"Adapter {adapter_shortname!r} does not support stateful history"
                )
            with state.adapter_turn(
                adapter_shortname,
                ticket=conversation_ticket,
                lane_identity=lane_identity,
            ) as conversation:
                coordinates = ConversationCoordinates(
                    session_id=conversation.session_id,
                    lane_identity=conversation.lane_identity,
                    cache_epoch=conversation.cache_epoch,
                    cache_key=conversation.cache_key,
                    history_count=len(state.snapshot()),
                )
                request = assembler.assemble(
                    prompt=active_prompt,
                    instructions=instructions,
                    conversation=coordinates,
                    tool_specs=tools,
                    output=output,
                    run_id=run_id,
                    turn_sequence=conversation.turn_sequence,
                    call_kind=call_kind,
                )
                adapter_scope = (
                    adapter.conversation_scope(conversation)
                    if capabilities.supports("stateful_history")
                    and isinstance(adapter, SupportsConversationState)
                    else nullcontext()
                )
                result = execute(
                    request,
                    resume_state=conversation.resume_state,
                    conversation_scope=adapter_scope,
                    live_conversation=conversation,
                )
                conversation.apply_outcome(
                    request.prompt.content,
                    AdapterRunOutcome(
                        value=result,
                        portable_result=model_value(result),
                        continuation_state=conversation.result_state,
                        cleanup=conversation.cleanup,
                        cache_prefix_changed=conversation.cache_prefix_changed,
                    ),
                )
                return result
    except BaseException:
        self._cancel_conversation_turn(state, conversation_ticket)
        raise


def _conversation_state_for_profile(
    self: KediRuntime,
    profile: AgentProfile,
) -> ConversationState | None:
    if profile.history is False:
        return None
    if self.conversation_state is not None:
        return self.conversation_state
    if profile.history is not True:
        return None
    with self._conversation_state_lock:
        if self._implicit_conversation_state is None:
            from kedi.agent_adapter.conversation import ConversationState

            self._implicit_conversation_state = ConversationState(artifacts=self.artifacts)
        return self._implicit_conversation_state


def _reserve_conversation_turn(
    self: KediRuntime,
    profile: AgentProfile,
    adapter: Any,
    tools: Sequence[ToolSpec],
) -> tuple[ConversationState | None, ConversationTurnTicket | None, str | None]:
    state = self._conversation_state_for_profile(profile)
    if state is None:
        return None, None, None
    from kedi.agent_adapter.request import conversation_lane_identity

    lane_identity = conversation_lane_identity(adapter, profile, tools)
    return state, state.reserve_turn(), lane_identity


def _cancel_conversation_turn(
    state: ConversationState | None,
    ticket: ConversationTurnTicket | None,
) -> None:
    if state is not None and ticket is not None:
        state.cancel_turn(ticket)


def _retire_conversation_turn_after_failed_future(
    cls: type[KediRuntime],
    state: ConversationState | None,
    ticket: ConversationTurnTicket | None,
    completed: Any,
) -> None:
    """Release a reservation when an engine fails before or outside the job body."""

    failed = completed.cancelled() or completed.exception() is not None
    if failed:
        cls._cancel_conversation_turn(state, ticket)
