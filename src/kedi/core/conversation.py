from __future__ import annotations

from contextlib import nullcontext
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Sequence,
)

from kedi.debugging import checkpoint, execution_frame, execution_observed, observation_only
from kedi.errors import KediExecutionError

if TYPE_CHECKING:
    from kedi.agent_adapter.conversation import (
        ConversationState,
        ConversationTurnTicket,
    )
    from kedi.agent_adapter.request import (
        AgentRunRequest,
        InstructionPart,
        OutputContract,
        PromptDraft,
    )
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
            payload: dict[str, Any] = {"rendered_input": prompt.content}
            if prompt.media:
                from dataclasses import asdict

                payload["media"] = [asdict(item.summary) for item in prompt.media]
            checkpoint(self, "model_input", payload=payload)
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
    retain_failed_turn: bool = False,
) -> Any:
    """Assemble and execute one adapter-neutral request transaction."""
    from functools import partial
    from uuid import uuid4

    from kedi.agent_adapter.hooks import HookRegistrationMixin, prepared_agent_hook_run
    from kedi.agent_adapter.request import (
        AdapterRunOutcome,
        ConversationCoordinates,
        RunRequestAssembler,
        media_summaries,
    )
    from kedi.agent_adapter.request_budget import ModelRequestBudget
    from kedi.core.request_execution import execute_request

    frames = self.current_trace_frames()
    execute = partial(
        execute_request,
        adapter=adapter,
        call=call,
        profile=profile,
        tools=tools,
        budget=self._request_budget.get() or ModelRequestBudget(),
        artifacts=self.artifacts,
        source_frame=frames[-1] if frames else None,
        checkpoint_event=partial(checkpoint, self),
    )
    assembler = RunRequestAssembler()

    if prompt.media and not adapter.capabilities.supports("media_input"):
        from kedi.media import MediaError

        raise MediaError(
            f"Adapter {adapter.shortname!r} does not accept media input; "
            "use the Pydantic AI or LangChain adapter for prompts with media"
        )
    runtime_adapter: Any = adapter
    if isinstance(runtime_adapter, HookRegistrationMixin):
        hook_context = prepared_agent_hook_run(
            runtime_adapter, prompt.content, media=media_summaries(prompt.media)
        )
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
                retain_failed_turn=retain_failed_turn,
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
        if not retain_failed_turn:
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
