"""Per-run LangChain lifecycle: exact archival, usage and bounded tool correction."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
from typing import Any, Iterator, Literal, Mapping, Sequence

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import AgentState, hook_config
from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Overwrite
from pydantic_ai import UsageLimits
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.usage import RunUsage

from kedi.agent_adapter.compaction.archive import HistoryArchiveSettings, cache_epoch_key
from kedi.agent_adapter.compaction.langchain_archive import LangChainArtifactHistoryProcessor
from kedi.agent_adapter.conversation import AdapterConversationContext
from kedi.agent_adapter.history_processing import (
    HistoryProcessingError,
    HistoryProcessorRunner,
    rebase_history_cache_key,
)
from kedi.agent_adapter.invocation import InvocationUsage
from kedi.agent_adapter.subagents import SubagentConversationContext
from kedi.agent_adapter.tools.dispatch import SequentialDispatchGate
from kedi.run_budget import admit_run_operation, assert_no_hidden_model_retries


@dataclass
class LangChainRunState:
    limits: UsageLimits
    usage: InvocationUsage = field(default_factory=InvocationUsage)
    native_usage: RunUsage = field(default_factory=RunUsage)
    corrections: Counter[str] = field(default_factory=Counter)
    pending_tools: int = 0
    revision: int = 0
    completed_tools: dict[str, ToolMessage] = field(default_factory=dict)
    dispatch_gate: SequentialDispatchGate = field(default_factory=SequentialDispatchGate)
    instruction_digest: str | None = None
    instructions_changed: bool = False


_RUN: ContextVar[LangChainRunState | None] = ContextVar("kedi_langchain_run", default=None)
_TOOL_CALL: ContextVar[ToolCall | None] = ContextVar("kedi_langchain_tool_call", default=None)


def record_live_tool_result(result: Any) -> None:
    from kedi.agent_adapter.subagents.control import current_live_input

    state = _RUN.get()
    call = _TOOL_CALL.get()
    if state is None or call is None or current_live_input() is None:
        return
    from langchain_core.tools.base import _format_output

    from kedi.agent_adapter.adapters.langchain import _langchain_tool_message_for_call

    # Match the SDK's result encoding before its asynchronous callback delivery.
    output = _format_output(result, None, call.get("id"), call["name"], "success")
    message = _langchain_tool_message_for_call(output, call.get("id"))
    if message is not None:
        state.completed_tools[message.tool_call_id] = message


@contextmanager
def langchain_run_scope(limits: UsageLimits | None) -> Iterator[LangChainRunState]:
    limits = limits if limits is not None else UsageLimits()
    if limits.count_tokens_before_request:
        raise ValueError(
            "LangChain usage limits do not support provider-exact preflight token counting"
        )
    if limits.cost_limit is not None:
        raise ValueError(
            "LangChain usage limits require token/request limits, not a provider cost_limit"
        )
    state = LangChainRunState(limits=limits)
    from kedi.agent_adapter.subagents.control import current_live_input, current_live_revision

    control = current_live_input()
    if control is not None:
        state.revision = current_live_revision(control)
    token = _RUN.set(state)
    try:
        yield state
    finally:
        _RUN.reset(token)


class LangChainRunMiddleware(AgentMiddleware):
    def __init__(
        self,
        *,
        archive: HistoryArchiveSettings | None,
        tool_retries: int,
        history_processor: HistoryProcessorRunner[Any] | None = None,
        model_id: str | None = None,
        conversation: AdapterConversationContext | SubagentConversationContext | None = None,
    ) -> None:
        self.archiver = LangChainArtifactHistoryProcessor(archive) if archive is not None else None
        self.tool_retries = tool_retries
        self.history_processor = history_processor
        self.model_id = model_id
        self.conversation = conversation

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        del runtime
        messages = state["messages"]
        updated = self.archiver(messages) if self.archiver is not None else messages
        from langchain_core.messages import HumanMessage

        from kedi.agent_adapter.subagents.control import current_live_input

        control = current_live_input()
        if control is not None:
            run_state = _RUN.get()
            items = control.take_pending(revision=run_state.revision if run_state else 0)
            if items:
                updated = [*updated, *(HumanMessage(content=item.render()) for item in items)]
        if self.history_processor is not None:
            outcome = await self.history_processor.process(updated, model_id=self.model_id)
            updated = outcome.messages
        if updated is messages:
            return None
        # Replace the graph's durable message state, not just this request's view.
        return {"messages": Overwrite(updated)}

    @hook_config(can_jump_to=["model"])
    async def aafter_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Literal["model"] | None] | None:
        del runtime
        from kedi.agent_adapter.subagents.control import current_live_input

        control = current_live_input()
        if control is not None and control.pending:
            last = state["messages"][-1]
            if (
                not getattr(last, "tool_calls", None)
                or state.get("structured_response") is not None
            ):
                return {"jump_to": "model", "structured_response": None}
        return None

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        from langchain_core.messages import HumanMessage

        from kedi.agent_adapter.adapters.langchain import _langchain_invocation_usage
        from kedi.agent_adapter.subagents.control import current_live_input

        state = _RUN.get()
        if state is not None:
            state.limits.check_before_request(state.native_usage)
        checkpoint_id = next(
            (
                message.additional_kwargs["kedi_compaction_checkpoint"]
                for message in reversed(request.messages)
                if isinstance(message.additional_kwargs.get("kedi_compaction_checkpoint"), str)
            ),
            None,
        )
        settings = dict(request.model_settings)
        changed = False
        if checkpoint_id is not None:
            key = settings.get("prompt_cache_key")
            if key is None:
                key = getattr(request.model, "model_kwargs", {}).get("prompt_cache_key")
            if isinstance(key, str) and key:
                settings["prompt_cache_key"] = cache_epoch_key(key, checkpoint_id)
                changed = True
        model = request.model
        fingerprint = None
        instructions_changed = False
        if self.conversation is not None and state is not None:
            content = getattr(request.system_message, "content", None)
            fingerprint = sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
            previous_digest = state.instruction_digest or self.conversation.resume_state.get(
                "langchain_instruction_digest"
            )
            has_history = state.instruction_digest is not None or bool(
                self.conversation.resume_state.get("langchain_messages")
            )
            # Responses trimming must not discard a changed system message.
            instructions_changed = has_history and previous_digest != fingerprint
            if instructions_changed or state.instructions_changed:
                for key in ("conversation", "conversation_id", "previous_response_id"):
                    changed = settings.pop(key, None) is not None or changed
                model = _without_stale_continuation(
                    model,
                    reset_previous_response=instructions_changed,
                    reason="LangChain instructions changed",
                )
                changed = model is not request.model or changed
        if self.history_processor is not None:
            directive = self.history_processor.prepare_transport()
            if directive.rewritten:
                for key in ("conversation", "conversation_id", "previous_response_id"):
                    changed = settings.pop(key, None) is not None or changed
                model = _without_stale_continuation(
                    model,
                    reset_previous_response=directive.reset_continuation,
                )
                changed = model is not request.model or changed
                cache_key = settings.get("prompt_cache_key")
                if cache_key is None:
                    cache_key = getattr(model, "model_kwargs", {}).get("prompt_cache_key")
                if directive.cache_key is not None and isinstance(cache_key, str) and cache_key:
                    settings["prompt_cache_key"] = rebase_history_cache_key(
                        directive.cache_key,
                        cache_key,
                    )
                    changed = True
        if changed:
            request = request.override(model=model, model_settings=settings)
        assert_no_hidden_model_retries(model)
        admit_run_operation("request")
        if state is not None:
            state.native_usage.requests += 1
        control = current_live_input()
        if control is not None:
            control.confirm_contents(
                tuple(
                    message.content
                    for message in request.messages
                    if isinstance(message, HumanMessage) and isinstance(message.content, str)
                )
            )
        response = await handler(request)
        if state is not None:
            usage = _langchain_invocation_usage({"messages": response.result})
            state.usage += usage
            state.native_usage.input_tokens += usage.input_tokens
            state.native_usage.output_tokens += usage.output_tokens
            state.native_usage.cache_read_tokens += usage.cache_read_tokens
            state.native_usage.cache_write_tokens += usage.cache_write_tokens
            state.limits.check_tokens(state.native_usage)
            state.limits.check_per_request_input_tokens(usage.input_tokens)
        _check_model_completion(getattr(response, "result", ()))
        if state is not None and fingerprint is not None:
            state.instruction_digest = fingerprint
            state.instructions_changed = state.instructions_changed or instructions_changed
        return response

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        from kedi.agent_adapter.adapters.langchain import _langchain_tool_message_for_call

        state = _RUN.get()
        if state is None:
            return await handler(request)
        projected = replace(state.native_usage)
        projected.tool_calls += state.pending_tools + 1
        state.limits.check_before_tool_call(projected)
        state.pending_tools += 1
        token = _TOOL_CALL.set(request.tool_call)
        try:
            metadata = getattr(getattr(request, "tool", None), "metadata", None) or {}
            async with state.dispatch_gate.enter(
                sequential=metadata.get("kedi_sequential") is True
            ):
                result = await handler(request)
        finally:
            _TOOL_CALL.reset(token)
            state.pending_tools -= 1
        call = request.tool_call
        message = _langchain_tool_message_for_call(result, call.get("id"))
        if message is not None:
            state.completed_tools[message.tool_call_id] = message
        if message is not None and message.status == "error":
            name = call["name"]
            if state.corrections[name] >= self.tool_retries:
                raise UnexpectedModelBehavior(
                    f"Tool {name!r} exceeded max retries count of {self.tool_retries}"
                )
            state.corrections[name] += 1
        else:
            state.native_usage.tool_calls += 1
        return result


def _check_model_completion(messages: Sequence[Any]) -> None:
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        metadata = message.response_metadata
        status = metadata.get("status")
        reason = metadata.get("finish_reason", metadata.get("stop_reason"))
        if status in {"incomplete", "failed"}:
            details = metadata.get("incomplete_details")
            reason = details.get("reason") if isinstance(details, Mapping) else status
        elif reason not in {"length", "max_tokens", "content_filter"}:
            return
        # Only expose a reason code, never arbitrary provider response text.
        code = (
            reason
            if isinstance(reason, str) and len(reason) <= 64 and reason.replace("_", "").isalnum()
            else "unknown"
        )
        raise UnexpectedModelBehavior(f"LangChain model response is incomplete (reason: {code})")


def _without_stale_continuation(
    model: Any,
    *,
    reset_previous_response: bool,
    reason: str = "history_processor rewrote LangChain history",
) -> Any:
    updates: dict[str, Any] = {}
    if reset_previous_response and getattr(model, "use_previous_response_id", False):
        updates["use_previous_response_id"] = False

    model_kwargs = getattr(model, "model_kwargs", None)
    if isinstance(model_kwargs, dict):
        filtered = dict(model_kwargs)
        for key in ("conversation", "conversation_id", "previous_response_id"):
            filtered.pop(key, None)
        if filtered != model_kwargs:
            updates["model_kwargs"] = filtered

    for attribute in ("conversation", "conversation_id"):
        if getattr(model, attribute, None) is not None:
            updates[attribute] = None

    if not updates:
        return model
    copy_model = getattr(model, "model_copy", None)
    if not callable(copy_model):
        raise HistoryProcessingError(
            f"{reason}, but the configured model cannot disable its stale server-side continuation"
        )
    return copy_model(update=updates)
