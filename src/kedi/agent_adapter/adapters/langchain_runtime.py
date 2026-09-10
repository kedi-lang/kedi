"""Per-run LangChain lifecycle: exact archival, usage and bounded tool correction."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any, Iterator

from langchain.agents.middleware import AgentMiddleware
from langgraph.types import Overwrite
from pydantic_ai import UsageLimits
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.usage import RunUsage

from ..compaction.archive import HistoryArchiveSettings, cache_epoch_key
from ..compaction.langchain_archive import LangChainArtifactHistoryProcessor
from ..invocation import InvocationUsage


@dataclass
class LangChainRunState:
    limits: UsageLimits
    usage: InvocationUsage = field(default_factory=InvocationUsage)
    native_usage: RunUsage = field(default_factory=RunUsage)
    corrections: Counter[str] = field(default_factory=Counter)
    pending_tools: int = 0


_RUN: ContextVar[LangChainRunState | None] = ContextVar("kedi_langchain_run", default=None)


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
    token = _RUN.set(state)
    try:
        yield state
    finally:
        _RUN.reset(token)


class LangChainRunMiddleware(AgentMiddleware):
    def __init__(self, *, archive: HistoryArchiveSettings | None, tool_retries: int) -> None:
        self.archiver = LangChainArtifactHistoryProcessor(archive) if archive is not None else None
        self.tool_retries = tool_retries

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        if self.archiver is None:
            return None
        messages = state["messages"]
        updated = self.archiver(messages)
        if updated is messages:
            return None
        # Replace the graph's durable message state, not just this request's view.
        return {"messages": Overwrite(updated)}

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        from .langchain import _langchain_invocation_usage

        state = _RUN.get()
        if state is not None:
            state.limits.check_before_request(state.native_usage)
            state.native_usage.requests += 1
        checkpoint_id = next(
            (
                message.additional_kwargs["kedi_compaction_checkpoint"]
                for message in reversed(request.messages)
                if isinstance(message.additional_kwargs.get("kedi_compaction_checkpoint"), str)
            ),
            None,
        )
        if checkpoint_id is not None:
            settings = dict(request.model_settings)
            key = settings.get("prompt_cache_key")
            if key is None:
                key = getattr(request.model, "model_kwargs", {}).get("prompt_cache_key")
            if isinstance(key, str) and key:
                settings["prompt_cache_key"] = cache_epoch_key(key, checkpoint_id)
                request = request.override(model_settings=settings)
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
        return response

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        from .langchain import _langchain_tool_message_for_call

        state = _RUN.get()
        if state is None:
            return await handler(request)
        projected = replace(state.native_usage)
        projected.tool_calls += state.pending_tools + 1
        state.limits.check_before_tool_call(projected)
        state.pending_tools += 1
        try:
            result = await handler(request)
        finally:
            state.pending_tools -= 1
        call = request.tool_call
        message = _langchain_tool_message_for_call(result, call.get("id"))
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
