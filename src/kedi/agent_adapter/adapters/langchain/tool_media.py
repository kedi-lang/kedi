"""Media that LangChain tools return: native blocks, placed where the provider reads them."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import AgentMiddleware

from kedi.agent_adapter.adapters.langchain.media import (
    split_tool_media,
    tool_message_kinds_allowed,
    tool_result_blocks,
)
from kedi.agent_adapter.tools.media import ToolMedia, codemode_media, tool_media
from kedi.agent_adapter.tools.scope import results_go_to_code


@dataclass(frozen=True, slots=True)
class _ToolMediaTarget:
    model: Any
    media_limit: int | None


_TARGET: ContextVar[_ToolMediaTarget | None] = ContextVar("kedi_langchain_tool_media", default=None)


async def lower_codemode_result(result: Any) -> Any:
    """A CodeMode result naming media from its nested calls, as content blocks."""

    media = codemode_media(result)
    if media is None:
        return result
    return await _blocks(media, tool_name="execute_code")


async def lower_tool_result(result: Any, *, tool_name: str) -> Any:
    """A tool result holding media as content blocks for the run's model; others unchanged."""

    if results_go_to_code():
        # A nested CodeMode call: the code receives placeholders at its boundary.
        return result
    media = tool_media(result, tool_name=tool_name)
    if media is None:
        return result
    return await _blocks(media, tool_name=tool_name)


async def _blocks(media: ToolMedia, *, tool_name: str) -> list[dict[str, Any]]:
    target = _TARGET.get()
    if target is None:
        from kedi.media import MediaError

        raise MediaError(f"Tool {tool_name!r} returned media outside a LangChain run")
    return await tool_result_blocks(media, model=target.model, media_limit=target.media_limit)


class LangChainToolMediaMiddleware(AgentMiddleware):
    """Give tool calls the run's model, and move media a provider reads only in user messages.

    Tools lower their media for ``model`` (its provider rules, profile and size limit). Before
    each model call, media blocks the model takes no tool messages for move into a user message
    after the tool messages; the stored history keeps them in place.
    """

    def __init__(self, model: Any, *, media_limit: int | None) -> None:
        super().__init__()
        self._target = _ToolMediaTarget(model=model, media_limit=media_limit)

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        token = _TARGET.set(self._target)
        try:
            return await handler(request)
        finally:
            _TARGET.reset(token)

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        # The model of this call decides, so a fallback model gets its own placement.
        allowed = tool_message_kinds_allowed(getattr(request, "model", None) or self._target.model)
        messages = split_tool_media(request.messages, allowed=allowed)
        if messages != list(request.messages):
            request = request.override(messages=messages)
        return await handler(request)


__all__ = ["LangChainToolMediaMiddleware", "lower_codemode_result", "lower_tool_result"]
