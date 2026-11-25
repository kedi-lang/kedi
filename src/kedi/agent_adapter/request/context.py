from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generator

from .models import AgentRunRequest

if TYPE_CHECKING:
    from .media import MediaAttachment

_CURRENT_AGENT_RUN_REQUEST: ContextVar[AgentRunRequest | None] = ContextVar(
    "kedi_current_agent_run_request",
    default=None,
)


@dataclass(slots=True)
class _PromptMediaClaim:
    media: tuple[MediaAttachment, ...]
    taken: bool = False


_PROMPT_MEDIA: ContextVar[_PromptMediaClaim | None] = ContextVar(
    "kedi_prompt_media",
    default=None,
)


@contextmanager
def agent_run_request_scope(request: AgentRunRequest) -> Generator[None, None, None]:
    """Expose one immutable semantic request while its adapter call is active."""

    token = _CURRENT_AGENT_RUN_REQUEST.set(request)
    media_token = _PROMPT_MEDIA.set(
        _PromptMediaClaim(request.prompt.media) if request.prompt.media else None
    )
    try:
        yield
    finally:
        _PROMPT_MEDIA.reset(media_token)
        _CURRENT_AGENT_RUN_REQUEST.reset(token)


def current_agent_run_request() -> AgentRunRequest | None:
    """Return the semantic request currently being lowered, when available."""

    return _CURRENT_AGENT_RUN_REQUEST.get()


@contextmanager
def prompt_media_scope(media: tuple[MediaAttachment, ...]) -> Generator[None, None, None]:
    """Give ``media`` to the first native run started inside this scope, as a request does.

    A subagent task's attachments travel this way: the child's run starts in a copy of this
    context and takes them; the parent's own runs are outside the scope.
    """

    token = _PROMPT_MEDIA.set(_PromptMediaClaim(media) if media else None)
    try:
        yield
    finally:
        _PROMPT_MEDIA.reset(token)


def take_prompt_media() -> tuple[MediaAttachment, ...]:
    """Claim the current request's attachments for the native run that sends its prompt.

    Adapter methods keep their ``prompt: str`` signatures, so attachments travel through the
    request scope. The first native run inside the scope takes them; runs nested in it (tool
    calls, subagents) get none.
    """

    claim = _PROMPT_MEDIA.get()
    if claim is None or claim.taken:
        return ()
    claim.taken = True
    return claim.media


__all__ = [
    "agent_run_request_scope",
    "current_agent_run_request",
    "prompt_media_scope",
    "take_prompt_media",
]
