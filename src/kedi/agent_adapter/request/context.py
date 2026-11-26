from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Generator

from .models import AgentRunRequest

if TYPE_CHECKING:
    from .media import MediaAttachment, MediaNamer

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


_TOOL_MEDIA_NAMER: ContextVar[MediaNamer | None] = ContextVar(
    "kedi_tool_media_namer",
    default=None,
)
# How many media the namer held when the request began: CodeMode attaches only media that
# tools returned during the request.
_TOOL_MEDIA_START: ContextVar[int] = ContextVar("kedi_tool_media_start", default=0)
_CONVERSATION_NAMER: ContextVar[MediaNamer | None] = ContextVar(
    "kedi_conversation_media_namer",
    default=None,
)


def tool_media_namer() -> MediaNamer:
    """The namer for media that tools return in the current request.

    One namer serves every tool call of a request, so ``snap_result_1`` and
    ``snap_result_2`` stay apart; outside a request each call gets its own.
    """

    from .media import MediaNamer

    return _TOOL_MEDIA_NAMER.get() or MediaNamer()


def request_tool_media() -> tuple[MediaAttachment, ...]:
    """The media tools returned during the current request, named by its namer."""

    return tool_media_namer().attachments[_TOOL_MEDIA_START.get() :]


@contextmanager
def conversation_media_scope(namer: MediaNamer | None) -> Generator[None, None, None]:
    """Name the requests inside this scope with ``namer``: a conversation's, or ``None``.

    Every Kedi request sets it, so a nested call outside the conversation never inherits it.
    """

    token = _CONVERSATION_NAMER.set(namer)
    try:
        yield
    finally:
        _CONVERSATION_NAMER.reset(token)


def prompt_media() -> tuple[MediaAttachment, ...]:
    """The attachments of the prompt the current native run sends, taken or not.

    A subagent task's child receives its attachments here rather than through a request.
    """

    claim = _PROMPT_MEDIA.get()
    return () if claim is None else claim.media


@contextmanager
def tool_media_scope() -> Generator[None, None, None]:
    """Keep tool media names unique in an adapter run started outside a Kedi request."""

    if _TOOL_MEDIA_NAMER.get() is not None:
        yield
        return
    from .media import MediaNamer

    token = _TOOL_MEDIA_NAMER.set(MediaNamer())
    try:
        yield
    finally:
        _TOOL_MEDIA_NAMER.reset(token)


_REQUEST_OBSERVERS: ContextVar[tuple[Callable[[AgentRunRequest], None], ...]] = ContextVar(
    "kedi_request_observers", default=()
)


@contextmanager
def observe_requests(
    callback: Callable[[AgentRunRequest], None],
) -> Generator[None, None, None]:
    """Call ``callback`` with each request Kedi hands an adapter inside this scope.

    The conformance kit uses it to compare what Kedi rendered with what reached the wire.
    """

    token = _REQUEST_OBSERVERS.set((*_REQUEST_OBSERVERS.get(), callback))
    try:
        yield
    finally:
        _REQUEST_OBSERVERS.reset(token)


@contextmanager
def agent_run_request_scope(request: AgentRunRequest) -> Generator[None, None, None]:
    """Expose one immutable semantic request while its adapter call is active."""

    for observer in _REQUEST_OBSERVERS.get():
        observer(request)
    token = _CURRENT_AGENT_RUN_REQUEST.set(request)
    # A request a subagent child makes through Kedi is a request of its own.
    entry_token = _SUBAGENT_ENTRY.set(False)
    media_token = _PROMPT_MEDIA.set(
        _PromptMediaClaim(request.prompt.media) if request.prompt.media else None
    )
    from .media import MediaNamer

    # A conversation's requests share its namer, so tool media never take an earlier name.
    namer = _CONVERSATION_NAMER.get() or MediaNamer()
    # A tool's media never take a prompt attachment's name: a model names either one.
    namer.reserve(attachment.name for attachment in request.prompt.media)
    namer_token = _TOOL_MEDIA_NAMER.set(namer)
    start_token = _TOOL_MEDIA_START.set(len(namer.attachments))
    try:
        yield
    finally:
        _TOOL_MEDIA_START.reset(start_token)
        _TOOL_MEDIA_NAMER.reset(namer_token)
        _PROMPT_MEDIA.reset(media_token)
        _SUBAGENT_ENTRY.reset(entry_token)
        _CURRENT_AGENT_RUN_REQUEST.reset(token)


def current_agent_run_request() -> AgentRunRequest | None:
    """Return the semantic request currently being lowered, when available."""

    return _CURRENT_AGENT_RUN_REQUEST.get()


_SUBAGENT_ENTRY: ContextVar[bool] = ContextVar("kedi_subagent_entry", default=False)


def has_current_request() -> bool:
    """Whether :func:`current_request` has a request for the adapter call in progress.

    False outside a Kedi call, and when Kedi calls ``invoke``, ``produce`` or
    ``produce_subagent`` to run a subagent child: use the call's arguments then.
    """

    return not _SUBAGENT_ENTRY.get() and _CURRENT_AGENT_RUN_REQUEST.get() is not None


def current_request() -> AgentRunRequest:
    """The request Kedi is making through the adapter, inside ``produce`` and ``invoke``.

    ``current_request().render()`` gives everything the model sees. Raises outside a Kedi
    call, and when Kedi runs a subagent child (its ``invoke``, ``produce`` or
    ``produce_subagent``), whose request is its parent's; check :func:`has_current_request`
    and use the call's arguments there.
    """

    if _SUBAGENT_ENTRY.get():
        raise RuntimeError(
            "current_request() is not available while Kedi runs a subagent child; check "
            "has_current_request() and use the call's arguments"
        )
    request = _CURRENT_AGENT_RUN_REQUEST.get()
    if request is None:
        raise RuntimeError("current_request() is only available inside an adapter call Kedi makes")
    return request


@contextmanager
def subagent_entry_scope() -> Generator[None, None, None]:
    """Mark a subagent entry point: the parent's request is not the child's."""

    token = _SUBAGENT_ENTRY.set(True)
    try:
        yield
    finally:
        _SUBAGENT_ENTRY.reset(token)


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
    "conversation_media_scope",
    "current_agent_run_request",
    "current_request",
    "has_current_request",
    "observe_requests",
    "prompt_media",
    "prompt_media_scope",
    "request_tool_media",
    "subagent_entry_scope",
    "take_prompt_media",
    "tool_media_namer",
    "tool_media_scope",
]
