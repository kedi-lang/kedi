from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, TypeAlias, cast

from ...errors import KediExecutionError
from ..request.media import MediaSummary
from ..tools.records import ToolCallRecord

HookEventName: TypeAlias = Literal[
    "user_prompt_submit",
    "pre_tool_use",
    "post_tool_use",
    "post_tool_use_failure",
    "post_tool_result_failure",
]
ToolOrigin: TypeAlias = Literal["kedi", "mcp", "adapter_native", "provider_builtin"]
HookDecisionKind: TypeAlias = Literal["continue", "deny", "edit"]

HOOK_EVENT_NAMES: tuple[HookEventName, ...] = (
    "user_prompt_submit",
    "pre_tool_use",
    "post_tool_use",
    "post_tool_use_failure",
    "post_tool_result_failure",
)


def validate_hook_event_name(value: str) -> HookEventName:
    if value not in HOOK_EVENT_NAMES:
        expected = ", ".join(HOOK_EVENT_NAMES)
        raise ValueError(f"Unknown Kedi hook event {value!r}; expected one of: {expected}")
    return cast(HookEventName, value)


def validate_hook_event_names(
    value: HookEventName | str | Sequence[HookEventName | str],
) -> tuple[HookEventName, ...]:
    raw_events = (value,) if isinstance(value, str) else tuple(value)
    if not raw_events:
        raise ValueError("At least one Kedi hook event is required")
    return tuple(validate_hook_event_name(event) for event in raw_events)


def readonly_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not value:
        return MappingProxyType({})
    return MappingProxyType(copy.deepcopy(dict(value)))


@dataclass(frozen=True, slots=True, kw_only=True)
class HookContext:
    event: HookEventName
    run_id: str
    sequence: int
    adapter_shortname: str
    parent_run_id: str | None = None
    agent_name: str | None = None
    profile_name: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class UserPromptSubmitRequest(HookContext):
    """The rendered prompt text, and a byte-free summary of each attachment.

    The attachments are read-only: an ``edit`` decision replaces only ``content``.
    """

    event: Literal["user_prompt_submit"] = field(default="user_prompt_submit", init=False)
    content: str = field(repr=False)
    media: tuple[MediaSummary, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolHookContext(HookContext):
    tool_call_id: str
    tool_name: str
    origin: ToolOrigin
    arguments: Mapping[str, Any] = field(repr=False)
    native_tool_name: str | None = None
    description: str | None = field(default=None, repr=False)
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", readonly_mapping(self.arguments))
        object.__setattr__(self, "metadata", readonly_mapping(self.metadata))


@dataclass(frozen=True, slots=True, kw_only=True)
class PreToolUseRequest(ToolHookContext):
    event: Literal["pre_tool_use"] = field(default="pre_tool_use", init=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class PostToolUseEvent(ToolHookContext):
    event: Literal["post_tool_use"] = field(default="post_tool_use", init=False)
    result: Any = field(repr=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class PostToolUseFailureEvent(ToolHookContext):
    event: Literal["post_tool_use_failure"] = field(
        default="post_tool_use_failure",
        init=False,
    )
    error_type: str
    error_message: str = field(repr=False)
    error: BaseException | None = field(default=None, repr=False, compare=False)
    duration_seconds: float | None = None
    interrupted: bool = False

    def __post_init__(self) -> None:
        ToolHookContext.__post_init__(self)
        if self.duration_seconds is not None and self.duration_seconds < 0:
            raise ValueError("Hook failure duration cannot be negative")


@dataclass(frozen=True, slots=True, kw_only=True)
class PostToolResultFailureEvent(ToolHookContext):
    event: Literal["post_tool_result_failure"] = field(
        default="post_tool_result_failure", init=False
    )
    record: ToolCallRecord = field(repr=False)
    error_type: str
    error_message: str = field(repr=False)
    error: BaseException | None = field(default=None, repr=False, compare=False)

    @property
    def tool_call(self) -> ToolCallRecord:
        return self.record


@dataclass(frozen=True, slots=True, kw_only=True)
class UserPromptSubmitDecision:
    decision: HookDecisionKind = "continue"
    reason: str | None = None
    edited_content: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _validate_decision(
            decision=self.decision,
            reason=self.reason,
            edited_value=self.edited_content,
            edited_name="edited_content",
        )

    @classmethod
    def continue_(cls) -> UserPromptSubmitDecision:
        return cls()

    @classmethod
    def deny(cls, reason: str) -> UserPromptSubmitDecision:
        return cls(decision="deny", reason=reason)

    @classmethod
    def edit(cls, content: str) -> UserPromptSubmitDecision:
        return cls(decision="edit", edited_content=content)


@dataclass(frozen=True, slots=True, kw_only=True)
class PreToolUseDecision:
    decision: HookDecisionKind = "continue"
    reason: str | None = None
    edit_arguments: Mapping[str, Any] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _validate_decision(
            decision=self.decision,
            reason=self.reason,
            edited_value=self.edit_arguments,
            edited_name="edit_arguments",
        )
        if self.edit_arguments is not None:
            object.__setattr__(self, "edit_arguments", readonly_mapping(self.edit_arguments))

    @classmethod
    def continue_(cls) -> PreToolUseDecision:
        return cls()

    @classmethod
    def deny(cls, reason: str) -> PreToolUseDecision:
        return cls(decision="deny", reason=reason)

    @classmethod
    def edit(cls, arguments: Mapping[str, Any]) -> PreToolUseDecision:
        return cls(decision="edit", edit_arguments=arguments)


def _validate_decision(
    *,
    decision: HookDecisionKind,
    reason: str | None,
    edited_value: object | None,
    edited_name: str,
) -> None:
    if decision == "continue":
        if reason is not None or edited_value is not None:
            raise ValueError("A continue hook decision cannot include a reason or edit")
        return
    if decision == "deny":
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("A deny hook decision requires a non-empty reason")
        if edited_value is not None:
            raise ValueError(f"A deny hook decision cannot include {edited_name}")
        return
    if decision == "edit":
        if edited_value is None:
            raise ValueError(f"An edit hook decision requires {edited_name}")
        if reason is not None:
            raise ValueError("An edit hook decision cannot include a denial reason")
        return
    raise ValueError(f"Unknown hook decision {decision!r}")


HookEvent: TypeAlias = (
    UserPromptSubmitRequest
    | PreToolUseRequest
    | PostToolUseEvent
    | PostToolUseFailureEvent
    | PostToolResultFailureEvent
)
HookResult: TypeAlias = UserPromptSubmitDecision | PreToolUseDecision | None
HookHandler: TypeAlias = Callable[[HookEvent], HookResult | Awaitable[HookResult]]


class HookExecutionError(KediExecutionError):
    def __init__(
        self,
        *,
        event: HookEventName,
        handler_name: str,
        cause: BaseException,
    ) -> None:
        self.event = event
        self.handler_name = handler_name
        super().__init__(
            message=f"Hook handler {handler_name!r} failed for {event}",
            original=cause,
        )


class HookTimeoutError(HookExecutionError):
    def __init__(
        self,
        *,
        event: HookEventName,
        handler_name: str,
        timeout: float,
    ) -> None:
        self.timeout = timeout
        super().__init__(
            event=event,
            handler_name=handler_name,
            cause=TimeoutError(f"Hook exceeded {timeout:g} seconds"),
        )
        self.message = f"Hook handler {handler_name!r} timed out during {event}"
        RuntimeError.__init__(self, self.message)


class HookDeniedError(KediExecutionError):
    def __init__(self, *, event: HookEventName, reason: str) -> None:
        self.event = event
        self.reason = reason
        super().__init__(message=f"Kedi hook denied {event}: {reason}")


__all__ = [
    "HOOK_EVENT_NAMES",
    "HookContext",
    "HookDecisionKind",
    "HookDeniedError",
    "HookEvent",
    "HookEventName",
    "HookExecutionError",
    "HookHandler",
    "HookResult",
    "HookTimeoutError",
    "PostToolUseEvent",
    "PostToolUseFailureEvent",
    "PostToolResultFailureEvent",
    "PreToolUseDecision",
    "PreToolUseRequest",
    "ToolHookContext",
    "ToolOrigin",
    "UserPromptSubmitDecision",
    "UserPromptSubmitRequest",
    "readonly_mapping",
    "validate_hook_event_name",
    "validate_hook_event_names",
]
