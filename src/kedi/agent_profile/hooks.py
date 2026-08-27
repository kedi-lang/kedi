from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias, cast

from ..agent_adapter.hooks.models import HOOK_EVENT_NAMES, HookEventName, HookHandler

HookInput: TypeAlias = Mapping[HookEventName | str, HookHandler | Sequence[HookHandler]] | None


@dataclass(frozen=True, slots=True, kw_only=True)
class HookSettings:
    enabled: bool | None = None
    user_prompt_submit: tuple[HookHandler, ...] | None = None
    pre_tool_use: tuple[HookHandler, ...] | None = None
    post_tool_use: tuple[HookHandler, ...] | None = None
    post_tool_use_failure: tuple[HookHandler, ...] | None = None

    def handlers_for(self, event: HookEventName) -> tuple[HookHandler, ...]:
        if self.enabled is False:
            return ()
        return getattr(self, event) or ()

    @property
    def active(self) -> bool:
        return self.enabled is not False and any(
            self.handlers_for(event) for event in HOOK_EVENT_NAMES
        )

    def append(self, event: HookEventName, handler: HookHandler) -> HookSettings:
        if not callable(handler):
            raise TypeError(f"Hook handler for {event} must be callable")
        values = {name: getattr(self, name) for name in HOOK_EVENT_NAMES}
        current = values[event] or ()
        values[event] = current + (handler,)
        return HookSettings(enabled=self.enabled, **values)

    def merge(self, other: HookSettings) -> HookSettings:
        values: dict[str, tuple[HookHandler, ...] | None] = {}
        for event in HOOK_EVENT_NAMES:
            left = getattr(self, event)
            right = getattr(other, event)
            if right is None:
                values[event] = left
            elif not right:
                values[event] = ()
            else:
                values[event] = (left or ()) + right
        return HookSettings(
            enabled=self.enabled if other.enabled is None else other.enabled,
            **values,
        )

    def compose_enforced(self, local: HookSettings | None) -> HookSettings:
        """Append active local handlers without allowing them to suppress enforcement."""

        if not self.active:
            return local or self
        values = {
            event: self.handlers_for(event) + (() if local is None else local.handlers_for(event))
            for event in HOOK_EVENT_NAMES
        }
        return HookSettings(enabled=True, **values)


def normalize_hook_settings(value: HookSettings | HookInput) -> HookSettings | None:
    if value is None or isinstance(value, HookSettings):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("hooks must be a HookSettings value or an event-to-handler mapping")

    values: dict[str, tuple[HookHandler, ...] | None] = {}
    for raw_event, raw_handlers in value.items():
        if raw_event not in HOOK_EVENT_NAMES:
            expected = ", ".join(HOOK_EVENT_NAMES)
            raise ValueError(f"Unknown Kedi hook event {raw_event!r}; expected one of: {expected}")
        event = cast(HookEventName, raw_event)
        if callable(raw_handlers):
            handlers = (cast(HookHandler, raw_handlers),)
        elif isinstance(raw_handlers, Sequence) and not isinstance(raw_handlers, (str, bytes)):
            handlers = tuple(raw_handlers)
            if not all(callable(handler) for handler in handlers):
                raise TypeError(f"Every hook handler for {event} must be callable")
        else:
            raise TypeError(f"Hook handlers for {event} must be callable or a sequence")
        values[event] = handlers
    return _settings_from_values(values)


def _settings_from_values(
    values: Mapping[str, tuple[HookHandler, ...] | None],
    *,
    enabled: bool | None = None,
) -> HookSettings:
    return HookSettings(
        enabled=enabled,
        user_prompt_submit=values.get("user_prompt_submit"),
        pre_tool_use=values.get("pre_tool_use"),
        post_tool_use=values.get("post_tool_use"),
        post_tool_use_failure=values.get("post_tool_use_failure"),
    )


__all__ = ["HookInput", "HookSettings", "normalize_hook_settings"]
