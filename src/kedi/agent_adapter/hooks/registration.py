from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType
from typing import Any, TypeVar, cast

from .models import HookEventName, HookHandler, validate_hook_event_names

_HandlerT = TypeVar("_HandlerT", bound=Callable[..., Any])


def handler_identity(handler: Callable[..., Any]) -> str:
    module = getattr(handler, "__module__", None)
    qualname = getattr(handler, "__qualname__", None)
    if isinstance(module, str) and isinstance(qualname, str):
        return f"{module}.{qualname}"
    handler_type = type(handler)
    return f"{handler_type.__module__}.{handler_type.__qualname__}"


@dataclass(frozen=True, slots=True)
class AdapterHookSnapshot:
    catch_all: tuple[HookHandler, ...]
    by_event: Mapping[HookEventName, tuple[HookHandler, ...]]

    @classmethod
    def empty(cls) -> AdapterHookSnapshot:
        return cls(catch_all=(), by_event=MappingProxyType({}))

    def handlers_for(self, event: HookEventName) -> tuple[HookHandler, ...]:
        return self.catch_all + self.by_event.get(event, ())

    @property
    def active(self) -> bool:
        return bool(self.catch_all or self.by_event)


class HookRegistrationMixin:
    """Adapter-instance hook registration without cooperative inheritance requirements."""

    _hook_registration_lock: RLock
    _hook_catch_all: tuple[HookHandler, ...]
    _hook_handlers: dict[HookEventName, list[HookHandler]]

    def _init_hook_registration(self, hook_handler: HookHandler | None = None) -> None:
        if hook_handler is not None and not callable(hook_handler):
            raise TypeError("hook_handler must be callable or None")
        self._hook_registration_lock = RLock()
        self._hook_catch_all = () if hook_handler is None else (hook_handler,)
        self._hook_handlers = {}

    def on(
        self,
        events: HookEventName | str | Sequence[HookEventName | str],
    ) -> Callable[[_HandlerT], _HandlerT]:
        event_names = validate_hook_event_names(events)

        def register(handler: _HandlerT) -> _HandlerT:
            if not callable(handler):
                raise TypeError("Kedi hook handler must be callable")
            with self._hook_registration_lock:
                for event_name in event_names:
                    self._hook_handlers.setdefault(event_name, []).append(
                        cast(HookHandler, handler)
                    )
            return handler

        return register

    def hook_snapshot(self) -> AdapterHookSnapshot:
        with self._hook_registration_lock:
            by_event = {event: tuple(handlers) for event, handlers in self._hook_handlers.items()}
            return AdapterHookSnapshot(
                catch_all=self._hook_catch_all,
                by_event=MappingProxyType(by_event),
            )

    def _replace_hook_snapshot(self, snapshot: AdapterHookSnapshot) -> None:
        """Replace registrations while transferring ownership through a lazy adapter."""

        with self._hook_registration_lock:
            self._hook_catch_all = snapshot.catch_all
            self._hook_handlers = {
                event: list(handlers) for event, handlers in snapshot.by_event.items()
            }

    def _prepend_hook_snapshot(self, snapshot: AdapterHookSnapshot) -> None:
        """Transfer outer registrations while preserving an adapter's own handlers."""

        if not snapshot.active:
            return
        with self._hook_registration_lock:
            self._hook_catch_all = snapshot.catch_all + self._hook_catch_all
            for event, handlers in snapshot.by_event.items():
                self._hook_handlers[event] = list(handlers) + self._hook_handlers.get(event, [])


__all__ = [
    "AdapterHookSnapshot",
    "HookRegistrationMixin",
    "handler_identity",
]
