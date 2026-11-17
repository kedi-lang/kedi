"""Rich display of ``> show:`` values for hosts such as Kedi Notebook.

``> show: `value``` prints the value's text. A host that can show more, an image or an audio
player, installs a hook for the run: the hook gets each value shown by a single backtick and
returns ``True`` when it displayed the value, and Kedi then prints nothing for it.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

DisplayHook = Callable[[Any], bool]

_hook: ContextVar[DisplayHook | None] = ContextVar("kedi_display_hook", default=None)


@contextmanager
def display_hook(hook: DisplayHook) -> Generator[None]:
    """Send values shown with ``> show: `value``` to ``hook`` while the block runs."""

    token = _hook.set(hook)
    try:
        yield
    finally:
        _hook.reset(token)


def display(value: Any) -> bool:
    """Offer ``value`` to the installed hook; ``False`` when there is none or it declined."""

    hook = _hook.get()
    return hook is not None and hook(value) is True


__all__ = ["DisplayHook", "display", "display_hook"]
