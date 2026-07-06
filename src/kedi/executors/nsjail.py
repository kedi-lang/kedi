from __future__ import annotations

from .playground import PlaygroundExecutor


class NsJailExecutor(PlaygroundExecutor):
    """Execute user-authored Python through an NsJail-backed bridge worker."""


__all__ = ["NsJailExecutor"]
