"""Harbor integration for running Kedi as an installed benchmark agent."""

# pyright: reportUnsupportedDunderAll=false

from __future__ import annotations

from typing import Any

__all__ = ["KediAgent"]


def __getattr__(name: str) -> Any:
    if name != "KediAgent":
        raise AttributeError(name)
    try:
        from .agent import KediAgent
    except ImportError as exc:
        if exc.name == "harbor" or (exc.name or "").startswith("harbor."):
            raise ImportError(
                "Kedi's Harbor agent requires the 'terminal-bench' optional dependency; "
                "install it with `pip install 'kedi[terminal-bench]'` on Python 3.12 or newer"
            ) from exc
        raise
    return KediAgent
