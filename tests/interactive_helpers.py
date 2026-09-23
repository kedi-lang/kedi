"""Inspect typed bindings in session tests without interactive root returns."""

from typing import Any

from kedi.interactive_session import InteractiveSession
from kedi.promise import force


def inspect_fragment(session: InteractiveSession, source: str, **kwargs: Any) -> Any:
    session.execute(source, **kwargs)
    return force(session._env["inspection_value"])
