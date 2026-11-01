"""Retry feedback in model requests: the retried call's own ``ToolReturnPart``."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic_ai.messages import ModelMessage, ToolReturnPart


def retry_feedback(messages: Iterable[ModelMessage]) -> list[Any]:
    return [
        part
        for message in messages
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.outcome == "retried"
    ]
