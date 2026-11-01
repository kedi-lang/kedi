"""Retry feedback in model requests, across Pydantic AI's two representations.

Before 2.6x a rejected output call came back as a ``RetryPromptPart``; later releases send a
``ToolReturnPart`` whose ``outcome`` is ``"retried"``. Both carry the feedback in ``content``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic_ai.messages import ModelMessage, RetryPromptPart, ToolReturnPart


def retry_feedback(messages: Iterable[ModelMessage]) -> list[Any]:
    return [
        part
        for message in messages
        for part in message.parts
        if isinstance(part, RetryPromptPart)
        or (isinstance(part, ToolReturnPart) and getattr(part, "outcome", None) == "retried")
    ]
