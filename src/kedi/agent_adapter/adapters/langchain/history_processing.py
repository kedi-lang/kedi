from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

from langchain_core.messages import BaseMessage

from kedi.agent_adapter.compaction.langchain_codec import LangChainHistoryCodec


class LangChainHistoryMessageCodec:
    adapter_shortname = "langchain"
    state_key = "langchain_messages"

    def __init__(self) -> None:
        self._codec = LangChainHistoryCodec()

    def clone(self, messages: Sequence[BaseMessage]) -> list[BaseMessage]:
        if not all(isinstance(message, BaseMessage) for message in messages):
            raise TypeError("all values must be LangChain BaseMessage instances")
        return deepcopy(list(messages))

    def inspect(self, messages: Sequence[BaseMessage]):
        return self._codec.inspect({self.state_key: list(messages)})


def langchain_model_id(model: Any) -> str | None:
    for attribute in ("model_name", "model", "model_id"):
        value = getattr(model, attribute, None)
        if isinstance(value, str):
            return value
    return None
