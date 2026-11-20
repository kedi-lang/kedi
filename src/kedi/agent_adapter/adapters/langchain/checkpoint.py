"""LangGraph checkpoint serialization for Kedi's runtime-built output models."""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel

_DYNAMIC_MODULE = "kedi.dynamic_models"


class KediCheckpointSerializer(JsonPlusSerializer):
    """Store Kedi's dynamic output models as plain data.

    Their classes are built per request and cannot be imported back, so LangGraph would
    deserialize them as dicts with an unregistered-type warning, and block them in strict
    MessagePack mode. Writing their data directly gives the same dict without either.
    """

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        return super().dumps_typed(_plain(obj))


def _plain(value: Any) -> Any:
    if isinstance(value, BaseModel) and type(value).__module__ == _DYNAMIC_MODULE:
        return value.model_dump(mode="python")
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_plain(item) for item in value)
    return value
