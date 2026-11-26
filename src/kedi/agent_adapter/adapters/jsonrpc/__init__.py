"""Kedi's adapter contract over JSON-RPC to a bridge process in any language (#197)."""

from .adapter import (
    REFERENCE_BRIDGE,
    JsonRpcAdapter,
    ModelCall,
    TestModel,
    reference_bridge_command,
)
from .protocol import PROTOCOL_VERSION, JsonRpcDisconnected, JsonRpcError

__all__ = [
    "PROTOCOL_VERSION",
    "REFERENCE_BRIDGE",
    "JsonRpcAdapter",
    "JsonRpcDisconnected",
    "JsonRpcError",
    "ModelCall",
    "TestModel",
    "reference_bridge_command",
]
