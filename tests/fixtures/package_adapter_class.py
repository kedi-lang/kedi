"""A Python adapter class a test package declares (`class: package_adapter_class:PeerAdapter`)."""

from __future__ import annotations

import sys
from pathlib import Path

from kedi.agent_adapter import JsonRpcAdapter

PEER = Path(__file__).parent / "jsonrpc_peer.py"


class PeerAdapter(JsonRpcAdapter):
    """The JSON-RPC test peer, launched by the adapter's own class."""

    def __init__(self, model: str | None = None) -> None:
        super().__init__([sys.executable, str(PEER), "default"], model=model)


class FrameworkStub:
    """A framework-kind class with nothing to close, declared as an agent by a test package."""

    kind = "agent-framework"

    def __init__(self, model: str | None = None) -> None:
        self.model = model
