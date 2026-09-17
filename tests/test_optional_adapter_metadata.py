from __future__ import annotations

import importlib
import sys


def test_dspy_metadata_does_not_import_optional_runtime(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "dspy", None)
    sys.modules.pop("kedi.agent_adapter.dspy_metadata", None)

    metadata = importlib.import_module("kedi.agent_adapter.dspy_metadata")

    assert metadata.DSPY_ADAPTER.shortname == "dspy"
    assert metadata.DSPY_ADAPTER.kind == "agent-framework"
    assert metadata.DSPY_ADAPTER.capabilities.supports_structured_output is True
