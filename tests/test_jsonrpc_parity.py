"""Semantic parity (#197): the reference JSON-RPC bridge and an in-process adapter agree.

The conformance kit checks one adapter at a time against the wire. These tests run the kit's
scenarios through Kedi's Pydantic AI adapter and through ``JsonRpcAdapter`` with the reference
bridge, against the same scripted endpoint. For each program, both must produce the same value
or the same error, from the same rendered requests, in the same number of model requests.
"""

from __future__ import annotations

import shutil
from typing import Any

import pytest

from kedi.agent_adapter import JsonRpcAdapter, PydanticAdapter, RenderedRequest
from kedi.agent_adapter.adapters.jsonrpc import reference_bridge_command
from kedi.agent_adapter.conformance.check import SCENARIOS, Scenario, _run
from kedi.agent_adapter.conformance.endpoint import RecordingEndpoint

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="the reference bridge needs Node.js"
)


def _bridge(base_url: str) -> JsonRpcAdapter:
    return JsonRpcAdapter(reference_bridge_command(f"{base_url}/v1", mode="native"))


FACTORIES = {"pydantic": PydanticAdapter.for_conformance, "jsonrpc": _bridge}


def _summary(rendered: RenderedRequest) -> dict[str, Any]:
    return {
        "system": rendered.system,
        "parts": [(part.identity, part.source) for part in rendered.instruction_parts],
        "user": rendered.user,
        "tools": [tool.name for tool in rendered.tools],
        "output": rendered.output_json_schema,
        "kind": (rendered.prompt_kind, rendered.call_kind),
    }


def _outcome(run: Any) -> tuple[str | None, Any]:
    error = run.error
    cause = getattr(error, "original", None) or error
    return (type(cause).__name__ if cause is not None else None, run.value)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda scenario: scenario.name)
def test_the_bridge_and_pydantic_ai_agree(scenario: Scenario) -> None:
    with RecordingEndpoint() as endpoint:
        if scenario.requires is not None:
            for factory in FACTORIES.values():
                adapter = factory(endpoint.base_url)
                capabilities = adapter.capabilities
                getattr(adapter, "close", lambda: None)()
                if not capabilities.supports(scenario.requires):
                    pytest.skip(f"{scenario.requires} is not supported by both adapters")
        runs = {
            name: [_run(endpoint, factory, variant) for variant in scenario.variants]
            for name, factory in FACTORIES.items()
        }
    for local, bridge in zip(runs["pydantic"], runs["jsonrpc"], strict=True):
        assert local.rendered and local.exchanges, "the scenario reached no model"
        assert _outcome(bridge) == _outcome(local)
        assert [_summary(r) for r in bridge.rendered] == [_summary(r) for r in local.rendered]
        assert len(bridge.exchanges) == len(local.exchanges)
