"""Semantic parity (#197): the reference JSON-RPC bridge and an in-process adapter agree.

The conformance kit checks one adapter at a time against the wire. These tests run the kit's
scenarios through Kedi's Pydantic AI adapter and through ``JsonRpcAdapter`` with the reference
bridge, against the same scripted endpoint. For each program, both must produce the same value
or the same error, from the same rendered requests, in the same number of model requests.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from typing import Any

import pytest

from kedi.agent_adapter import JsonRpcAdapter, PydanticAdapter, RenderedRequest
from kedi.agent_adapter.adapters.jsonrpc import reference_bridge_command
from kedi.agent_adapter.conformance.check import (
    ANY_ERROR,
    SCENARIOS,
    Scenario,
    Variant,
    _run,
    _wire,
)
from kedi.agent_adapter.conformance.endpoint import RecordingEndpoint, Script, _openai_answer
from kedi.agent_adapter.request.context import observe_requests
from kedi.lang import compile_program, parse_program

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


def _outcome(run: Any, variant: Variant) -> tuple[str | bool | None, Any]:
    error = run.error
    cause = getattr(error, "original", None) or error
    if variant.error == ANY_ERROR:
        # A failed model request: each stack reports it in its own error type.
        return (cause is not None, run.value)
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
    for variant, local, bridge in zip(
        scenario.variants, runs["pydantic"], runs["jsonrpc"], strict=True
    ):
        assert local.rendered and local.exchanges, "the scenario reached no model"
        assert _outcome(bridge, variant) == _outcome(local, variant)
        assert [_summary(r) for r in bridge.rendered] == [_summary(r) for r in local.rendered]
        assert len(bridge.exchanges) == len(local.exchanges)


class DelegatingEndpoint(RecordingEndpoint):
    """The parent model delegates once and reports; the child answers in text."""

    def _answer(self, protocol: Any, body: dict[str, Any]) -> dict[str, Any]:
        offered = {tool["function"]["name"] for tool in body.get("tools") or []}
        delegated = any(message.get("role") == "tool" for message in body.get("messages") or [])
        if "delegate_task" in offered and not delegated:
            task = {"subagent": "worker", "task": "Name the largest ocean."}
            script = Script(final=None, text=True, tool="delegate_task", arguments=task)
        elif "delegate_task" in offered:
            script = Script(final="The worker says: the Pacific.", text=True)
        else:
            script = Script(final="The Pacific.", text=True)
        return _openai_answer(script, body)


DELEGATION = """
> profile: worker:
    ###
    Answers one question.
    ###
    > instructions: Answer in two words.

> profile: manager:
    > subagent: worker

> use: manager
[reply] << Ask the worker for the largest ocean, then report its answer.
= <reply>
"""


def _delegate(factory: Any) -> tuple[Any, list[dict[str, Any]], list[Any]]:
    with DelegatingEndpoint() as endpoint:
        adapter = factory(endpoint.base_url)
        rendered: list[dict[str, Any]] = []
        runtime = compile_program(parse_program(DELEGATION), adapter=adapter)
        try:
            with observe_requests(lambda request: rendered.append(_summary(request.render()))):
                value = runtime.run_main()
        finally:
            asyncio.run(runtime.aclose())
            getattr(adapter, "close", lambda: None)()
        return value, rendered, endpoint.take()


def test_a_child_runs_on_the_bridge_as_it_runs_in_process() -> None:
    local, bridge = _delegate(PydanticAdapter.for_conformance), _delegate(_bridge)
    assert local[0] == bridge[0] == "The worker says: the Pacific."
    # The parent's request is the same, but for live input to children, which the bridge
    # protocol does not carry yet.
    (local_parent,), (bridge_parent,) = local[1], bridge[1]
    live_input = [tool for tool in local_parent["tools"] if tool not in bridge_parent["tools"]]
    assert live_input == ["send_subagent_input"]
    assert bridge_parent == {**local_parent, "tools": bridge_parent["tools"]}
    assert bridge_parent["tools"] == [
        tool for tool in local_parent["tools"] if tool not in live_input
    ]
    # The parent's two requests and the child's one, through either stack.
    assert len(bridge[2]) == len(local[2]) == 3
    # Kedi renders the child's request; both stacks send its system text and task as given.
    children = [
        [
            _wire(exchange)
            for exchange in exchanges
            if "delegate_task" not in json.dumps(exchange.body)
        ]
        for exchanges in (local[2], bridge[2])
    ]
    (local_child,), (bridge_child,) = children
    assert (bridge_child.system, bridge_child.user) == (local_child.system, local_child.user)
    assert "Answer in two words." in "".join(local_child.system)
    assert local_child.user == ("Name the largest ocean.",)
