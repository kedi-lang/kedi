from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

pytest.importorskip("langchain_core")

from kedi import tool
from kedi.agent_adapter import LangChainAdapter
from kedi.agent_adapter.tools.conversion import python_callable_tool_spec


@pytest.mark.parametrize("explicit_schema", [False, True])
def test_langchain_codemode_keeps_callable_parameters_and_tool_policy(explicit_schema):
    attempts = []

    @tool(risk="read_only", retries=1, retry_on=(TimeoutError,))
    def lookup(identifier: str, count: int = 2) -> str:
        """Read a record by its exact identifier."""
        attempts.append((identifier, count))
        if len(attempts) == 1:
            raise TimeoutError("Transient fixture read")
        return identifier * count

    spec = python_callable_tool_spec("lookup", lookup)
    if explicit_schema:
        spec = replace(
            spec,
            json_schema={
                "type": "object",
                "properties": {
                    "identifier": {"type": "string", "enum": ["R7"]},
                    "count": {"type": "integer", "default": 2},
                },
                "required": ["identifier"],
                "additionalProperties": False,
            },
        )

    async def run():
        adapter = LangChainAdapter(codemode={"enabled": True, "preload_tools": ["lookup"]})
        with adapter.tool_scope([spec]):
            async with adapter._projected_tools_for_run() as projection:
                preloaded = projection.preloaded_tools[0]
                schema = preloaded["parameters_json_schema"]
                assert schema["required"] == ["identifier"]
                assert schema["properties"]["identifier"]["type"] == "string"
                assert schema["properties"]["count"]["default"] == 2
                if explicit_schema:
                    assert schema == spec.json_schema
                assert preloaded["return_schema"] == {"type": "string"}
                result = await projection.bridge.execute_code(
                    code='await lookup(identifier="R7", count=3)'
                )
                assert result == "R7R7R7"

    asyncio.run(run())
    assert attempts == [("R7", 3), ("R7", 3)]
    if not explicit_schema:
        assert spec.json_schema is None
