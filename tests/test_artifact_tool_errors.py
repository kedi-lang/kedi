from __future__ import annotations

import asyncio
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from kedi.agent_adapter import PydanticAdapter
from kedi.agent_adapter.adapters.claude import _claude_sdk_tool
from kedi.agent_adapter.adapters.langchain import _langchain_tool_from_spec
from kedi.agent_adapter.tools.specs import ToolSpec
from kedi.artifacts import (
    ArtifactAccessError,
    ArtifactHandle,
    ArtifactInputError,
    ArtifactManager,
    ArtifactPolicy,
    ArtifactQuotaExceededError,
    ArtifactSerializationError,
    artifact_tool_specs,
)
from kedi.artifacts.errors import artifact_tool_error_message


@pytest.fixture(params=["memory", "file"])
def artifact_tools(
    request: pytest.FixtureRequest, tmp_path: Path
) -> Iterator[tuple[ArtifactManager, dict[str, ToolSpec], str]]:
    manager = ArtifactManager(session_id="tool-errors")
    policy = ArtifactPolicy(store=request.param, path=tmp_path, threshold_bytes=1)
    handle = manager.artifactize({"stdout": "row\n" * 1000, "items": [1, 2]}, policy=policy)
    assert isinstance(handle, ArtifactHandle)
    try:
        yield manager, {tool.name: tool for tool in artifact_tool_specs(manager)}, handle.ref.ref_id
    finally:
        manager.close()


@pytest.mark.parametrize(
    "arguments",
    [
        {"pattern": "row", "path": "/stdout"},
        {"pattern": "row", "offset": 1},
        {"pattern": "row", "offset_from": "end"},
        {"pattern": ""},
        {"pattern": "row", "max_matches": 0},
        {"offset": -1},
        {"offset_from": "invalid"},
        {"max_chars": 0},
        {"max_chars": -2},
        {"path": "stdout"},
        {"path": "/missing"},
        {"path": "/items/100"},
        {"path": "/items/not-an-index"},
        {"path": "/stdout/child"},
    ],
)
def test_invalid_reads_are_typed_and_do_not_damage_artifact(
    artifact_tools: tuple[ArtifactManager, dict[str, ToolSpec], str],
    arguments: dict[str, Any],
) -> None:
    manager, tools, ref_id = artifact_tools
    with pytest.raises(ArtifactInputError):
        tools["read_artifact"].fn(ref_id=ref_id, **arguments)
    assert tools["read_artifact"].fn(ref_id=ref_id, path="/items/0")["content"] == "1"
    assert manager.get_value(ref_id)["items"] == [1, 2]
    with pytest.raises(ArtifactInputError):
        tools["search_artifacts"].fn(limit=0)


def test_schema_explains_mutually_exclusive_modes(
    artifact_tools: tuple[ArtifactManager, dict[str, ToolSpec], str],
) -> None:
    _, tools, _ = artifact_tools
    spec = tools["read_artifact"]
    assert spec.json_schema is not None
    properties = spec.json_schema["properties"]
    assert "without path or offsets" in (spec.description or "")
    assert "path=null" in properties["pattern"]["description"]
    assert "offset=0" in properties["pattern"]["description"]
    assert "offset_from='start'" in properties["pattern"]["description"]


@pytest.mark.parametrize(
    "invalid", ["pattern_path", "pattern_offset", "missing_path", "missing_ref", "released_ref"]
)
def test_pydantic_model_corrects_artifact_call_without_experiment_capability(
    artifact_tools: tuple[ArtifactManager, dict[str, ToolSpec], str], invalid: str
) -> None:
    manager, tools, ref_id = artifact_tools
    arguments: dict[str, Any] = {"ref_id": ref_id}
    if invalid == "pattern_path":
        arguments.update(pattern="row", path="/stdout")
    elif invalid == "pattern_offset":
        arguments.update(pattern="row", offset=1)
    elif invalid == "missing_path":
        arguments["path"] = "/missing"
    elif invalid == "missing_ref":
        arguments["ref_id"] = "unknown"
    else:
        old = manager.artifactize("old", policy=ArtifactPolicy(threshold_bytes=1))
        assert isinstance(old, ArtifactHandle)
        manager.release(old.ref.ref_id)
        arguments["ref_id"] = old.ref.ref_id
    calls = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(
                parts=[ToolCallPart("read_artifact", arguments, tool_call_id="bad")]
            )
        if calls == 2:
            retries = [
                part for msg in messages for part in msg.parts if isinstance(part, RetryPromptPart)
            ]
            assert len(retries) == 1
            assert retries[0].tool_call_id == "bad"
            assert "Artifact" in str(retries[0].content)
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "read_artifact",
                        {
                            "ref_id": ref_id,
                            "path": "/items/0",
                        },
                        tool_call_id="fixed",
                    )
                ]
            )
        assert calls == 3
        results = [
            part for msg in messages for part in msg.parts if isinstance(part, ToolReturnPart)
        ]
        assert results[-1].tool_call_id == "fixed"
        content = results[-1].content
        assert isinstance(content, dict)
        assert cast(dict[str, Any], content)["content"] == "1"
        return ModelResponse(parts=[TextPart("done")])

    adapter = PydanticAdapter(FunctionModel(respond))
    adapter.register_tool(tools["read_artifact"])
    assert asyncio.run(adapter.invoke(prompt="Inspect the stored result.")) == "done"
    assert calls == 3


@pytest.mark.parametrize("async_tool", [False, True])
def test_langchain_native_tool_returns_error_then_accepts_correction(
    artifact_tools: tuple[ArtifactManager, dict[str, ToolSpec], str], async_tool: bool
) -> None:
    pytest.importorskip("langchain_core.tools")
    _, tools, ref_id = artifact_tools
    spec = tools["read_artifact"]
    if async_tool:
        original = spec.fn

        async def read(**kwargs: Any) -> Any:
            return original(**kwargs)

        spec = replace(spec, fn=read)
    native = _langchain_tool_from_spec(spec)

    async def run() -> None:
        error = await native.ainvoke(
            {
                "name": spec.name,
                "type": "tool_call",
                "id": "bad",
                "args": {
                    "ref_id": ref_id,
                    "pattern": "row",
                    "path": "/stdout",
                },
            }
        )
        assert error.status == "error"
        assert error.tool_call_id == "bad"
        assert "JSON Pointer" in error.content
        success = await native.ainvoke(
            {
                "name": spec.name,
                "type": "tool_call",
                "id": "fixed",
                "args": {
                    "ref_id": ref_id,
                    "path": "/items/0",
                },
            }
        )
        assert success.status == "success"
        assert success.tool_call_id == "fixed"

    asyncio.run(run())


def test_claude_tool_returns_error_then_accepts_correction(
    artifact_tools: tuple[ArtifactManager, dict[str, ToolSpec], str],
) -> None:
    _, tools, ref_id = artifact_tools
    sdk = SimpleNamespace(tool=lambda **kwargs: lambda fn: fn)
    native = _claude_sdk_tool(sdk, tools["read_artifact"])
    error = asyncio.run(native({"ref_id": ref_id, "pattern": "row", "path": "/stdout"}))
    assert error["is_error"] is True
    assert "JSON Pointer" in error["content"][0]["text"]
    success = asyncio.run(native({"ref_id": ref_id, "path": "/items/0"}))
    assert not success.get("is_error")


@pytest.mark.parametrize(
    "failure",
    [
        ValueError("internal bug"),
        OSError("disk failed"),
        ArtifactSerializationError("corrupt payload"),
        ArtifactAccessError("unsafe store"),
        ArtifactQuotaExceededError("quota"),
        RuntimeError("closed"),
    ],
)
def test_runtime_failures_are_not_reclassified(failure: Exception) -> None:
    assert artifact_tool_error_message(failure) is None


@pytest.mark.parametrize("metadata", [{"kind": "artifact_management"}, {}])
@pytest.mark.parametrize("framework", ["pydantic", "langchain", "claude"])
@pytest.mark.parametrize(
    "failure", [ValueError("internal bug"), ArtifactSerializationError("corrupt payload")]
)
def test_unexpected_tool_failures_still_propagate(
    framework: str, failure: Exception, metadata
) -> None:
    def fail(**kwargs: Any) -> Any:
        raise failure

    spec = ToolSpec(
        fn=fail,
        name="read_artifact",
        metadata=metadata,
        risk="read_only",
        json_schema={"type": "object", "properties": {}},
    )
    with pytest.raises(type(failure), match=str(failure)):
        if framework == "pydantic":

            def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
                return ModelResponse(parts=[ToolCallPart(spec.name, {})])

            adapter = PydanticAdapter(FunctionModel(respond))
            adapter.register_tool(spec)
            asyncio.run(adapter.invoke(prompt="inspect"))
        elif framework == "langchain":
            pytest.importorskip("langchain_core.tools")
            _langchain_tool_from_spec(spec).invoke({})
        else:
            native = _claude_sdk_tool(SimpleNamespace(tool=lambda **kwargs: lambda fn: fn), spec)
            asyncio.run(native({}))
