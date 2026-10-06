from __future__ import annotations

import asyncio
from functools import partial, wraps
from typing import Annotated, Any

import pytest
from pydantic import Field
from typing_extensions import TypedDict

from kedi.agent_adapter.adapters.pydantic.tools import pydantic_tool_from_spec
from kedi.agent_adapter.codemode.core import CodeModeLimits
from kedi.agent_adapter.codemode.tool_specs import CodeModeToolSpecBridge
from kedi.agent_adapter.tools.conversion import python_callable_tool_spec
from kedi.agent_adapter.tools.specs import ToolSpec
from kedi.artifacts.stream import ArtifactStream


class ReviewRecord(TypedDict):
    blocker_ids: list[str]
    evidence_code: str


def snapshot(stream: str = "quality") -> list[ReviewRecord]:
    return [{"blocker_ids": [], "evidence_code": stream}]


@wraps(snapshot)
def wrapped_snapshot(*args, **kwargs):
    return snapshot(*args, **kwargs)


class ReviewReader:
    def read(self) -> list[ReviewRecord]:
        return snapshot()

    async def read_async(self) -> list[ReviewRecord]:
        return snapshot()

    def __call__(self) -> list[ReviewRecord]:
        return snapshot()


@pytest.mark.parametrize(
    "fn",
    [
        snapshot,
        wrapped_snapshot,
        partial(snapshot, "security"),
        ReviewReader().read,
        ReviewReader().read_async,
        ReviewReader(),
    ],
)
def test_python_tool_return_schema_survives_adapter_and_codemode_projection(fn):
    spec = python_callable_tool_spec("snapshot", fn)
    schema = spec.return_schema
    assert schema["type"] == "array"
    assert set(schema["$defs"]["ReviewRecord"]["required"]) == {"blocker_ids", "evidence_code"}
    assert pydantic_tool_from_spec(spec).tool_def.return_schema == schema
    bridge = CodeModeToolSpecBridge(
        [spec],
        limits=CodeModeLimits(),
        policies=[],
        adapter_shortname="langchain",
        preload_tools=["snapshot"],
    )
    try:
        assert bridge.preloaded_tools[0]["return_schema"] == schema
        assert (
            bridge.get_tool_schema(tool_names=["snapshot"])["tools"][0]["return_schema"] == schema
        )
    finally:
        asyncio.run(bridge.close())


def test_explicit_pydantic_tool_spec_return_schema_is_not_discarded():
    schema = {"type": "string", "enum": ["ready"]}
    spec = ToolSpec(
        lambda: "ready",
        "state",
        json_schema={"type": "object", "properties": {}},
        return_schema=schema,
    )
    assert pydantic_tool_from_spec(spec).tool_def.return_schema == schema


def test_python_tool_preserves_return_metadata_and_logical_stream_type():
    def rating() -> Annotated[int, Field(ge=0, le=9)]:
        return 3

    def stream() -> ArtifactStream[str]:
        return ArtifactStream.text(["ready"])

    assert python_callable_tool_spec("rating", rating).return_schema == {
        "type": "integer",
        "minimum": 0,
        "maximum": 9,
    }
    assert python_callable_tool_spec("stream", stream).return_schema == {"type": "string"}


def test_unknown_or_unsupported_return_schema_does_not_block_tool_registration():
    class Opaque:
        pass

    def missing():
        return "ready"

    def unresolved() -> UnknownRecord:  # noqa: F821
        return "ready"

    def opaque():
        return Opaque()

    opaque.__annotations__["return"] = Opaque
    for fn in (missing, unresolved, opaque):
        spec = python_callable_tool_spec(fn.__name__, fn)
        assert spec.return_schema is None
        assert spec.fn() is not None

    def unspecified() -> Any:
        return "ready"

    def nothing() -> None:
        return None

    assert python_callable_tool_spec("unspecified", unspecified).return_schema == {}
    assert python_callable_tool_spec("nothing", nothing).return_schema == {"type": "null"}


def test_unresolved_argument_does_not_discard_resolved_return_type():
    def count(value):
        return len(value)

    count.__annotations__ = {"value": "MissingInput", "return": int}
    spec = python_callable_tool_spec("count", count)
    assert spec.return_schema == {"type": "integer"}
    assert spec.fn("abc") == 3
