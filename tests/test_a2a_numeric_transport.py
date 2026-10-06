from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace
from typing import Literal
from unittest.mock import AsyncMock

import pytest
from a2a.types import Artifact, Part, Task, TaskState, TaskStatus
from google.protobuf.json_format import MessageToDict, ParseDict
from pydantic import BaseModel, StrictBool, StrictInt

from kedi.a2a.protocol import (
    KEDI_STRUCTURED_OUTPUT_ARTIFACT_NAME,
    KEDI_STRUCTURED_OUTPUT_EXTENSION_URI,
    KEDI_STRUCTURED_OUTPUT_METADATA_KEY,
    A2ARemoteTaskError,
    structured_output_request,
    structured_output_schema,
)
from kedi.a2a.result import _ResponseAccumulator
from kedi.a2a.runtime import prepare_a2a_program
from kedi.a2a.server import _KediAgentExecutor, create_a2a_app
from kedi.agent_adapter.adapters.a2a import A2AAdapter
from kedi.agent_profile import A2AConnection
from tests.test_a2a import _ServedAdapter, _unused_loopback_port, _wait_for_server


class Numbers(BaseModel):
    count: StrictInt
    identifier: Literal[9007199254740993]
    weights: list[float]
    optional: dict[str, StrictInt | None]
    enabled: StrictBool


NUMBERS = {
    "count": 78,
    "identifier": 9007199254740993,
    "weights": [78.0, 0.125],
    "optional": {"large": 10**80 + 1, "missing": None},
    "enabled": True,
}


def result_from_parts(parts, *, limit=10000):
    task = Task(
        id="task",
        context_id="context",
        status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
        artifacts=[
            Artifact(artifact_id="result", name=KEDI_STRUCTURED_OUTPUT_ARTIFACT_NAME, parts=parts)
        ],
    )
    # Exercise the same protobuf/JSON boundary as JSONRPC and persistent task storage.
    wire = json.loads(json.dumps(MessageToDict(task)))
    accumulator = _ResponseAccumulator(max_bytes=limit)
    accumulator.add_task(ParseDict(wire, Task()))
    return accumulator.result(structured=True).data


@pytest.mark.parametrize("value", [78, 2**53 + 1, -(2**53 + 1), 10**80 + 1])
def test_lossless_schema_survives_request_metadata(value):
    from google.protobuf import struct_pb2

    schema = {"type": "object", "properties": {"count": {"const": value}}}
    metadata = structured_output_request(schema)
    legacy = metadata[KEDI_STRUCTURED_OUTPUT_METADATA_KEY].get("schema")
    assert (legacy is not None) == (value == 78)
    struct = ParseDict(metadata, getattr(struct_pb2, "Struct")())
    received = structured_output_schema(MessageToDict(struct))
    assert received == schema
    assert type(received["properties"]["count"]["const"]) is int


@pytest.mark.parametrize("encoded", [None, "[", "null", "NaN", '{"maximum":1e400}', " " * 33000])
def test_invalid_canonical_schema_never_falls_back_to_legacy(encoded):
    metadata = structured_output_request({"type": "object"})
    metadata[KEDI_STRUCTURED_OUTPUT_METADATA_KEY]["schemaJson"] = encoded
    from kedi.a2a.server import _requested_output_schema

    with pytest.raises(ValueError):
        _requested_output_schema(
            SimpleNamespace(
                metadata=metadata, requested_extensions=[KEDI_STRUCTURED_OUTPUT_EXTENSION_URI]
            )
        )


@pytest.mark.parametrize("legacy_first", [False, True])
def test_raw_result_preserves_nested_numbers_and_overrides_legacy(legacy_first):
    canonical = Part(raw=json.dumps(NUMBERS).encode(), media_type="application/json")
    legacy = ParseDict({"data": {"count": 1}}, Part())
    result = result_from_parts([legacy, canonical] if legacy_first else [canonical, legacy])
    assert Numbers.model_validate(result, strict=True).model_dump() == NUMBERS
    assert type(result["count"]) is int
    assert type(result["weights"][0]) is float
    assert type(result["optional"]["large"]) is int


@pytest.mark.parametrize("payload", [b"{", b"\xff", b'{"count":NaN}', b'{"count":1e400}'])
def test_malformed_raw_result_never_falls_back_to_legacy(payload):
    with pytest.raises(A2ARemoteTaskError, match="invalid structured JSON"):
        result_from_parts(
            [
                ParseDict({"data": {"count": 78}}, Part()),
                Part(raw=payload, media_type="application/json"),
            ]
        )


def test_wrong_media_duplicate_and_oversized_raw_results_are_rejected():
    with pytest.raises(A2ARemoteTaskError, match="without its structured"):
        result_from_parts([Part(raw=b"{}", media_type="text/plain")])
    part = Part(raw=b"{}", media_type="application/json")
    with pytest.raises(A2ARemoteTaskError, match="invalid structured JSON"):
        result_from_parts([part, part])
    with pytest.raises(A2ARemoteTaskError, match="exceeded"):
        result_from_parts([Part(raw=b" " * 300, media_type="application/json")], limit=200)


def test_legacy_schema_and_results_remain_readable():
    metadata = {KEDI_STRUCTURED_OUTPUT_METADATA_KEY: {"version": 1, "schema": {"type": "object"}}}
    assert structured_output_schema(metadata) == {"type": "object"}
    assert result_from_parts([ParseDict({"data": {"answer": "legacy"}}, Part())]) == {
        "answer": "legacy"
    }


@pytest.mark.parametrize(
    "encoding,called", [("json-utf8", True), ("protobuf-value", True), ("invalid", False)]
)
@pytest.mark.parametrize("result", [NUMBERS, {"count": 78}])
def test_executor_negotiates_encoding_without_silently_rounding(encoding, called, result):
    async def scenario():
        program = SimpleNamespace(invoke=AsyncMock(return_value=result))
        metadata = structured_output_request(Numbers.model_json_schema())
        metadata[KEDI_STRUCTURED_OUTPUT_METADATA_KEY]["resultEncoding"] = encoding
        context = SimpleNamespace(
            task_id="task",
            context_id="context",
            current_task=Task(id="task"),
            requested_extensions=[KEDI_STRUCTURED_OUTPUT_EXTENSION_URI],
            metadata=metadata,
            call_context=SimpleNamespace(user=SimpleNamespace(user_name="user")),
            get_user_input=lambda: "numbers",
        )
        events = SimpleNamespace(enqueue_event=AsyncMock())
        await _KediAgentExecutor(program).execute(context, events)
        assert program.invoke.called is called
        states = [
            call.args[0].status.state
            for call in events.enqueue_event.call_args_list
            if hasattr(call.args[0], "status")
        ]
        completed = encoding == "json-utf8" or (
            encoding == "protobuf-value" and result == {"count": 78}
        )
        assert states[-1] == (
            TaskState.TASK_STATE_COMPLETED if completed else TaskState.TASK_STATE_FAILED
        )
        if completed:
            parts = [
                part
                for call in events.enqueue_event.call_args_list
                if hasattr(call.args[0], "artifact")
                for part in call.args[0].artifact.parts
            ]
            assert result_from_parts(parts) == result

    asyncio.run(scenario())


def test_official_sdk_validates_strict_numbers_over_http(tmp_path):
    import uvicorn

    class NumericAdapter(_ServedAdapter):
        async def produce_subagent(self, *, output_schema, **kwargs):
            assert output_schema["properties"]["identifier"]["const"] == NUMBERS["identifier"]
            return NUMBERS

    source = tmp_path / "remote.kedi"
    source.write_text(
        "> profile: remote:\n  > instructions: Return numbers.\n> export:\n  remote\n"
    )
    port = _unused_loopback_port()
    endpoint = f"http://127.0.0.1:{port}"
    served = prepare_a2a_program(
        source, entry="remote", model="test:model", adapter=NumericAdapter()
    )
    app = create_a2a_app(
        served, public_url=endpoint, auth=None, allow_unauthenticated_loopback=True
    )
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="critical"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        _wait_for_server(server)
        adapter = A2AAdapter(connection=A2AConnection(endpoint=endpoint, task_timeout=10))
        result = adapter.produce_sync("Return numbers", output_type=Numbers)
        assert result.model_dump() == NUMBERS
        assert type(result.count) is int
    finally:
        server.should_exit = True
        thread.join(timeout=10)
    assert not thread.is_alive()
