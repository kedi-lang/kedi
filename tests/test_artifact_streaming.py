from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Annotated, Any, get_type_hints

import pytest

from kedi import ArtifactStream
from kedi.agent_adapter.tool_override import ToolSpec, with_artifact_result
from kedi.agent_adapter.utils import kedi_procedure_tool_spec, python_callable_tool_spec
from kedi.artifacts import (
    ArtifactHandle,
    ArtifactManager,
    ArtifactPolicy,
    ArtifactQuotaExceededError,
    ArtifactStreamConsumedError,
    ArtifactStreamError,
    ArtifactStreamKind,
    ToolOutputTooLargeError,
)
from kedi.artifacts import streaming as streaming_module
from kedi.artifacts.stream import (
    artifact_stream_matches_annotation,
    callable_with_logical_stream_return,
)
from kedi.artifacts.streaming import (
    ArtifactStreamEncoder,
    StreamedArtifact,
    materialize_stream_payload,
)
from kedi.errors import KediTypeExecutionError
from kedi.lang import compile_program, parse_program
from tests.telemetry_helpers import installed_recording_backend


def test_text_stream_is_single_consumption_and_preserves_metadata() -> None:
    stream = ArtifactStream.text(
        ["hello", " ", "world"],
        metadata={"source": "test"},
        content_length=11,
    )

    assert stream.kind is ArtifactStreamKind.TEXT
    assert stream.media_type == "text/plain; charset=utf-8"
    assert stream.logical_type == "str"
    assert stream.content_length == 11
    assert stream.metadata == {"source": "test"}
    assert not stream.consumed
    assert list(stream.iter_chunks()) == ["hello", " ", "world"]
    assert stream.consumed
    with pytest.raises(ArtifactStreamConsumedError, match="already been consumed"):
        list(stream.iter_chunks())


def test_stream_normalizes_bytes_and_json_items() -> None:
    binary = ArtifactStream.bytes([bytearray(b"a"), memoryview(b"b")])
    structured = ArtifactStream.json_items([{"ok": True}, [1, 2]])

    assert list(binary.iter_chunks()) == [b"a", b"b"]
    assert list(structured.iter_chunks()) == [{"ok": True}, [1, 2]]


@pytest.mark.parametrize(
    ("stream", "annotation", "expected"),
    [
        (ArtifactStream.text([]), Annotated[str, "content"], True),
        (ArtifactStream.text([]), str | None, True),
        (ArtifactStream.text([]), int | None, False),
        (ArtifactStream.bytes([]), bytes, True),
        (ArtifactStream.bytes([]), str, False),
        (ArtifactStream.json_items([]), list[dict[str, Any]], True),
        (ArtifactStream.json_items([]), tuple[str, ...], False),
        (ArtifactStream.text([]), Any, True),
    ],
)
def test_stream_transport_matches_its_logical_annotation(
    stream: ArtifactStream[Any], annotation: Any, expected: bool
) -> None:
    assert artifact_stream_matches_annotation(stream, annotation) is expected


@pytest.mark.parametrize(
    ("factory", "chunks", "message"),
    [
        (ArtifactStream.text, [b"wrong"], "must be str"),
        (ArtifactStream.bytes, ["wrong"], "must be bytes-like"),
        (ArtifactStream.json_items, [{"bad": object()}], "must be JSON-compatible"),
    ],
)
def test_stream_rejects_invalid_chunks(factory: object, chunks: list[object], message: str) -> None:
    stream = factory(chunks)  # type: ignore[operator]
    with pytest.raises(ArtifactStreamError, match=message):
        list(stream.iter_chunks())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"media_type": " "}, "media_type cannot be empty"),
        ({"logical_type": ""}, "logical_type cannot be empty"),
        ({"content_length": -1}, "content_length cannot be negative"),
        ({"metadata": {"ref_id": "mine"}}, "cannot override: ref_id"),
        ({"metadata": {1: "bad"}}, "keys must be strings"),
        ({"metadata": {"bad": object()}}, "must be JSON-compatible"),
    ],
)
def test_stream_rejects_invalid_configuration(kwargs: dict[str, object], message: str) -> None:
    options: dict[str, object] = {
        "media_type": "text/plain",
        "logical_type": "str",
        **kwargs,
    }
    with pytest.raises(ArtifactStreamError, match=message):
        ArtifactStream(["value"], kind=ArtifactStreamKind.TEXT, **options)


@pytest.mark.parametrize("chunks", ["whole text", b"whole bytes", object()])
def test_stream_requires_a_chunk_iterable(chunks: object) -> None:
    with pytest.raises(ArtifactStreamError, match="chunks must be"):
        ArtifactStream.text(chunks)  # type: ignore[arg-type]


def test_stream_closes_generator_after_failure() -> None:
    closed = False

    def chunks() -> Iterator[str]:
        nonlocal closed
        try:
            yield "ok"
            raise RuntimeError("chunk failed")
        finally:
            closed = True

    stream = ArtifactStream.text(chunks())
    with pytest.raises(RuntimeError, match="chunk failed"):
        list(stream.iter_chunks())
    assert closed


def test_async_stream_and_sync_source_async_iteration_close_cleanly() -> None:
    async def run() -> None:
        async_closed = False

        async def chunks() -> AsyncIterator[str]:
            nonlocal async_closed
            try:
                yield "a"
                yield "b"
            finally:
                async_closed = True

        stream = ArtifactStream.text(chunks())
        assert [chunk async for chunk in stream.aiter_chunks()] == ["a", "b"]
        assert async_closed

        sync_stream = ArtifactStream.text(["c"])
        assert [chunk async for chunk in sync_stream.aiter_chunks()] == ["c"]

    asyncio.run(run())


def test_async_stream_rejects_sync_iteration_and_can_be_closed() -> None:
    async def run() -> None:
        class ClosableChunks:
            def __init__(self) -> None:
                self.closed = False

            def __aiter__(self) -> ClosableChunks:
                return self

            async def __anext__(self) -> str:
                raise StopAsyncIteration

            async def aclose(self) -> None:
                self.closed = True

        chunks = ClosableChunks()
        stream = ArtifactStream.text(chunks)
        with pytest.raises(ArtifactStreamError, match="requires async iteration"):
            list(stream.iter_chunks())
        await stream.aclose()
        assert chunks.closed
        with pytest.raises(ArtifactStreamConsumedError, match="closed"):
            [chunk async for chunk in stream.aiter_chunks()]

    asyncio.run(run())


def test_close_is_idempotent_and_prevents_consumption() -> None:
    stream = ArtifactStream.text(["value"])
    stream.close()
    stream.close()
    with pytest.raises(ArtifactStreamConsumedError, match="closed"):
        list(stream.iter_chunks())


def test_stream_return_annotation_is_hidden_from_tool_introspection() -> None:
    def tool() -> ArtifactStream[str]:
        return ArtifactStream.text(["value"])

    wrapped, logical = callable_with_logical_stream_return(tool)
    spec = python_callable_tool_spec("tool", tool)

    assert logical is str
    assert inspect.signature(wrapped).return_annotation is str
    assert get_type_hints(wrapped)["return"] is str
    assert isinstance(wrapped(), ArtifactStream)
    assert spec.return_schema == {"type": "string"}
    assert inspect.signature(spec.fn).return_annotation is str


def test_python_callable_tool_survives_unresolved_argument_annotations() -> None:
    def tool(value: str) -> str:
        return value

    tool.__annotations__["value"] = "MissingType"
    spec = python_callable_tool_spec("tool", tool)

    assert spec.argument_validator is not None
    assert spec.argument_validator({"value": "ok"}) == {"value": "ok"}


def test_async_stream_return_annotation_keeps_async_callable() -> None:
    async def run() -> None:
        async def tool() -> ArtifactStream[bytes]:
            return ArtifactStream.bytes([b"value"])

        wrapped, logical = callable_with_logical_stream_return(tool)

        assert logical is bytes
        assert inspect.iscoroutinefunction(wrapped)
        assert isinstance(await wrapped(), ArtifactStream)

    asyncio.run(run())


def test_non_stream_callable_is_not_wrapped() -> None:
    def tool() -> str:
        return "value"

    wrapped, logical = callable_with_logical_stream_return(tool)

    assert wrapped is tool
    assert logical is None


@pytest.mark.parametrize(
    ("stream", "expected"),
    [
        (ArtifactStream.text(["Mer", "haba", " ", "dunya"]), "Merhaba dunya"),
        (ArtifactStream.bytes([b"abc", b"def"]), b"abcdef"),
        (ArtifactStream.json_items([{"a": 1}, [2, 3], "x"]), [{"a": 1}, [2, 3], "x"]),
    ],
)
def test_stream_encoder_produces_canonical_materializable_payload(
    stream: ArtifactStream[object], expected: object
) -> None:
    encoder = ArtifactStreamEncoder(stream, preview_chars=8)
    payload = bytearray(encoder.start())
    for chunk in stream.iter_chunks():
        payload.extend(encoder.encode(chunk))
    tail, streamed = encoder.finish()
    payload.extend(tail)

    assert materialize_stream_payload(streamed, bytes(payload)) == expected
    assert streamed.size_bytes == len(payload)
    assert len(streamed.preview) <= 8
    assert len(streamed.digest) == 64


def test_stream_validates_declared_content_length() -> None:
    manager = ArtifactManager(session_id="stream-length")
    try:
        with pytest.raises(ArtifactStreamError, match="content_length does not match"):
            manager.artifactize_stream(
                ArtifactStream.text(["hello"], content_length=4),
                policy=_policy(),
            )
        assert manager.stats()["active_artifacts"] == 0
    finally:
        manager.close()


def _policy(**values: object) -> ArtifactPolicy:
    return ArtifactPolicy.patch(
        {
            "enabled": True,
            "threshold": 5,
            "cleanup_interval": "1h",
            **values,
        }
    ).resolved()


def test_manager_streams_above_threshold_and_keeps_small_values_native() -> None:
    manager = ArtifactManager(session_id="stream-manager")
    try:
        small = manager.artifactize_stream(ArtifactStream.text(["12", "34"]), policy=_policy())
        large = manager.artifactize_stream(
            ArtifactStream.text(["12", "345", "678"]),
            policy=_policy(),
            source="tool:lookup",
        )

        assert small == "1234"
        assert isinstance(large, ArtifactHandle)
        assert large.ref.ref_id == "tool_call_result_1"
        assert large.get_value() == "12345678"
        assert manager.stats()["active_artifacts"] == 1
    finally:
        manager.close()


def test_streamed_artifact_records_one_bounded_store_span() -> None:
    manager = ArtifactManager(session_id="stream-telemetry")
    try:
        with installed_recording_backend() as backend:
            artifact = manager.artifactize_stream(
                ArtifactStream.text(["12", "345", "678"]),
                policy=_policy(),
                source="tool:lookup",
            )

        assert isinstance(artifact, ArtifactHandle)
        store_spans = [
            span for span in backend.finished_spans if span.operation == "store_artifact"
        ]
        assert len(store_spans) == 1
        [store_span] = store_spans
        assert store_span.name == "store artifact"
        assert store_span.attributes["kedi.artifact.streamed"] is True
        assert store_span.attributes["kedi.artifact.size_bytes"] == 8
        assert store_span.attributes["kedi.artifact.chunk_count"] == 3
    finally:
        manager.close()


def test_failed_streamed_artifact_closes_store_span_and_stream() -> None:
    closed = False

    def chunks() -> Iterator[str]:
        nonlocal closed
        try:
            yield "123456"
            raise RuntimeError("stream failed")
        finally:
            closed = True

    manager = ArtifactManager(session_id="stream-telemetry-failure")
    try:
        with installed_recording_backend() as backend:
            with pytest.raises(RuntimeError, match="stream failed"):
                manager.artifactize_stream(
                    ArtifactStream.text(chunks()),
                    policy=_policy(),
                    source="tool:lookup",
                )

        assert closed
        assert manager.stats()["active_artifacts"] == 0
        store_spans = [
            span for span in backend.finished_spans if span.operation == "store_artifact"
        ]
        assert len(store_spans) == 1
        [store_span] = store_spans
        assert store_span.attributes["error.type"] == "RuntimeError"
        assert [type(exc).__name__ for exc in store_span.exceptions] == ["RuntimeError"]
    finally:
        manager.close()


def test_threshold_crossing_writes_large_chunk_without_copying_it_into_prebuffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ArtifactManager(session_id="stream-prebuffer")
    write_sizes: list[int] = []
    begin_write = manager._memory_store.begin_write

    def traced_begin_write(session_id: str, ref_id: str) -> object:
        writer = begin_write(session_id, ref_id)
        write = writer.write

        def traced_write(chunk: bytes) -> None:
            write_sizes.append(len(chunk))
            write(chunk)

        monkeypatch.setattr(writer, "write", traced_write)
        return writer

    monkeypatch.setattr(manager._memory_store, "begin_write", traced_begin_write)
    try:
        artifact = manager.artifactize_stream(
            ArtifactStream.text(["ab", "x" * 100]),
            policy=_policy(),
        )

        assert isinstance(artifact, ArtifactHandle)
        assert write_sizes == [2, 100]
        assert artifact.get_value() == "ab" + "x" * 100
    finally:
        manager.close()


def test_manager_stream_quota_failure_aborts_without_visible_artifact(tmp_path: Path) -> None:
    manager = ArtifactManager(session_id="stream-quota")
    try:
        with pytest.raises(ArtifactQuotaExceededError, match="session quota"):
            manager.artifactize_stream(
                ArtifactStream.text(["12345", "67890"]),
                policy=_policy(store="file", path=tmp_path, session_quota="8b"),
                source="tool:lookup",
            )

        assert manager.stats()["active_artifacts"] == 0
        assert manager._reserved_artifacts == 0
        assert manager._reserved_bytes == 0
        assert not tuple(tmp_path.rglob("*.tmp"))
        assert not tuple(tmp_path.rglob("*.payload"))
    finally:
        manager.close()


@pytest.mark.parametrize("store_kind", ["memory", "file"])
def test_stream_registration_failure_discards_committed_payload(
    store_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ArtifactManager(session_id=f"stream-registration-{store_kind}")

    def fail_history(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("history registration failed")

    monkeypatch.setattr(manager.history, "append", fail_history)
    values: dict[str, object] = {"store": store_kind}
    if store_kind == "file":
        values["path"] = tmp_path
    try:
        with pytest.raises(RuntimeError, match="history registration failed"):
            manager.artifactize_stream(
                ArtifactStream.text(["large", " value"]),
                policy=_policy(**values),
                source="tool:lookup",
            )

        assert manager.stats()["active_artifacts"] == 0
        assert manager._ref_stores == {}
        assert manager._ref_policies == {}
        assert manager._reserved_artifacts == 0
        assert manager._reserved_bytes == 0
        assert not tuple(tmp_path.rglob("*.payload"))
        assert not tuple(tmp_path.rglob("*.json"))
    finally:
        manager.close()


def test_file_stream_is_not_retained_as_a_native_store_cache(tmp_path: Path) -> None:
    manager = ArtifactManager(session_id="stream-file")
    try:
        artifact = manager.artifactize_stream(
            ArtifactStream.text(["large", " value"]),
            policy=_policy(store="file", path=tmp_path),
            source="tool:file",
        )
        assert isinstance(artifact, ArtifactHandle)
        store = manager._ref_stores[artifact.ref.ref_id]
        assert not hasattr(store, "_native_cache")
        assert artifact.get_value() == "large value"
    finally:
        manager.close()


def test_file_stream_bounded_reads_do_not_materialize_native_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = ArtifactManager(session_id="stream-range")
    try:
        artifact = manager.artifactize_stream(
            ArtifactStream.text(["abc", "def", "ghi", "jkl"]),
            policy=_policy(store="file", path=tmp_path),
            source="tool:file",
        )
        assert isinstance(artifact, ArtifactHandle)
        store = manager._ref_stores[artifact.ref.ref_id]

        def reject_materialization(*args: object, **kwargs: object) -> object:
            raise AssertionError("bounded reads must not call get_value")

        monkeypatch.setattr(store, "get_value", reject_materialization)
        start = manager.read(artifact.ref.ref_id, offset=3, max_chars=4)
        end = manager.read(
            artifact.ref.ref_id,
            offset=2,
            offset_from="end",
            max_chars=4,
        )

        assert start.content == "defg"
        assert start.next_offset == 7
        assert end.content == "ghij"
        assert end.offset_from == "end"
        assert end.next_offset == 6
    finally:
        manager.close()


def test_file_stream_bounded_read_preserves_unicode_boundaries(tmp_path: Path) -> None:
    manager = ArtifactManager(session_id="stream-unicode")
    try:
        artifact = manager.artifactize_stream(
            ArtifactStream.text(["Ağ", "β", "🙂Z"]),
            policy=_policy(store="file", path=tmp_path),
        )
        assert isinstance(artifact, ArtifactHandle)
        assert manager.read(artifact.ref.ref_id, offset=1, max_chars=3).content == "ğβ🙂"
    finally:
        manager.close()


def test_file_json_stream_pointer_reads_do_not_materialize_native_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = ArtifactManager(session_id="stream-json-pointer")
    try:
        artifact = manager.artifactize_stream(
            ArtifactStream.json_items(
                [
                    {"city/name": "İstanbul"},
                    {"nested": {"~value": ["zero", "Ağβ🙂Z"]}},
                ]
            ),
            policy=_policy(store="file", path=tmp_path),
        )
        assert isinstance(artifact, ArtifactHandle)
        store = manager._ref_stores[artifact.ref.ref_id]

        def reject_materialization(*args: object, **kwargs: object) -> object:
            raise AssertionError("JSON pointer reads must not call get_value")

        monkeypatch.setattr(store, "get_value", reject_materialization)
        first = manager.read(artifact.ref.ref_id, path="/0/city~1name")
        nested = manager.read(
            artifact.ref.ref_id,
            path="/1/nested/~0value/1",
            offset=1,
            max_chars=3,
        )

        assert first.content == '"İstanbul"'
        assert nested.content == "Ağβ"
        assert nested.next_offset == 4
    finally:
        manager.close()


def test_async_manager_consumes_async_stream() -> None:
    async def run() -> None:
        async def chunks() -> AsyncIterator[str]:
            yield "abc"
            yield "def"

        manager = ArtifactManager(session_id="async-stream")
        try:
            artifact = await manager.artifactize_stream_async(
                ArtifactStream.text(chunks()),
                policy=_policy(),
                source="template:result",
            )
            assert isinstance(artifact, ArtifactHandle)
            assert artifact.ref.ref_id == "template_output_1"
            assert artifact.get_value() == "abcdef"
        finally:
            manager.close()

    asyncio.run(run())


def test_tool_interceptor_hides_stream_transport_from_model() -> None:
    def tool() -> ArtifactStream[str]:
        return ArtifactStream.text(["abc", "def"])

    manager = ArtifactManager(session_id="stream-tool")
    try:
        wrapped = with_artifact_result(
            python_callable_tool_spec("stream_tool", tool),
            manager=manager,
            policy=_policy(),
        )
        result = wrapped.fn()

        assert result["ref_id"] == "tool_call_result_1"
        assert manager.get_value(result["ref_id"]) == "abcdef"
    finally:
        manager.close()


def test_tool_interceptor_consumes_async_stream() -> None:
    async def run() -> None:
        async def chunks() -> AsyncIterator[str]:
            yield "abc"
            yield "def"

        async def tool() -> ArtifactStream[str]:
            return ArtifactStream.text(chunks())

        manager = ArtifactManager(session_id="async-stream-tool")
        try:
            wrapped = with_artifact_result(
                ToolSpec(fn=tool, name="async_stream_tool"),
                manager=manager,
                policy=_policy(),
            )
            result = await wrapped.fn()
            assert result["ref_id"] == "tool_call_result_1"
            assert manager.get_value(result["ref_id"]) == "abcdef"
        finally:
            manager.close()

    asyncio.run(run())


def test_disabled_artifacts_reject_oversized_materialized_tool_output() -> None:
    marker = "payload-must-not-appear"

    def tool() -> str:
        return marker + "x" * 100_000

    manager = ArtifactManager(session_id="disabled-materialized")
    try:
        wrapped = with_artifact_result(
            ToolSpec(fn=tool, name="large_tool"),
            manager=manager,
            policy=ArtifactPolicy.patch({"enabled": False}).resolved(),
        )
        with pytest.raises(ToolOutputTooLargeError) as raised:
            wrapped.fn()
        assert marker not in str(raised.value)
        events = manager.history.snapshot()
        assert [event.kind for event in events] == ["tool_call", "tool_result"]
        assert events[-1].data["error"] == "tool_output_too_large"
    finally:
        manager.close()


def test_disabled_artifacts_stop_oversized_stream_incrementally() -> None:
    produced = 0

    def chunks() -> Iterator[str]:
        nonlocal produced
        for _ in range(100):
            produced += 1
            yield "x" * 10_000

    manager = ArtifactManager(session_id="disabled-stream")
    try:
        wrapped = with_artifact_result(
            ToolSpec(fn=lambda: ArtifactStream.text(chunks()), name="stream_tool"),
            manager=manager,
            policy=ArtifactPolicy.patch({"enabled": False}).resolved(),
        )
        with pytest.raises(ToolOutputTooLargeError):
            wrapped.fn()
        assert produced == 11
        assert manager.stats()["active_artifacts"] == 0
        events = manager.history.snapshot()
        assert [event.kind for event in events] == ["tool_call", "tool_result"]
        assert events[-1].data["error"] == "tool_output_too_large"
    finally:
        manager.close()


def test_async_tool_interceptor_records_small_and_quota_results() -> None:
    async def run() -> None:
        async def small_chunks() -> AsyncIterator[str]:
            yield "ok"

        async def large_chunks() -> AsyncIterator[str]:
            yield "12345"
            yield "67890"

        small_manager = ArtifactManager(session_id="async-small-result")
        quota_manager = ArtifactManager(session_id="async-quota-result")
        try:
            small = with_artifact_result(
                ToolSpec(
                    fn=lambda: ArtifactStream.text(small_chunks()),
                    name="small_stream",
                ),
                manager=small_manager,
                policy=_policy(threshold=100),
            )
            assert await small.fn() == "ok"
            assert [event.kind for event in small_manager.history.snapshot()] == [
                "tool_call",
                "tool_result",
            ]

            quota = with_artifact_result(
                ToolSpec(
                    fn=lambda: ArtifactStream.text(large_chunks()),
                    name="quota_stream",
                ),
                manager=quota_manager,
                policy=_policy(session_quota="8b"),
            )
            result = await quota.fn()
            assert result["error"] == "artifact_quota_exceeded"
            events = quota_manager.history.snapshot()
            assert [event.kind for event in events] == ["tool_call", "tool_result"]
            assert events[-1].data["error"] == "artifact_quota_exceeded"
        finally:
            small_manager.close()
            quota_manager.close()

    asyncio.run(run())


def test_async_tool_interceptor_records_oversized_disabled_stream() -> None:
    async def run() -> None:
        async def chunks() -> AsyncIterator[str]:
            for _ in range(20):
                yield "x" * 10_000

        manager = ArtifactManager(session_id="async-disabled-stream")
        try:
            wrapped = with_artifact_result(
                ToolSpec(
                    fn=lambda: ArtifactStream.text(chunks()),
                    name="disabled_stream",
                ),
                manager=manager,
                policy=ArtifactPolicy.patch({"enabled": False}).resolved(),
            )
            with pytest.raises(ToolOutputTooLargeError):
                await wrapped.fn()
            events = manager.history.snapshot()
            assert [event.kind for event in events] == ["tool_call", "tool_result"]
            assert events[-1].data["error"] == "tool_output_too_large"
        finally:
            manager.close()

    asyncio.run(run())


def test_sync_tool_returning_awaitable_and_async_stream_is_compacted() -> None:
    async def run() -> None:
        async def value() -> str:
            return "abcdef"

        async def chunks() -> AsyncIterator[str]:
            yield "abc"
            yield "def"

        manager = ArtifactManager(session_id="sync-awaitable-tool")
        try:
            awaitable_tool = with_artifact_result(
                ToolSpec(fn=lambda: value(), name="awaitable_tool"),
                manager=manager,
                policy=_policy(),
            )
            stream_tool = with_artifact_result(
                ToolSpec(
                    fn=lambda: ArtifactStream.text(chunks()),
                    name="async_stream_tool",
                ),
                manager=manager,
                policy=_policy(),
            )
            first = await awaitable_tool.fn()
            second = await stream_tool.fn()
            assert manager.get_value(first["ref_id"]) == "abcdef"
            assert manager.get_value(second["ref_id"]) == "abcdef"
        finally:
            manager.close()

    asyncio.run(run())


def test_stream_close_and_aclose_cover_sync_and_nonawaitable_close_hooks() -> None:
    class SyncChunks:
        def __init__(self) -> None:
            self.closed = False
            self.done = False

        def __iter__(self) -> SyncChunks:
            return self

        def __next__(self) -> str:
            if self.done:
                raise StopIteration
            self.done = True
            return "value"

        def close(self) -> None:
            self.closed = True

    class AsyncChunks:
        def __init__(self) -> None:
            self.closed = False
            self.done = False

        def __aiter__(self) -> AsyncChunks:
            return self

        async def __anext__(self) -> str:
            if self.done:
                raise StopAsyncIteration
            self.done = True
            return "value"

        def aclose(self) -> None:
            self.closed = True

    async def run() -> None:
        sync_chunks = SyncChunks()
        sync_stream = ArtifactStream.text(sync_chunks)
        assert [item async for item in sync_stream.aiter_chunks()] == ["value"]
        assert sync_chunks.closed

        async_chunks = AsyncChunks()
        async_stream = ArtifactStream.text(async_chunks)
        assert [item async for item in async_stream.aiter_chunks()] == ["value"]
        assert async_chunks.closed
        await async_stream.aclose()

        direct_async_chunks = AsyncChunks()
        await ArtifactStream.text(direct_async_chunks).aclose()
        assert direct_async_chunks.closed

        unopened = SyncChunks()
        unopened_stream = ArtifactStream.text(unopened)
        await unopened_stream.aclose()
        assert unopened.closed

    asyncio.run(run())


def test_stream_annotation_unwraps_bare_transport_and_survives_unresolved_hints() -> None:
    def bare() -> ArtifactStream:
        return ArtifactStream.text(["value"])

    def unresolved() -> str:
        return "value"

    unresolved.__annotations__["return"] = "MissingType"

    wrapped, logical = callable_with_logical_stream_return(bare)
    unresolved_wrapped, unresolved_logical = callable_with_logical_stream_return(unresolved)

    assert str(logical) == "typing.Any"
    assert isinstance(wrapped(), ArtifactStream)
    assert unresolved_wrapped is unresolved
    assert unresolved_logical is None


def test_stream_encoder_rejects_reuse_caps_json_index_and_handles_tiny_preview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = ArtifactStream.text(["ab"])
    encoder = ArtifactStreamEncoder(stream, preview_chars=1)
    encoder.encode("ab")
    _, artifact = encoder.finish()
    assert artifact.preview == "a"
    with pytest.raises(RuntimeError, match="already finished"):
        encoder.encode("again")
    with pytest.raises(RuntimeError, match="already finished"):
        encoder.finish()

    monkeypatch.setattr(streaming_module, "_MAX_JSON_POINTER_INDEX_ENTRIES", 2)
    json_stream = ArtifactStream.json_items([{"nested": 1}, {"second": 2}])
    json_encoder = ArtifactStreamEncoder(json_stream, preview_chars=20)
    payload = bytearray(json_encoder.start())
    for item in json_stream.iter_chunks():
        payload.extend(json_encoder.encode(item))
    tail, streamed = json_encoder.finish()
    payload.extend(tail)
    assert streamed.json_pointer_index_complete is False
    assert materialize_stream_payload(streamed, bytes(payload)) == [
        {"nested": 1},
        {"second": 2},
    ]


def test_stream_materialization_rejects_unknown_codec_and_binary_rendering() -> None:
    unknown = StreamedArtifact(
        codec="unknown",
        media_type="application/octet-stream",
        logical_type="bytes",
        size_bytes=0,
        char_count=None,
        preview="",
        summary="",
        digest="",
    )
    with pytest.raises(RuntimeError, match="unsupported stream codec"):
        materialize_stream_payload(unknown, b"")

    encoder = ArtifactStreamEncoder(ArtifactStream.bytes([b"a"]), preview_chars=4)
    with pytest.raises(RuntimeError, match="cannot observe rendered text"):
        encoder._observe(b"a", rendered="a")


def test_async_stream_without_close_hook_uses_plain_close_fallback() -> None:
    class AsyncChunks:
        def __aiter__(self) -> AsyncChunks:
            return self

        async def __anext__(self) -> str:
            raise StopAsyncIteration

    async def run() -> None:
        stream = ArtifactStream.text(AsyncChunks())
        assert [item async for item in stream.aiter_chunks()] == []

        unopened = ArtifactStream.text(AsyncChunks())
        await unopened.aclose()
        with pytest.raises(ArtifactStreamConsumedError, match="closed"):
            [item async for item in unopened.aiter_chunks()]

    asyncio.run(run())


def test_async_stream_close_failure_still_marks_stream_closed() -> None:
    class AsyncChunks:
        def __aiter__(self) -> AsyncChunks:
            return self

        async def __anext__(self) -> str:
            raise StopAsyncIteration

        async def aclose(self) -> None:
            raise RuntimeError("close failed")

    async def run() -> None:
        stream = ArtifactStream.text(AsyncChunks())
        with pytest.raises(RuntimeError, match="close failed"):
            await stream.aclose()
        assert stream.consumed is False
        with pytest.raises(ArtifactStreamConsumedError, match="closed"):
            [item async for item in stream.aiter_chunks()]

    asyncio.run(run())


def test_kedi_procedure_can_stream_its_logical_result_only_as_a_tool() -> None:
    source = """\
```\n
from kedi import ArtifactStream\n
from kedi.tool_context import inside_kedi_tool_call\n
```\n
\n
@read_value() -> str:\n
    = ```\n
    if inside_kedi_tool_call():\n
        return ArtifactStream.text([\"abc\", \"def\"])\n
    return \"abcdef\"\n
    ```\n
\n
= `read_value()`\n
"""
    program = parse_program(source)
    runtime = compile_program(program)
    assert runtime.run_main() == "abcdef"

    manager = ArtifactManager(session_id="kedi-procedure-stream")
    try:
        wrapped = with_artifact_result(
            kedi_procedure_tool_spec(runtime, program.procedures[0]),
            manager=manager,
            policy=_policy(),
        )
        result = wrapped.fn()

        assert result["ref_id"] == "tool_call_result_1"
        assert manager.get_value(result["ref_id"]) == "abcdef"
        assert wrapped.return_schema == {"type": "string"}
    finally:
        manager.close()


def test_kedi_procedure_rejects_stream_with_wrong_logical_return_type() -> None:
    source = """\
```\n
from kedi import ArtifactStream\n
```\n
\n
@bad_stream() -> int:\n
    = `ArtifactStream.text([\"not an integer\"])`\n
"""
    program = parse_program(source)
    runtime = compile_program(program)
    tool = kedi_procedure_tool_spec(runtime, program.procedures[0])

    with pytest.raises(KediTypeExecutionError, match="does not match"):
        tool.fn()


def test_kedi_procedure_rejects_artifact_transport_from_direct_call() -> None:
    source = """\
```\n
from kedi import ArtifactStream\n
```\n
\n
@leaking_stream() -> str:\n
    = `ArtifactStream.text([\"value\"])`\n
\n
= `leaking_stream()`\n
"""

    with pytest.raises(KediTypeExecutionError, match="tool transport"):
        compile_program(parse_program(source)).run_main()
