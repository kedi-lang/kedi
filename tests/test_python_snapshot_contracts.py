"""Rejection, restoration and integrity contracts of interactive Python snapshots."""

from __future__ import annotations

import io
import json
import pickle
import struct
import sys
import zipfile
from concurrent.futures import Future
from pathlib import Path
from typing import Any, cast

import pytest

from kedi._interactive import bundle
from kedi._interactive import python_graph as graph
from kedi._interactive.persistence import SessionDumpError, SessionLoadError
from kedi._interactive.resources import restore_resources
from kedi.agent_adapter.conversation import ConversationState
from kedi.agent_adapter.subagents.limits import SubagentUsageLimits
from kedi.agent_profile.capture import _CaptureCell
from kedi.artifacts import ArtifactPolicy
from kedi.decisions import DecisionReference
from kedi.executors.base import DefaultExecutor
from kedi.interactive_session import InteractiveSession
from kedi.lang.compiler.environment import ScopeFrame
from kedi.promise import KediPromise, force
from tests.mock_adapter import MockAdapter

pytest.importorskip("cloudpickle")


def _value(session: InteractiveSession, expression: str) -> Any:
    session.execute(f"[inspection_value: Any] = `{expression}`")
    return force(session._env["inspection_value"])


def _hold(session: InteractiveSession, *values: Any) -> None:
    """Place arbitrary Python objects in a user binding."""
    session.execute("[holder: Any] = `[]`")
    force(session._env["holder"]).extend(values)


def _pending_promise() -> KediPromise:
    return KediPromise(Future(), None)


class _HidesPendingPromise:
    """Only the reducer, not the graph inventory, can see this promise."""

    def __reduce__(self) -> Any:
        return (list, ([_pending_promise()],))


class _HidesPendingDecision:
    def __reduce__(self) -> Any:
        return (list, ([DecisionReference(Future(), ("answer",))],))


class _HidesEvaluatingCapture:
    def __reduce__(self) -> Any:
        cell = _CaptureCell()
        cell._evaluating = True
        return (list, ([cell],))


class _Unpicklable:
    def __reduce__(self) -> Any:
        raise TypeError("refuses serialization")


def _closure_with_empty_cell() -> Any:
    late = None

    def read_late() -> Any:
        return late  # noqa: F821 - deleted below to leave an empty closure cell

    del late
    return read_late


def _evaluating_capture() -> _CaptureCell:
    cell = _CaptureCell()
    cell._evaluating = True
    return cell


# ---------------------------------------------------------------------------
# Dump rejections leave no snapshot behind.


def _reject(session: InteractiveSession, path: Path, match: str) -> None:
    with pytest.raises(SessionDumpError, match=match):
        session.dump(path, format="python")
    assert not path.exists()


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda s: s._persistence_blockers.append("blocked by test"), "blocked by test"),
        (lambda s: s._compiler.rt._outstanding.update({object(): [1]}), "unfinished model work"),
        (
            lambda s: s._compiler.rt._subagent_coordinator._retained_native_runs.add("run"),
            "unfinished subagent work",
        ),
        (lambda s: s._compiler.rt._lifecycle._owners.add(object()), "active invocation"),
        (lambda s: setattr(s._compiler.rt._lifecycle, "_closed", True), "runtime 0 is closed"),
        (
            lambda s: s._compiler.module_loader._loading.append(Path("module.kedi")),
            "module initialization is incomplete",
        ),
        (lambda s: _hold(s, s), "references the live interactive session"),
        (lambda s: _hold(s, _pending_promise()), "unfinished Kedi promise"),
        (lambda s: _hold(s, _HidesPendingPromise()), "unfinished Kedi promise"),
        (
            lambda s: _hold(s, DecisionReference(Future(), ("answer",))),
            "unfinished decision evidence",
        ),
        (lambda s: _hold(s, _HidesPendingDecision()), "unfinished decision evidence"),
        (lambda s: _hold(s, _evaluating_capture()), "profile capture is being evaluated"),
        (lambda s: _hold(s, _HidesEvaluatingCapture()), "profile capture is being evaluated"),
        (lambda s: _hold(s, _Unpicklable()), r"capture failed \(TypeError\): refuses"),
    ],
)
def test_unsafe_session_state_rejects_python_dump(tmp_path: Path, mutate: Any, match: str) -> None:
    with InteractiveSession() as session:
        session.execute("[value: int] = `1`")
        mutate(session)
        try:
            _reject(session, tmp_path / "rejected.state", match)
        finally:
            # Undo runtime-lifecycle fakes so the session can close normally.
            session._compiler.rt._outstanding.clear()
            session._compiler.rt._lifecycle._owners.clear()
            session._compiler.rt._lifecycle._closed = False


def test_foreign_runtime_and_procedure_reject_python_dump(tmp_path: Path) -> None:
    with InteractiveSession() as other, InteractiveSession() as session:
        other.execute("@foreign() -> int:\n  = `1`")
        _hold(session, other._compiler.rt)
        _reject(session, tmp_path / "runtime.state", "runtime not owned by this session")
        # The compiled implementation itself, not its runtime-closing wrapper.
        _hold(session, _value(other, "foreign").__kedi_procedure_target__)
        _reject(session, tmp_path / "procedure.state", "procedure belongs to a runtime outside")


def test_external_subagent_state_rejects_python_dump(tmp_path: Path) -> None:
    with InteractiveSession(subagent_state_path=tmp_path / "subagents") as session:
        _reject(session, tmp_path / "subagents.state", "external persistent subagent state")


def test_python_dump_requires_optional_serializer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "cloudpickle", None)
    with InteractiveSession() as session:
        _reject(session, tmp_path / "missing.state", r"install kedi\[snapshots\]")


def test_unknown_dump_format_is_rejected(tmp_path: Path) -> None:
    with InteractiveSession() as session:
        with pytest.raises(ValueError, match="'portable' or 'python'"):
            session.dump(tmp_path / "x.state", format=cast(Any, "pickle"))


# ---------------------------------------------------------------------------
# Artifact and conversation boundaries.


def _artifact(session: InteractiveSession, value: Any, **policy: Any) -> Any:
    manager = session._compiler.rt.artifacts
    resolved = ArtifactPolicy.patch({"threshold": "1b", **policy}).resolved()
    handle = manager.artifactize(value, policy=resolved)
    cast(ScopeFrame, session._env).declare("handle", handle)
    return manager, handle


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda manager, record: setattr(manager, "_reserved_bytes", 1), "unfinished producer"),
        (lambda manager, record: setattr(record, "active_readers", 1), "active readers"),
        (lambda manager, record: setattr(record, "digest", "0" * 64), "failed its digest check"),
        (lambda manager, record: setattr(record, "codec", "opaque"), "is not persistable"),
        (lambda manager, record: setattr(record, "codec", "bytes"), "changed codec"),
    ],
)
def test_artifact_owner_state_rejects_python_dump(tmp_path: Path, mutate: Any, match: str) -> None:
    with InteractiveSession() as session:
        manager, _ = _artifact(session, "payload text")
        record = manager._all_records()[0]
        mutate(manager, record)
        try:
            _reject(session, tmp_path / "artifact.state", match)
        finally:
            manager._reserved_bytes = 0
            record.active_readers = 0


def test_file_artifact_payload_must_be_a_regular_file(tmp_path: Path) -> None:
    with InteractiveSession() as session:
        manager, _ = _artifact(session, {"name": "Ada"}, store="file", path=str(tmp_path / "store"))
        record = manager._all_records()[0]
        link = tmp_path / "link.json"
        link.symlink_to(record.payload)
        record.payload = link
        _reject(session, tmp_path / "unsafe.state", "unsafe payload")


def test_conversation_checkpoint_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conversation = ConversationState(session_id="lanes")
    with InteractiveSession(conversation_state=conversation) as session:
        conversation._adapter_states.update({"pydantic": {}, "langchain": {}})
        _reject(session, tmp_path / "lanes.state", "multiple native adapter lanes")
        conversation._adapter_states.clear()

        def refuse(lane: str) -> Any:
            raise ValueError(f"{lane} lane is opaque")

        monkeypatch.setattr(conversation, "export_checkpoint", refuse)
        _reject(session, tmp_path / "opaque.state", "cannot be checkpointed: pydantic lane")
    conversation.close()


def test_durable_artifact_owner_stays_durable_after_restore(tmp_path: Path) -> None:
    path = tmp_path / "durable.state"
    with InteractiveSession() as session:
        manager, _ = _artifact(session, "durable text")
        manager._durable_root = tmp_path / "durable-root"
        session.dump(path, format="python")
    with InteractiveSession.load(path, trusted=True) as restored:
        assert restored._compiler.rt.artifacts._durable_root is not None
        assert _value(restored, "handle") == "durable text"


# ---------------------------------------------------------------------------
# Restoration semantics that only a live Python graph can express.


def test_infrastructure_references_bind_to_supplied_adapter(tmp_path: Path) -> None:
    path = tmp_path / "adapter.state"
    original = MockAdapter()
    with InteractiveSession(adapter=original) as session:
        _hold(session, original, session._compiler.rt.executor)
        session.dump(path, format="python")
    supplied = MockAdapter()
    with InteractiveSession.load(path, trusted=True, adapter=supplied) as restored:
        adapter, executor = _value(restored, "holder")
        assert adapter is supplied
        assert executor is restored._compiler.rt.executor


def test_empty_closure_cells_properties_and_by_value_closures(tmp_path: Path) -> None:
    path = tmp_path / "python-shapes.state"
    prelude = """```
class Temperature:
    def __init__(self, celsius):
        self.celsius = celsius

    @property
    def fahrenheit(self):
        return self.celsius * 9 / 5 + 32

def make_reader():
    late = None
    def read_late():
        return late  # noqa: F821 - deleted below to leave an empty closure cell
    del late
    return read_late

reader = make_reader()
```"""
    with InteractiveSession() as session:
        session.execute(prelude)
        session.execute("[reading: Any] = `Temperature(100)`")
        _hold(session, _closure_with_empty_cell())
        session.dump(path, format="python")
    with InteractiveSession.load(path, trusted=True) as restored:
        assert _value(restored, "reading.fahrenheit") == 212
        for expression in ("reader()", "holder[0]()"):
            with pytest.raises(Exception, match="late"):
                _value(restored, expression)


def test_failed_fragment_keeps_definition_site_capture_across_restore(tmp_path: Path) -> None:
    from kedi.agent_profile.capture import CapturedAgentProfile
    from kedi.lang.procedure_introspection import get_procedure_def

    path = tmp_path / "capability.state"
    # Execution is non-transactional: the fragment fails before it reaches
    # ``main``, so the precompiled procedure keeps its definition-site capture.
    source = (
        "> capability: evidence:\n"
        "    > instructions: Cite evidence.\n"
        "@helper() -> str:\n"
        "  = helped\n"
        "> use: evidence\n"
        "`1/0`\n"
        "@main() -> str:\n"
        "  = <helper()>\n"
    )
    with InteractiveSession(adapter=MockAdapter()) as session:
        with pytest.raises(Exception, match="ZeroDivisionError|division"):
            session.execute(source)
        assert _value(session, "main()") == "helped"
        session.dump(path, format="python")
    with InteractiveSession.load(path, trusted=True, adapter=MockAdapter()) as restored:
        definition = get_procedure_def(_value(restored, "main"))
        assert definition is not None
        capture = definition.captured_agent_profile
        assert isinstance(capture, CapturedAgentProfile)
        assert _value(restored, "main()") == "helped"
        # The capture keeps the exact compiled helper, not a later rebinding.
        restored.execute("@helper() -> str:\n  = redefined")
        assert capture._env["helper"]() == "helped"


def test_branch_scope_decision_evidence_survives_restore(tmp_path: Path) -> None:
    path = tmp_path / "decision.state"
    source = (
        "[reader: Any] = `None`\n"
        "> if: `True`:\n"
        "  >> The capital is [capital].\n"
        "  @read() -> str:\n"
        "    = <capital>\n"
        "  [reader] := `read`\n"
    )
    with InteractiveSession(adapter=MockAdapter()) as session:
        session.execute(source)
        session.dump(path, format="python")
    with InteractiveSession.load(path, trusted=True, adapter=MockAdapter()) as restored:
        assert _value(restored, "reader()") == "capital-value"
        target = _value(restored, "reader").__kedi_procedure_target__
        owner = graph._procedure_owner(target)
        assert owner is not None and isinstance(owner[1], ScopeFrame)
        metadata = owner[1].binding_metadata("capital")
        assert metadata is not None and metadata.decision is not None
        assert metadata.decision.record_future.done()


def test_restore_keeps_session_limits(tmp_path: Path) -> None:
    path = tmp_path / "limits.state"
    limits = SubagentUsageLimits(request_limit=3, tool_calls_limit=4)
    with InteractiveSession(subagent_usage_limits=limits) as session:
        session.dump(path, format="python")
    with InteractiveSession.load(path, trusted=True, close_conversation_state=True) as restored:
        assert restored._subagent_usage_limits == limits
        assert restored._close_conversation_state is True


# ---------------------------------------------------------------------------
# Load-time boundaries.


@pytest.fixture
def snapshot(tmp_path: Path) -> Path:
    path = tmp_path / "base.state"
    with InteractiveSession() as session:
        session.execute("[value: int] = `7`")
        session.dump(path, format="python")
    return path


def test_load_rejects_unavailable_cwd(tmp_path: Path) -> None:
    cwd = tmp_path / "workspace"
    cwd.mkdir()
    path = tmp_path / "cwd.state"
    with InteractiveSession(cwd=cwd) as session:
        session.dump(path, format="python")
    cwd.rmdir()
    with pytest.raises(SessionLoadError, match="working directory is unavailable"):
        InteractiveSession.load(path, trusted=True)


def test_load_requires_default_executor_and_serializer(
    snapshot: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ExternalExecutor(DefaultExecutor):
        pass

    with pytest.raises(SessionLoadError, match="default local executor"):
        InteractiveSession.load(snapshot, trusted=True, executor=ExternalExecutor())
    monkeypatch.setitem(sys.modules, "cloudpickle", None)
    with pytest.raises(SessionLoadError, match=r"install kedi\[snapshots\]"):
        InteractiveSession.load(snapshot, trusted=True)


def test_supplied_conversation_must_match_saved_owner(snapshot: Path, tmp_path: Path) -> None:
    unrelated = ConversationState(session_id="unrelated")
    with pytest.raises(SessionLoadError, match="does not match the saved session"):
        InteractiveSession.load(snapshot, trusted=True, conversation_state=unrelated)
    unrelated.close()


def test_missing_snapshot_is_reported(tmp_path: Path) -> None:
    with pytest.raises(SessionLoadError, match="Cannot read interactive session snapshot"):
        InteractiveSession.load(tmp_path / "absent.state", trusted=True)


def _graph_payload(header: Any, *records: Any, header_bytes: bytes | None = None) -> bytes:
    rendered = pickle.dumps(header) if header_bytes is None else header_bytes
    body = b"".join(pickle.dumps(record) for record in records)
    return graph._MAGIC + struct.pack(">Q", len(rendered)) + rendered + body


class _UnknownReferencePickler(pickle.Pickler):
    def persistent_id(self, obj: Any) -> Any:
        return ("unknown",) if obj == "sentinel" else None


def _unknown_reference_payload() -> bytes:
    stream = io.BytesIO()
    _UnknownReferencePickler(stream).dump({"value": "sentinel"})
    header = pickle.dumps({"version": 1, "runtime_count": 1, "objects": []})
    return graph._MAGIC + struct.pack(">Q", len(header)) + header + stream.getvalue()


_HEADER = {"version": 1, "runtime_count": 1, "objects": [{"kind": "cell"}]}


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (b"not a graph", "graph signature"),
        (graph._MAGIC + struct.pack(">Q", 99), "Truncated Python session graph"),
        (_graph_payload({**_HEADER, "version": 2}), "Unsupported Python session graph"),
        (
            _graph_payload({**_HEADER, "objects": [{"kind": "unknown"}]}),
            "Unknown Python graph object kind",
        ),
        (
            _graph_payload(
                {**_HEADER, "objects": [{"kind": "cell"}, {"kind": "cell"}]},
                {},
                (0, {"populated": False}),
                (0, {"populated": False}),
            ),
            "Duplicate or invalid Python graph record",
        ),
        (
            _graph_payload(_HEADER, {}, (0, {"populated": False}), "trailing"),
            "Unexpected trailing Python graph data",
        ),
        (_unknown_reference_payload(), "Unknown Python graph persistent reference"),
        (_graph_payload(None, header_bytes=b"\x80\x05garbage"), "hydration failed"),
    ],
)
def test_corrupt_graph_payload_is_rejected(payload: bytes, match: str) -> None:
    with InteractiveSession() as session:
        with pytest.raises(SessionLoadError, match=match):
            graph.loads_session(session, payload, {})


def test_graph_resource_references_require_restored_owners(snapshot: Path) -> None:
    loaded = bundle.read_python_bundle(snapshot, trusted=True)
    try:
        with InteractiveSession() as session:
            with pytest.raises(SessionLoadError, match="Missing restored resource"):
                graph.loads_session(session, loaded.graph, {})
    finally:
        loaded.cleanup()


@pytest.mark.parametrize(
    ("metadata", "match"),
    [
        (
            {"managers": [{"token": 1, "root": "resources/x"}], "conversations": []},
            "artifact owner",
        ),
        (
            {"managers": [{"token": "a", "root": "../escape"}], "conversations": []},
            "artifact owner",
        ),
        ({"managers": [], "conversations": [{"token": 1}]}, "Duplicate conversation owner"),
        ({"managers": [{"token": "a", "root": "resources/x"}]}, "Cannot restore Python snapshot"),
    ],
)
def test_corrupt_resource_metadata_is_rejected(
    tmp_path: Path, metadata: dict[str, Any], match: str
) -> None:
    with pytest.raises(SessionLoadError, match=match):
        restore_resources(metadata, tmp_path)


# ---------------------------------------------------------------------------
# Session close ownership.


def test_close_attempts_every_owner_and_raises_first_failure() -> None:
    conversation = ConversationState(session_id="closing")
    session = InteractiveSession(conversation_state=conversation, close_conversation_state=True)
    calls: list[str] = []

    def failing(name: str) -> Any:
        def cleanup() -> None:
            calls.append(name)
            raise RuntimeError(f"{name} failed")

        return cleanup

    session._snapshot_cleanups.extend([failing("first"), failing("second")])
    with pytest.raises(RuntimeError, match="first failed") as raised:
        session.close()
    assert calls == ["first", "second"]
    assert conversation._closed
    assert any("second failed" in note for note in raised.value.__notes__)


# ---------------------------------------------------------------------------
# Archive writer and reader boundaries.


def _archive_parts(path: Path) -> tuple[bytes, dict[str, bytes]]:
    content = path.read_bytes()
    prefix = content[: content.index(b"PK\x03\x04")]
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        return prefix, {name: archive.read(name) for name in archive.namelist()}


def _write_archive(path: Path, prefix: bytes, members: dict[str, bytes]) -> None:
    with path.open("wb") as stream:
        stream.write(prefix)
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, data in members.items():
                archive.writestr(name, data)


def _mutate_manifest(path: Path, mutate: Any) -> None:
    prefix, members = _archive_parts(path)
    manifest = json.loads(members["manifest.json"])
    mutate(manifest)
    members["manifest.json"] = json.dumps(manifest).encode()
    _write_archive(path, prefix, members)


def _bundle(tmp_path: Path) -> Path:
    path = tmp_path / "bundle.state"
    bundle.write_python_bundle(path, b"graph", {"resource_limit_bytes": 16}, {})
    return path


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda m: m.update(extra=True), "invalid snapshot manifest"),
        (lambda m: m.update(format_version=2), "unsupported Python snapshot format"),
        (lambda m: m.update(metadata=[]), "metadata must be an object"),
        (lambda m: m["environment"].pop("packages"), "invalid compatibility metadata"),
        (lambda m: m["environment"].update(packages={"x": 1}), "invalid installed package"),
        (lambda m: m["members"].clear(), "member allowlist does not match"),
        (lambda m: m["members"]["graph.pkl"].update(sha256="zz"), "invalid archive member"),
    ],
)
def test_manifest_structure_is_validated(tmp_path: Path, mutate: Any, match: str) -> None:
    path = _bundle(tmp_path)
    _mutate_manifest(path, mutate)
    with pytest.raises(SessionLoadError, match=match):
        bundle.read_python_bundle(path, trusted=True)


def test_duplicate_manifest_keys_and_missing_members_are_rejected(tmp_path: Path) -> None:
    path = _bundle(tmp_path)
    prefix, members = _archive_parts(path)
    duplicated = dict(members)
    duplicated["manifest.json"] = b'{"format": 1, ' + members["manifest.json"][1:]
    _write_archive(path, prefix, duplicated)
    with pytest.raises(SessionLoadError, match="duplicate manifest key"):
        bundle.read_python_bundle(path, trusted=True)
    _write_archive(
        path, prefix, {"manifest.json": members["manifest.json"], "resources/x": b"payload"}
    )
    with pytest.raises(SessionLoadError, match="missing required archive members"):
        bundle.read_python_bundle(path, trusted=True)


def test_zip_directory_bounds_are_validated(tmp_path: Path) -> None:
    path = tmp_path / "long-names.state"
    # Long names leave room in the directory for one fabricated entry header.
    files = {f"resources/{'n' * 200}": b"x"}
    bundle.write_python_bundle(path, b"graph", {"resource_limit_bytes": 1}, files)
    content = path.read_bytes()
    path.write_bytes(content[: len(content) // 2])
    with pytest.raises(SessionLoadError, match="missing ZIP end record"):
        bundle.read_python_bundle(path, trusted=True)
    # Claim one more central-directory entry than the archive contains.
    end = content.rfind(b"PK\x05\x06")
    count = struct.unpack_from("<H", content, end + 10)[0]
    inflated = bytearray(content)
    struct.pack_into("<HH", inflated, end + 8, count + 1, count + 1)
    path.write_bytes(bytes(inflated))
    with pytest.raises(SessionLoadError, match="entry count does not match"):
        bundle.read_python_bundle(path, trusted=True)


def test_archive_size_limits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _bundle(tmp_path)
    monkeypatch.setattr(bundle, "_MAX_ARCHIVE_BYTES", 10)
    with pytest.raises(SessionLoadError, match="archive exceeds hard size limit"):
        bundle.read_python_bundle(path, trusted=True)
    monkeypatch.undo()
    monkeypatch.setattr(bundle, "MAX_SNAPSHOT_BYTES", 10)
    monkeypatch.setattr(bundle, "_MAX_ARTIFACT_BYTES", 0)
    with pytest.raises(SessionLoadError, match="expanded snapshot exceeds hard size limit"):
        bundle.read_python_bundle(path, trusted=True)


def _nested(depth: int) -> Any:
    value: Any = 0
    for _ in range(depth):
        value = [value]
    return value


@pytest.mark.parametrize(
    ("arguments", "limits", "match"),
    [
        (("graph", {"resource_limit_bytes": 0}, {}), {}, "graph must be bytes"),
        ((b"g", {"resource_limit_bytes": 0, "deep": _nested(102)}, {}), {}, "nesting limit"),
        ((b"g", {"resource_limit_bytes": 0, "set": {1}}, {}), {}, "finite JSON primitives"),
        ((b"g", {"resource_limit_bytes": 9}, {"evil/../x": b""}), {}, "under resources/"),
        (
            (b"g", {"resource_limit_bytes": 9}, {"resources/a": b"", "resources/b": b""}),
            {"_MAX_MEMBERS": 3},
            "member count exceeds limit",
        ),
        ((b"graph", {"resource_limit_bytes": 0}, {}), {"MAX_SNAPSHOT_BYTES": 10}, "exceed 64 MiB"),
    ],
)
def test_writer_rejects_invalid_bundle_contents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arguments: tuple[Any, ...],
    limits: dict[str, int],
    match: str,
) -> None:
    for name, value in limits.items():
        monkeypatch.setattr(bundle, name, value)
    path = tmp_path / "invalid.state"
    with pytest.raises(SessionDumpError, match=match):
        bundle.write_python_bundle(path, *arguments)
    assert list(tmp_path.iterdir()) == []


def test_writer_requires_serializer_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from importlib import metadata

    installed = metadata.version

    def version(name: str) -> str:
        if name == "cloudpickle":
            raise metadata.PackageNotFoundError(name)
        return installed(name)

    monkeypatch.setattr(bundle.package_metadata, "version", version)
    with pytest.raises(SessionDumpError, match=r"kedi\[snapshots\]"):
        bundle.write_python_bundle(tmp_path / "x.state", b"g", {"resource_limit_bytes": 0}, {})


def test_uninstalled_dependency_distributions_are_not_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bundle.package_metadata,
        "packages_distributions",
        lambda: {"phantom": ["kedi-snapshot-phantom-distribution"]},
    )
    pinned = bundle._required_packages(["phantom.module"])
    assert "kedi-snapshot-phantom-distribution" not in pinned
    assert "pydantic" in pinned
