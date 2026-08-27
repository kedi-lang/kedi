from __future__ import annotations

import json
import math
import operator
import re
from concurrent.futures import Future
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID

import pytest
from pydantic import BaseModel

import kedi
from kedi._interactive import persistence, state
from kedi._interactive.persistence import (
    SessionDumpError,
    SessionLoadError,
    SessionValueDecoder,
    SessionValueEncoder,
    UnsupportedSessionValue,
)
from kedi._interactive.state import ExecutedFragment
from kedi.agent_adapter.approval import ApprovalDecision, ApprovalPolicy
from kedi.agent_adapter.conversation import ConversationState
from kedi.agent_adapter.hooks import HookHandler
from kedi.agent_adapter.subagents.limits import SubagentUsageLimits
from kedi.agent_adapter.tool_override import ToolSpec
from kedi.agent_profile import AgentProfile, HookSettings
from kedi.agent_profile.codemode import CodeModeSettings
from kedi.agent_profile.compaction import CompactionSettings
from kedi.agent_profile.mcp import McpServerSpec
from kedi.agent_profile.skills import SkillsSettings
from kedi.artifacts import ArtifactPolicy
from kedi.lang.ast import IfStmt, ProcDefStmt, TypeDefStmt
from kedi.lang.compiler.environment import BindingMetadata, ScopeFrame
from kedi.lang.compiler.type_system import KediType
from kedi.promise import KediPromise


class SnapshotModel(BaseModel):
    name: str
    count: int


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        42,
        "text",
        1.5,
        b"binary",
        2 + 3j,
        Decimal("12.340"),
        UUID("12345678-1234-5678-1234-567812345678"),
        datetime(2026, 8, 26, 12, 30, tzinfo=timezone.utc, fold=1),
        date(2026, 8, 26),
        time(12, 30, fold=1),
        timedelta(days=2, seconds=3, microseconds=4),
        Path("relative/file.txt"),
        [1, "two"],
        (1, "two"),
        {1, 2},
        frozenset({1, 2}),
        {1: "one", (2, 3): ["nested"]},
    ],
)
def test_session_value_codec_round_trips_supported_values(value: Any) -> None:
    encoded = SessionValueEncoder(kedi_models={}).encode(value)
    restored = SessionValueDecoder(kedi_models={}).decode(encoded)
    assert restored == value
    assert type(restored) is type(value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_session_value_codec_round_trips_non_finite_floats(value: float) -> None:
    encoded = SessionValueEncoder(kedi_models={}).encode(value)
    restored = SessionValueDecoder(kedi_models={}).decode(encoded)
    if math.isnan(value):
        assert math.isnan(restored)
    else:
        assert restored == value


def test_session_value_codec_round_trips_non_finite_complex_and_regex() -> None:
    encoder = SessionValueEncoder(kedi_models={})
    decoder = SessionValueDecoder(kedi_models={})

    complex_value = complex(float("inf"), float("nan"))
    restored_complex = decoder.decode(encoder.encode(complex_value))
    assert math.isinf(restored_complex.real)
    assert math.isnan(restored_complex.imag)

    pattern = re.compile(r"kedi\s+state", re.IGNORECASE)
    restored_pattern = decoder.decode(encoder.encode(pattern))
    assert restored_pattern.pattern == pattern.pattern
    assert restored_pattern.flags == pattern.flags


def test_session_value_codec_round_trips_source_backed_model() -> None:
    value = SnapshotModel(name="Kedi", count=2)
    encoded = SessionValueEncoder(kedi_models={SnapshotModel: "SnapshotModel"}).encode(value)
    restored = SessionValueDecoder(kedi_models={"SnapshotModel": SnapshotModel}).decode(encoded)
    assert restored == value
    assert type(restored) is SnapshotModel


def test_session_value_encoder_rejects_unsafe_values_aliases_and_depth() -> None:
    with pytest.raises(UnsupportedSessionValue, match="only source-backed Kedi models"):
        SessionValueEncoder(kedi_models={}).encode(SnapshotModel(name="x", count=1))
    with pytest.raises(UnsupportedSessionValue, match="bytearray"):
        SessionValueEncoder(kedi_models={}).encode(bytearray(b"x"))

    shared: list[int] = []
    with pytest.raises(UnsupportedSessionValue, match="shared or cyclic object"):
        SessionValueEncoder(kedi_models={}).encode([shared, shared])
    cyclic: list[Any] = []
    cyclic.append(cyclic)
    with pytest.raises(UnsupportedSessionValue, match="shared or cyclic object"):
        SessionValueEncoder(kedi_models={}).encode(cyclic)

    nested: Any = None
    for _ in range(persistence.MAX_VALUE_DEPTH + 2):
        nested = [nested]
    with pytest.raises(UnsupportedSessionValue, match="maximum nesting depth"):
        SessionValueEncoder(kedi_models={}).encode(nested)


@pytest.mark.parametrize(
    "encoded, expected",
    [
        ([], "invalid encoded session value"),
        ({"$kedi": "unknown"}, "unknown session value tag"),
        ({"$kedi": "bytes", "data": "not-base64"}, "invalid 'bytes' session value"),
        ({"$kedi": "dict", "items": [[1]]}, "invalid encoded dict entry"),
        (
            {"$kedi": "dict", "items": [["same", 1], ["same", 2]]},
            "duplicate decoded dict key",
        ),
        (
            {"$kedi": "kedi_model", "name": "Missing", "fields": {}},
            "unknown source-backed Kedi model",
        ),
    ],
)
def test_session_value_decoder_rejects_invalid_values(encoded: Any, expected: str) -> None:
    with pytest.raises(SessionLoadError, match=expected):
        SessionValueDecoder(kedi_models={}).decode(encoded)


def test_session_value_decoder_rejects_excessive_depth() -> None:
    decoder = SessionValueDecoder(kedi_models={})
    with pytest.raises(SessionLoadError, match="maximum nesting depth"):
        decoder.decode(None, depth=persistence.MAX_VALUE_DEPTH + 1)


def test_profile_codec_round_trips_full_portable_profile() -> None:
    profile = AgentProfile(
        name="reviewer",
        model="model-id",
        adapter_kind="agent-framework",
        adapter_shortname="pydantic",
        agent_command=("agent", "run"),
        effort="high",
        system="Review carefully.",
        settings={"temperature": 0.2, "stop": ["done"]},
        tools=("read_file",),
        mcp_servers=(McpServerSpec(command="server", args=("--stdio",)),),
        approval=ApprovalPolicy.allow(),
        skills=SkillsSettings(
            enabled=True,
            cwd=Path("skills"),
            max_skills=5,
            include_registry=False,
            include_all=True,
            exclude_paths=(Path("ignored"),),
        ),
        subagents=("planner",),
        max_agents=3,
        workflow_mode="dynamic",
        history=True,
        compaction=CompactionSettings(mode="native", threshold=1_000),
        codemode=CodeModeSettings(enabled=True),
        output_type=int,
        artifacts=ArtifactPolicy.patch({"enabled": False}),
    )
    encoder = SessionValueEncoder(kedi_models={})
    encoded = persistence.encode_profile(
        profile,
        encoder=encoder,
        model_names={},
        path="profile",
    )
    restored = persistence.decode_profile(
        encoded,
        decoder=SessionValueDecoder(kedi_models={}),
        kedi_models={},
    )
    assert restored == profile
    assert restored.agent_command == ("agent", "run")


@pytest.mark.parametrize(
    "output_type",
    [bool, bytes, complex, dict, float, frozenset, int, list, set, str],
)
def test_profile_codec_round_trips_builtin_output_types(output_type: type[Any]) -> None:
    profile = AgentProfile(output_type=output_type)
    encoded = persistence.encode_profile(
        profile,
        encoder=SessionValueEncoder(kedi_models={}),
        model_names={},
        path="profile",
    )
    restored = persistence.decode_profile(
        encoded,
        decoder=SessionValueDecoder(kedi_models={}),
        kedi_models={},
    )
    assert restored.output_type is output_type


def test_profile_codec_round_trips_kedi_output_type_and_boolean_settings() -> None:
    profile = AgentProfile(skills=True, codemode=False, output_type=SnapshotModel)
    encoded = persistence.encode_profile(
        profile,
        encoder=SessionValueEncoder(kedi_models={SnapshotModel: "SnapshotModel"}),
        model_names={SnapshotModel: "SnapshotModel"},
        path="profile",
    )
    restored = persistence.decode_profile(
        encoded,
        decoder=SessionValueDecoder(kedi_models={"SnapshotModel": SnapshotModel}),
        kedi_models={"SnapshotModel": SnapshotModel},
    )
    assert restored.output_type is SnapshotModel
    assert isinstance(restored.skills, SkillsSettings) and restored.skills.enabled is True
    assert isinstance(restored.codemode, CodeModeSettings) and restored.codemode.enabled is False


def test_profile_codec_boolean_helpers_expand_short_forms() -> None:
    assert persistence._encode_skills(True) == {
        "enabled": True,
        "cwd": None,
        "max_skills": 20,
        "include_registry": True,
        "include_all": False,
        "exclude_paths": [],
    }
    assert persistence._dataclass_values(False) == {
        item.name: getattr(CodeModeSettings(enabled=False), item.name)
        for item in persistence.fields(CodeModeSettings)
    }


@pytest.mark.parametrize(
    "profile, expected",
    [
        (
            AgentProfile(tool_bindings={"tool": cast(ToolSpec, object())}),
            "tool_bindings",
        ),
        (AgentProfile(subagent_bindings={"agent": cast(Any, object())}), "subagent_bindings"),
        (
            AgentProfile(
                approval=ApprovalPolicy.dynamic(lambda _request: ApprovalDecision.allow())
            ),
            "process-local",
        ),
        (AgentProfile(output_type=object), "output_type"),
    ],
)
def test_profile_encoder_rejects_process_bound_state(
    profile: AgentProfile,
    expected: str,
) -> None:
    with pytest.raises(UnsupportedSessionValue, match=expected):
        persistence.encode_profile(
            profile,
            encoder=SessionValueEncoder(kedi_models={}),
            model_names={},
            path="profile",
        )


@pytest.mark.parametrize(
    "output_type, expected",
    [
        ({"kind": "builtin", "name": "missing"}, "Unknown builtin output type"),
        ({"kind": "kedi", "name": "Missing"}, "Unknown Kedi output type"),
        ({"kind": "other", "name": "x"}, "Invalid profile output type"),
    ],
)
def test_profile_decoder_rejects_invalid_output_types(
    output_type: dict[str, str],
    expected: str,
) -> None:
    encoded = persistence.encode_profile(
        AgentProfile(),
        encoder=SessionValueEncoder(kedi_models={}),
        model_names={},
        path="profile",
    )
    encoded["output_type"] = output_type
    with pytest.raises(SessionLoadError, match=expected):
        persistence.decode_profile(
            encoded,
            decoder=SessionValueDecoder(kedi_models={}),
            kedi_models={},
        )


def test_snapshot_file_round_trip_and_validation(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot.json"
    payload = {"value": 42}
    assert persistence.write_snapshot(snapshot, payload) == snapshot
    assert persistence.read_snapshot(snapshot) == payload
    assert snapshot.stat().st_mode & 0o777 == 0o600


def test_snapshot_write_and_read_enforce_size_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(persistence, "MAX_SNAPSHOT_BYTES", 1)
    with pytest.raises(SessionDumpError, match="snapshot is"):
        persistence.write_snapshot(tmp_path / "large.json", {"value": "large"})

    oversized = tmp_path / "oversized.json"
    oversized.write_text("{}", encoding="utf-8")
    with pytest.raises(SessionLoadError, match="exceeds"):
        persistence.read_snapshot(oversized)


def test_snapshot_read_rejects_missing_invalid_and_incompatible_documents(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(SessionLoadError, match="Cannot read"):
        persistence.read_snapshot(missing)

    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text("not json", encoding="utf-8")
    with pytest.raises(SessionLoadError, match="Cannot read"):
        persistence.read_snapshot(snapshot)

    cases = [
        ([], "root must be an object"),
        ({"format": "other"}, "Not a Kedi"),
        (
            {"format": persistence.SNAPSHOT_FORMAT, "format_version": 99},
            "Unsupported interactive session snapshot version",
        ),
        (
            {
                "format": persistence.SNAPSHOT_FORMAT,
                "format_version": persistence.SNAPSHOT_FORMAT_VERSION,
                "kedi_version": "other",
            },
            "current version",
        ),
        (
            {
                "format": persistence.SNAPSHOT_FORMAT,
                "format_version": persistence.SNAPSHOT_FORMAT_VERSION,
                "kedi_version": persistence.__version__,
                "payload": [],
            },
            "payload must be an object",
        ),
        (
            {
                "format": persistence.SNAPSHOT_FORMAT,
                "format_version": persistence.SNAPSHOT_FORMAT_VERSION,
                "kedi_version": persistence.__version__,
                "payload": {},
                "digest": "wrong",
            },
            "integrity check",
        ),
    ]
    for document, expected in cases:
        snapshot.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(SessionLoadError, match=expected):
            persistence.read_snapshot(snapshot)


def test_atomic_write_cleans_temporary_file_when_initialization_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "snapshot.json"

    def fail_chmod(_fd: int, _mode: int) -> None:
        raise OSError("chmod failed")

    monkeypatch.setattr(persistence.os, "fchmod", fail_chmod)
    with pytest.raises(OSError, match="chmod failed"):
        persistence._atomic_write(destination, b"value")
    assert not destination.exists()
    assert not tuple(tmp_path.iterdir())


def test_session_load_restores_limits_and_honors_conversation_ownership_override(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "conversation.kedi-state"
    conversation = ConversationState(session_id="portable-session")
    limits = SubagentUsageLimits(request_limit=3, tool_calls_limit=7)
    session = kedi.InteractiveSession(
        conversation_state=conversation,
        subagent_usage_limits=limits,
    )
    try:
        session.execute("[value: int] = `42`")
        session.dump(snapshot)
    finally:
        session.close()
        conversation.close()

    supplied = ConversationState(session_id="portable-session")
    restored = kedi.load_session(
        snapshot,
        conversation_state=supplied,
        close_conversation_state=True,
    )
    assert restored._subagent_usage_limits == limits
    restored.close()
    assert supplied._closed is True


def test_session_load_restores_binding_types_and_python_writeback_contracts(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "typed-bindings.kedi-state"
    with kedi.InteractiveSession() as session:
        session.execute("[count: int] = `1`")
        session.dump(snapshot)

    with kedi.load_session(snapshot) as restored:
        restored.execute("[count] := `2`")
        assert restored.execute("= `count`") == 2

        with pytest.raises(TypeError, match="annotated type"):
            restored.execute('[count] := `"invalid"`')
        assert restored.execute("= `count`") == 2

        with pytest.raises(TypeError, match="annotated type"):
            restored.execute('`count = "invalid"`')
        assert restored.execute("= `count`") == 2


def test_session_load_preserves_runtime_owned_reserved_binding(tmp_path: Path) -> None:
    snapshot = tmp_path / "reserved-binding.kedi-state"
    with kedi.InteractiveSession(runtime_globals={"args": ["value"]}) as session:
        session.dump(snapshot)

    with kedi.load_session(snapshot) as restored:
        assert restored.execute("= `args`") == ["value"]
        with pytest.raises(ValueError, match="Reserved identifier"):
            restored.execute('[args] := `["replacement"]`')


def test_binding_metadata_codec_rejects_nonlexical_and_invalid_entries() -> None:
    issues: list[str] = []
    assert state._encode_binding_metadata({}, issues) == {}
    assert issues == ["interactive environment does not expose lexical binding metadata"]

    frame = ScopeFrame()
    frame.declare("invalid", 1, BindingMetadata(type_contract=object()))
    issues = []
    assert state._encode_binding_metadata(frame, issues) == {}
    assert "unsupported type contract" in issues[0]

    valid = {
        "value": {
            "kind": "value",
            "type_contract": "int",
            "assignable": True,
            "synthetic": False,
        }
    }
    decoded = state._decode_binding_metadata(valid, {"value": 1})
    assert str(decoded["value"].type_contract) == str(KediType("int"))

    invalid_cases = [
        ({}, {"value": 1}, "match the environment"),
        ({"value": []}, {"value": 1}, "entries must be objects"),
        ({1: {}}, {1: 1}, "entries must be objects"),
        (
            {"value": {"kind": 1, "type_contract": None, "assignable": True, "synthetic": False}},
            {"value": 1},
            "invalid kind",
        ),
        (
            {
                "value": {
                    "kind": "value",
                    "type_contract": 1,
                    "assignable": True,
                    "synthetic": False,
                }
            },
            {"value": 1},
            "invalid type contract",
        ),
        (
            {
                "value": {
                    "kind": "value",
                    "type_contract": None,
                    "assignable": 1,
                    "synthetic": False,
                }
            },
            {"value": 1},
            "invalid flags",
        ),
    ]
    for encoded, environment, message in invalid_cases:
        with pytest.raises(SessionLoadError, match=message):
            state._decode_binding_metadata(cast(Any, encoded), cast(Any, environment))


def test_restore_payload_rejects_nonlexical_interactive_environment() -> None:
    with kedi.InteractiveSession() as session:
        payload = state._build_snapshot_payload(session)
        session._env = cast(Any, {})
        with pytest.raises(SessionLoadError, match="not a lexical scope frame"):
            state._restore_payload(session, payload)


def test_session_load_validates_supplied_conversation_state(tmp_path: Path) -> None:
    without_conversation = tmp_path / "without-conversation.kedi-state"
    with kedi.InteractiveSession() as session:
        session.dump(without_conversation)

    supplied = ConversationState()
    try:
        with pytest.raises(SessionLoadError, match="did not own conversation state"):
            kedi.load_session(without_conversation, conversation_state=supplied)
    finally:
        supplied.close()

    with_conversation = tmp_path / "with-conversation.kedi-state"
    original = ConversationState(session_id="expected-session")
    with kedi.InteractiveSession(conversation_state=original) as session:
        session.dump(with_conversation)
    original.close()

    nonempty = ConversationState(session_id="expected-session")
    nonempty.record_exchange("question", "answer")
    try:
        with pytest.raises(SessionLoadError, match="not empty"):
            kedi.load_session(with_conversation, conversation_state=nonempty)
    finally:
        nonempty.close()

    wrong = ConversationState(session_id="wrong-session")
    try:
        with pytest.raises(SessionLoadError, match="different session_id"):
            kedi.load_session(with_conversation, conversation_state=wrong)
    finally:
        wrong.close()


def test_session_load_wraps_invalid_metadata_and_payload(tmp_path: Path) -> None:
    snapshot = tmp_path / "base.kedi-state"
    with kedi.InteractiveSession() as session:
        session.execute("[value: int] = `42`")
        session.dump(snapshot)
    payload = persistence.read_snapshot(snapshot)

    invalid_metadata = dict(payload)
    invalid_metadata["default_profile"] = {}
    metadata_snapshot = tmp_path / "invalid-metadata.kedi-state"
    persistence.write_snapshot(metadata_snapshot, invalid_metadata)
    with pytest.raises(SessionLoadError, match="Invalid interactive session metadata"):
        kedi.load_session(metadata_snapshot)

    unsupported_metadata = dict(payload)
    unsupported_metadata["default_profile"] = dict(payload["default_profile"])
    unsupported_metadata["default_profile"]["output_type"] = {
        "kind": "builtin",
        "name": "missing",
    }
    unsupported_snapshot = tmp_path / "unsupported-metadata.kedi-state"
    persistence.write_snapshot(unsupported_snapshot, unsupported_metadata)
    with pytest.raises(SessionLoadError, match="Unknown builtin output type"):
        kedi.load_session(unsupported_snapshot)

    invalid_payload = dict(payload)
    invalid_payload.pop("environment")
    payload_snapshot = tmp_path / "invalid-payload.kedi-state"
    persistence.write_snapshot(payload_snapshot, invalid_payload)
    with pytest.raises(SessionLoadError, match="Invalid interactive session payload"):
        kedi.load_session(payload_snapshot)


def test_session_load_closes_partial_session_for_unexpected_restore_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "session.kedi-state"
    with kedi.InteractiveSession() as session:
        session.dump(snapshot)

    created: list[kedi.InteractiveSession] = []

    class CapturingSession(kedi.InteractiveSession):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            created.append(self)

    def fail_restore(_session: object, _payload: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(state, "_restore_payload", fail_restore)
    with pytest.raises(KeyboardInterrupt):
        kedi.load_session(snapshot, session_type=CapturingSession)
    assert len(created) == 1
    assert created[0]._closed is True


def test_session_dump_reports_all_runtime_only_boundaries(tmp_path: Path) -> None:
    future: Future[Any] = Future()
    future.set_exception(RuntimeError("promise failed"))
    session = kedi.InteractiveSession(subagent_state_path=tmp_path / "subagents.json")
    try:
        session._env[cast(Any, 1)] = "invalid key"
        session._env["failed_promise"] = KediPromise(future, None)
        session._compiler.rt._outstanding["pending"] = cast(Any, object())
        session._compiler.rt._procedures["runtime_proc"] = cast(Any, object())
        session._compiler.rt._type_env_stack[0]["RuntimeModel"] = SnapshotModel
        session._compiler.rt._profile_registry["runtime_profile"] = AgentProfile(
            name="runtime_profile"
        )
        with session._compiler.rt.artifacts._lock:
            session._compiler.rt.artifacts._ref_stores["artifact"] = cast(Any, object())

        with pytest.raises(SessionDumpError) as info:
            session.dump(tmp_path / "rejected.kedi-state")
    finally:
        session.close()

    rendered = str(info.value)
    assert "environment key 1 is not a string" in rendered
    assert "promise failed" in rendered
    assert "unfinished model work" in rendered
    assert "external persistent subagent state" in rendered
    assert "procedures without restorable fragment source: runtime_proc" in rendered
    assert "types without restorable fragment source: RuntimeModel" in rendered
    assert "profiles without restorable fragment source: runtime_profile" in rendered
    assert "runtime contains active or retained artifacts" in rendered


def test_session_dump_reports_unparseable_fragments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = kedi.InteractiveSession()
    session._fragments.append(ExecutedFragment("broken", "broken.kedi"))

    def fail_parse(_source: str, *, source_path: str) -> None:
        raise ValueError(f"cannot parse {source_path}")

    monkeypatch.setattr(state, "parse_program", fail_parse)
    try:
        with pytest.raises(SessionDumpError, match="no longer parses"):
            session.dump(tmp_path / "rejected.kedi-state")
    finally:
        session.close()


def test_session_dump_reports_unsupported_active_profile(tmp_path: Path) -> None:
    with kedi.InteractiveSession() as session:
        session._top_level_state.active_profile = AgentProfile(
            approval=ApprovalPolicy.dynamic(lambda _request: ApprovalDecision.allow())
        )
        with pytest.raises(SessionDumpError, match="active_profile.approval"):
            session.dump(tmp_path / "rejected.kedi-state")


def test_session_dump_load_restores_import_addressable_hook_handlers(tmp_path: Path) -> None:
    snapshot = tmp_path / "hooks.kedi-state"
    with kedi.InteractiveSession() as session:
        session._top_level_state.active_profile = AgentProfile(
            hooks=HookSettings(pre_tool_use=(cast(HookHandler, operator.neg),)),
        )
        session.dump(snapshot)

    with kedi.load_session(snapshot) as restored:
        hooks = restored._top_level_state.active_profile.hooks
        assert hooks is not None
        assert hooks.pre_tool_use == (operator.neg,)


def test_session_dump_rejects_non_importable_hook_handlers(tmp_path: Path) -> None:
    def local_handler(_event: object) -> None:
        return None

    with kedi.InteractiveSession() as session:
        session._top_level_state.active_profile = AgentProfile(
            hooks=HookSettings(pre_tool_use=(local_handler,)),
        )
        with pytest.raises(SessionDumpError, match="local functions and lambdas"):
            session.dump(tmp_path / "rejected.kedi-state")


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (42, "profile.hooks must be an object"),
        ({"enabled": "yes"}, "profile.hooks.enabled must be a boolean"),
        ({"pre_tool_use": "handler"}, "profile.hooks.pre_tool_use must be a list"),
    ],
)
def test_session_hook_decoder_rejects_malformed_settings(value: Any, message: str) -> None:
    with pytest.raises(SessionLoadError, match=message):
        persistence._decode_hooks(value)


def test_session_hook_callable_encoder_rejects_nonportable_addresses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Handler:
        def bound(self, _event: object) -> None:
            return None

        def __call__(self, _event: object) -> None:
            return None

    handler = Handler()
    with pytest.raises(UnsupportedSessionValue, match="bound callables"):
        persistence._encode_importable_callable(handler.bound, path="handler")
    with pytest.raises(UnsupportedSessionValue, match="no import address"):
        persistence._encode_importable_callable(handler, path="handler")

    def main_handler(_event: object) -> None:
        return None

    monkeypatch.setattr(main_handler, "__module__", "__main__")
    monkeypatch.setattr(main_handler, "__qualname__", "main_handler")
    with pytest.raises(UnsupportedSessionValue, match="__main__"):
        persistence._encode_importable_callable(main_handler, path="handler")

    def missing_handler(_event: object) -> None:
        return None

    monkeypatch.setattr(missing_handler, "__module__", "missing_hook_module")
    monkeypatch.setattr(missing_handler, "__qualname__", "handler")
    with pytest.raises(UnsupportedSessionValue, match="cannot resolve callable"):
        persistence._encode_importable_callable(missing_handler, path="handler")

    def impostor(_event: object) -> None:
        return None

    monkeypatch.setattr(impostor, "__module__", "operator")
    monkeypatch.setattr(impostor, "__qualname__", "neg")
    with pytest.raises(UnsupportedSessionValue, match="same callable"):
        persistence._encode_importable_callable(impostor, path="handler")


@pytest.mark.parametrize(
    ("descriptor", "message"),
    [
        (42, "callable descriptor must be an object"),
        ({"module": "operator"}, "requires module and qualname"),
        ({"module": "__main__", "qualname": "handler"}, "unsafe callable"),
        ({"module": "operator", "qualname": "<locals>"}, "unsafe callable"),
    ],
)
def test_session_hook_callable_decoder_rejects_malformed_descriptors(
    descriptor: Any,
    message: str,
) -> None:
    with pytest.raises(SessionLoadError, match=message):
        persistence._decode_importable_callable(descriptor, path="handler")


@pytest.mark.parametrize(
    ("module", "qualname", "message"),
    [
        ("operator", ".neg", "cannot resolve callable"),
        ("operator", "missing", "cannot resolve callable"),
        ("math", "pi", "not callable"),
    ],
)
def test_session_hook_callable_resolution_rejects_invalid_targets(
    module: str,
    qualname: str,
    message: str,
) -> None:
    with pytest.raises(SessionLoadError, match=message):
        persistence._resolve_importable_callable(module, qualname, path="handler")


def test_session_dump_wraps_late_value_encoding_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_encoder = state.SessionValueEncoder
    instances = 0

    class FailingSecondEncoder(real_encoder):
        def __init__(self, *, kedi_models: Any) -> None:
            nonlocal instances
            super().__init__(kedi_models=kedi_models)
            instances += 1
            self._instance_number = instances

        def encode(self, value: Any, *, path: str = "value", depth: int = 0) -> Any:
            if self._instance_number == 2:
                raise UnsupportedSessionValue(path, value, "late failure")
            return super().encode(value, path=path, depth=depth)

    monkeypatch.setattr(state, "SessionValueEncoder", FailingSecondEncoder)
    with kedi.InteractiveSession() as session:
        with pytest.raises(SessionDumpError, match="late failure"):
            session.dump(tmp_path / "rejected.kedi-state")


def test_session_load_rejects_unsafe_fragments_bad_digests_and_environment(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "base.kedi-state"
    with kedi.InteractiveSession() as session:
        session.dump(snapshot)
    payload = persistence.read_snapshot(snapshot)

    bad_digest = dict(payload)
    bad_digest["fragments"] = [
        {"source": "[value: int] = `1`", "source_name": "cell.kedi", "digest": "wrong"}
    ]
    digest_snapshot = tmp_path / "bad-digest.kedi-state"
    persistence.write_snapshot(digest_snapshot, bad_digest)
    with pytest.raises(SessionLoadError, match="failed its integrity check"):
        kedi.load_session(digest_snapshot)

    inline_source = "```\nvalue = 1\n```"
    unsafe = dict(payload)
    unsafe["fragments"] = [
        {
            "source": inline_source,
            "source_name": "cell.kedi",
            "digest": persistence.hashlib.sha256(inline_source.encode()).hexdigest(),
        }
    ]
    unsafe_snapshot = tmp_path / "unsafe.kedi-state"
    persistence.write_snapshot(unsafe_snapshot, unsafe)
    with pytest.raises(SessionLoadError, match="Unsafe fragment in snapshot"):
        kedi.load_session(unsafe_snapshot)

    invalid_environment = dict(payload)
    invalid_environment["environment"] = 42
    environment_snapshot = tmp_path / "invalid-environment.kedi-state"
    persistence.write_snapshot(environment_snapshot, invalid_environment)
    with pytest.raises(SessionLoadError, match="string-keyed mapping"):
        kedi.load_session(environment_snapshot)


def test_conversation_boundary_reports_every_nonportable_lane() -> None:
    assert state._conversation_persistence_issues(None) == []

    conversation = ConversationState()
    conversation._closed = True
    conversation._adapter_states["pydantic"] = {"state": True}
    conversation._adapter_cleanups["pydantic"] = lambda: None
    conversation._cache_epoch = 1
    conversation.history.append("artifact_created", ref_id="artifact")
    with conversation.artifacts._lock:
        conversation.artifacts._ref_stores["artifact"] = cast(Any, object())

    issues = state._conversation_persistence_issues(conversation)
    assert "conversation state is closed" in issues
    assert "conversation contains adapter-native continuation state" in issues
    assert "conversation owns process-local adapter cleanup callbacks" in issues
    assert "conversation cache epoch is not initial" in issues
    assert "conversation contains artifact history" in issues
    assert "conversation contains active or retained artifacts" in issues


def test_fragment_boundary_walks_runtime_scoped_declarations() -> None:
    program = SimpleNamespace(
        imports=[],
        prelude=None,
        type_defs=[],
        ordered_toplevel=[
            IfStmt(
                body=[ProcDefStmt()],
                else_body=[TypeDefStmt()],
            )
        ],
        toplevel_body=[],
    )
    issues = state._fragment_persistence_issues(program, "nested.kedi")
    assert "runtime-scoped procedure definition" in issues[0]
    assert "runtime-scoped type definition" in issues[1]


def test_kedi_models_filters_hidden_and_non_model_types() -> None:
    compiler = SimpleNamespace(
        rt=SimpleNamespace(
            _type_env_stack=[
                {
                    "__hidden": SnapshotModel,
                    "plain": int,
                    "SnapshotModel": SnapshotModel,
                }
            ]
        )
    )
    assert state._kedi_models(cast(Any, compiler)) == {"SnapshotModel": SnapshotModel}
