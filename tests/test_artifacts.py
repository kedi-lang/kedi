from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from kedi import ArtifactPolicy as PublicArtifactPolicy
from kedi.agent_adapter import PydanticAdapter
from kedi.agent_adapter.approvals.policy import ApprovalDecision, ApprovalPolicy, ToolApprovalDenied
from kedi.agent_adapter.conversation import (
    AdapterConversationContext,
    ConversationState,
)
from kedi.agent_adapter.meta import AdapterCapabilities
from kedi.agent_adapter.tools.execution import with_approval, with_artifact_result
from kedi.agent_adapter.tools.scope import defer_artifact_admission
from kedi.agent_adapter.tools.specs import ToolSpec
from kedi.artifacts import (
    ArtifactAccessError,
    ArtifactExpiredError,
    ArtifactHandle,
    ArtifactHistory,
    ArtifactInputError,
    ArtifactManager,
    ArtifactPolicy,
    ArtifactPolicyError,
    ArtifactQuotaExceededError,
    ArtifactRef,
    ArtifactReleasedError,
    ArtifactSerializationError,
    ArtifactStatus,
    FileArtifactStore,
    artifact_instructions,
    deserialize_artifact,
    estimate_object_size,
    normalize_artifact_policy,
    parse_byte_size,
    parse_duration,
    render_artifact_content,
    serialize_artifact,
)
from kedi.engine import ThreadEngine
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program
from kedi.lang.ast import ArtifactsDirectiveStmt, ArtifactsField
from kedi.lang.parser.diagnostics import KediParseError
from kedi.utils import AttrDict
from tests.mock_adapter import MockAdapter


class Clock:
    def __init__(self) -> None:
        self.monotonic = 0.0
        self.wall = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def tick(self, seconds: float) -> None:
        self.monotonic += seconds
        self.wall += timedelta(seconds=seconds)


class Person(BaseModel):
    name: str


@dataclass
class Payload:
    value: int


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def manager(clock: Clock) -> Iterator[ArtifactManager]:
    value = ArtifactManager(
        session_id="session_test",
        monotonic=lambda: clock.monotonic,
        wall_clock=lambda: clock.wall,
    )
    yield value
    value.close()


def artifact_policy(**values: object) -> ArtifactPolicy:
    return ArtifactPolicy.patch(
        {
            "enabled": True,
            "threshold": 1,
            "cleanup_interval": "1h",
            **values,
        }
    ).resolved()


def test_policy_parses_units_and_overlays_only_specified_fields(tmp_path: Path) -> None:
    base = ArtifactPolicy.patch(
        {
            "enabled": True,
            "store": "file",
            "path": tmp_path,
            "threshold": "100kb",
            "ttl": "1h",
            "idle_ttl": "30m",
            "session_quota": "2MiB",
            "cleanup_interval": "250ms",
        }
    )
    result = normalize_artifact_policy({"preview_chars": 42}, base=base)

    assert result is not None
    assert result.enabled is True
    assert result.store == "file"
    assert result.threshold_bytes == 100_000
    assert result.ttl_seconds == 3_600
    assert result.idle_ttl_seconds == 1_800
    assert result.session_quota_bytes == 2_097_152
    assert result.cleanup_interval_seconds == 0.25
    assert result.preview_chars == 42
    assert parse_byte_size(5.0) == 5
    assert parse_duration(timedelta(seconds=2)) == 2
    disabled = normalize_artifact_policy(False, base=base)
    assert disabled is not None
    assert disabled.enabled is False
    assert normalize_artifact_policy(None, base=base) is base
    assert PublicArtifactPolicy is ArtifactPolicy


def test_artifacts_are_enabled_by_default_with_cache_stable_contract() -> None:
    class LargeOutputAdapter(ArtifactRecordingAdapter):
        def produce_sync(
            self,
            *,
            template: str,
            output_schema: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> AttrDict[str, Any]:
            super().produce_sync(
                template=template,
                output_schema=output_schema,
                **kwargs,
            )
            return AttrDict({"payload": "x" * 100_000})

    adapter = LargeOutputAdapter()
    runtime = compile_program(
        parse_program(">> A large value is [payload].\n= `payload`\n"),
        adapter=adapter,
    )

    result = runtime.run_main()

    assert ArtifactPolicy().enabled is True
    assert result == "x" * 100_000
    assert runtime.artifacts.has_active_artifacts()
    assert adapter.tool_calls == [
        ("read_artifact", "search_artifacts", "release_artifact", "run_artifact_code")
    ]
    assert "<ARTIFACT_INSTRUCTIONS>" in (adapter.instruction_calls[0] or "")
    runtime.artifacts.close()


def test_artifact_contract_routes_simple_and_complex_inspection() -> None:
    instructions = artifact_instructions()
    usage_hint = ArtifactRef.model_fields["usage_hint"].default

    assert "read_artifact for a known literal or bounded inspection" in instructions
    assert "filtering, aggregation, joins, or multi-artifact transformations" in instructions
    assert "use the exact ref_id returned by each tool" in instructions
    assert "never infer identity or order from its suffix" in instructions
    assert "do not call artifact tools or search for" in instructions
    assert "before a ref appears in the conversation" in instructions
    assert "task_id, or context_id is not an artifact ref_id" in instructions
    assert "release is not required before answering" in instructions
    assert "pattern searches literal text without paging" in instructions
    assert "targeted search instead of paging from the beginning" in instructions
    assert "when sequential coverage is needed" in instructions
    assert "query_artifact" not in instructions
    assert "Use query_artifact" in artifact_instructions(query_artifact=True)
    assert isinstance(usage_hint, str)
    assert "Copy this exact ref_id" in usage_hint
    assert "read_artifact for a known literal (pattern)" in usage_hint
    assert "if available, query_artifact" in usage_hint
    assert "release is optional" in usage_hint
    assert "filtering, aggregation, joins, or multi-artifact transformations" in usage_hint


def test_default_artifact_contract_is_stable_for_mcp_only_calls() -> None:
    adapter = MockAdapter()
    runtime = compile_program(
        parse_program(
            "> mcp:\n"
            "  transport: http\n"
            "  url: http://localhost:8000/mcp\n"
            "\n"
            ">> Return [answer].\n"
            "= <answer>\n"
        ),
        adapter=adapter,
    )

    assert runtime.run_main() == "answer-value"
    assert adapter.tool_calls == [
        ("read_artifact", "search_artifacts", "release_artifact", "run_artifact_code")
    ]
    assert "<ARTIFACT_INSTRUCTIONS>" in (adapter.instruction_calls[0] or "")


def test_default_artifact_contract_does_not_change_between_model_turns() -> None:
    adapter = ArtifactRecordingAdapter()
    runtime = compile_program(
        parse_program("[first] << First turn.\n[second] << Second turn.\n= <first> <second>\n"),
        adapter=adapter,
    )

    assert runtime.run_main() == "response:First turn.response:Second turn."
    assert len(adapter.instruction_calls) == 2
    assert adapter.instruction_calls[0] == adapter.instruction_calls[1]
    assert adapter.tool_calls == [
        ("read_artifact", "search_artifacts", "release_artifact", "run_artifact_code"),
        ("read_artifact", "search_artifacts", "release_artifact", "run_artifact_code"),
    ]
    runtime.artifacts.close()


def test_default_artifacts_do_not_require_contract_from_unsupported_adapter() -> None:
    class TextOnlyAdapter(MockAdapter):
        capabilities = AdapterCapabilities(
            supports_structured_output=True,
            supports_tool_registration=True,
            supports_profile_override=True,
        )

    adapter = TextOnlyAdapter()
    runtime = compile_program(
        parse_program("[answer] << A plain text turn.\n= <answer>\n"),
        adapter=adapter,
    )

    assert runtime.run_main() == "response:A plain text turn."
    assert adapter.tool_calls == [()]
    assert adapter.instruction_calls == [None]
    runtime.artifacts.close()


def test_artifacts_directive_parses_top_level_and_profile_members() -> None:
    program = parse_program(
        "> artifacts:\n"
        "  enabled: true\n"
        "  threshold: 100kb\n"
        "\n"
        "> profile: compact:\n"
        "  > artifacts:\n"
        "    preview_chars: `40 + 2`\n"
    )

    directive = program.toplevel_body[0]
    profile_field = program.profile_defs["compact"].artifacts_members[0].fields[0]
    assert isinstance(directive, ArtifactsDirectiveStmt)
    assert directive.fields[0].plain_value == "true"
    assert directive.fields[1].plain_value == "100kb"
    assert profile_field.name == "preview_chars"
    assert profile_field.backtick_code == "40 + 2"


def test_artifact_documentation_example_parses() -> None:
    source = Path("examples/artifact_investigation/main.kedi").read_text(encoding="utf-8")

    parse_program(source)


def test_artifact_directives_apply_inside_procedures_and_profiles() -> None:
    adapter = ArtifactRecordingAdapter()
    runtime = compile_program(
        parse_program(
            "> artifacts:\n"
            "  enabled: true\n"
            "  threshold: 1b\n"
            "\n"
            "> profile: compact:\n"
            "  > artifacts:\n"
            "    enabled: true\n"
            "    threshold: `1`\n"
            "\n"
            "@profiled():\n"
            "  > use: compact\n"
            "  >> Give [profile_value].\n"
            "  = `profile_value`\n"
            "\n"
            "@local():\n"
            "  > artifacts:\n"
            "    enabled: true\n"
            "    threshold: 1b\n"
            "  >> Give [local_value].\n"
            "  = `local_value`\n"
            "\n"
            "= `profiled(), local()`\n"
        ),
        adapter=adapter,
    )

    assert runtime.run_main() == ("profile_value-value", "local_value-value")
    assert runtime.artifacts.stats()["active_artifacts"] == 2
    runtime.artifacts.close()


def test_artifact_directive_accepts_comments() -> None:
    program = parse_program(
        "> artifacts:\n"
        "  # Keep policy documentation beside the field.\n"
        "  enabled: true\n"
        "  ###\n"
        "  block comment\n"
        "  ###\n"
        "  threshold: 1b\n"
    )

    directive = program.toplevel_body[0]
    assert isinstance(directive, ArtifactsDirectiveStmt)
    assert [field.name for field in directive.fields] == ["enabled", "threshold"]


def test_artifact_profile_evaluation_rejects_missing_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kedi.agent_profile.directives as profile_model

    assert (
        profile_model.eval_artifacts_field_value(
            object(),  # type: ignore[arg-type]
            ArtifactsField(name="enabled"),
            {},
        )
        is None
    )
    monkeypatch.setattr(profile_model, "normalize_artifact_policy", lambda values: None)
    with pytest.raises(KediExecutionError, match="did not produce a policy"):
        profile_model.eval_artifacts_fields(object(), (), {})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("source", "code"),
    [
        (
            "> artifacts:\n  enabled: true\n  enabled: false\n",
            "K-ARTIFACTS-DUP",
        ),
        (
            "> artifacts:\n  mystery: true\n",
            "K-ARTIFACTS-FIELD",
        ),
        (
            "> artifacts:\n  threshold: lots\n",
            "K-ARTIFACTS-VALUE",
        ),
        (
            "> artifacts:\n  enabled: maybe\n",
            "K-ARTIFACTS-VALUE",
        ),
        (
            "> artifacts:\n  max_artifacts: 0\n",
            "K-ARTIFACTS-VALUE",
        ),
    ],
)
def test_artifacts_directive_rejects_invalid_fields(source: str, code: str) -> None:
    with pytest.raises(KediParseError) as exc_info:
        parse_program(source)

    assert any(diagnostic.code == code for diagnostic in exc_info.value.diagnostics)


def test_artifact_history_is_append_only_across_release() -> None:
    history = ArtifactHistory()
    history.append("artifact_created", ref_id="artifact_1")
    history.append("artifact_read", ref_id="artifact_1")
    before_release = history.snapshot()

    history.append("artifact_released", ref_id="artifact_1")

    after_release = history.snapshot()
    assert after_release[: len(before_release)] == before_release
    assert after_release[-1].kind == "artifact_released"


class ArtifactRecordingAdapter(MockAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.templates: list[str] = []

    def produce_sync(
        self,
        *,
        template: str,
        output_schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AttrDict[str, Any]:
        self.templates.append(template)
        return super().produce_sync(
            template=template,
            output_schema=output_schema,
            **kwargs,
        )


class ConversationAdapter(MockAdapter):
    capabilities = AdapterCapabilities(
        supports_structured_output=True,
        supports_tool_registration=True,
        supports_profile_override=True,
        supports_artifacts=True,
        supports_stateful_history=True,
        supports_history_replay=True,
    )

    def __init__(self) -> None:
        super().__init__()
        self._conversation: ContextVar[AdapterConversationContext | None] = ContextVar(
            "test_artifact_conversation",
            default=None,
        )
        self.prompts: list[str] = []
        self.resume_states: list[dict[str, Any]] = []
        self.active_calls = 0
        self.max_active_calls = 0
        self.fail = False

    @contextmanager
    def conversation_scope(
        self,
        conversation: AdapterConversationContext,
    ) -> Iterator[None]:
        token = self._conversation.set(conversation)
        try:
            yield
        finally:
            self._conversation.reset(token)

    def invoke_sync(self, *, prompt: str, **kwargs: Any) -> str:
        conversation = self._conversation.get()
        assert conversation is not None
        with self._record_lock:
            self.prompts.append(prompt)
            self.resume_states.append(dict(conversation.resume_state))
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            if self.fail:
                raise RuntimeError("conversation failure")
            time.sleep(0.01)
            turn = int(conversation.resume_state.get("turn", 0)) + 1
            conversation.result_state["turn"] = turn
            return f"response-{turn}"
        finally:
            with self._record_lock:
                self.active_calls -= 1


def test_runtime_returns_native_output_but_reuses_compact_model_reference() -> None:
    adapter = ArtifactRecordingAdapter()
    runtime = compile_program(
        parse_program(
            "> artifacts:\n"
            "  enabled: true\n"
            "  threshold: 1b\n"
            "  preview_chars: 8\n"
            "\n"
            ">> Give [payload].\n"
            "\n"
            ">> Describe <payload> as [summary].\n"
            "= `payload`\n"
        ),
        adapter=adapter,
    )

    result = runtime.run_main()

    assert result == "payload-value"
    assert "payload-value" not in adapter.templates[1]
    assert '"ref_id": "template_output_1"' in adapter.templates[1]
    assert '"preview": "payload…"' in adapter.templates[1]
    assert runtime.artifacts.stats()["active_artifacts"] == 2
    runtime.artifacts.close()


def test_inline_python_receives_native_artifact_value() -> None:
    runtime = compile_program(
        parse_program(
            "> artifacts:\n"
            "  enabled: true\n"
            "  threshold: 1b\n"
            "\n"
            ">> Give [payload].\n"
            "= `payload.upper()`\n"
        ),
        adapter=MockAdapter(),
    )

    assert runtime.run_main() == "PAYLOAD-VALUE"
    runtime.artifacts.close()


def test_adapter_output_is_validated_before_artifact_storage() -> None:
    class InvalidAdapter(MockAdapter):
        def produce_sync(
            self,
            *,
            template: str,
            output_schema: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> AttrDict[str, Any]:
            return AttrDict({"count": "not-an-integer"})

    runtime = compile_program(
        parse_program("> artifacts:\n  enabled: true\n  threshold: 1b\n\n>> Count [count: int].\n"),
        adapter=InvalidAdapter(),
    )

    with pytest.raises(Exception):
        runtime.run_main()
    assert runtime.artifacts.stats()["active_artifacts"] == 0
    runtime.artifacts.close()


def test_parallel_outputs_are_artifacted_once_each() -> None:
    runtime = compile_program(
        parse_program(
            "> artifacts:\n"
            "  enabled: true\n"
            "  threshold: 1b\n"
            "\n"
            ">> Give [first].\n"
            "\n"
            ">> Give [second].\n"
            "= `second`\n"
        ),
        adapter=MockAdapter(),
        engine=ThreadEngine(2),
    )

    assert runtime.run_main() == "second-value"
    assert runtime.artifacts.stats()["active_artifacts"] == 2
    runtime.artifacts.close()


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"unknown": True}, "unknown artifact policy field"),
        ({"store": "database"}, "store must be"),
        ({"threshold": "1widget"}, "unsupported threshold_bytes unit"),
        ({"ttl": "none"}, "cannot be none"),
        ({"max_artifacts": 0}, "positive integer"),
        ({"enabled": "yes"}, "boolean"),
    ],
)
def test_policy_rejects_invalid_values(value: dict[str, object], message: str) -> None:
    with pytest.raises(ArtifactPolicyError, match=message):
        ArtifactPolicy.patch(value)


def test_serialization_round_trips_supported_values() -> None:
    text = serialize_artifact("hello", preview_chars=10)
    binary = serialize_artifact(b"\x00\x01", preview_chars=10)
    model = serialize_artifact(Person(name="Ada"), preview_chars=10)
    dataclass_value = serialize_artifact(Payload(value=3), preview_chars=10)

    assert deserialize_artifact(text.codec, text.payload_bytes or b"") == "hello"
    assert deserialize_artifact(binary.codec, binary.payload_bytes or b"") == b"\x00\x01"
    assert model.memory_value == Person(name="Ada")
    assert dataclass_value.memory_value == Payload(value=3)
    assert render_artifact_content("json", {"value": 3}) == '{\n  "value": 3\n}'
    assert text.digest is not None
    assert binary.digest is not None


def test_serialization_snapshots_mutable_json_values_deterministically() -> None:
    value = {"items": ["é", 2], "enabled": True}
    first = serialize_artifact(value, preview_chars=100)
    second = serialize_artifact(
        {"enabled": True, "items": ["é", 2]},
        preview_chars=100,
    )
    value["items"].append(3)

    assert first.memory_value == {"items": ["é", 2], "enabled": True}
    assert first.payload_bytes == second.payload_bytes
    assert first.digest == second.digest
    assert first.size_bytes == len(first.payload_bytes or b"")
    assert first.size_bytes > len((first.payload_bytes or b"").decode("utf-8"))


def test_serialization_treats_cycles_and_non_finite_json_as_opaque() -> None:
    cyclic: list[object] = []
    cyclic.append(cyclic)

    serialized_cycle = serialize_artifact(cyclic, preview_chars=100)
    serialized_nan = serialize_artifact(float("nan"), preview_chars=100)

    assert serialized_cycle.codec == "opaque"
    assert serialized_cycle.memory_value is cyclic
    assert serialized_cycle.preview == "<opaque builtins.list>"
    assert serialized_nan.codec == "opaque"
    with pytest.raises(ArtifactSerializationError):
        serialize_artifact(cyclic, preview_chars=100, require_persistable=True)


def test_opaque_preview_does_not_execute_user_repr() -> None:
    class DangerousRepr:
        def __repr__(self) -> str:
            raise AssertionError("user repr must not execute")

    serialized = serialize_artifact(DangerousRepr(), preview_chars=200)

    assert serialized.preview.endswith(".DangerousRepr>")
    assert "Python object" in serialized.summary


def test_opaque_memory_values_are_not_persistable_or_deduplicated() -> None:
    value = object()
    serialized = serialize_artifact(value, preview_chars=20)

    assert serialized.codec == "opaque"
    assert serialized.persistable is False
    assert serialized.digest is None
    assert estimate_object_size(value) > 0
    with pytest.raises(ArtifactSerializationError, match="cannot be safely stored"):
        serialize_artifact(value, preview_chars=20, require_persistable=True)


def test_object_size_handles_cycles() -> None:
    value: list[object] = []
    value.append(value)
    assert estimate_object_size(value) > 0
    assert estimate_object_size([1, 2], max_nodes=1) == sys.maxsize


def test_manager_artifacts_only_values_at_or_above_threshold(
    manager: ArtifactManager,
) -> None:
    policy = artifact_policy(threshold=5)

    assert manager.artifactize("1234", policy=policy) == "1234"
    artifact = manager.artifactize("12345", policy=policy, source="field:value")

    assert isinstance(artifact, ArtifactHandle)
    assert artifact.get_value() == "12345"
    assert artifact.as_ref().size_bytes == 5
    assert "run_artifact_code" in artifact.as_ref().usage_hint
    assert "read_artifact for a known literal" in artifact.as_ref().usage_hint


def test_manager_keeps_equal_payloads_as_distinct_logical_artifacts(
    manager: ArtifactManager,
) -> None:
    policy = artifact_policy(max_artifacts=2, session_quota=10)
    first = manager.artifactize(
        "12345",
        policy=policy,
        source="tool:first",
        sensitive=False,
        provenance={"owner": "first"},
    )
    second = manager.artifactize(
        "12345",
        policy=policy,
        source="tool:second",
        sensitive=True,
        provenance={"owner": "second"},
    )

    assert isinstance(first, ArtifactHandle)
    assert isinstance(second, ArtifactHandle)
    assert first.ref.ref_id != second.ref.ref_id
    first_metadata = manager.metadata(first.ref.ref_id)
    second_metadata = manager.metadata(second.ref.ref_id)
    assert first_metadata["source"] == "tool:first"
    assert second_metadata["source"] == "tool:second"
    assert first_metadata["sensitive"] is False
    assert second_metadata["sensitive"] is True
    assert first_metadata["provenance"] == {"owner": "first"}
    assert second_metadata["provenance"] == {"owner": "second"}

    with pytest.raises(ArtifactQuotaExceededError, match="count"):
        manager.artifactize("other", policy=policy)
    manager.release(first.ref.ref_id)
    assert second.get_value() == "12345"


def test_manager_reads_searches_and_releases(manager: ArtifactManager) -> None:
    artifact = manager.artifactize(
        {"person": {"name": "Ada"}, "items": [1, 2]},
        policy=artifact_policy(read_max_chars=8),
        source="tool:lookup",
    )
    assert isinstance(artifact, ArtifactHandle)

    chunk = manager.read(artifact.ref.ref_id, max_chars=100, path="/person/name")
    results = manager.search("lookup")
    released = manager.release(artifact.ref.ref_id)

    assert chunk.content == '"Ada"'
    assert chunk.complete is True
    assert results[0].ref_id == artifact.ref.ref_id
    assert released.status is ArtifactStatus.RELEASED
    assert [event.kind for event in manager.history.snapshot()] == [
        "artifact_created",
        "artifact_read",
        "artifact_released",
    ]
    with pytest.raises(ArtifactReleasedError):
        artifact.get_value()
    assert manager.release(artifact.ref.ref_id).already_released is True


def test_manager_expires_on_fixed_and_idle_deadlines(
    manager: ArtifactManager,
    clock: Clock,
) -> None:
    fixed = manager.artifactize("fixed", policy=artifact_policy(ttl="2s"))
    idle = manager.artifactize(
        "idle",
        policy=artifact_policy(ttl="1h", idle_ttl="1s"),
    )
    assert isinstance(fixed, ArtifactHandle)
    assert isinstance(idle, ArtifactHandle)

    clock.tick(1.5)
    manager.cleanup_expired()
    with pytest.raises(ArtifactExpiredError):
        idle.get_value()
    assert fixed.get_value() == "fixed"
    clock.tick(0.6)
    manager.cleanup_expired()
    with pytest.raises(ArtifactExpiredError):
        fixed.get_value()


def test_release_waits_for_active_reader(manager: ArtifactManager) -> None:
    artifact = manager.artifactize("payload", policy=artifact_policy())
    assert isinstance(artifact, ArtifactHandle)

    with manager._lease(artifact.ref.ref_id):
        result = manager.release(artifact.ref.ref_id)
        assert result.status is ArtifactStatus.PENDING_RELEASE

    with pytest.raises(ArtifactReleasedError):
        artifact.get_value()


def test_file_store_restores_session_and_fixed_expiry(
    tmp_path: Path,
    clock: Clock,
) -> None:
    policy = artifact_policy(store="file", path=tmp_path, ttl="10s")
    first = ArtifactManager(
        session_id="session_restart",
        monotonic=lambda: clock.monotonic,
        wall_clock=lambda: clock.wall,
    )
    artifact = first.artifactize("persisted", policy=policy)
    assert isinstance(artifact, ArtifactHandle)
    ref_id = artifact.ref.ref_id
    first.close()

    clock.tick(4)
    second = ArtifactManager(
        session_id="session_restart",
        monotonic=lambda: clock.monotonic,
        wall_clock=lambda: clock.wall,
    )
    duplicate = second.artifactize("persisted", policy=policy)
    assert isinstance(duplicate, ArtifactHandle)
    assert duplicate.ref.ref_id != ref_id
    assert second.get_value(ref_id) == "persisted"
    assert duplicate.get_value() == "persisted"
    clock.tick(7)
    assert second.cleanup_expired() == 1
    with pytest.raises(ArtifactExpiredError):
        second.get_value(ref_id)
    assert duplicate.get_value() == "persisted"
    second.close()


def test_file_store_rejects_symlink_root(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(ArtifactAccessError, match="symlink"):
        FileArtifactStore(link)


def test_manager_rejects_invalid_json_pointer(manager: ArtifactManager) -> None:
    artifact = manager.artifactize({"value": 1}, policy=artifact_policy())
    assert isinstance(artifact, ArtifactHandle)

    with pytest.raises(ValueError, match="RFC 6901"):
        manager.read(artifact.ref.ref_id, path="value")
    with pytest.raises(ArtifactInputError, match="was not found"):
        manager.read(artifact.ref.ref_id, path="/missing")


def test_closed_manager_rejects_new_artifacts(manager: ArtifactManager) -> None:
    manager.close()
    with pytest.raises(RuntimeError, match="closed"):
        manager.artifactize("value", policy=artifact_policy())


def test_tool_result_artifact_wrapper_preserves_approval_order_and_signature(
    manager: ArtifactManager,
) -> None:
    calls: list[str] = []

    def lookup(*, query: str) -> str:
        calls.append(query)
        return "large-result"

    approved = with_approval(
        ToolSpec(fn=lookup, name="lookup"),
        policy=ApprovalPolicy.allow(),
        adapter_shortname="test",
    )
    wrapped = with_artifact_result(
        approved,
        manager=manager,
        policy=artifact_policy(),
    )

    result = wrapped.fn(query="requested")

    assert calls == ["requested"]
    assert result["ref_id"] == "tool_call_result_1"
    assert manager.get_value(result["ref_id"]) == "large-result"
    assert wrapped.fn.__name__ == "lookup"
    assert wrapped.metadata == {"kedi_artifact_wrapped": True}
    assert (
        with_artifact_result(
            wrapped,
            manager=manager,
            policy=artifact_policy(),
        )
        is wrapped
    )


def test_approval_evaluates_defaulted_arguments_before_risk() -> None:
    calls: list[str] = []

    def read_file(*, path: str) -> str:
        calls.append(path)
        return path

    tool = with_approval(
        ToolSpec(
            fn=read_file,
            name="read_file",
            risk="mutating",
            risk_resolver=lambda values: "sensitive" if values["path"] == ".env" else "mutating",
            argument_validator=lambda values: {"path": values.get("path", ".env")},
        ),
        policy=None,
        adapter_shortname="test",
    )

    with pytest.raises(ToolApprovalDenied, match="no approval policy configured"):
        tool.fn()
    assert calls == []


def test_approval_rechecks_handler_edits_that_increase_risk() -> None:
    requests: list[tuple[str, str]] = []
    calls: list[str] = []

    def handler(request: Any) -> ApprovalDecision:
        requests.append((request.arguments["path"], request.risk))
        if len(requests) == 1:
            return ApprovalDecision.edit({"path": ".env"})
        return ApprovalDecision.deny(reason="secret path")

    def write_file(*, path: str) -> None:
        calls.append(path)

    tool = with_approval(
        ToolSpec(
            fn=write_file,
            name="write_file",
            risk="mutating",
            risk_resolver=lambda values: "sensitive" if values["path"] == ".env" else "mutating",
            argument_validator=lambda values: {"path": str(values["path"])},
        ),
        policy=ApprovalPolicy.dynamic(handler),
        adapter_shortname="test",
    )

    with pytest.raises(ToolApprovalDenied, match="secret path"):
        tool.fn(path="notes.txt")
    assert requests == [("notes.txt", "mutating"), (".env", "sensitive")]
    assert calls == []


def test_async_approval_rechecks_and_executes_canonical_edited_arguments() -> None:
    requests: list[tuple[str, str]] = []

    async def handler(request: Any) -> ApprovalDecision:
        requests.append((request.arguments["path"], request.risk))
        if len(requests) == 1:
            return ApprovalDecision.edit({"path": ".env"})
        return ApprovalDecision.allow()

    async def read_file(*, path: str) -> str:
        return path

    tool = with_approval(
        ToolSpec(
            fn=read_file,
            name="read_file",
            risk="mutating",
            risk_resolver=lambda values: "sensitive" if values["path"] == ".env" else "mutating",
            argument_validator=lambda values: {"path": str(values["path"])},
        ),
        policy=ApprovalPolicy.dynamic(handler),
        adapter_shortname="test",
    )

    assert asyncio.run(tool.fn(path="notes.txt")) == ".env"
    assert requests == [("notes.txt", "mutating"), (".env", "sensitive")]


def test_tool_result_artifact_wrapper_handles_async_and_awaitable_results(
    manager: ArtifactManager,
) -> None:
    async def async_tool(*, value: str) -> str:
        return value

    def awaitable_tool(*, value: str) -> Any:
        async def resolve() -> str:
            return value

        return resolve()

    policy = artifact_policy()
    async_spec = with_artifact_result(
        ToolSpec(fn=async_tool, name="async_tool"),
        manager=manager,
        policy=policy,
    )
    awaitable_spec = with_artifact_result(
        ToolSpec(fn=awaitable_tool, name="awaitable_tool"),
        manager=manager,
        policy=policy,
    )

    async_result = asyncio.run(async_spec.fn(value="async-result"))
    awaitable_result = asyncio.run(awaitable_spec.fn(value="awaitable-result"))

    assert manager.get_value(async_result["ref_id"]) == "async-result"
    assert manager.get_value(awaitable_result["ref_id"]) == "awaitable-result"


def test_tool_result_artifact_wrapper_can_defer_async_admission(
    manager: ArtifactManager,
) -> None:
    async def async_tool() -> str:
        return "async-raw"

    def awaitable_tool() -> Any:
        async def resolve() -> str:
            return "awaitable-raw"

        return resolve()

    policy = artifact_policy()
    async_spec = with_artifact_result(
        ToolSpec(async_tool, "async_tool"),
        manager=manager,
        policy=policy,
    )
    awaitable_spec = with_artifact_result(
        ToolSpec(awaitable_tool, "awaitable_tool"),
        manager=manager,
        policy=policy,
    )

    with defer_artifact_admission():
        assert asyncio.run(async_spec.fn()) == "async-raw"
        assert asyncio.run(awaitable_spec.fn()) == "awaitable-raw"
    assert manager.history.snapshot() == ()


def test_tool_result_artifact_wrapper_records_only_successful_approved_calls(
    manager: ArtifactManager,
) -> None:
    calls: list[str] = []

    def lookup(*, query: str) -> str:
        calls.append(query)
        return query

    denied = with_artifact_result(
        with_approval(
            ToolSpec(fn=lookup, name="lookup"),
            policy=ApprovalPolicy.deny(),
            adapter_shortname="test",
        ),
        manager=manager,
        policy=artifact_policy(),
    )
    with pytest.raises(ToolApprovalDenied):
        denied.fn(query="blocked")

    edited = with_artifact_result(
        with_approval(
            ToolSpec(fn=lookup, name="lookup"),
            policy=ApprovalPolicy.dynamic(
                lambda _request: ApprovalDecision.edit({"query": "edited"})
            ),
            adapter_shortname="test",
        ),
        manager=manager,
        policy=artifact_policy(),
    )
    result = edited.fn(query="original")

    assert calls == ["edited"]
    assert manager.get_value(result["ref_id"]) == "edited"
    events = manager.history.snapshot()
    assert [event.kind for event in events].count("tool_call") == 1
    assert [event.kind for event in events].count("tool_result") == 1


def test_tool_result_artifact_wrapper_does_not_record_failed_or_cancelled_calls(
    manager: ArtifactManager,
) -> None:
    def fail() -> str:
        raise LookupError("failed")

    async def cancel() -> str:
        raise asyncio.CancelledError

    failed = with_artifact_result(
        ToolSpec(fn=fail, name="fail"),
        manager=manager,
        policy=artifact_policy(),
    )
    cancelled = with_artifact_result(
        ToolSpec(fn=cancel, name="cancel"),
        manager=manager,
        policy=artifact_policy(),
    )

    with pytest.raises(LookupError, match="failed"):
        failed.fn()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(cancelled.fn())

    assert manager.history.snapshot() == ()
    assert manager.stats()["active_artifacts"] == 0


def test_tool_result_artifact_wrapper_honors_never_mode(
    manager: ArtifactManager,
) -> None:
    tool = ToolSpec(
        fn=lambda: "large-result",
        name="read_artifact",
        artifact_result="never",
    )

    assert (
        with_artifact_result(
            tool,
            manager=manager,
            policy=artifact_policy(),
        )
        is tool
    )
    assert tool.fn() == "large-result"
    assert manager.history.snapshot() == ()


def test_tool_result_wrapper_records_small_native_results_without_artifacting(
    manager: ArtifactManager,
) -> None:
    wrapped = with_artifact_result(
        ToolSpec(fn=lambda: "small", name="small"),
        manager=manager,
        policy=artifact_policy(threshold="1kb"),
    )

    assert wrapped.fn() == "small"
    events = manager.history.snapshot()
    assert [event.kind for event in events] == ["tool_call", "tool_result"]
    assert events[-1].data["artifacted"] is False


def test_tool_result_wrapper_returns_recoverable_quota_error(
    manager: ArtifactManager,
) -> None:
    wrapped = with_artifact_result(
        ToolSpec(fn=lambda: "oversized", name="load_all"),
        manager=manager,
        policy=artifact_policy(threshold="1b", session_quota=4),
    )

    result = wrapped.fn()

    assert result["ok"] is False
    assert result["error"] == "artifact_quota_exceeded"
    assert "active=0" in result["message"]
    assert "requested=9" in result["message"]
    assert "narrower query" in result["recovery"]
    events = manager.history.snapshot()
    assert [event.kind for event in events] == ["tool_call", "tool_result"]
    assert events[-1].data["error"] == "artifact_quota_exceeded"


def test_artifact_management_tools_are_registered_and_never_artifacted() -> None:
    adapter = ArtifactRecordingAdapter()
    runtime = compile_program(
        parse_program("> artifacts:\n  enabled: true\n  threshold: 1b\n\n>> Give [payload].\n"),
        adapter=adapter,
    )

    runtime.run_main()

    assert adapter.tool_calls == [
        ("read_artifact", "search_artifacts", "release_artifact", "run_artifact_code")
    ]
    assert "<ARTIFACT_INSTRUCTIONS>" in (adapter.instruction_calls[0] or "")
    runtime.artifacts.close()


def test_pydantic_agent_reads_compact_tool_artifact_before_final_output() -> None:
    calls: list[tuple[str, ...]] = []
    tool_returns: list[tuple[str, object]] = []

    async def respond(messages: list[object], info: AgentInfo) -> ModelResponse:
        returns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        calls.append(tuple(tool.name for tool in info.function_tools))
        tool_returns.extend((part.tool_name, part.content) for part in returns[len(tool_returns) :])

        if not returns:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "load_reference_material",
                        {},
                        tool_call_id="load-reference",
                    )
                ]
            )

        last = returns[-1]
        if last.tool_name == "load_reference_material":
            assert isinstance(last.content, dict)
            assert "VERIFIED_RELEASE_CODENAME" not in str(last.content)
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "read_artifact",
                        {
                            "ref_id": last.content["ref_id"],
                            "offset": 0,
                            "max_chars": 4000,
                        },
                        tool_call_id="read-reference",
                    )
                ]
            )

        assert last.tool_name == "read_artifact"
        assert "VERIFIED_RELEASE_CODENAME=Lale" in str(last.content)
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {"codename": "Lale"},
                    tool_call_id="final-output",
                )
            ]
        )

    source = """
> approval: allow
> artifacts:
    enabled: true
    threshold: 32b
    preview_chars: 16
    read_max_chars: 4000

@load_reference_material() -> str:
    = `("background " * 100) + "VERIFIED_RELEASE_CODENAME=Lale"`

> use: load_reference_material

>> Call load_reference_material. Read its returned artifact before filling [codename].
= <codename>
"""
    runtime = compile_program(
        parse_program(source),
        adapter=PydanticAdapter(FunctionModel(respond)),
    )

    assert runtime.run_main() == "Lale"
    assert [name for name, _content in tool_returns] == [
        "load_reference_material",
        "read_artifact",
    ]
    assert all(
        {
            "load_reference_material",
            "read_artifact",
            "search_artifacts",
            "release_artifact",
            "run_artifact_code",
        }
        <= set(call)
        for call in calls
    )
    events = runtime.artifacts.history.snapshot()
    assert sum(event.kind == "artifact_created" for event in events) == 1
    assert sum(event.kind == "artifact_read" for event in events) == 1
    runtime.artifacts.close()


def test_artifact_tools_require_adapter_capability() -> None:
    class UnsupportedAdapter(MockAdapter):
        capabilities = AdapterCapabilities(
            supports_structured_output=True,
            supports_tool_registration=True,
        )

    runtime = compile_program(
        parse_program("> artifacts:\n  enabled: true\n\n>> Give [payload].\n"),
        adapter=UnsupportedAdapter(),
    )
    with pytest.raises(KediExecutionError, match="does not support"):
        runtime.run_main()
    runtime.artifacts.close()


def test_native_approval_adapter_receives_unwrapped_artifact_tools() -> None:
    class NativeApprovalAdapter(ArtifactRecordingAdapter):
        capabilities = AdapterCapabilities(
            supports_structured_output=True,
            supports_tool_registration=True,
            supports_profile_override=True,
            supports_native_approvals=True,
            supports_artifacts=True,
        )

    adapter = NativeApprovalAdapter()
    runtime = compile_program(
        parse_program("> artifacts:\n  enabled: true\n\n>> Give [payload].\n"),
        adapter=adapter,
    )

    runtime.run_main()

    assert adapter.tool_calls == [
        ("read_artifact", "search_artifacts", "release_artifact", "run_artifact_code")
    ]
    runtime.artifacts.close()


def test_runtime_artifact_scope_covers_pydantic_constructor_tools() -> None:
    async def respond(messages: list[Any], _info: AgentInfo) -> ModelResponse:
        if not any(
            isinstance(part, ToolReturnPart) for message in messages for part in message.parts
        ):
            return ModelResponse(
                parts=[ToolCallPart("native_tool", {}, tool_call_id="native-call")]
            )
        return ModelResponse(parts=[TextPart("done")])

    def native_tool() -> str:
        return "runtime-native-result" * 20

    source = """
> approval: allow
> artifacts:
    enabled: true
    threshold: 10b

[answer] << Call native_tool once, then answer done.
= <answer>
"""
    runtime = compile_program(
        parse_program(source),
        adapter=PydanticAdapter(FunctionModel(respond), tools=[native_tool]),
    )

    assert runtime.run_main() == "done"
    assert runtime.artifacts.get_value("tool_call_result_1") == "runtime-native-result" * 20
    events = runtime.artifacts.history.snapshot()
    assert events[0].data["call_id"] == "native-call"
    assert events[-1].data == {
        "tool_name": "native_tool",
        "call_id": "native-call",
        "artifacted": True,
    }
    runtime.artifacts.close()


def test_disabled_artifacts_do_not_register_management_tools() -> None:
    adapter = ArtifactRecordingAdapter()
    runtime = compile_program(
        parse_program("> artifacts:\n  enabled: false\n\n>> Give [payload].\n"),
        adapter=adapter,
    )

    runtime.run_main()

    assert adapter.tool_calls == [()]
    assert adapter.instruction_calls == [None]
    runtime.artifacts.close()


def test_explicit_conversation_continues_state_across_runtimes() -> None:
    state = ConversationState(session_id="conversation_test")
    adapter = ConversationAdapter()

    first = compile_program(
        parse_program("[answer] << First turn.\n= <answer>\n"),
        adapter=adapter,
        conversation_state=state,
    )
    second = compile_program(
        parse_program("[answer] << Second turn.\n= <answer>\n"),
        adapter=adapter,
        conversation_state=state,
    )

    assert first.run_main() == "response-1"
    assert second.run_main() == "response-2"
    assert adapter.resume_states == [{}, {"turn": 1}]
    assert [turn.role for turn in state.snapshot()] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    json.dumps(state.as_jsonable())
    state.close()


def test_failed_conversation_turn_does_not_commit_checkpoint_or_history() -> None:
    state = ConversationState()
    adapter = ConversationAdapter()
    runtime = compile_program(
        parse_program("[answer] << Fail.\n= <answer>\n"),
        adapter=adapter,
        conversation_state=state,
    )
    adapter.fail = True

    with pytest.raises(Exception, match="conversation failure"):
        runtime.run_main()

    assert state.snapshot() == ()
    adapter.fail = False
    assert runtime.run_main() == "response-1"
    assert adapter.resume_states == [{}, {}]
    state.close()


def test_release_preserves_checkpoint_and_append_only_conversation_history() -> None:
    state = ConversationState()
    adapter = ConversationAdapter()
    artifact = state.artifacts.artifactize(
        "private-payload",
        policy=artifact_policy(),
    )
    assert isinstance(artifact, ArtifactHandle)
    ref_id = artifact.ref.ref_id
    state.record_exchange("store it", artifact.as_ref().model_dump(mode="json"))
    with state.adapter_turn(adapter.shortname) as conversation:
        conversation.result_state["turn"] = 7

    state.artifacts.release(ref_id)
    runtime = compile_program(
        parse_program("[answer] << Continue.\n= <answer>\n"),
        adapter=adapter,
        conversation_state=state,
    )

    turns_before_run = state.snapshot()

    assert runtime.run_main() == "response-8"
    assert adapter.resume_states == [{"turn": 7}]
    assert adapter.prompts == ["Continue."]
    assert state.snapshot()[: len(turns_before_run)] == turns_before_run
    assert [event.kind for event in state.history.snapshot()] == [
        "artifact_created",
        "artifact_released",
    ]
    state.close()


def test_conversation_serializes_concurrent_turns() -> None:
    state = ConversationState()
    adapter = ConversationAdapter()
    runtimes = [
        compile_program(
            parse_program(f"[answer] << Turn {index}.\n= <answer>\n"),
            adapter=adapter,
            conversation_state=state,
        )
        for index in range(2)
    ]
    results: list[str] = []

    threads = [
        threading.Thread(target=lambda runtime=runtime: results.append(runtime.run_main()))
        for runtime in runtimes
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(results) == ["response-1", "response-2"]
    assert adapter.max_active_calls == 1
    state.close()


def test_conversation_keeps_released_refs_in_immutable_prior_turns() -> None:
    state = ConversationState()
    first = state.artifacts.artifactize("first-secret", policy=artifact_policy())
    second = state.artifacts.artifactize("second-secret", policy=artifact_policy())
    assert isinstance(first, ArtifactHandle)
    assert isinstance(second, ArtifactHandle)
    first_ref = first.ref.ref_id
    second_ref = second.ref.ref_id
    state.record_exchange(
        "store both",
        {
            "first": first.as_ref().model_dump(mode="json"),
            "second": second.as_ref().model_dump(mode="json"),
        },
    )

    turns_before_release = state.snapshot()
    events_before_release = state.history.snapshot()
    state.artifacts.release(first_ref)
    with state.adapter_turn("test"):
        pass

    assistant_turn = state.snapshot()[-1]
    assert state.snapshot() == turns_before_release
    assert first_ref in assistant_turn.content
    assert second_ref in assistant_turn.content
    assert state.history.snapshot()[: len(events_before_release)] == events_before_release
    assert state.history.snapshot()[-1].kind == "artifact_released"
    state.close()


def test_closed_conversation_rejects_new_turns() -> None:
    state = ConversationState()
    state.close()

    with pytest.raises(RuntimeError, match="conversation state is closed"):
        with state.adapter_turn("test"):
            pass
