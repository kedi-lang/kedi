from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pytest

from kedi.artifacts import (
    ArtifactCodeError,
    ArtifactCodeLimits,
    ArtifactCodeRuntime,
    ArtifactHandle,
    ArtifactManager,
    ArtifactPolicy,
    artifact_tool_specs,
)
from kedi.artifacts import codemode as codemode_module
from kedi.artifacts.codemode import _ArtifactHostBridge


def _policy(**changes: object) -> ArtifactPolicy:
    values: dict[str, object] = {
        "enabled": True,
        "threshold": "1b",
        "read_max_chars": 32,
        "session_quota": "2mb",
    }
    values.update(changes)
    return ArtifactPolicy.patch(values).resolved()


def _artifact(
    manager: ArtifactManager,
    value: object,
    *,
    source: str = "tool:test",
    policy: ArtifactPolicy | None = None,
) -> ArtifactHandle[object]:
    result = manager.artifactize(value, policy=policy or _policy(), source=source)
    assert isinstance(result, ArtifactHandle)
    return result


def test_artifact_code_combines_allowed_artifacts_and_captures_stdout() -> None:
    manager = ArtifactManager(session_id="code-combine")
    text = _artifact(manager, "alpha\nbeta\ngamma")
    data = _artifact(manager, {"values": [1, 2, 3]})
    runtime = ArtifactCodeRuntime()
    try:
        result = runtime.execute(
            manager=manager,
            artifact_refs=[text.ref.ref_id, data.ref.ref_id],
            code=(
                f"text = ''.join(iter_artifact({text.ref.ref_id!r}, 5))\n"
                f"data = get_artifact({data.ref.ref_id!r})\n"
                "print('reduced')\n"
                "{'has_beta': 'beta' in text, 'total': sum(data['values'])}"
            ),
        )
    finally:
        runtime.close()
        manager.close()
    assert result.result == {"has_beta": True, "total": 6}
    assert result.stdout == "reduced\n"
    assert result.source_refs == (text.ref.ref_id, data.ref.ref_id)


def test_artifact_code_rejects_refs_outside_explicit_allowlist() -> None:
    manager = ArtifactManager(session_id="code-allowlist")
    allowed = _artifact(manager, "allowed")
    hidden = _artifact(manager, "hidden")
    runtime = ArtifactCodeRuntime()
    try:
        with pytest.raises(ArtifactCodeError, match="allowlist"):
            runtime.execute(
                manager=manager,
                artifact_refs=[allowed.ref.ref_id],
                code=f"read_artifact({hidden.ref.ref_id!r})",
            )
    finally:
        runtime.close()
        manager.close()


def test_artifact_code_get_requires_bounded_artifact() -> None:
    manager = ArtifactManager(session_id="code-materialize")
    artifact = _artifact(manager, "x" * 100)
    runtime = ArtifactCodeRuntime(
        limits=ArtifactCodeLimits(max_materialize_bytes=16),
    )
    try:
        with pytest.raises(ArtifactCodeError, match="use iter_artifact"):
            runtime.execute(
                manager=manager,
                artifact_refs=[artifact.ref.ref_id],
                code=f"get_artifact({artifact.ref.ref_id!r})",
            )
    finally:
        runtime.close()
        manager.close()


def test_artifact_code_enforces_read_and_stdout_budgets() -> None:
    manager = ArtifactManager(session_id="code-budgets")
    artifact = _artifact(manager, "x" * 100)
    read_runtime = ArtifactCodeRuntime(
        limits=ArtifactCodeLimits(max_read_chars=32, max_total_read_bytes=8),
    )
    try:
        with pytest.raises(ArtifactCodeError, match="read budget"):
            read_runtime.execute(
                manager=manager,
                artifact_refs=[artifact.ref.ref_id],
                code=f"read_artifact({artifact.ref.ref_id!r}, max_chars=20)",
            )
    finally:
        read_runtime.close()

    stdout_runtime = ArtifactCodeRuntime(limits=ArtifactCodeLimits(max_stdout_bytes=4))
    try:
        with pytest.raises(ArtifactCodeError, match="memory limit exceeded"):
            stdout_runtime.execute(
                manager=manager,
                artifact_refs=[],
                code="print('too long')\nNone",
            )
    finally:
        stdout_runtime.close()
        manager.close()


def test_artifact_code_failure_does_not_poison_reused_pool() -> None:
    manager = ArtifactManager(session_id="code-recovery")
    runtime = ArtifactCodeRuntime()
    try:
        with pytest.raises(ArtifactCodeError):
            runtime.execute(manager=manager, artifact_refs=[], code="1 / 0")
        assert runtime.execute(manager=manager, artifact_refs=[], code="6 * 7").result == 42
    finally:
        runtime.close()
        manager.close()


@pytest.mark.parametrize("operation", ["eval", "exec"])
def test_unsupported_dynamic_code_has_bounded_actionable_recovery(operation: str) -> None:
    manager = ArtifactManager()
    runtime = ArtifactCodeRuntime()
    try:
        with pytest.raises(ArtifactCodeError) as error:
            runtime.execute(manager=manager, artifact_refs=[], code=f"{operation}('1 + 1')")
        diagnostic = str(error.value)
        assert f"name '{operation}' is not defined" in diagnostic
        assert "direct expressions, loops, or comprehensions" in diagnostic
        assert "final expression" in diagnostic
        assert len(diagnostic) < 800
        assert error.value.__cause__ is not None
        assert runtime.execute(manager=manager, artifact_refs=[], code="1 + 1").result == 2
        spec = next(s for s in artifact_tool_specs(manager) if s.name == "run_artifact_code")
        assert codemode_module.ARTIFACT_COMPUTE_GUIDANCE in spec.description
    finally:
        runtime.close()
        manager.close()


def test_compute_provenance_tracks_transitive_depth_and_retains_released_parents() -> None:
    manager = ArtifactManager()
    source = _artifact(manager, "original")
    run = next(s for s in artifact_tool_specs(manager, _policy()) if s.name == "run_artifact_code")
    try:
        parent = source.ref.ref_id
        for depth in (1, 2, 3):
            result = run.fn(code=f"get_artifact({parent!r}) + '!'", artifact_refs=[parent])
            derived = result["result"]["ref_id"]
            metadata = manager.metadata(derived)
            assert metadata["provenance"]["derivation_depth"] == depth
            assert metadata["provenance"]["source_refs"] == [parent]
            manager.release(parent)
            assert manager.derivation_depth((parent,)) == depth
            assert manager.get_value(derived) == "original" + "!" * depth
            parent = derived
        assert manager.derivation_depth((source.ref.ref_id, parent)) == 4
        assert manager.derivation_depth(()) == 1
    finally:
        manager.close()


@pytest.mark.parametrize("depth", [-1, True, "2"])
def test_compute_rejects_invalid_parent_provenance_depth(depth: object) -> None:
    manager = ArtifactManager()
    try:
        source = manager.artifactize(
            "payload", policy=_policy(), provenance={"derivation_depth": depth}
        )
        with pytest.raises(ValueError, match="derivation_depth"):
            manager.derivation_depth((source.ref.ref_id,))
    finally:
        manager.close()


@pytest.mark.parametrize("code", ["open('secret')", "import os\nos.getcwd()"])
def test_artifact_code_has_no_host_filesystem_or_imports(code: str) -> None:
    manager = ArtifactManager(session_id="code-isolation")
    runtime = ArtifactCodeRuntime()
    try:
        with pytest.raises(ArtifactCodeError):
            runtime.execute(manager=manager, artifact_refs=[], code=code)
    finally:
        runtime.close()
        manager.close()


def test_run_artifact_code_tool_recursively_artifactizes_large_result(tmp_path: Path) -> None:
    manager = ArtifactManager(session_id="code-tool")
    source = _artifact(manager, "abcdefgh", policy=_policy(store="file", path=tmp_path))
    specs = artifact_tool_specs(manager, _policy(store="file", path=tmp_path, threshold="4b"))
    run = next(spec for spec in specs if spec.name == "run_artifact_code")
    try:
        result = run.fn(
            code=f"get_artifact({source.ref.ref_id!r}) * 2",
            artifact_refs=[source.ref.ref_id],
        )
        ref = result["result"]
        assert ref["ref_id"] == "artifact_code_result_1"
        assert manager.get_value(ref["ref_id"]) == "abcdefghabcdefgh"
        provenance = manager.metadata(ref["ref_id"])["provenance"]
        assert provenance == {
            "operation": "run_artifact_code",
            "source_refs": [source.ref.ref_id],
            "code_digest": provenance["code_digest"],
            "derivation_depth": 1,
        }
        manager.release(source.ref.ref_id)
        assert manager.get_value(ref["ref_id"]) == "abcdefghabcdefgh"
    finally:
        manager.close()


def test_read_artifact_tool_supports_literal_pattern() -> None:
    manager = ArtifactManager(session_id="code-pattern")
    artifact = _artifact(manager, "alpha beta alpha")
    read = next(spec for spec in artifact_tool_specs(manager) if spec.name == "read_artifact")
    try:
        matches = read.fn(ref_id=artifact.ref.ref_id, pattern="alpha", max_matches=2)
        assert [match["offset"] for match in matches] == [0, 11]
        with pytest.raises(ValueError, match="offset"):
            read.fn(ref_id=artifact.ref.ref_id, pattern="alpha", offset=1)
    finally:
        manager.close()


def test_artifact_code_rejects_empty_code_excess_refs_and_closed_runtime() -> None:
    manager = ArtifactManager(session_id="code-runtime-guards")
    artifact = _artifact(manager, "value")
    runtime = ArtifactCodeRuntime(limits=ArtifactCodeLimits(max_artifacts=0))
    try:
        with pytest.raises(ArtifactCodeError, match="must not be empty"):
            runtime.execute(manager=manager, artifact_refs=[], code="  ")
        with pytest.raises(ArtifactCodeError, match="at most 0 references"):
            runtime.execute(
                manager=manager,
                artifact_refs=[artifact.ref.ref_id],
                code="None",
            )
        runtime.close()
        runtime.close()
        with pytest.raises(ArtifactCodeError, match="runtime is closed"):
            runtime.execute(manager=manager, artifact_refs=[], code="None")
    finally:
        runtime.close()
        manager.close()


def test_artifact_code_bridge_validates_search_and_host_budgets() -> None:
    manager = ArtifactManager(session_id="code-bridge")
    artifact = _artifact(manager, "alpha beta alpha")
    ref_id = artifact.ref.ref_id
    bridge = _ArtifactHostBridge(
        manager,
        (ref_id,),
        ArtifactCodeLimits(max_host_calls=20, max_read_chars=8),
    )
    try:
        assert bridge.metadata(ref_id)["ref_id"] == ref_id
        assert bridge.read(ref_id, max_chars=-1)["content"] == "alpha be"
        assert [match["offset"] for match in bridge.find(ref_id, "alpha")] == [0, 11]
        with pytest.raises(ArtifactCodeError, match="cannot be empty"):
            bridge.find(ref_id, "")
        with pytest.raises(ArtifactCodeError, match="exceeds 4096"):
            bridge.find(ref_id, "x" * 4_097)
        with pytest.raises(ArtifactCodeError, match="max_matches must be positive"):
            bridge.find(ref_id, "alpha", max_matches=0)
        with pytest.raises(ArtifactCodeError, match="context_chars"):
            bridge.find(ref_id, "alpha", context_chars=-1)
        with pytest.raises(ArtifactCodeError, match="max_chars"):
            bridge.read(ref_id, max_chars=0)
        with pytest.raises(ArtifactCodeError, match="allowlist"):
            bridge.metadata("missing")

        exhausted = _ArtifactHostBridge(
            manager,
            (ref_id,),
            ArtifactCodeLimits(max_host_calls=1),
        )
        exhausted.metadata(ref_id)
        with pytest.raises(ArtifactCodeError, match="host-call budget"):
            exhausted.metadata(ref_id)
    finally:
        manager.close()


def test_artifact_code_bridge_rejects_cumulative_materialization_budget() -> None:
    manager = ArtifactManager(session_id="code-get-budget")
    artifact = _artifact(manager, "abcdef")
    bridge = _ArtifactHostBridge(
        manager,
        (artifact.ref.ref_id,),
        ArtifactCodeLimits(max_materialize_bytes=10, max_total_read_bytes=5),
    )
    try:
        with pytest.raises(ArtifactCodeError, match="cumulative read budget"):
            bridge.get(artifact.ref.ref_id)
    finally:
        manager.close()


@pytest.mark.parametrize(
    ("value", "limits", "message"),
    [
        (math.inf, ArtifactCodeLimits(), "non-finite"),
        ({1: "value"}, ArtifactCodeLimits(), "keys must be strings"),
        ({"value"}, ArtifactCodeLimits(), "not JSON-compatible"),
        ([1, 2], ArtifactCodeLimits(max_result_nodes=1), "too many values"),
        ([[1]], ArtifactCodeLimits(max_result_depth=1), "nested too deeply"),
        ("abcdef", ArtifactCodeLimits(max_result_bytes=3), "exceeds 3 bytes"),
    ],
)
def test_artifact_code_result_validation_rejects_unsafe_values(
    value: object,
    limits: ArtifactCodeLimits,
    message: str,
) -> None:
    with pytest.raises(ArtifactCodeError, match=message):
        codemode_module._validate_result(value, limits)


def test_artifact_code_result_validation_handles_bytes_tuples_and_cycles() -> None:
    cycle: list[object] = []
    cycle.append(cycle)
    with pytest.raises(ArtifactCodeError, match="cycle"):
        codemode_module._validate_result(cycle, ArtifactCodeLimits())

    assert codemode_module._validate_result(
        {"items": (b"a", 1.5)},
        ArtifactCodeLimits(),
    ) == {"items": [b"a", 1.5]}
    assert codemode_module._compact_monty_error(Exception("\n")) == (
        "artifact code failed: Exception"
    )
    with pytest.raises(TypeError):
        codemode_module._json_bytes(object())


def test_artifact_code_preserves_bridge_errors_from_monty_session() -> None:
    class Session:
        def feed_run(self, *_args: object, **_kwargs: object) -> object:
            raise ArtifactCodeError("bridge failed")

    class Pool:
        @contextmanager
        def checkout(self, **_kwargs: object) -> Iterator[Session]:
            yield Session()

    manager = ArtifactManager(session_id="code-bridge-error")
    runtime = ArtifactCodeRuntime()
    runtime._pool = Pool()  # type: ignore[assignment]
    try:
        with pytest.raises(ArtifactCodeError, match="bridge failed"):
            runtime.execute(manager=manager, artifact_refs=[], code="None")
    finally:
        runtime._pool = None
        runtime.close()
        manager.close()


def test_artifact_code_find_stops_at_limit_and_deduplicates_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ArtifactManager(session_id="code-find-overlap")
    artifact = _artifact(manager, "alpha")
    calls = 0

    def repeated_read(
        _ref_id: str,
        *,
        offset: int = 0,
        max_chars: int = -1,
        **_kwargs: Any,
    ) -> Any:
        nonlocal calls
        calls += 1
        return type(
            "Page",
            (),
            {
                "content": "alpha",
                "complete": calls > 1,
                "next_offset": None if calls > 1 else 0,
            },
        )()

    bridge = _ArtifactHostBridge(manager, (artifact.ref.ref_id,), ArtifactCodeLimits())
    try:
        assert len(bridge.find(artifact.ref.ref_id, "alpha", max_matches=1)) == 1
        calls = 0
        monkeypatch.setattr(manager, "read", repeated_read)
        matches = bridge.find(artifact.ref.ref_id, "alpha")
        assert matches == [{"offset": 0, "match": "alpha", "context": "alpha"}]
    finally:
        manager.close()
