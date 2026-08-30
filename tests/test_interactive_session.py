from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

import kedi
from kedi._interactive import persistence as persistence_module
from kedi.agent_adapter import ConversationState
from kedi.errors import KediExecutionError
from kedi.executors.base import Executor
from kedi.lang.compiler.program_compiler import ProgramCompiler
from kedi.lang.parser import parse_program
from tests.mock_adapter import MockAdapter


def test_interactive_session_persists_native_values_without_replay() -> None:
    calls: list[str] = []

    def record(value: str) -> None:
        calls.append(value)

    with kedi.interactive(env={"record": record}) as session:
        assert session.execute("`record('first')`\n[value: int] = `40`") is None
        assert session.execute("= `value + 2`") == 42

    assert calls == ["first"]


def test_interactive_session_persists_map_aggregate_without_iteration_locals() -> None:
    with kedi.interactive() as session:
        session.execute("[results: list] = `[]`")
        session.execute(
            "> loop [item]: `[1, 2, 3]`:\n"
            "  [scaled: int] = `item * 11`\n"
            "> map:\n"
            "  `results.append((item, scaled))`\n"
        )

        assert session.execute("= `sorted(results)`") == [(1, 11), (2, 22), (3, 33)]
        with pytest.raises(KediExecutionError, match="item"):
            session.execute("= `item`")
        with pytest.raises(KediExecutionError, match="scaled"):
            session.execute("= `scaled`")


def test_interactive_session_procedure_reads_latest_global_value() -> None:
    with kedi.interactive() as session:
        session.execute(
            """
[value: int] = `1`

@read_value() -> int:
  = `value`
""".strip()
        )
        session.execute("[value: int] = `2`")

        assert session.execute("= `read_value()`") == 2


def test_interactive_session_preludes_share_live_python_globals() -> None:
    with kedi.interactive() as session:
        session.execute(
            """
```
value = 1

def read_value():
    return value
```
""".strip()
        )
        session.execute("```\nvalue = 2\n```")

        assert session.execute("= `read_value()`") == 2


def test_interactive_session_preserves_types_across_fragments() -> None:
    with kedi.interactive() as session:
        session.execute("~Person(name: str, age: int)")
        person = cast(Any, session.execute('= `Person(name="Ada", age=42)`'))

    assert person.name == "Ada"
    assert person.age == 42


def test_interactive_session_preserves_profile_selection_across_fragments() -> None:
    adapter = MockAdapter()
    with kedi.interactive(adapter=adapter) as session:
        session.execute(
            """
> profile: retained:
    > model: retained-model
""".strip()
        )
        session.execute("> use: retained")

        assert session.execute(">> The result is [answer].\n= `answer`") == "answer-value"

    assert adapter.model_calls == ["retained-model"]


def test_interactive_session_preserves_direct_model_selection_across_fragments() -> None:
    adapter = MockAdapter()
    with kedi.interactive(adapter=adapter) as session:
        session.execute("> model: retained-model")

        assert session.execute(">> The result is [answer].\n= `answer`") == "answer-value"

    assert adapter.model_calls == ["retained-model"]


def test_interactive_session_keeps_prior_state_after_failure() -> None:
    with kedi.interactive() as session:
        with pytest.raises(KediExecutionError, match="division by zero"):
            session.execute("[saved: int] = `7`\n= `1 / 0`")

        assert session.execute("= `saved`") == 7


def test_interactive_session_traceback_retains_defining_fragment() -> None:
    with kedi.interactive() as session:
        session.execute(
            """
@explode() -> int:
  = `1 / 0`
""".strip()
        )

        with pytest.raises(KediExecutionError) as info:
            session.execute("= `explode()`")

    rendered = str(info.value)
    assert 'File "<interactive:1>", line 2' in rendered
    assert 'File "<interactive:2>", line 1' in rendered


def test_interactive_session_resolves_relative_import_once(tmp_path: Path) -> None:
    calls = 0

    def mark() -> int:
        nonlocal calls
        calls += 1
        return calls

    (tmp_path / "helper.kedi").write_text(
        "[imported_value: int] = `mark()`\n\n> export:\n  imported_value\n",
        encoding="utf-8",
    )

    with kedi.interactive(cwd=tmp_path, env={"mark": mark}) as session:
        assert session.execute("> import: helper\n= `imported_value`") == 1
        assert session.execute("> import: helper\n= `imported_value`") == 1

    assert calls == 1


def test_interactive_session_resolves_import_relative_to_explicit_source_name(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "helper.kedi").write_text(
        "[answer: int] = `42`\n\n> export:\n  answer\n",
        encoding="utf-8",
    )

    with kedi.interactive(cwd=tmp_path / "elsewhere") as session:
        result = session.execute(
            "> import: helper\n= `answer`",
            source_name=str(package / "cell.kedi"),
        )

    assert result == 42


def test_interactive_session_uses_active_python_api_context() -> None:
    with kedi.context(env={"base": 40}):
        with kedi.interactive(env={"increment": 2}) as session:
            assert session.execute("= `base + increment`") == 42


def test_interactive_session_accepts_supplied_conversation_and_parallel_engine() -> None:
    conversation = ConversationState()
    session = kedi.interactive(
        conversation=conversation,
        parallel=True,
        max_workers=1,
    )
    try:
        assert session.execute("= `42`") == 42
        assert session._compiler.rt._engine.parallel is True
    finally:
        session.close()
        conversation.close()


def test_interactive_sessions_are_isolated() -> None:
    first = kedi.interactive()
    second = kedi.interactive()
    try:
        first.execute("[value: int] = `1`")
        second.execute("[value: int] = `2`")

        assert first.execute("= `value`") == 1
        assert second.execute("= `value`") == 2
    finally:
        first.close()
        second.close()


def test_interactive_session_rejects_reentry_and_execution_after_close() -> None:
    session: kedi.InteractiveSession

    def reenter() -> Any:
        return session.execute("= `1`")

    session = kedi.interactive(env={"reenter": reenter})
    with pytest.raises(KediExecutionError, match="already executing"):
        session.execute("`reenter()`")

    session.close()
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        session.execute("= `1`")
    with pytest.raises(RuntimeError, match="closed"):
        session.__enter__()


def test_interactive_session_rejects_duplicate_sources_and_close_while_executing() -> None:
    session = kedi.interactive()
    try:
        assert session.execute("= `1`", source_name="cell.kedi") == 1
        with pytest.raises(ValueError, match="already used"):
            session.execute("= `2`", source_name="cell.kedi")

        session._execute_lock.acquire()
        try:
            with pytest.raises(RuntimeError, match="while it is executing"):
                session.close()
        finally:
            session._execute_lock.release()
    finally:
        session.close()


def test_interactive_session_rejects_executor_without_incremental_prelude_support() -> None:
    class NonIncrementalExecutor:
        def set_debug_exporter(self, _exporter: object) -> None:
            pass

    with kedi.interactive(executor=cast(Executor, NonIncrementalExecutor())) as session:
        with pytest.raises(TypeError, match="does not support incremental preludes"):
            session.execute("```\nvalue = 1\n```")


def test_fragment_preparation_tolerates_missing_source_map(tmp_path: Path) -> None:
    compiler = ProgramCompiler(None)
    compiler.initialize_fragments()

    prepared = compiler.prepare_fragment(
        parse_program("= `42`"),
        import_source_path=tmp_path / "cell.kedi",
    )

    assert prepared.program.source_map is None


@pytest.mark.parametrize(
    "source, expected",
    [
        ("> export:\n  value", "export directives"),
        ("@test: target:\n  > case: basic:\n    `assert True`", "validation suites"),
    ],
)
def test_interactive_session_rejects_non_incremental_constructs(
    source: str,
    expected: str,
) -> None:
    with kedi.interactive() as session:
        with pytest.raises(ValueError, match=expected):
            session.execute(source)


def test_interactive_session_rejects_package_metadata_after_parsing() -> None:
    with kedi.interactive() as session:
        with pytest.raises(ValueError, match="package metadata"):
            session.execute(
                "> package: demo:\n  version: 1.0",
                source_name="package.kedi",
            )


def test_interactive_session_dump_load_restores_values_types_and_procedures_without_replay(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "marker.txt"
    snapshot = tmp_path / "session.kedi-state"
    source = f"""
`open({str(marker)!r}, "a").write("executed\\n")`
~Person(name: str, age: int)

@double(value: int) -> int:
  = `value * 2`

[value: int] = `20`
[person: Person] = `Person(name="Ada", age=2)`
[data: dict[str, list[int]]] = `{{"items": [1, 2, 3]}}`
""".strip()

    with kedi.interactive() as session:
        session.execute(source, source_name="setup.kedi")
        assert session.dump(snapshot) == snapshot

    with kedi.InteractiveSession.load(snapshot) as restored:
        assert restored.execute("= `double(value) + person.age`") == 42
        assert restored.execute('= `data["items"]`') == [1, 2, 3]

    assert marker.read_text(encoding="utf-8") == "executed\n"


def test_interactive_session_dump_load_preserves_active_profile(tmp_path: Path) -> None:
    snapshot = tmp_path / "profile.kedi-state"
    with kedi.interactive(adapter=MockAdapter()) as session:
        session.execute("> model: retained-model")
        session.dump(snapshot)

    adapter = MockAdapter()
    with kedi.InteractiveSession.load(snapshot, adapter=adapter) as restored:
        assert restored.execute(">> The result is [answer].\n= `answer`") == "answer-value"

    assert adapter.model_calls == ["retained-model"]


def test_public_session_factories_round_trip_with_custom_session_type(
    tmp_path: Path,
) -> None:
    class CustomInteractiveSession(kedi.InteractiveSession):
        pass

    snapshot = tmp_path / "public-api.kedi-state"
    with kedi.interactive() as session:
        session.execute("[value: int] = `42`")
        assert kedi.dump_session(session, snapshot) == snapshot

    with kedi.load_session(
        snapshot,
        session_type=CustomInteractiveSession,
    ) as restored:
        assert type(restored) is CustomInteractiveSession
        assert restored.execute("= `value`") == 42

    load = cast(Any, kedi.load_session)
    with pytest.raises(TypeError, match="positional"):
        load(snapshot, CustomInteractiveSession)


def test_interactive_session_dump_preserves_state_committed_before_execution_failure(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "failed-fragment.kedi-state"
    with kedi.interactive() as session:
        with pytest.raises(KediExecutionError, match="division by zero"):
            session.execute("[saved: int] = `7`\n= `1 / 0`")
        session.dump(snapshot)

    with kedi.InteractiveSession.load(snapshot) as restored:
        assert restored.execute("= `saved`") == 7


def test_interactive_session_dump_rejects_all_unsupported_state_atomically(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "existing.kedi-state"
    snapshot.write_text("previous", encoding="utf-8")
    shared: list[int] = []

    with kedi.interactive(
        env={"callback": lambda: None, "left": shared, "right": shared}
    ) as session:
        assert session._conversation_state is not None
        session._conversation_state.record_exchange("hello", "world")
        with pytest.raises(kedi.SessionDumpError) as info:
            session.dump(snapshot)

    rendered = str(info.value)
    assert "environment['callback']" in rendered
    assert "shared or cyclic object" in rendered
    assert "conversation contains portable turns" in rendered
    assert snapshot.read_text(encoding="utf-8") == "previous"
    assert not tuple(tmp_path.glob(".existing.kedi-state.*.tmp"))


@pytest.mark.parametrize(
    "source, expected",
    [
        ("```\nvalue = 1\n```", "inline Python prelude"),
        ("> import: helper", "imports modules"),
        ("~Value(count: int = `1`)", "executable defaults"),
    ],
)
def test_interactive_session_dump_rejects_fragments_that_require_reexecution(
    tmp_path: Path,
    source: str,
    expected: str,
) -> None:
    if source.startswith("> import"):
        (tmp_path / "helper.kedi").write_text(
            "[answer: int] = `42`\n\n> export:\n  answer\n",
            encoding="utf-8",
        )
    with kedi.interactive(cwd=tmp_path) as session:
        session.execute(source)
        with pytest.raises(kedi.SessionDumpError, match=expected):
            session.dump(tmp_path / "rejected.kedi-state")


def test_interactive_session_load_rejects_modified_snapshot(tmp_path: Path) -> None:
    snapshot = tmp_path / "session.kedi-state"
    with kedi.interactive() as session:
        session.execute("[value: int] = `42`")
        session.dump(snapshot)

    document = json.loads(snapshot.read_text(encoding="utf-8"))
    document["payload"]["cell_number"] = 999
    snapshot.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(kedi.SessionLoadError, match="integrity check"):
        kedi.InteractiveSession.load(snapshot)


def test_interactive_session_dump_publish_is_atomic_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "session.kedi-state"
    snapshot.write_text("previous", encoding="utf-8")

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(persistence_module.os, "replace", fail_replace)
    with kedi.interactive() as session:
        session.execute("[value: int] = `42`")
        with pytest.raises(OSError, match="replace failed"):
            session.dump(snapshot)

    assert snapshot.read_text(encoding="utf-8") == "previous"
    assert not tuple(tmp_path.glob(".session.kedi-state.*.tmp"))


def test_interactive_session_dump_rejects_execution_and_closed_session(tmp_path: Path) -> None:
    session = kedi.interactive()
    session._execute_lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="while it is executing"):
            session.dump(tmp_path / "busy.kedi-state")
    finally:
        session._execute_lock.release()
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        session.dump(tmp_path / "closed.kedi-state")
