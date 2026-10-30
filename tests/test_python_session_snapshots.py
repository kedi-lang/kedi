from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from kedi._interactive.persistence import SessionDumpError, SessionLoadError
from kedi.agent_adapter.conversation import ConversationState
from kedi.artifacts import ArtifactManager, ArtifactPolicy
from kedi.artifacts.errors import ArtifactExpiredError, ArtifactReleasedError
from kedi.interactive_session import InteractiveSession, load_session
from kedi.lang.compiler.environment import ScopeFrame
from tests.interactive_helpers import inspect_fragment
from tests.mock_adapter import MockAdapter

pytest.importorskip("cloudpickle")


def _value(session: InteractiveSession, expression: str) -> Any:
    return inspect_fragment(session, f"[inspection_value: Any] = `{expression}`")


def _fresh_process(script: str, directory: Path, *args: str) -> subprocess.CompletedProcess[str]:
    path = directory / "consumer.py"
    path.write_text(script, encoding="utf-8")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(sys.path)
    result = subprocess.run(
        [sys.executable, str(path), *args],
        text=True,
        capture_output=True,
        env=environment,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_fresh_process_restores_shared_prelude_graph_without_replay(tmp_path: Path) -> None:
    marker = tmp_path / "initialization"
    snapshot = tmp_path / "rich.state"
    source = f"""```
from pathlib import Path
Path({str(marker)!r}).write_text("once")
count = 10
class Counter:
    def value(self):
        return count
def get():
    return count
def bump():
    global count
    count += 1
    return count
def factory():
    number = 40
    def step():
        nonlocal number
        number += 1
        return number
    return step
```
[item: Any] = `Counter()`
[step: Any] = `factory()`
[shared: Any] = `[]`
[alias: Any] = `shared`
[cycle: Any] = `[]`
`cycle.append(cycle)`
~Person(name: str, age: int)
[person: Person] = `Person(name="Ada", age=37)`
[base: int] = `40`
@answer() -> int:
    = `base + person.age`
[procedure: Any] = `answer`
"""
    with InteractiveSession() as session:
        session.execute(source)
        session.dump(snapshot, format="python")
    marker.write_text("do not overwrite", encoding="utf-8")
    _fresh_process(
        """
import sys
from kedi.interactive_session import InteractiveSession
with InteractiveSession.load(sys.argv[1], trusted=True) as s:
    s.execute("[inspection: Any] = `(get(), item.value(), bump(), get(), item.value())`")
    assert s._env['inspection'] == (10,10,11,11,11)
    s.execute("[inspection: Any] = `(step(), step(), shared is alias, cycle[0] is cycle)`")
    assert s._env['inspection'] == (41,42,True,True)
    s.execute("[inspection: Any] = `isinstance(person, Person)`")
    assert s._env['inspection'] is True
    s.execute("[inspection: int] = `procedure()`")
    assert s._env['inspection'] == 77
    s.execute("[base] := `50`")
    s.execute("[inspection: int] = `procedure()`")
    assert s._env['inspection'] == 87
    s.dump(sys.argv[2], format='python')
with InteractiveSession.load(sys.argv[2], trusted=True) as s:
    s.execute("[inspection: Any] = `(step(), get(), procedure())`")
    assert s._env['inspection'] == (43,11,87)
""",
        tmp_path,
        str(snapshot),
        str(tmp_path / "second.state"),
    )
    assert marker.read_text() == "do not overwrite"


def test_saved_declaration_generations_defaults_and_lexical_scopes(tmp_path: Path) -> None:
    path = tmp_path / "generations.state"
    with InteractiveSession() as session:
        session.execute("""```
count = 0
def next_default():
    global count
    count += 1
    return count
```
~Person(name: str, index: int = `next_default()`)
[old: Any] = `Person("old")`
@read(value: int = `next_default()`) -> int:
    = `value`
[old_read: Any] = `read`
@factory(value: int) -> Any:
    [offset: int] = `value + 1`
    @inner() -> int:
        = `offset`
    = `inner.__kedi_procedure_target__`
[closed: Any] = `factory(40)`
""")
        session.execute("""~Person(name: str, index: int = `next_default()`)
[new: Person] = `Person("new")`
@read(value: int = `next_default()`) -> int:
    = `value * 10`
""")
        expected = _value(session, "(old.index, new.index, old_read(), read(), closed(), count)")
        session.dump(path, format="python")
    with InteractiveSession.load(path, trusted=True) as restored:
        assert (
            _value(restored, "(old.index, new.index, old_read(), read(), closed(), count)")
            == expected
        )
        assert _value(restored, "type(old) is type(new)") is False
        assert _value(restored, "type(new) is Person") is True
        restored.execute("[another: Person] = `Person('another')`")
        assert _value(restored, "another.index") == expected[1]
        assert _value(restored, "count") == expected[-1]


def test_initialized_import_keeps_private_state_without_reinitialization(tmp_path: Path) -> None:
    module = tmp_path / "saved_module.kedi"
    marker = tmp_path / "module_initialized"
    module.write_text(
        f"""```
from pathlib import Path
Path({str(marker)!r}).write_text("once")
```
[private: int] = `40`
@next_value() -> int:
    [private] := `private + 1`
    = `private`
> export:
    next_value
""",
        encoding="utf-8",
    )
    snapshot = tmp_path / "module.state"
    with InteractiveSession(cwd=tmp_path) as session:
        session.execute("> import: saved_module")
        assert _value(session, "next_value()") == 41
        session.dump(snapshot, format="python")
    marker.write_text("preserve", encoding="utf-8")
    module.write_text("this source must not be executed", encoding="utf-8")
    with InteractiveSession.load(snapshot, trusted=True) as restored:
        assert _value(restored, "next_value()") == 42
        restored.execute("> import: saved_module")
        assert _value(restored, "next_value()") == 43
    assert marker.read_text() == "preserve"


def test_restored_procedures_keep_definition_profile_not_callers(tmp_path: Path) -> None:
    snapshot = tmp_path / "profiles.state"
    original_adapter = MockAdapter()
    with InteractiveSession(adapter=original_adapter) as session:
        session.execute("""> model: definition-model
> instructions: Definition instructions.
@answer() -> str:
    [result] << Answer now.
    = `result`
""")
        session.execute("> model: later-model\n> instructions: Later instructions.")
        expected = _value(session, "answer()")
        expected_instructions = original_adapter.instruction_calls[-1]
        session.dump(snapshot, format="python")
    adapter = MockAdapter()
    with InteractiveSession.load(snapshot, trusted=True, adapter=adapter) as restored:
        assert _value(restored, "answer()") == expected
        assert adapter.model_calls == ["definition-model"]
        assert adapter.instruction_calls == [expected_instructions]
        restored.execute("[later] << Later prompt.")
        assert adapter.model_calls[-1] == "later-model"


def test_snapshot_artifacts_keep_owners_lifecycle_and_survive_redump(tmp_path: Path) -> None:
    snapshot = tmp_path / "artifacts.state"
    second = tmp_path / "artifacts-second.state"
    with InteractiveSession() as session:
        manager = session._compiler.rt.artifacts
        policy = ArtifactPolicy.patch({"threshold": "1b", "ttl": "1h"}).resolved()
        active = manager.artifactize("complete report", policy=policy)
        released = manager.artifactize("released report", policy=policy)
        manager.release(released.ref.ref_id)
        cast(ScopeFrame, session._env).declare("report", active)
        cast(ScopeFrame, session._env).declare("released_ref", released.ref.ref_id)
        session.dump(snapshot, format="python")
        assert active.manager is manager
        assert active.get_value() == "complete report"
    with InteractiveSession.load(snapshot, trusted=True) as restored:
        assert _value(restored, "report.upper()") == "COMPLETE REPORT"
        manager = restored._compiler.rt.artifacts
        with pytest.raises(ArtifactReleasedError):
            manager.get_value(_value(restored, "released_ref"))
        restored.dump(second, format="python")
    snapshot.unlink()
    with InteractiveSession.load(second, trusted=True) as restored:
        assert _value(restored, "report") == "complete report"
        assert restored._compiler.rt.artifacts.stats()["released_artifacts"] == 1


def test_expired_artifacts_are_not_renewed_on_restore(tmp_path: Path) -> None:
    from datetime import datetime, timedelta, timezone

    snapshot = tmp_path / "expired.state"
    with InteractiveSession() as session:
        manager = session._compiler.rt.artifacts
        handle = manager.artifactize("expired", policy=ArtifactPolicy.patch({"threshold": "1b"}))
        record = manager._all_records()[0]
        record.ref = record.ref.model_copy(
            update={"expires_at": datetime.now(timezone.utc) - timedelta(seconds=1)}
        )
        cast(ScopeFrame, session._env).declare("ref_id", handle.ref.ref_id)
        session.dump(snapshot, format="python")
    with InteractiveSession.load(snapshot, trusted=True) as restored:
        with pytest.raises(ArtifactExpiredError):
            restored._compiler.rt.artifacts.get_value(_value(restored, "ref_id"))


def test_completed_pydantic_conversation_continues_after_restore(tmp_path: Path) -> None:
    from pydantic_ai.messages import ModelResponse, TextPart, UserPromptPart
    from pydantic_ai.models.function import FunctionModel

    from kedi.agent_adapter import PydanticAdapter

    prompts = []

    def model(messages: Any, info: Any) -> Any:
        user_prompts = [
            part.content
            for message in messages
            for part in message.parts
            if isinstance(part, UserPromptPart)
        ]
        prompts.append(user_prompts)
        return ModelResponse(parts=[TextPart("remembered" if len(user_prompts) > 1 else "stored")])

    snapshot = tmp_path / "history.state"
    adapter = PydanticAdapter(FunctionModel(model))
    with InteractiveSession(adapter=adapter) as session:
        session.execute("> history: enabled\n[first] << Remember release amber.")
        assert _value(session, "first") == "stored"
        session.dump(snapshot, format="python")
    prompts.clear()
    with InteractiveSession.load(snapshot, trusted=True, adapter=adapter) as restored:
        restored.execute("[second] << Which release?")
        assert _value(restored, "second") == "remembered"
        assert prompts[-1] == ["Remember release amber.", "Which release?"]


@pytest.mark.parametrize("expression", ["(x for x in range(2))", "__import__('threading').Lock()"])
def test_unsupported_live_values_do_not_replace_existing_snapshot(
    tmp_path: Path, expression: str
) -> None:
    path = tmp_path / "existing.state"
    with InteractiveSession() as session:
        session.execute("[value: int] = `42`")
        session.dump(path)
        original = path.read_bytes()
        session.execute(f"[live: Any] = `{expression}`")
        with pytest.raises(SessionDumpError, match="resource|generator|lock"):
            session.dump(path, format="python")
        assert path.read_bytes() == original


def test_subclass_restore_and_external_conversation_ownership(tmp_path: Path) -> None:
    class Session(InteractiveSession):
        pass

    path = tmp_path / "subclass.state"
    conversation = ConversationState(session_id="owned-conversation")
    with Session(conversation_state=conversation) as session:
        session.execute("[answer: Any] = `lambda: 42`")
        session.dump(path, format="python")
    supplied = ConversationState(session_id="owned-conversation")
    with load_session(
        path, trusted=True, session_type=Session, conversation_state=supplied
    ) as restored:
        assert type(restored) is Session
        assert restored._conversation_state is supplied
        assert _value(restored, "answer()") == 42
    assert not supplied._closed
    supplied.close()
    conversation.close()


def test_bridged_executor_has_no_silent_local_fallback(tmp_path: Path) -> None:
    from kedi.executors.base import DefaultExecutor

    class ExternalExecutor(DefaultExecutor):
        pass

    with InteractiveSession(executor=ExternalExecutor()) as session:
        session.execute("[value: int] = `42`")
        with pytest.raises(SessionDumpError, match="executor|transfer"):
            session.dump(tmp_path / "unsupported.state", format="python")


def test_memory_artifact_native_types_and_aliases_survive_restore(tmp_path: Path) -> None:
    path = tmp_path / "native-artifact.state"
    policy = ArtifactPolicy.patch({"threshold": "1b"}).resolved()
    with InteractiveSession() as session:
        session.execute("~Report(answer: int)\n[value: Report] = `Report(42)`")
        manager = session._compiler.rt.artifacts
        handle = manager.artifactize(session._env["value"], policy=policy)
        env = cast(ScopeFrame, session._env)
        env.declare("report", handle)
        env.declare("backing", handle.get_value())
        session.dump(path, format="python")
    with InteractiveSession.load(path, trusted=True) as restored:
        assert _value(restored, "isinstance(report, Report)") is True
        assert _value(restored, "report is backing") is True
        assert _value(restored, "report.answer") == 42
        manager = restored._compiler.rt.artifacts
        another = manager.artifactize(_value(restored, "Report(43)"), policy=policy)
        assert another.get_value().answer == 43


def test_file_artifact_json_pointer_reads_survive_restore(tmp_path: Path) -> None:
    path = tmp_path / "json-artifact.state"
    policy = ArtifactPolicy.patch(
        {"store": "file", "path": str(tmp_path / "original-store"), "threshold": "1b"}
    ).resolved()
    with InteractiveSession() as session:
        manager = session._compiler.rt.artifacts
        handle = manager.artifactize({"customer": {"name": "Ada"}}, policy=policy)
        cast(ScopeFrame, session._env).declare("ref_id", handle.ref.ref_id)
        session.dump(path, format="python")
    with InteractiveSession.load(path, trusted=True) as restored:
        result = restored._compiler.rt.artifacts.read(
            _value(restored, "ref_id"), path="/customer/name"
        )
        assert result.content == '"Ada"'


@pytest.mark.parametrize("supplied", [False, True])
def test_retained_conversation_survives_cache_rotation_and_session_close(
    tmp_path: Path, supplied: bool
) -> None:
    path = tmp_path / "retained-conversation.state"
    original = ConversationState(session_id="retained-conversation")
    with InteractiveSession(conversation_state=original) as session:
        handle = original.artifacts.artifactize(
            "retained report", policy=ArtifactPolicy.patch({"threshold": "1b"})
        )
        cast(ScopeFrame, session._env).declare("ref_id", handle.ref.ref_id)
        session.dump(path, format="python")
    owner = ConversationState(session_id=original.session_id) if supplied else None
    with InteractiveSession.load(
        path, trusted=True, conversation_state=owner, close_conversation_state=False
    ) as restored:
        conversation = restored._conversation_state
        assert conversation is not None
        ref_id = _value(restored, "ref_id")
        assert conversation.artifacts.get_value(ref_id) == "retained report"
        conversation.advance_cache_epoch()
        assert conversation.artifacts.get_value(ref_id) == "retained report"
        root = next(iter(conversation.artifacts._file_stores))
    assert not conversation._closed
    assert root.exists()
    assert conversation.artifacts.get_value(ref_id) == "retained report"
    conversation.close()
    assert not root.exists()
    original.close()


def test_portable_session_does_not_import_optional_serializer(tmp_path: Path) -> None:
    _fresh_process(
        """
import importlib.abc
import sys
from pathlib import Path

class NoSerializer(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'cloudpickle' or fullname.startswith('cloudpickle.'):
            raise ModuleNotFoundError('cloudpickle deliberately unavailable')

sys.meta_path.insert(0, NoSerializer())
from kedi.interactive_session import InteractiveSession
from kedi._interactive.persistence import SessionDumpError
path = Path(sys.argv[1])
with InteractiveSession() as session:
    session.execute('[base: int] = `40`')
    session.dump(path)
    try:
        session.dump(path, format='python')
    except SessionDumpError as exc:
        assert 'kedi[snapshots]' in str(exc)
    else:
        raise AssertionError('Python mode must require the optional serializer')
with InteractiveSession.load(path) as restored:
    restored.execute('[inspection: int] = `base + 2`')
    assert restored._env['inspection'] == 42
assert 'cloudpickle' not in sys.modules
""",
        tmp_path,
        str(tmp_path / "portable.state"),
    )


def test_nested_generated_models_keep_their_validation_schema(tmp_path: Path) -> None:
    from pydantic import ValidationError

    path = tmp_path / "nested-models.state"
    with InteractiveSession() as session:
        session.execute("""~Person(name: str)
~Team(leader: Person)
[team: Team] = `Team(Person("Ada"))`
""")
        expected = _value(session, "Team.model_json_schema()")
        session.dump(path, format="python")
    with InteractiveSession.load(path, trusted=True) as restored:
        assert _value(restored, "Team.model_json_schema()") == expected
        team_type = _value(restored, "Team")
        assert team_type.model_validate({"leader": {"name": "Bob"}}).leader.name == "Bob"
        with pytest.raises(ValidationError):
            team_type.model_validate({"leader": {"name": 42}})


def test_completed_langchain_conversation_restores_to_a_fresh_adapter(tmp_path: Path) -> None:
    pytest.importorskip("langchain_core")
    from langchain_core.messages import AIMessage, HumanMessage

    from kedi.agent_adapter import LangChainAdapter
    from tests.test_langchain_user_history_processor import HistoryScriptedModel

    path = tmp_path / "langchain-history.state"
    first_model = HistoryScriptedModel(responses=[AIMessage(content="stored")])
    with InteractiveSession(adapter=LangChainAdapter(first_model)) as session:
        session.execute("> history: enabled\n[first] << Remember release amber.")
        assert _value(session, "first") == "stored"
        session.dump(path, format="python")
    next_model = HistoryScriptedModel(responses=[AIMessage(content="continued")])
    with InteractiveSession.load(
        path, trusted=True, adapter=LangChainAdapter(next_model)
    ) as restored:
        restored.execute("[second] << Which release?")
        assert _value(restored, "second") == "continued"
        assert [
            message.content for message in next_model.seen[-1] if isinstance(message, HumanMessage)
        ] == ["Remember release amber.", "Which release?"]


@pytest.mark.parametrize("failure_stage", ["resources", "graph"])
def test_failed_restore_drains_all_resources_and_preserves_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_stage: str
) -> None:
    from kedi._interactive import python_graph, python_state, resources

    path = tmp_path / "failed-restore.state"
    original = ConversationState(session_id="caller-owned")
    child = ConversationState(session_id="child-owned")
    try:
        with InteractiveSession(conversation_state=original) as session:
            session._compiler.rt._implicit_conversation_state = child
            session.execute("[answer: int] = `42`")
            session.dump(path, format="python")
    finally:
        original.close()
        child.close()

    supplied = ConversationState(session_id="caller-owned")
    supplied_manager = supplied.artifacts
    supplied_history = supplied.history
    roots: list[Path] = []
    closed: list[ArtifactManager] = []
    read_bundle = python_state.read_python_bundle
    close_manager = resources.RestoredArtifactManager.close

    def track_bundle(*args: Any, **kwargs: Any) -> Any:
        bundle = read_bundle(*args, **kwargs)
        roots.append(bundle.root)
        return bundle

    def fail_cleanup(manager: Any) -> None:
        if manager._closed:
            return
        close_manager(manager)
        closed.append(manager)
        raise RuntimeError("injected manager cleanup failure")

    class FailingCloseSession(InteractiveSession):
        def close(self) -> None:
            super().close()
            raise RuntimeError("injected session cleanup failure")

    monkeypatch.setattr(python_state, "read_python_bundle", track_bundle)
    monkeypatch.setattr(resources.RestoredArtifactManager, "close", fail_cleanup)
    if failure_stage == "resources":

        def fail_conversation(cls: Any, *args: Any, **kwargs: Any) -> Any:
            raise ValueError("injected primary restore failure")

        monkeypatch.setattr(ConversationState, "from_checkpoint", classmethod(fail_conversation))
    else:
        hydrate = python_graph.loads_session

        def fail_graph(*args: Any, **kwargs: Any) -> None:
            hydrate(*args, **kwargs)
            raise ValueError("injected primary restore failure")

        monkeypatch.setattr(python_graph, "loads_session", fail_graph)

    try:
        with pytest.raises(SessionLoadError, match="injected primary restore failure") as error:
            FailingCloseSession.load(path, trusted=True, conversation_state=supplied)
        assert len(closed) == 2
        assert all(manager._closed for manager in closed)
        assert len(roots) == 1 and not roots[0].exists()
        assert isinstance(error.value.__cause__, ValueError)
        assert any(
            "injected manager cleanup failure" in note for note in error.value.__cause__.__notes__
        )
        assert not supplied._closed and not supplied_manager._closed
        assert supplied.artifacts is supplied_manager
        assert supplied.history is supplied_history
        assert supplied.snapshot() == ()
        assert supplied._owns_artifacts
    finally:
        supplied.close()


@pytest.mark.parametrize("external_artifacts", [False, True])
def test_supplied_conversation_preserves_previous_artifact_ownership(
    tmp_path: Path, external_artifacts: bool
) -> None:
    path = tmp_path / "supplied-owner.state"
    original = ConversationState(session_id="supplied-owner")
    try:
        with InteractiveSession(conversation_state=original) as session:
            session.dump(path, format="python")
    finally:
        original.close()
    external = ArtifactManager(session_id="supplied-owner") if external_artifacts else None
    supplied = ConversationState(session_id="supplied-owner", artifacts=external)
    previous = supplied.artifacts
    with InteractiveSession.load(path, trusted=True, conversation_state=supplied) as restored:
        assert restored._conversation_state is supplied
        assert supplied.artifacts is not previous
        assert not previous._closed
    assert not supplied._closed
    supplied.close()
    assert previous._closed is (not external_artifacts)
    previous.close()


def test_retained_owner_cleanup_failure_still_releases_other_owners_and_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "retained-cleanup.state"
    original = ConversationState(session_id="retained-cleanup")
    child = ConversationState(session_id="retained-child")
    try:
        with InteractiveSession(conversation_state=original) as session:
            session._compiler.rt._implicit_conversation_state = child
            session.dump(path, format="python")
    finally:
        original.close()
        child.close()
    supplied = ConversationState(session_id="retained-cleanup")
    previous = supplied.artifacts
    close_previous = previous.close

    def fail_previous_cleanup() -> None:
        close_previous()
        raise RuntimeError("previous owner cleanup failed")

    monkeypatch.setattr(previous, "close", fail_previous_cleanup)
    with InteractiveSession.load(path, trusted=True, conversation_state=supplied) as restored:
        restored_child = restored._compiler.rt._implicit_conversation_state
        assert restored_child is not None
        root = next(iter(supplied.artifacts._file_stores))
        supplied.advance_cache_epoch()
    assert root.exists()
    assert not restored_child.artifacts._closed
    with pytest.raises(RuntimeError, match="previous owner cleanup failed"):
        supplied.close()
    assert previous._closed
    assert supplied.artifacts._closed
    assert restored_child._closed
    assert restored_child.artifacts._closed
    assert not root.exists()
    supplied.close()


def test_reserved_conversation_turn_rejects_dump_without_replacing_destination(
    tmp_path: Path,
) -> None:
    path = tmp_path / "reserved-turn.state"
    conversation = ConversationState()
    try:
        with InteractiveSession(conversation_state=conversation) as session:
            session.dump(path, format="python")
            original = path.read_bytes()
            ticket = conversation.reserve_turn()
            with pytest.raises(SessionDumpError, match="unfinished reserved turns"):
                session.dump(path, format="python")
            assert path.read_bytes() == original
            conversation.cancel_turn(ticket)
            session.dump(path, format="python")
    finally:
        conversation.close()


@pytest.mark.parametrize("unavailable", ["reserved_turn", "closed_artifacts", "producer"])
def test_supplied_conversation_must_have_quiescent_open_artifacts(
    tmp_path: Path, unavailable: str
) -> None:
    path = tmp_path / "unavailable-owner.state"
    original = ConversationState(session_id="unavailable-owner")
    try:
        with InteractiveSession(conversation_state=original) as session:
            session.dump(path, format="python")
    finally:
        original.close()
    supplied = ConversationState(session_id="unavailable-owner")
    manager = supplied.artifacts
    ticket = None
    if unavailable == "reserved_turn":
        ticket = supplied.reserve_turn()
    elif unavailable == "closed_artifacts":
        manager.close()
    else:
        manager._reserved_artifacts = 1
    try:
        with pytest.raises(SessionLoadError, match="empty and open"):
            InteractiveSession.load(path, trusted=True, conversation_state=supplied)
        assert supplied.artifacts is manager
        assert not supplied._closed
    finally:
        if ticket is not None:
            supplied.cancel_turn(ticket)
        manager._reserved_artifacts = 0
        supplied.close()


def test_repeated_snapshots_preserve_native_artifact_aliases_and_cycles(tmp_path: Path) -> None:
    path = tmp_path / "native-cycle-0.state"
    policy = ArtifactPolicy.patch({"threshold": "1b"}).resolved()
    with InteractiveSession() as session:
        session.execute("~Report(answer: int)\n[value: Report] = `Report(42)`")
        handle = session._compiler.rt.artifacts.artifactize(session._env["value"], policy=policy)
        backing = handle.get_value()
        aliases: dict[str, Any] = {"direct": backing, "nested": [backing]}
        aliases["self"] = aliases
        env = cast(ScopeFrame, session._env)
        env.declare("report", handle)
        env.declare("backing", backing)
        env.declare("aliases", aliases)
        env.declare("read_backing", lambda: backing)
        session.dump(path, format="python")
    for generation in range(1, 4):
        next_path = tmp_path / f"native-cycle-{generation}.state"
        with InteractiveSession.load(path, trusted=True) as restored:
            assert _value(restored, "isinstance(report, Report)") is True
            assert _value(restored, "report is backing is read_backing()") is True
            assert _value(restored, "aliases['direct'] is aliases['nested'][0] is backing") is True
            assert _value(restored, "aliases['self'] is aliases") is True
            assert _value(restored, "report.answer") == 42
            manager = restored._compiler.rt.artifacts
            added = manager.artifactize(_value(restored, f"Report({generation})"), policy=policy)
            cast(ScopeFrame, restored._env).declare(f"added_{generation}", added)
            for previous_generation in range(1, generation + 1):
                assert (
                    _value(restored, f"added_{previous_generation}.answer") == previous_generation
                )
            restored.dump(next_path, format="python")
        path.unlink()
        path = next_path
    with InteractiveSession.load(path, trusted=True) as restored:
        assert _value(restored, "report is backing is read_backing()") is True
        assert _value(restored, "[added_1.answer, added_2.answer, added_3.answer]") == [1, 2, 3]


def test_staging_setup_failure_closes_owner_without_masking_primary_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kedi._interactive import resources

    path = tmp_path / "failed-staging.state"
    staged: list[ArtifactManager] = []

    class FailingCleanupManager(ArtifactManager):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            staged.append(self)

        def close(self) -> None:
            super().close()
            raise RuntimeError("injected staging cleanup failure")

    def fail_store(*args: Any, **kwargs: Any) -> Any:
        raise OSError("injected staging setup failure")

    with InteractiveSession() as session:
        session.dump(path, format="python")
        original = path.read_bytes()
        monkeypatch.setattr(resources, "ArtifactManager", FailingCleanupManager)
        monkeypatch.setattr(resources, "FileArtifactStore", fail_store)
        with pytest.raises(SessionDumpError, match="injected staging setup failure") as error:
            session.dump(path, format="python")
        assert len(staged) == 1 and staged[0]._closed
        assert path.read_bytes() == original
        assert isinstance(error.value.__cause__, OSError)
        assert any(
            "injected staging cleanup failure" in note for note in error.value.__cause__.__notes__
        )
        assert not session._compiler.rt.artifacts._closed


def test_unclosed_restored_session_releases_extracted_files(tmp_path: Path) -> None:
    import gc

    path = tmp_path / "unclosed.state"
    with InteractiveSession() as session:
        session.execute("[value: int] = `1`")
        session.dump(path, format="python")
    restored = InteractiveSession.load(path, trusted=True)
    root = next(
        cleanup.__self__.root
        for cleanup in restored._snapshot_cleanups
        if getattr(cleanup, "__name__", "") == "cleanup"
    )
    assert root.is_dir()
    del restored
    gc.collect()
    assert not root.exists()
