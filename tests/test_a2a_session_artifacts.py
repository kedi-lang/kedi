import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic_ai.models.test import TestModel

from kedi.a2a.runtime import prepare_a2a_program
from kedi.agent_adapter import LangChainAdapter, PydanticAdapter
from kedi.artifacts import ArtifactManager
from kedi.artifacts.errors import ArtifactNotFoundError
from tests.test_a2a_persistence import _ScriptedChatModel


@pytest.mark.asyncio
@pytest.mark.parametrize("history", [False, True])
async def test_failed_session_construction_closes_owned_resources(tmp_path, monkeypatch, history):
    from kedi.a2a import runtime as module

    source = tmp_path / "agent.kedi"
    source.write_text(
        "> profile: worker:\n  > history: "
        + ("enabled" if history else "disabled")
        + "\n> export:\n  worker\n"
    )
    served = prepare_a2a_program(
        source, entry="worker", model="test:model", adapter=PydanticAdapter(TestModel())
    )
    artifacts, conversation = Mock(), Mock()
    monkeypatch.setattr(module, "ArtifactManager", Mock(return_value=artifacts))
    make_conversation = Mock(return_value=conversation)
    monkeypatch.setattr(module, "ConversationState", make_conversation)
    monkeypatch.setattr(module, "AgentSurface", Mock(side_effect=ValueError("invalid surface")))
    try:
        with pytest.raises(ValueError, match="invalid surface"):
            served._new_session()
        artifacts.close.assert_called_once_with()
        assert conversation.close.call_count == int(history)
        assert make_conversation.call_count == int(history)
        assert not served._sessions
    finally:
        await served.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["pydantic", "langchain"])
async def test_a2a_tools_use_each_session_store_and_restore_bindings(tmp_path, backend):
    source = tmp_path / "agent.kedi"
    source.write_text(
        "```\nimport asyncio\nasync def echo(value: str) -> str:\n"
        "    await asyncio.sleep(0)\n    return value * 2000\n```\n"
        "> profile: worker:\n  > approval: allow\n  > artifacts:\n"
        "    threshold: 1kb\n  > use: echo\n> export:\n  worker\n"
    )
    adapter = (
        PydanticAdapter(TestModel())
        if backend == "pydantic"
        else LangChainAdapter(_ScriptedChatModel(reply="unused"))
    )
    served = prepare_a2a_program(source, entry="worker", model="test:model", adapter=adapter)
    checkpoints = {}

    async def load(owner, context):
        return checkpoints.get((owner, context))

    served.configure_persistence(
        SimpleNamespace(artifact_root=tmp_path / "artifacts", load_checkpoint=load)
    )
    sessions = [served._new_session(), served._new_session()]
    active = peak = 0

    async def call(session, name, **kwargs):
        nonlocal active, peak
        with session.surface.run_scope():
            active += 1
            peak = max(peak, active)
            try:
                tool = next(t for t in session.surface.tool_specs if t.name == name)
                result = tool.fn(**kwargs)
                return await result if inspect.isawaitable(result) else result
            finally:
                active -= 1

    try:
        first, second = await asyncio.gather(
            call(sessions[0], "echo", value="first"),
            call(sessions[1], "echo", value="second"),
        )
        assert peak == 2 and active == 0
        assert sessions[0].artifacts.stats()["active_artifacts"] == 1
        assert sessions[1].artifacts.stats()["active_artifacts"] == 1
        assert sessions[0].artifacts.session_id != sessions[1].artifacts.session_id
        assert served.runtime.artifacts.stats()["active_artifacts"] == 0
        for session, value, ref in zip(sessions, ("first", "second"), (first, second)):
            checkpoint = served._session_checkpoint(session)
            assert len(checkpoint["artifacts"]["records"]) == 1
            checkpoint_root = tmp_path / "artifacts"
            restored = ArtifactManager.from_checkpoint(
                checkpoint["artifacts"], durable_root=checkpoint_root
            )
            restored.close()
            checkpoints[("owner", value)] = checkpoint
            old = session.artifacts
            await served._restore_session(session, "owner", value)
            assert session.artifacts is not old
            answer = await call(session, "read_artifact", ref_id=ref["ref_id"], max_chars=30)
            assert value in str(answer)
        foreign = await call(sessions[0], "echo", value="only-first")
        with pytest.raises(ArtifactNotFoundError):
            await call(sessions[1], "read_artifact", ref_id=foreign["ref_id"])
    finally:
        for session in sessions:
            session.close()
        await served.aclose()
