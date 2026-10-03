from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kedi.a2a import runtime, server
from kedi.a2a.protocol import KEDI_STRUCTURED_OUTPUT_EXTENSION_URI, structured_output_request
from kedi.agent_profile import AgentProfile
from tests.test_a2a import _ServedAdapter


@pytest.mark.parametrize(
    "options",
    [
        {"rpc_path": "relative"},
        {"max_request_bytes": 0},
        {"max_tasks_per_principal": 0},
        {"public_url": "https://agent.example"},
    ],
)
def test_app_rejects_invalid_configuration(options):
    with pytest.raises(ValueError):
        server.create_a2a_app(
            SimpleNamespace(), **({"public_url": "http://localhost:8000", "auth": None} | options)
        )


def test_app_reports_missing_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "a2a.server.routes", None)
    with pytest.raises(RuntimeError, match="not installed"):
        server.create_a2a_app(None, public_url="http://localhost", auth=None)


@pytest.mark.parametrize(
    "url", ["relative", "https://user:pass@agent.example", "http://agent.example"]
)
def test_public_url_validation(url):
    with pytest.raises(ValueError):
        server._validated_public_url(url)


def test_loopback_and_unknown_task_states():
    assert not server._is_loopback("relative")
    assert server._is_loopback("http://localhost")
    assert not server._is_loopback("https://agent.example")
    from a2a.types import Task, TaskStatus

    assert server._stored_task_state(Task()) == "unspecified"
    assert server._stored_task_state(Task(status=TaskStatus(state=999))) == "999"


@pytest.mark.parametrize(
    "extensions,metadata",
    [
        ([KEDI_STRUCTURED_OUTPUT_EXTENSION_URI], {}),
        ([], structured_output_request({"type": "object"})),
        (
            [KEDI_STRUCTURED_OUTPUT_EXTENSION_URI],
            structured_output_request({"description": "x" * 40000}),
        ),
    ],
)
def test_output_schema_requires_negotiation_and_bounded_size(extensions, metadata):
    with pytest.raises(ValueError):
        server._requested_output_schema(
            SimpleNamespace(requested_extensions=extensions, metadata=metadata)
        )


def test_executor_rejects_missing_ids_and_reports_execution_errors():
    from a2a.types import Message, Role, Task

    async def scenario():
        program = SimpleNamespace(invoke=AsyncMock(side_effect=ValueError("invalid task")))
        executor = server._KediAgentExecutor(program)
        events = SimpleNamespace(enqueue_event=AsyncMock())
        context = SimpleNamespace(task_id=None, context_id=None)
        with pytest.raises(RuntimeError, match="identifiers"):
            await executor.execute(context, events)
        await executor.cancel(context, events)
        context = SimpleNamespace(
            task_id="task",
            context_id="context",
            current_task=Task(id="task"),
            requested_extensions=[],
            metadata={},
            message=Message(role=Role.ROLE_USER),
            call_context=SimpleNamespace(user=SimpleNamespace(user_name="user")),
            get_user_input=lambda: "prompt",
        )
        await executor.execute(context, events)
        states = [
            getattr(event.args[0], "status", None) for event in events.enqueue_event.call_args_list
        ]
        assert any(status is not None and "invalid task" in str(status) for status in states)
        program.invoke.side_effect = asyncio.CancelledError
        with pytest.raises(asyncio.CancelledError):
            await executor.execute(context, events)

    asyncio.run(scenario())


def test_task_store_rolls_back_failed_reservation_and_deletes_entries():
    from a2a.types import Task, TaskState, TaskStatus

    async def scenario():
        underlying = SimpleNamespace(
            save=AsyncMock(), delete=AsyncMock(), list=AsyncMock(return_value="listing")
        )
        store = server._BoundedTaskStore(underlying, max_tasks_per_principal=1)
        context = SimpleNamespace(user=SimpleNamespace(user_name="user"))
        task = Task(id="t", status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED))
        underlying.save.side_effect = RuntimeError("storage unavailable")
        with pytest.raises(RuntimeError):
            await store.save(task, context)
        assert store._tasks == {}
        underlying.save.side_effect = None
        await store.save(task, context)
        underlying.save.side_effect = RuntimeError("storage unavailable")
        with pytest.raises(RuntimeError):
            await store.save(task, context)
        assert "t" in store._tasks["user"]
        assert await store.list(None, context) == "listing"
        await store.delete("absent", context)
        await store.delete("t", context)
        await store.delete("t", context)
        assert store._tasks == {}

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "declared,started", [(b"100", False), (b"invalid", False), (None, False), (None, True)]
)
def test_request_limit_catches_declared_and_streamed_oversize_bodies(declared, started):
    async def scenario():
        sent = []

        async def send(message):
            sent.append(message)

        async def receive():
            return {"type": "http.request", "body": b"12345"}

        async def app(scope, receive, send):
            if started:
                await send({"type": "http.response.start", "status": 200})
            await receive()

        middleware = server._RequestSizeLimitMiddleware(app, limit=4)
        await middleware(
            {
                "type": "http",
                "headers": [] if declared is None else [(b"content-length", declared)],
            },
            receive,
            send,
        )
        assert sent[0]["status"] == (200 if started else 413)

    asyncio.run(scenario())


def test_request_limit_allows_disconnect_messages():
    async def scenario():
        async def receive():
            return {"type": "http.disconnect"}

        async def app(scope, receive, send):
            assert await receive() == {"type": "http.disconnect"}

        await server._RequestSizeLimitMiddleware(app, limit=1)(
            {"type": "http"}, receive, AsyncMock()
        )

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "options",
    [
        {"max_concurrency": 0},
        {"max_concurrency": 2, "max_pending_tasks": 1},
        {"max_input_chars": 0},
    ],
)
def test_runtime_limits_validate_before_execution(options):
    with pytest.raises(ValueError):
        runtime.A2AServedProgram(
            **(
                {
                    "runtime": None,
                    "entry": "worker",
                    "description": "test",
                    "profile": AgentProfile.empty(),
                    "adapter": _ServedAdapter(),
                    "tools": (),
                    "max_concurrency": 1,
                }
                | options
            )
        )


def _program(tmp_path, **kwargs):
    source = tmp_path / "worker.kedi"
    source.write_text(
        "> profile: worker:\n    > instructions: Answer precisely.\n> export:\n    worker\n"
    )
    return runtime.prepare_a2a_program(
        source, entry="worker", model="test:model", adapter=_ServedAdapter(), **kwargs
    )


def test_runtime_input_output_limits_and_capacity(tmp_path):
    async def scenario():
        program = _program(
            tmp_path,
            max_concurrency=1,
            max_pending_tasks=1,
            max_input_chars=8,
            max_response_bytes=8,
        )
        try:
            for prompt in (" ", "x" * 9):
                with pytest.raises(ValueError):
                    await program.invoke(
                        principal="u", context_id="c", prompt=prompt, output_schema=None
                    )
            with pytest.raises(ValueError, match="response limit"):
                await program.invoke(
                    principal="u", context_id="c", prompt="test", output_schema=None
                )
            await program._reserve_task("u")
            with pytest.raises(RuntimeError, match="capacity"):
                await program._reserve_task("u")
            with pytest.raises(RuntimeError, match="active invocation"):
                await program.aclose()
            await program._release_task("u")
            program._pending_by_principal["u"] = 1
            with pytest.raises(RuntimeError, match="principal"):
                await program._reserve_task("u")
        finally:
            await program.aclose()
        with pytest.raises(RuntimeError, match="closed"):
            program._session("u", "new")

    asyncio.run(scenario())


def test_runtime_stringifies_non_string_outputs(tmp_path):
    async def scenario():
        program = _program(tmp_path)
        program.adapter.invoke = AsyncMock(return_value=42)
        try:
            assert (
                await program.invoke(
                    principal="u", context_id="c", prompt="test", output_schema=None
                )
                == "42"
            )
        finally:
            await program.aclose()

    asyncio.run(scenario())


def test_runtime_cleanup_continues_after_resource_errors(tmp_path):
    async def scenario():
        program = _program(tmp_path)
        program._owns_adapter = True
        program._sessions[("u", "c")] = SimpleNamespace(
            close=lambda: (_ for _ in ()).throw(ValueError("session failure"))
        )
        program.runtime.aclose = AsyncMock(side_effect=RuntimeError("runtime failure"))
        program.adapter.aclose = AsyncMock(side_effect=RuntimeError("adapter failure"))
        with pytest.raises(ValueError, match="session failure"):
            await program.aclose()
        program.runtime.aclose.assert_awaited_once()
        program.adapter.aclose.assert_awaited_once()

    asyncio.run(scenario())


def test_adapter_cleanup_supports_sync_and_async_interfaces():
    closed = []
    asyncio.run(runtime._aclose_adapter(SimpleNamespace(close=lambda: closed.append("sync"))))
    asyncio.run(
        runtime._aclose_adapter(SimpleNamespace(aclose=lambda: closed.append("async-name")))
    )
    asyncio.run(runtime._aclose_adapter(SimpleNamespace()))
    assert closed == ["sync", "async-name"]


def test_prepare_requires_export_model_and_matching_adapter(tmp_path):
    source = tmp_path / "worker.kedi"
    for text, message, adapter in [
        ("", "not an agent profile", None),
        ("> profile: worker:\n    > instructions: Answer.\n", "explicitly exported", None),
        (
            "> profile: worker:\n    > instructions: Answer.\n> export:\n    worker\n",
            "requires a model",
            None,
        ),
        (
            "> profile: worker:\n    > agent: a2a:\n        endpoint: https://agent.example\n> export:\n    worker\n",
            "cannot delegate",
            None,
        ),
        (
            "> profile: worker:\n    > adapter: pydantic\n    > model: test:model\n> export:\n    worker\n",
            "does not match",
            _ServedAdapter(),
        ),
    ]:
        source.write_text(text)
        with pytest.raises(ValueError, match=message):
            runtime.prepare_a2a_program(source, entry="worker", adapter=adapter)
    assert runtime._is_exported(
        SimpleNamespace(exports=[SimpleNamespace(names=[], export_all=True)]), "worker"
    )


def test_prepare_rejects_interactive_approval():
    from kedi.agent_adapter.approvals.policy import ApprovalPolicy

    def handler(request):
        return "deny"

    handler.telemetry_mechanism = "cli"
    profile = AgentProfile(approval=ApprovalPolicy(handler=handler))
    with pytest.raises(ValueError, match="interactive"):
        runtime._validate_headless_profile(profile)


def test_failed_prepare_preserves_primary_error_when_cleanup_fails():
    async def fail():
        raise RuntimeError("cleanup failure")

    error = ValueError("primary")
    runtime._cleanup_failed_prepare(
        SimpleNamespace(aclose=fail), owned_adapter=None, original_error=error
    )
    if hasattr(error, "__notes__"):
        assert "cleanup failure" in error.__notes__[0]


def test_failed_prepare_without_exception_notes():
    async def fail():
        raise RuntimeError("cleanup failure")

    class LegacyError(Exception):
        add_note = None

    runtime._cleanup_failed_prepare(
        SimpleNamespace(aclose=fail), owned_adapter=None, original_error=LegacyError("primary")
    )
    assert not runtime._is_exported(
        SimpleNamespace(exports=[SimpleNamespace(names=[], export_all=False)]), "worker"
    )
    assert not runtime._is_exported(
        SimpleNamespace(exports=[SimpleNamespace(names=[], export_all=True)]), "_private"
    )


def test_served_session_closes_conversation_artifacts():
    closed = []
    session = runtime._ServedSession(
        surface=SimpleNamespace(close=lambda: closed.append("surface")),
        lock=asyncio.Lock(),
        artifacts=None,
        conversation=SimpleNamespace(close=lambda: closed.append("conversation")),
    )
    session.close()
    assert closed == ["surface", "conversation"]


def test_served_program_rejects_incompatible_adapters(tmp_path, monkeypatch):
    with pytest.raises(TypeError, match="child execution"):
        runtime.A2AServedProgram(
            runtime=None,
            entry="worker",
            description="test",
            profile=AgentProfile.empty(),
            adapter=object(),
            tools=(),
            max_concurrency=1,
        )
    from kedi.core import KediRuntime

    monkeypatch.setattr(
        KediRuntime,
        "bind_agent",
        lambda *args, **kwargs: SimpleNamespace(
            agent=SimpleNamespace(shortname="unsupported"), close=lambda: None
        ),
    )
    with pytest.raises(TypeError, match="cannot execute"):
        _program(tmp_path)


def test_cli_reports_preparation_error_before_server_exists(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from kedi.a2a import cli
    from kedi.cli import main

    source = tmp_path / "agent.kedi"
    source.write_text("")

    def fail(*args, **kwargs):
        raise ValueError("invalid program")

    monkeypatch.setattr(cli, "prepare_a2a_program", fail)
    result = CliRunner().invoke(
        main, ["a2a", "serve", str(source), "--entry", "worker", "--allow-unauthenticated"]
    )
    assert result.exit_code == 1
    assert "invalid program" in result.output


@pytest.mark.parametrize("concurrent_action", ["delete", "save"])
def test_task_store_failed_save_preserves_concurrent_changes(concurrent_action):
    from a2a.types import Task, TaskState, TaskStatus

    async def scenario():
        underlying = SimpleNamespace(save=AsyncMock(), delete=AsyncMock())
        store = server._BoundedTaskStore(underlying, max_tasks_per_principal=2)
        context = SimpleNamespace(user=SimpleNamespace(user_name="user"))
        task = Task(id="first", status=TaskStatus(state=TaskState.TASK_STATE_WORKING))

        async def racing_save(task, context):
            if concurrent_action == "delete":
                await store.delete(task.id, context)
            else:
                underlying.save.side_effect = None
                await store.save(
                    Task(id="second", status=TaskStatus(state=TaskState.TASK_STATE_WORKING)),
                    context,
                )
            raise RuntimeError("storage failure")

        underlying.save.side_effect = racing_save
        with pytest.raises(RuntimeError, match="storage failure"):
            await store.save(task, context)
        assert store._tasks == (
            {} if concurrent_action == "delete" else {"user": {"second": "working"}}
        )

    asyncio.run(scenario())
