from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from a2a.client.errors import A2AClientError, AgentCardResolutionError
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentExtension,
    AgentInterface,
    APIKeySecurityScheme,
    Artifact,
    Message,
    Part,
    Role,
    SecurityRequirement,
    SecurityScheme,
    StreamResponse,
    StringList,
    Task,
    TaskState,
    TaskStatus,
)

from kedi.a2a.protocol import (
    KEDI_STRUCTURED_OUTPUT_EXTENSION_URI,
    A2AAmbiguousSendError,
    A2ARemoteResult,
    A2ARemoteTaskError,
    A2ATaskTimeoutError,
    A2ATransportError,
)
from kedi.agent_adapter.adapters import a2a as client_module
from kedi.agent_adapter.adapters.a2a import A2AAdapter
from kedi.agent_profile import A2AAuthConfig, A2AConnection, AgentProfile
from tests.test_a2a import _Output


@pytest.mark.parametrize("options", [{"unexpected": 1}, {"max_response_bytes": 0}])
def test_adapter_constructor_rejects_invalid_settings(options):
    with pytest.raises((TypeError, ValueError)):
        A2AAdapter(**options)


def test_adapter_rejects_local_tools_and_missing_connection():
    adapter = A2AAdapter()
    assert adapter.default_model() is None
    assert adapter.telemetry_model_name() is None
    assert adapter.scoped_tools == ()
    with adapter.tool_scope():
        pass
    with pytest.raises(RuntimeError, match="requires an endpoint"):
        adapter._active_connection()
    for action in (lambda: adapter.register_tool(object()), lambda: adapter.tool_scope([object()])):
        with pytest.raises(Exception, match="local .*tools"):
            action()
    with pytest.raises(Exception, match="local Kedi tools"):
        with adapter.subagent_scope(SimpleNamespace(tools=[object()])):
            pass


@pytest.mark.parametrize(
    "profile",
    [
        AgentProfile(model="model"),
        AgentProfile(effort="high"),
        AgentProfile(mcp_servers=(object(),)),
        AgentProfile(settings={"a": 1}),
        AgentProfile(codemode=True),
        AgentProfile(compaction=SimpleNamespace(enabled=True)),
    ],
)
def test_adapter_rejects_local_runtime_overrides(profile):
    with pytest.raises(Exception, match="unsupported local"):
        with A2AAdapter().profile_override(profile):
            pass


def test_adapter_invocation_validation_and_sync_wrappers(monkeypatch):
    adapter = A2AAdapter(connection=A2AConnection(endpoint="http://localhost:8000"))
    response = A2ARemoteResult(
        text="answer", data={"answer": "typed"}, task_id="t", context_id="c", state="completed"
    )
    send = AsyncMock(return_value=response)
    monkeypatch.setattr(adapter, "_send_official", send)
    assert adapter.invoke_sync(prompt="test") == "answer"
    assert adapter.produce_sync("test", output_type=_Output).answer == "typed"
    assert adapter.continue_task_sync(task_id="t", context_id="c", prompt="more") == "answer"
    assert asyncio.run(adapter.invoke(prompt="test", instructions=" guide ")) == "answer"
    assert send.call_args.args[1] == " guide\n\ntest"
    assert asyncio.run(adapter.invoke(prompt="test", instructions=" ")) == "answer"
    for kwargs in ({"unknown": True}, {"instructions": 42}):
        with pytest.raises(TypeError):
            asyncio.run(adapter.invoke(prompt="test", **kwargs))
    with pytest.raises(ValueError, match="schema or type"):
        adapter.produce_sync("test")
    with pytest.raises(ValueError, match="exceeds"):
        asyncio.run(
            adapter.produce_subagent(template="test", output_schema={"description": "x" * 40000})
        )
    send.return_value = A2ARemoteResult(
        text=None, data=None, task_id="t", context_id="c", state="completed"
    )
    with pytest.raises(A2ARemoteTaskError, match="text result"):
        adapter.invoke_sync(prompt="test")
    with pytest.raises(A2ARemoteTaskError, match="text result"):
        adapter.continue_task_sync(task_id="t", prompt="test")
    with pytest.raises(A2ARemoteTaskError, match="non-object"):
        asyncio.run(adapter.produce_subagent(template="test", output_schema={"type": "object"}))
    with pytest.raises(ValueError, match="non-empty"):
        adapter.continue_task_sync(task_id=" ", prompt="test")


def test_status_and_cancel_preserve_task_identifiers(monkeypatch):
    adapter = A2AAdapter(connection=A2AConnection(endpoint="http://localhost", poll_interval=0.001))
    task = Task(id="t", status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED))
    client = SimpleNamespace(
        get_task=AsyncMock(side_effect=[A2AClientError("transient"), task]),
        cancel_task=AsyncMock(return_value=task),
    )

    @asynccontextmanager
    async def scope(*args, **kwargs):
        yield client, None

    monkeypatch.setattr(adapter, "_client_scope", scope)
    assert adapter.get_task_sync("t").task_id == "t"
    assert adapter.cancel_task_sync("t").task_id == "t"
    client.cancel_task.side_effect = A2AClientError("disconnected")
    with pytest.raises(A2ATransportError) as exc:
        adapter.cancel_task_sync("t")
    assert exc.value.task_id == "t"
    client.get_task.side_effect = A2AClientError("disconnected")
    with pytest.raises(A2ATransportError) as exc:
        adapter.get_task_sync("t")
    assert exc.value.task_id == "t"
    with pytest.raises(ValueError):
        adapter.get_task_sync("t", history_length=-1)


@pytest.mark.parametrize("operation", ["get", "cancel", "send", "scope", "recover", "read"])
def test_missing_optional_sdk_has_actionable_error(monkeypatch, operation):
    adapter = A2AAdapter(connection=A2AConnection(endpoint="http://localhost"))
    monkeypatch.setitem(sys.modules, "a2a.types", None)
    monkeypatch.setitem(sys.modules, "a2a.client", None)
    monkeypatch.setitem(sys.modules, "a2a.client.errors", None)

    async def scenario():
        if operation == "get":
            await adapter.get_task("t")
        elif operation == "cancel":
            await adapter.cancel_task("t")
        elif operation == "send":
            await adapter._send_official(
                adapter._connection, "test", output_schema=None, task_id=None, context_id=None
            )
        elif operation == "scope":
            async with adapter._client_scope(adapter._connection, structured=False):
                pass
        elif operation == "read":
            await client_module._safe_status_read(
                AsyncMock(), connection=adapter._connection, task_id="t"
            )
        else:
            await client_module._recover_observation(
                None,
                connection=adapter._connection,
                call_context=None,
                accumulator=client_module._ResponseAccumulator(max_bytes=1024),
                deadline=0,
                event_resolver=None,
            )

    with pytest.raises(RuntimeError, match="kedi\\[a2a\\]"):
        asyncio.run(scenario())


@pytest.mark.parametrize(
    "mode", ["ambiguous", "recover", "empty", "unfinished", "timeout", "direct"]
)
def test_send_does_not_repeat_ambiguous_execution(monkeypatch, mode):
    adapter = A2AAdapter(connection=A2AConnection(endpoint="http://localhost", task_timeout=0.01))
    calls = []

    async def send(request, **kwargs):
        calls.append(request)
        if mode == "timeout":
            await asyncio.sleep(1)
        if mode in {"recover", "unfinished"}:
            yield StreamResponse(
                task=Task(id="t", status=TaskStatus(state=TaskState.TASK_STATE_WORKING))
            )
        if mode in {"ambiguous", "recover"}:
            raise A2AClientError("connection lost")
        if mode == "direct":
            yield StreamResponse(message=Message(role=Role.ROLE_AGENT, parts=[Part(text="answer")]))

    @asynccontextmanager
    async def scope(*args, **kwargs):
        yield SimpleNamespace(send_message=send), None

    async def recover(client, **kwargs):
        kwargs["accumulator"].add_task(
            Task(
                id="t",
                status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
                artifacts=[Artifact(artifact_id="a", parts=[Part(text="recovered")])],
            )
        )

    monkeypatch.setattr(adapter, "_client_scope", scope)
    monkeypatch.setattr(client_module, "_recover_observation", recover)
    if mode in {"ambiguous", "empty", "timeout"}:
        expected = {
            "ambiguous": A2AAmbiguousSendError,
            "empty": A2ATransportError,
            "timeout": A2ATaskTimeoutError,
        }[mode]
        with pytest.raises(expected):
            adapter.invoke_sync(prompt="test")
    else:
        assert adapter.invoke_sync(prompt="test") == ("answer" if mode == "direct" else "recovered")
    assert len(calls) == 1


def test_agent_card_validation_rejects_unsafe_routes_and_auth():
    endpoint = "https://agent.example"
    for interfaces in (
        [],
        [AgentInterface(url="invalid", protocol_binding="JSONRPC")],
        [AgentInterface(url="https://other.example/rpc", protocol_binding="JSONRPC")],
    ):
        with pytest.raises(A2ARemoteTaskError):
            client_module._validate_agent_card(endpoint, AgentCard(supported_interfaces=interfaces))
    assert (
        client_module._url_origin(
            __import__("urllib.parse", fromlist=["urlsplit"]).urlsplit("http://localhost")
        )[2]
        == 80
    )
    auth = A2AAuthConfig(scheme="api_key", secret_env="TEST", header="X-Key")
    scheme = SecurityScheme(
        api_key_security_scheme=APIKeySecurityScheme(location="header", name="X-Key")
    )
    assert client_module._auth_matches_scheme(auth, scheme)
    assert not client_module._auth_matches_scheme(auth, SecurityScheme())
    card = AgentCard(
        security_requirements=[SecurityRequirement(schemes={"key": StringList()})],
        security_schemes={"key": scheme},
    )
    client_module._validate_card_auth(card, auth)
    for configured in (None, A2AAuthConfig(scheme="bearer", secret_env="TEST")):
        with pytest.raises(A2ATransportError, match="authentication scheme"):
            client_module._validate_card_auth(card, configured)
    client_module._validate_card_auth(
        AgentCard(security_requirements=[SecurityRequirement()]), None
    )
    card.security_requirements[0].schemes["second"].CopyFrom(StringList())
    with pytest.raises(A2ATransportError):
        client_module._validate_card_auth(card, auth)
    assert not client_module._supports_kedi_structured_output(
        AgentCard(
            capabilities=AgentCapabilities(
                extensions=[
                    AgentExtension(uri="other"),
                    AgentExtension(uri=KEDI_STRUCTURED_OUTPUT_EXTENSION_URI),
                ]
            )
        )
    )


def test_agent_card_resolution_retries_transient_failures(monkeypatch):
    adapter = A2AAdapter()
    resolver = SimpleNamespace(
        get_agent_card=AsyncMock(side_effect=AgentCardResolutionError("unavailable"))
    )
    connection = A2AConnection(endpoint="https://agent.example", poll_interval=0.001)
    with pytest.raises(A2ATransportError, match="could not be resolved"):
        asyncio.run(adapter._agent_card(connection, None, lambda *a: resolver))
    assert resolver.get_agent_card.await_count == 3


def test_record_suspended_coordinates_and_deadlines():
    conversation = SimpleNamespace(result_state={}, resume_state={})
    error = A2ARemoteTaskError("needs input", task_id="t", context_id="c", state="input_required")
    client_module._record_error_coordinates(conversation, "https://agent.example", error)
    assert conversation.resume_state["a2a"]["task_id"] == "t"
    client_module._record_error_coordinates(
        conversation, "https://agent.example", A2ATransportError("no task")
    )
    assert conversation.resume_state["a2a"]["task_id"] == "t"

    async def scenario():
        with pytest.raises(asyncio.TimeoutError):
            client_module._remaining(0)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "mode",
    [
        "subscribe",
        "timeout",
        "failures",
        "read_timeout",
        "expired_read",
        "expire_after_read",
        "pending",
        "no_id",
    ],
)
def test_observation_recovery_states(mode):
    async def scenario():
        connection = A2AConnection(
            endpoint="http://localhost", poll_interval=0.001, request_timeout=0.01
        )
        accumulator = client_module._ResponseAccumulator(max_bytes=1024)
        if mode != "no_id":
            accumulator.add_task(
                Task(id="t", status=TaskStatus(state=TaskState.TASK_STATE_WORKING))
            )
        deadline = asyncio.get_running_loop().time() + (0.015 if mode == "expired_read" else 0.1)
        calls = []

        async def subscribe(*args, **kwargs):
            if mode == "subscribe":
                yield StreamResponse(
                    task=Task(id="t", status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED))
                )
                return
            if mode == "timeout":
                raise asyncio.TimeoutError
            raise A2AClientError("disconnected")

        async def get(*args, **kwargs):
            calls.append(True)
            if mode == "failures":
                raise A2AClientError("unavailable")
            if mode == "expired_read":
                await asyncio.sleep(1)
            if mode == "read_timeout" and len(calls) < 2:
                raise asyncio.TimeoutError
            if mode == "expire_after_read":
                import time

                time.sleep(0.11)
            pending = mode == "expire_after_read" or (mode == "pending" and len(calls) == 1)
            return Task(
                id="t",
                status=TaskStatus(
                    state=TaskState.TASK_STATE_WORKING
                    if pending
                    else TaskState.TASK_STATE_COMPLETED,
                    message=Message(role=Role.ROLE_AGENT, parts=[Part(text="progress")]),
                ),
            )

        arguments = dict(
            connection=connection,
            call_context=None,
            accumulator=accumulator,
            deadline=deadline,
            event_resolver=SimpleNamespace(complete_message=lambda *args: None),
        )
        client = SimpleNamespace(subscribe=subscribe, get_task=get)
        if mode in {"timeout", "expired_read", "expire_after_read", "failures", "no_id"}:
            with pytest.raises((A2ATransportError, A2ATaskTimeoutError)):
                await client_module._recover_observation(client, **arguments)
        else:
            await client_module._recover_observation(client, **arguments)
            assert accumulator.is_complete

    asyncio.run(scenario())


def test_profile_connection_and_noncontinuable_error_checkpoint():
    adapter = A2AAdapter()
    connection = A2AConnection(endpoint="http://localhost")
    with adapter.profile_override(AgentProfile(a2a_connection=connection)):
        assert adapter._active_connection() is connection
    conversation = SimpleNamespace(result_state={}, resume_state={})
    client_module._record_error_coordinates(
        conversation, connection.endpoint, A2ATransportError("lost", task_id="t", state="working")
    )
    assert conversation.result_state["a2a"]["task_id"] == "t"
    assert not conversation.resume_state
    assert not client_module._supports_kedi_structured_output(AgentCard())
    client_module._validate_card_auth(AgentCard(), None)


def test_client_rejects_unadvertised_structured_output(monkeypatch):
    adapter = A2AAdapter()
    connection = A2AConnection(endpoint="https://agent.example")
    monkeypatch.setattr(
        adapter,
        "_agent_card",
        AsyncMock(
            return_value=AgentCard(
                supported_interfaces=[
                    AgentInterface(url="https://agent.example/rpc", protocol_binding="JSONRPC")
                ]
            )
        ),
    )

    async def scenario():
        with pytest.raises(A2ARemoteTaskError, match="does not advertise"):
            async with adapter._client_scope(connection, structured=True):
                pass

    asyncio.run(scenario())


def test_stream_events_report_direct_message_text(monkeypatch):
    from contextlib import contextmanager

    events = []
    resolver = SimpleNamespace(
        complete_message=lambda *args: events.append(args),
        complete_run=lambda **kw: events.append(kw),
    )

    @contextmanager
    def event_run(*args):
        yield resolver

    async def send(*args, **kwargs):
        yield StreamResponse(message=Message(role=Role.ROLE_AGENT, parts=[Part(text="answer")]))

    @asynccontextmanager
    async def scope(*args, **kwargs):
        yield SimpleNamespace(send_message=send), None

    monkeypatch.setattr(client_module, "agent_event_run", event_run)
    adapter = A2AAdapter(connection=A2AConnection(endpoint="http://localhost"))
    monkeypatch.setattr(adapter, "_client_scope", scope)
    assert adapter.invoke_sync(prompt="test") == "answer"
    assert events[-1] == {"final_text": "answer"}
