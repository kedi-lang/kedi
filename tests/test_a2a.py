from __future__ import annotations

import asyncio
import base64
import socket
import sys
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from click.testing import CliRunner
from google.protobuf.json_format import MessageToDict
from pydantic import BaseModel

from kedi.a2a.auth import A2AServerAuth
from kedi.a2a.protocol import (
    DEFAULT_A2A_MAX_SCHEMA_BYTES,
    KEDI_STRUCTURED_OUTPUT_ARTIFACT_NAME,
    KEDI_STRUCTURED_OUTPUT_EXTENSION_URI,
    KEDI_STRUCTURED_OUTPUT_VERSION,
)
from kedi.a2a.runtime import prepare_a2a_program
from kedi.a2a.server import (
    _agent_card,
    _authentication_error,
    _BoundedTaskStore,
    _safe_failure_message,
    create_a2a_app,
)
from kedi.agent_adapter.adapters.a2a import (
    _MAX_AGENT_CARD_CACHE_ENTRIES,
    A2AAdapter,
    _recover_observation,
    _ResponseAccumulator,
)
from kedi.agent_adapter.conversation import AdapterConversationContext
from kedi.agent_adapter.meta import AdapterCapabilities, AgentAdapterKind
from kedi.agent_adapter.registry import adapter_capabilities, build_adapter
from kedi.agent_adapter.subagents import MAX_FINAL_SCHEMA_BYTES
from kedi.agent_profile import A2AAuthConfig, A2AConnection, AgentProfile
from kedi.agent_profile.directives import eval_agent_directive
from kedi.cli import main as cli_main
from kedi.core import KediRuntime
from kedi.lang import compile_program, parse_program
from kedi.lang.ast import AgentDirectiveStmt

pytest.importorskip("a2a")


class _Output(BaseModel):
    answer: str


def test_a2a_schema_limit_matches_runtime_validation() -> None:
    assert DEFAULT_A2A_MAX_SCHEMA_BYTES == MAX_FINAL_SCHEMA_BYTES


@pytest.mark.parametrize("field", ["request_timeout", "task_timeout", "poll_interval"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a2a_connection_rejects_nonfinite_timeouts(field: str, value: float) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        A2AConnection(endpoint="https://agent.example.com", **{field: value})


def test_a2a_client_rejects_unicode_header_names() -> None:
    with pytest.raises(ValueError, match="HTTP header name"):
        A2AAuthConfig(scheme="api_key", secret_env="SECRET", header="X-Key-\u00e9")


@pytest.mark.parametrize("handler_fails", [False, True])
def test_a2a_lifespan_closes_sdk_before_program(
    monkeypatch: pytest.MonkeyPatch, handler_fails: bool
) -> None:
    from a2a.server.request_handlers import DefaultRequestHandler

    closed: list[str] = []

    async def close_handler(_self: Any) -> None:
        closed.append("handler")
        if handler_fails:
            raise RuntimeError("shutdown failure")

    async def close_program() -> None:
        closed.append("program")

    monkeypatch.setattr(DefaultRequestHandler, "aclose", close_handler)
    program: Any = SimpleNamespace(entry="worker", description="Worker", aclose=close_program)
    app = create_a2a_app(
        program, public_url="http://127.0.0.1:8000", auth=None, allow_unauthenticated_loopback=True
    )

    async def scenario() -> None:
        async with app.router.lifespan_context(app):
            assert not closed

    if handler_fails:
        with pytest.raises(RuntimeError, match="shutdown failure"):
            asyncio.run(scenario())
    else:
        asyncio.run(scenario())
    assert closed == ["handler", "program"]


def test_a2a_http_auth_handles_unicode_without_server_errors() -> None:
    import httpx

    async def scenario() -> None:
        program: Any = SimpleNamespace(entry="worker", description="Worker")
        auth = A2AServerAuth(scheme="basic", username="user-\u00e9", secret="secret-\u00e9")
        app = create_a2a_app(program, public_url="http://127.0.0.1:8000", auth=auth)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8000"
        ) as client:
            for secret, status in (("secret-\u00e9", 200), ("wrong-\u00e9", 401)):
                encoded = base64.b64encode(f"user-\u00e9:{secret}".encode()).decode()
                response = await client.get(
                    "/.well-known/agent-card.json",
                    headers={"Authorization": f"Basic {encoded}"},
                )
                assert response.status_code == status
        for scheme, headers in (
            ("bearer", {b"Authorization": b"Bearer \xe9"}),
            ("api_key", {b"X-API-Key": b"\xe9"}),
        ):
            app = create_a2a_app(
                program,
                public_url="http://127.0.0.1:8000",
                auth=A2AServerAuth(scheme=scheme, secret="secret"),
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8000"
            ) as client:
                response = await client.get("/.well-known/agent-card.json", headers=headers)
                assert response.status_code == 401

    asyncio.run(scenario())


def test_a2a_agent_card_advertises_the_versioned_structured_contract() -> None:
    program: Any = SimpleNamespace(entry="worker", description="Worker")
    card = _agent_card(
        program,
        endpoint="http://127.0.0.1:8000",
        rpc_path="/a2a",
        auth=None,
    )

    extension = card.capabilities.extensions[0]
    params = MessageToDict(extension.params)
    assert extension.uri == KEDI_STRUCTURED_OUTPUT_EXTENSION_URI
    assert extension.required is False
    assert params == {
        "version": float(KEDI_STRUCTURED_OUTPUT_VERSION),
        "mode": "request-schema",
        "schemaDialect": "https://json-schema.org/draft/2020-12/schema",
        "maxSchemaBytes": float(DEFAULT_A2A_MAX_SCHEMA_BYTES),
        "resultArtifactName": KEDI_STRUCTURED_OUTPUT_ARTIFACT_NAME,
    }


class _ServedAdapter:
    kind: AgentAdapterKind = "agent-framework"
    shortname = "a2a-test"
    capabilities = AdapterCapabilities(
        supports_structured_output=True,
        supports_profile_override=True,
        supports_subagents=True,
        supports_stateful_history=False,
    )

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.closed = False
        self.slow_started = threading.Event()
        self.release_slow = threading.Event()

    def default_model(self) -> str:
        return "test:model"

    def telemetry_model_name(self) -> str:
        return "test:model"

    @contextmanager
    def profile_override(self, _profile: AgentProfile) -> Iterator[None]:
        yield

    @contextmanager
    def tool_scope(self, _tools: Sequence[Any] = ()) -> Iterator[None]:
        yield

    @property
    def scoped_tools(self) -> tuple[Any, ...]:
        return ()

    def subagent_scope(self, _request: Any) -> AbstractContextManager[None]:
        return self.profile_override(AgentProfile.empty())

    @staticmethod
    def subagent_run_kwargs(_limits: Any) -> dict[str, Any]:
        return {}

    async def invoke(self, *, prompt: str, **_kwargs: Any) -> str:
        self.calls.append(prompt)
        if prompt.startswith("slow:"):
            self.slow_started.set()
            await asyncio.to_thread(self.release_slow.wait)
        return f"remote:{prompt}"

    def invoke_sync(self, *, prompt: str, **_kwargs: Any) -> str:
        return f"remote:{prompt}"

    async def produce(
        self,
        *,
        template: str,
        output_schema: dict[str, Any] | None = None,
        output_type: type[Any] | None = None,
        **_kwargs: Any,
    ) -> Mapping[str, Any]:
        return {"answer": f"structured:{template}"}

    def produce_sync(
        self,
        *,
        template: str,
        output_schema: dict[str, Any] | None = None,
        output_type: type[Any] | None = None,
        **_kwargs: Any,
    ) -> Mapping[str, Any]:
        return {"answer": f"structured:{template}"}

    async def produce_subagent(
        self,
        *,
        template: str,
        output_schema: dict[str, Any],
        **_kwargs: Any,
    ) -> Mapping[str, Any]:
        assert output_schema["type"] == "object"
        if "task_summary" in output_schema.get("properties", {}):
            return {
                "task_summary": f"structured:{template}",
                "final_result": {"value": 11},
            }
        return {"answer": f"structured:{template}"}

    async def aclose(self) -> None:
        self.closed = True


def test_a2a_auth_sources_headers_and_redaction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("KEDI_TEST_A2A_PASSWORD", "correct horse")
    basic = A2AAuthConfig(
        scheme="basic",
        username="kedi",
        secret_env="KEDI_TEST_A2A_PASSWORD",
    )
    assert basic.headers()["Authorization"].startswith("Basic ")
    assert "correct horse" not in repr(basic)
    assert "correct horse" not in repr(basic.identity())

    secret_file = tmp_path / "token"
    secret_file.write_text("token-value\n", encoding="utf-8")
    bearer = A2AAuthConfig(scheme="bearer", secret_file=str(secret_file))
    assert bearer.headers() == {"Authorization": "Bearer token-value"}

    api_key = A2AAuthConfig(
        scheme="api_key",
        provider=lambda: "api-secret",
        header="X-Kedi-Key",
    )
    assert api_key.headers() == {"X-Kedi-Key": "api-secret"}
    assert "api-secret" not in repr(api_key)


def test_a2a_server_auth_accepts_case_insensitive_http_schemes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KEDI_TEST_A2A_PASSWORD", "secret")
    basic_header = A2AAuthConfig(
        scheme="basic",
        username="kedi",
        secret_env="KEDI_TEST_A2A_PASSWORD",
    ).headers()["Authorization"]
    basic = A2AServerAuth(scheme="basic", username="kedi", secret="secret")
    assert basic.authenticate({"authorization": basic_header.replace("Basic", "bAsIc")}) == "kedi"

    bearer = A2AServerAuth(scheme="bearer", secret="token")
    assert bearer.authenticate({"authorization": "bEaReR token"}) == bearer.principal


@pytest.mark.parametrize(
    ("auth", "challenge"),
    [
        (
            A2AServerAuth(scheme="basic", username="kedi", secret="secret"),
            'Basic realm="kedi-a2a", charset="UTF-8"',
        ),
        (
            A2AServerAuth(scheme="bearer", secret="secret"),
            'Bearer realm="kedi-a2a"',
        ),
        (A2AServerAuth(scheme="api_key", secret="secret"), None),
    ],
)
def test_a2a_authentication_errors_advertise_only_the_configured_scheme(
    auth: A2AServerAuth,
    challenge: str | None,
) -> None:
    response = _authentication_error(None, Exception(), auth=auth)

    assert response.headers["cache-control"] == "no-store"
    assert response.headers.get("www-authenticate") == challenge


def test_a2a_failure_messages_redact_credentials() -> None:
    message = _safe_failure_message(
        RuntimeError(
            "upstream rejected api_key=do-not-leak and Authorization: Bearer also-do-not-leak"
        )
    )

    assert "do-not-leak" not in message
    assert "also-do-not-leak" not in message
    assert "api_key=<redacted>" in message
    assert "Bearer <redacted>" in message


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://example.com",
        "https://user:secret@example.com",
        "https://example.com?token=secret",
        "file:///tmp/agent",
    ],
)
def test_a2a_connection_rejects_unsafe_endpoints(endpoint: str) -> None:
    with pytest.raises(ValueError):
        A2AConnection(endpoint=endpoint)


def test_a2a_agent_syntax_parses_at_root_procedure_and_profile() -> None:
    source = """
> agent: a2a:
    endpoint: http://127.0.0.1:8000
    auth:
        scheme: bearer
        token_env: A2A_TOKEN

> profile: remote:
    > agent: a2a:
        endpoint: https://research.example.com/agent
        auth:
            scheme: api_key
            api_key_env: RESEARCH_KEY
            header: X-Research-Key

@ask(endpoint: str):
    > agent: a2a:
        endpoint: `endpoint`
        auth:
            scheme: basic
            username: kedi
            password_env: A2A_PASSWORD
    >> The answer is [answer].
"""
    program = parse_program(source)
    root = program.ordered_toplevel[0]
    assert isinstance(root, AgentDirectiveStmt)
    assert root.plain_value == "a2a"
    assert [field.name for field in root.connection_fields] == ["endpoint", "auth"]
    assert program.profile_defs["remote"].agent_members[0].plain_value == "a2a"
    procedure_agent = program.procedures[0].body[0]
    assert isinstance(procedure_agent, AgentDirectiveStmt)
    assert procedure_agent.connection_fields[0].backtick_code == "endpoint"


def test_a2a_profile_evaluation_and_transport_selection_are_atomic() -> None:
    statement = parse_program(
        """
> agent: a2a:
    endpoint: http://localhost:9000
    auth:
        scheme: basic
        username: kedi
        password_env: A2A_PASSWORD
"""
    ).ordered_toplevel[0]
    assert isinstance(statement, AgentDirectiveStmt)
    profile = eval_agent_directive(KediRuntime(), statement, {})
    assert profile.adapter_shortname == "a2a"
    assert profile.a2a_connection is not None
    assert profile.a2a_connection.endpoint == "http://localhost:9000"
    assert profile.a2a_connection.auth is not None
    assert profile.a2a_connection.auth.secret_env == "A2A_PASSWORD"

    replaced = profile.merge(AgentProfile(adapter_kind="agent-harness", adapter_shortname="codex"))
    assert replaced.adapter_shortname == "codex"
    assert replaced.a2a_connection is None
    assert replaced.agent_command is None


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            "> agent: a2a:\n"
            "    endpoint: http://localhost:8000\n"
            "    endpoint: http://localhost:9000\n",
            "duplicate agent connection field 'endpoint'",
        ),
        (
            "> agent: a2a:\n    command: run\n",
            "A2A agent connection has unsupported field(s): command",
        ),
        (
            "> agent: acp:\n    endpoint: http://localhost:8000\n",
            "ACP agent connection has unsupported field(s): endpoint",
        ),
        (
            "> agent: codex:\n    command: run\n",
            "Agent 'codex' does not accept a connection body",
        ),
        (
            "> agent: a2a:\n    endpoint: `42`\n",
            "A2A agent connection requires a string 'endpoint'",
        ),
    ],
)
def test_agent_connection_bodies_reject_invalid_shapes(source: str, message: str) -> None:
    statement = parse_program(source).ordered_toplevel[0]
    assert isinstance(statement, AgentDirectiveStmt)
    with pytest.raises(Exception) as raised:
        eval_agent_directive(KediRuntime(), statement, {})
    assert message in str(raised.value)


@pytest.mark.parametrize("extra_source", ["credential_file", "credential_env", "password_env"])
def test_a2a_auth_rejects_ambiguous_credential_sources(extra_source: str) -> None:
    statement = parse_program(
        f"""
> agent: a2a:
    endpoint: http://localhost:8000
    auth:
        scheme: bearer
        token_env: A2A_TOKEN
        {extra_source}: A2A_OTHER
"""
    ).ordered_toplevel[0]
    assert isinstance(statement, AgentDirectiveStmt)
    with pytest.raises(ValueError, match="exactly one credential source"):
        eval_agent_directive(KediRuntime(), statement, {})


def test_a2a_auth_rejects_credential_source_for_another_scheme() -> None:
    statement = parse_program(
        "> agent: a2a:\n    endpoint: http://localhost:8000\n"
        "    auth:\n        scheme: bearer\n        password_env: SECRET\n"
    ).ordered_toplevel[0]
    assert isinstance(statement, AgentDirectiveStmt)
    with pytest.raises(ValueError, match="does not match auth scheme"):
        eval_agent_directive(KediRuntime(), statement, {})


def test_a2a_adapter_registry_metadata() -> None:
    adapter = build_adapter("a2a")
    assert isinstance(adapter, A2AAdapter)
    assert adapter.kind == "agent-harness"
    assert adapter_capabilities("a2a") == A2AAdapter.capabilities
    with pytest.raises(TypeError, match="model selection"):
        A2AAdapter(model="local:model")


def test_a2a_nested_cli_help_reaches_serve_command() -> None:
    result = CliRunner().invoke(cli_main, ["a2a", "serve", "--help"])

    assert result.exit_code == 0
    assert "Usage: kedi a2a serve" in result.output
    assert "--max-pending-tasks" in result.output


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10"])
def test_a2a_unauthenticated_serve_rejects_public_bind_with_loopback_url(
    tmp_path: Path, host: str
) -> None:
    source = tmp_path / "agent.kedi"
    source.write_text("", encoding="utf-8")
    result = CliRunner().invoke(
        cli_main,
        [
            "a2a",
            "serve",
            str(source),
            "--entry",
            "remote",
            "--host",
            host,
            "--public-url",
            "http://127.0.0.1:8000",
            "--allow-unauthenticated",
        ],
    )
    assert result.exit_code == 2
    assert "requires a loopback bind host" in result.output


def test_a2a_serve_reports_missing_optional_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "agent.kedi"
    source.write_text("", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "kedi.a2a.server", None)

    result = CliRunner().invoke(
        cli_main,
        [
            "a2a",
            "serve",
            str(source),
            "--entry",
            "remote",
            "--allow-unauthenticated",
        ],
    )

    assert result.exit_code == 1
    assert "uv add 'kedi[a2a]'" in result.output


def test_prepare_a2a_program_closes_owned_resources_when_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import kedi.a2a.runtime as runtime_module

    source = tmp_path / "agent.kedi"
    source.write_text(
        "> profile: remote:\n    > system: Answer exactly.\n\n> export:\n    remote\n",
        encoding="utf-8",
    )
    adapter = _ServedAdapter()
    captured: dict[str, KediRuntime] = {}
    compile_original = runtime_module.compile_program

    def capture_runtime(*args: Any, **kwargs: Any) -> KediRuntime:
        runtime = compile_original(*args, **kwargs)
        captured["runtime"] = runtime
        return runtime

    monkeypatch.setattr(runtime_module, "compile_program", capture_runtime)
    monkeypatch.setattr(runtime_module, "build_adapter", lambda *_args, **_kwargs: adapter)

    with pytest.raises(ValueError, match="limits must be positive"):
        prepare_a2a_program(
            source,
            entry="remote",
            model="test:model",
            max_concurrency=0,
        )

    assert captured["runtime"]._closed is True
    assert adapter.closed is True


def test_served_a2a_program_close_is_idempotent_and_terminal(tmp_path: Path) -> None:
    source = tmp_path / "agent.kedi"
    source.write_text(
        "> profile: remote:\n    > system: Answer exactly.\n\n> export:\n    remote\n",
        encoding="utf-8",
    )
    adapter = _ServedAdapter()
    served = prepare_a2a_program(
        source,
        entry="remote",
        model="test:model",
        adapter=adapter,
    )

    async def scenario() -> None:
        await served.aclose()
        await served.aclose()
        with pytest.raises(RuntimeError, match="served program is closed"):
            await served.invoke(
                principal="test",
                context_id="context",
                prompt="hello",
                output_schema=None,
            )

    asyncio.run(scenario())
    assert adapter.closed is False


@pytest.mark.parametrize("cancel_queued", [False, True])
def test_served_a2a_queued_sessions_are_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel_queued: bool
) -> None:
    source = tmp_path / "agent.kedi"
    source.write_text(
        "> profile: remote:\n    > system: Answer exactly.\n\n> export:\n    remote\n",
        encoding="utf-8",
    )
    adapter = _ServedAdapter()
    served = prepare_a2a_program(
        source,
        entry="remote",
        model="test:model",
        adapter=adapter,
        max_concurrency=1,
        max_sessions=2,
    )

    async def scenario() -> None:
        started, release = asyncio.Event(), asyncio.Event()

        async def invoke(*, prompt: str, **_kwargs: Any) -> str:
            if prompt == "first":
                started.set()
                await release.wait()
            return prompt

        monkeypatch.setattr(adapter, "invoke", invoke)

        async def call(context: str) -> Any:
            return await served.invoke(
                principal="user", context_id=context, prompt=context, output_schema=None
            )

        first = asyncio.create_task(call("first"))
        await started.wait()
        second = asyncio.create_task(call("second"))
        await asyncio.sleep(0)
        session = served._sessions[("user", "second")]
        assert session.borrowers == 1
        if cancel_queued:
            second.cancel()
            with pytest.raises(asyncio.CancelledError):
                await second
            assert session.borrowers == 0
        else:
            with pytest.raises(RuntimeError, match="session capacity"):
                await call("third")
            assert not session.surface._closed
        release.set()
        assert await first == "first"
        if not cancel_queued:
            assert await second == "second"
        assert await call("third") == "third"
        assert served._pending_tasks == 0
        assert not served._invocations
        await served.aclose()

    asyncio.run(asyncio.wait_for(scenario(), timeout=5))


@pytest.mark.parametrize("owns_adapter", [False, True])
def test_served_a2a_shutdown_drains_active_and_queued_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, owns_adapter: bool
) -> None:
    import kedi.a2a.runtime as runtime_module

    source = tmp_path / "agent.kedi"
    source.write_text(
        "> profile: remote:\n    > system: Answer exactly.\n\n> export:\n    remote\n",
        encoding="utf-8",
    )
    adapter = _ServedAdapter()
    monkeypatch.setattr(runtime_module, "build_adapter", lambda *_args: adapter)
    served = prepare_a2a_program(
        source,
        entry="remote",
        model="test:model",
        adapter=None if owns_adapter else adapter,
        max_concurrency=1,
    )

    async def scenario() -> None:
        started, cancelling, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def invoke(**_kwargs: Any) -> str:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelling.set()
                await release.wait()
            return "unreachable"

        monkeypatch.setattr(adapter, "invoke", invoke)

        async def call(context: str) -> Any:
            return await served.invoke(
                principal="user", context_id=context, prompt="work", output_schema=None
            )

        active = asyncio.create_task(call("first"))
        await started.wait()
        queued = asyncio.create_task(call("second"))
        await asyncio.sleep(0)
        sessions = tuple(served._sessions.values())
        closing = asyncio.create_task(served.aclose())
        await cancelling.wait()
        assert all(not session.surface._closed for session in sessions)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        with pytest.raises(RuntimeError, match="served program is closed"):
            await call("new")
        release.set()
        await asyncio.gather(served.aclose(), served.aclose())
        assert active.cancelled() and queued.cancelled()
        assert all(session.surface._closed for session in sessions)
        assert served.runtime._closed
        assert served._pending_tasks == 0
        assert not served._invocations
        assert not served._sessions
        assert adapter.closed is owns_adapter

    asyncio.run(asyncio.wait_for(scenario(), timeout=5))


def test_a2a_auth_supports_utf8_basic_and_rejects_unicode_mismatches() -> None:
    auth = A2AServerAuth(scheme="basic", username="user-\u00e9", secret="secret-\u00e9")
    encoded = base64.b64encode("user-\u00e9:secret-\u00e9".encode()).decode()
    assert auth.authenticate({"authorization": f"Basic {encoded}"}) == "user-\u00e9"
    for scheme, header, value in (
        ("basic", "authorization", f"Basic {encoded}"),
        ("bearer", "authorization", "Bearer secret-\u00e9"),
        ("api_key", "X-API-Key", "secret-\u00e9"),
    ):
        auth = A2AServerAuth(
            scheme=scheme, secret="secret", username="user" if scheme == "basic" else None
        )
        assert auth.authenticate({header: value}) is None
    with pytest.raises(ValueError, match="header is invalid"):
        A2AServerAuth(scheme="api_key", secret="secret", header="X-Key-\u00e9")


def test_a2a_cli_closes_prepared_program_when_app_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import kedi.a2a.cli as cli_module
    import kedi.a2a.server as server_module

    source = tmp_path / "agent.kedi"
    source.write_text("", encoding="utf-8")

    class Prepared:
        closed = 0

        async def aclose(self) -> None:
            self.closed += 1

    prepared = Prepared()
    monkeypatch.setattr(cli_module, "prepare_a2a_program", lambda *_args, **_kwargs: prepared)

    def reject_app(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError("invalid server configuration")

    monkeypatch.setattr(server_module, "create_a2a_app", reject_app)

    result = CliRunner().invoke(
        cli_main,
        [
            "a2a",
            "serve",
            str(source),
            "--entry",
            "remote",
            "--allow-unauthenticated",
        ],
    )

    assert result.exit_code == 1
    assert "invalid server configuration" in result.output
    assert prepared.closed == 1


def test_a2a_stream_recovery_polls_known_task_without_resending() -> None:
    from a2a.client.errors import A2AClientError
    from a2a.types import Artifact, Part, Task, TaskState, TaskStatus

    class RecoveringClient:
        send_calls = 0
        status_calls = 0

        async def subscribe(self, _request: Any, *, context: Any):
            raise A2AClientError("stream disconnected")
            yield  # pragma: no cover

        async def get_task(self, _request: Any, *, context: Any) -> Any:
            self.status_calls += 1
            return Task(
                id="task-1",
                context_id="context-1",
                status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
                artifacts=[
                    Artifact(
                        artifact_id="result",
                        name="result",
                        parts=[Part(text="recovered")],
                    )
                ],
            )

    async def scenario() -> None:
        client = RecoveringClient()
        accumulator = _ResponseAccumulator(max_bytes=1_024)
        accumulator.add_task(
            Task(
                id="task-1",
                context_id="context-1",
                status=TaskStatus(state=TaskState.TASK_STATE_WORKING),
            )
        )
        await _recover_observation(
            client,
            connection=A2AConnection(
                endpoint="http://127.0.0.1:8000",
                request_timeout=1,
                task_timeout=2,
                poll_interval=0.01,
            ),
            call_context=None,
            accumulator=accumulator,
            deadline=asyncio.get_running_loop().time() + 2,
            event_resolver=None,
        )
        assert client.send_calls == 0
        assert client.status_calls == 1
        assert accumulator.result(structured=False).text == "recovered"

    asyncio.run(scenario())


def test_a2a_agent_card_cache_is_bounded() -> None:
    from a2a.types import AgentCard

    class Resolver:
        def __init__(self, _client: Any, _endpoint: str) -> None:
            pass

        async def get_agent_card(self, **_kwargs: Any) -> AgentCard:
            return AgentCard()

    async def scenario() -> None:
        adapter = A2AAdapter()
        for index in range(_MAX_AGENT_CARD_CACHE_ENTRIES + 1):
            await adapter._agent_card(
                A2AConnection(endpoint=f"https://agent-{index}.example.com"),
                None,
                Resolver,
            )
        assert len(adapter._card_cache) == _MAX_AGENT_CARD_CACHE_ENTRIES
        assert all(key[0] != "https://agent-0.example.com" for key in adapter._card_cache)

    asyncio.run(scenario())


def test_a2a_bounded_task_store_evicts_terminal_tasks_and_keeps_active_tasks() -> None:
    from a2a.types import Task, TaskState, TaskStatus

    class MemoryStore:
        def __init__(self) -> None:
            self.tasks: dict[str, Any] = {}

        async def save(self, task: Any, _context: Any) -> None:
            self.tasks[task.id] = task

        async def get(self, task_id: str, _context: Any) -> Any:
            return self.tasks.get(task_id)

        async def list(self, _params: Any, _context: Any) -> Sequence[Any]:
            return list(self.tasks.values())

        async def delete(self, task_id: str, _context: Any) -> None:
            self.tasks.pop(task_id, None)

    context = SimpleNamespace(user=SimpleNamespace(user_name="alice"))

    async def terminal_scenario() -> None:
        delegate = MemoryStore()
        store = _BoundedTaskStore(delegate, max_tasks_per_principal=1)
        await store.save(
            Task(id="old", status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED)),
            context,
        )
        await store.save(
            Task(id="new", status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED)),
            context,
        )
        assert set(delegate.tasks) == {"new"}

    async def active_scenario() -> None:
        store = _BoundedTaskStore(MemoryStore(), max_tasks_per_principal=1)
        await store.save(
            Task(id="active", status=TaskStatus(state=TaskState.TASK_STATE_WORKING)),
            context,
        )
        with pytest.raises(RuntimeError, match="capacity"):
            await store.save(
                Task(id="overflow", status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED)),
                context,
            )

    asyncio.run(terminal_scenario())
    asyncio.run(active_scenario())


def test_official_a2a_sdk_text_structured_status_and_context_continuation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    uvicorn = pytest.importorskip("uvicorn")
    source = tmp_path / "remote.kedi"
    source.write_text(
        "> profile: remote:\n    > system: Answer exactly.\n\n> export:\n    remote\n",
        encoding="utf-8",
    )
    port = _unused_loopback_port()
    public_url = f"http://127.0.0.1:{port}"
    served_adapter = _ServedAdapter()
    served = prepare_a2a_program(
        source,
        entry="remote",
        model="test:model",
        adapter=served_adapter,
        max_concurrency=2,
    )
    app = create_a2a_app(
        served,
        public_url=public_url,
        auth=A2AServerAuth(scheme="basic", username="kedi", secret="secret"),
    )
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="critical",
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _wait_for_server(server)
    monkeypatch.setenv("KEDI_TEST_A2A_SECRET", "secret")
    connection = A2AConnection(
        endpoint=public_url,
        auth=A2AAuthConfig(
            scheme="basic",
            username="kedi",
            secret_env="KEDI_TEST_A2A_SECRET",
        ),
        task_timeout=10,
    )

    sync_adapter = A2AAdapter(connection=connection)
    sync_structured = sync_adapter.produce_sync("sync question", output_type=_Output)
    assert sync_structured.answer == "structured:sync question"

    async def scenario() -> None:
        adapter = A2AAdapter(connection=connection)
        first_context = AdapterConversationContext()
        with adapter.conversation_scope(first_context):
            assert await adapter.invoke(prompt="hello") == "remote:hello"
        first = first_context.result_state["a2a"]
        assert first["state"] == "completed"

        snapshot = await adapter.get_task(first["task_id"])
        assert snapshot.state == "completed"
        assert snapshot.context_id == first["context_id"]
        assert snapshot.text == "remote:hello"

        second_context = AdapterConversationContext(resume_state=first_context.result_state)
        with adapter.conversation_scope(second_context):
            assert await adapter.invoke(prompt="again") == "remote:again"
        second = second_context.result_state["a2a"]
        assert second["context_id"] == first["context_id"]
        assert second["task_id"] != first["task_id"]

        structured = await adapter.produce("question", output_type=_Output)
        assert structured.answer == "structured:question"

        kedi_source = f"""
> profile: remote:
    > requires: structured_output
    > agent: a2a:
        endpoint: {public_url}
        auth:
            scheme: basic
            username: kedi
            password_env: KEDI_TEST_A2A_SECRET

@ask() -> str:
    > use: remote
    >> The remote answer is [answer].
    = <answer>

= <ask()>
"""
        runtime = compile_program(parse_program(kedi_source), adapter=None)
        try:
            remote_answer = await asyncio.to_thread(runtime.run_main)
            assert remote_answer.startswith("structured:")
            assert "The remote answer is [answer]." in remote_answer
        finally:
            await runtime.aclose()

        coordinator_source = f"""
> profile: remote:
    > agent: a2a:
        endpoint: {public_url}
        auth:
            scheme: basic
            username: kedi
            password_env: KEDI_TEST_A2A_SECRET

> profile: manager:
    > subagent: remote
    > artifacts:
        enabled: false
"""
        coordinator = compile_program(
            parse_program(coordinator_source),
            adapter=_ServedAdapter(),
        )
        try:
            manager = coordinator._evaluate_profile_definition(
                coordinator._profile_registry["manager"]
            )
            delegate = next(
                tool
                for tool in coordinator._resolve_agent_tool_specs(manager, coordinator.adapter)
                if tool.name == "delegate_task"
            )
            raw_child = await delegate.fn(subagent="remote", task="inspect")
            assert raw_child["task_summary"] == "remote:inspect"
            typed_child = await delegate.fn(
                subagent="remote",
                task="build value",
                final_schema={
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            )
            assert typed_child["task_summary"] == "structured:build value"
            assert typed_child["final_result"] == {"value": 11}
        finally:
            await coordinator.aclose()

        detached_task_id = await _start_detached_task(adapter, connection, "slow:detached")
        assert await asyncio.to_thread(served_adapter.slow_started.wait, 2)
        served_adapter.release_slow.set()
        detached = await _wait_for_remote_state(adapter, detached_task_id, "completed")
        assert detached.text == "remote:slow:detached"
        assert served_adapter.calls.count("slow:detached") == 1

        served_adapter.slow_started.clear()
        served_adapter.release_slow.clear()
        canceled_task_id = await _start_detached_task(adapter, connection, "slow:cancel")
        assert await asyncio.to_thread(served_adapter.slow_started.wait, 2)
        canceled = await adapter.cancel_task(canceled_task_id)
        assert canceled.state == "canceled"
        served_adapter.release_slow.set()
        canceled = await _wait_for_remote_state(adapter, canceled_task_id, "canceled")
        assert canceled.state == "canceled"

        monkeypatch.setenv("KEDI_TEST_A2A_SECRET", "wrong")
        unauthorized = A2AAdapter(connection=connection)
        with pytest.raises(Exception) as raised:
            await unauthorized.invoke(prompt="must not run")
        assert "secret" not in str(raised.value).lower()

    try:
        asyncio.run(scenario())
    finally:
        server.should_exit = True
        thread.join(timeout=10)
    assert not thread.is_alive()


def _unused_loopback_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_server(server: Any) -> None:
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.02)
    assert server.started


async def _start_detached_task(
    adapter: A2AAdapter,
    connection: A2AConnection,
    prompt: str,
) -> str:
    from a2a.types import (
        Message,
        Part,
        Role,
        SendMessageConfiguration,
        SendMessageRequest,
    )

    async with adapter._client_scope(connection, structured=False) as (client, context):
        stream = client.send_message(
            SendMessageRequest(
                message=Message(
                    message_id=uuid4().hex,
                    role=Role.ROLE_USER,
                    parts=[Part(text=prompt)],
                ),
                configuration=SendMessageConfiguration(
                    accepted_output_modes=["text/plain"],
                    history_length=1,
                    return_immediately=False,
                ),
            ),
            context=context,
        )
        async for response in stream:
            payload = response.WhichOneof("payload")
            if payload == "task":
                task_id = response.task.id
            elif payload == "status_update":
                task_id = response.status_update.task_id
            else:
                continue
            await stream.aclose()
            return task_id
    raise AssertionError("A2A server did not acknowledge a task")


async def _wait_for_remote_state(
    adapter: A2AAdapter,
    task_id: str,
    expected: str,
) -> Any:
    deadline = asyncio.get_running_loop().time() + 5
    while True:
        snapshot = await adapter.get_task(task_id)
        if snapshot.state == expected:
            return snapshot
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError(
                f"A2A task {task_id} stayed in {snapshot.state!r}, expected {expected!r}"
            )
        await asyncio.sleep(0.05)
