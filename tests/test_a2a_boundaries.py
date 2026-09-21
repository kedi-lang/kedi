from __future__ import annotations

import base64
import sys
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner

from kedi.a2a import cli, result
from kedi.a2a.auth import A2AServerAuth
from kedi.a2a.protocol import response_size
from kedi.agent_profile.connection import A2AAuthConfig, A2AConnection, normalize_a2a_endpoint
from kedi._interactive import persistence
from kedi.cli import main


@pytest.mark.parametrize(
    "values",
    [
        {"scheme": "unknown"},
        {"secret": ""},
        {"secret": "x\ny"},
        {"scheme": "basic"},
        {"scheme": "basic", "username": "a:b"},
        {"username": "user"},
        {"header": "Custom"},
    ],
)
def test_server_auth_rejects_invalid_configuration(values):
    with pytest.raises(ValueError):
        A2AServerAuth(**({"scheme": "bearer", "secret": "test-secret"} | values))


@pytest.mark.parametrize(
    "header",
    [
        None,
        "Basic",
        "Bearer secret",
        "Basic !!!",
        "Basic " + base64.b64encode(b"no-separator").decode(),
        "Basic " + base64.b64encode(b"user:wrong").decode(),
    ],
)
def test_basic_auth_rejects_malformed_or_wrong_credentials(header):
    auth = A2AServerAuth(scheme="basic", username="user", secret="secret")
    assert auth.authenticate({"authorization": header}) is None


def test_bearer_auth_requires_matching_scheme():
    auth = A2AServerAuth(scheme="bearer", secret="secret")
    assert auth.authenticate({"authorization": "Basic secret"}) is None


@pytest.mark.parametrize(
    "values",
    [
        {"scheme": "unknown"},
        {"scheme": "basic"},
        {"scheme": "basic", "username": "a:b"},
        {"username": "user"},
        {"scheme": "api_key", "header": "Bad Header"},
        {"header": "Custom"},
        {"secret_env": "1BAD"},
        {"secret_env": None, "secret_file": " "},
        {"secret_env": None, "provider": 42},
    ],
)
def test_client_auth_rejects_invalid_configuration(values):
    with pytest.raises(ValueError):
        A2AAuthConfig(**({"scheme": "bearer", "secret_env": "A2A_TEST_SECRET"} | values))


def test_client_credentials_report_missing_invalid_files_and_providers(tmp_path, monkeypatch):
    auth = A2AAuthConfig(scheme="bearer", secret_env="A2A_TEST_SECRET")
    for env in ({}, {"A2A_TEST_SECRET": ""}, {"A2A_TEST_SECRET": "x\ny"}):
        with pytest.raises(RuntimeError):
            auth.resolve_secret(env=env)
    path = tmp_path / "secret"
    file_auth = A2AAuthConfig(scheme="bearer", secret_file=str(path))
    with pytest.raises(RuntimeError, match="regular file"):
        file_auth.resolve_secret()
    path.write_bytes(b"x" * 65537)
    with pytest.raises(RuntimeError, match="exceeds"):
        file_auth.resolve_secret()
    path.write_bytes(b"\xff")
    with pytest.raises(RuntimeError, match="Could not read"):
        file_auth.resolve_secret()

    def broken_provider():
        raise ValueError("provider details must not leak")

    from kedi.agent_profile.connection import A2ACredentialError

    def credential_error():
        raise A2ACredentialError("safe credential error")

    for provider, message in (
        (broken_provider, "provider failed"),
        (credential_error, "safe credential"),
        (lambda: 42, "return str"),
    ):
        with pytest.raises(RuntimeError, match=message):
            A2AAuthConfig(scheme="bearer", provider=provider).resolve_secret()
    key = A2AAuthConfig(scheme="api_key", provider=lambda config: config.scheme)
    assert key.headers() == {"X-API-Key": "api_key"}


@pytest.mark.parametrize("endpoint", ["", " ", None, "https:///path"])
def test_endpoint_requires_nonempty_host(endpoint):
    with pytest.raises(ValueError):
        normalize_a2a_endpoint(endpoint)


def test_connection_identity_and_loopback_variants():
    connection = A2AConnection(endpoint="http://localhost:8000/")
    assert connection.identity() == ("http://localhost:8000", None, 30.0, 600.0, 0.5)
    assert A2AConnection(endpoint="http://[::1]:8000").endpoint == "http://[::1]:8000"


def test_response_size_handles_non_json_values():
    assert response_size({("tuple",): "value"}) > 0


@pytest.mark.parametrize(
    "value", [[], {"auth": []}, {"auth": {}}, {"endpoint": "http://remote.example"}, {}]
)
def test_persisted_connection_rejects_invalid_shape(value):
    with pytest.raises(persistence.SessionLoadError):
        persistence._decode_a2a_connection(value)


def test_persisted_connection_without_auth_and_provider_rejection():
    connection = persistence._decode_a2a_connection({"endpoint": "https://agent.example"})
    assert connection.auth is None
    with pytest.raises(persistence.UnsupportedSessionValue, match="not persisted"):
        persistence._encode_a2a_connection(
            A2AConnection(
                endpoint="https://agent.example",
                auth=A2AAuthConfig(scheme="bearer", provider=lambda: "secret"),
            ),
            path="profile",
        )


def test_cli_secret_sources(tmp_path, monkeypatch):
    monkeypatch.delenv("KEDI_A2A_SECRET", raising=False)
    monkeypatch.delenv("A2A_TEST_SECRET", raising=False)
    monkeypatch.setattr(cli.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    with pytest.raises(click.ClickException, match="requires"):
        cli._server_secret(secret_env=None, secret_file=None)
    with pytest.raises(click.ClickException, match="not set"):
        cli._server_secret(secret_env="A2A_TEST_SECRET", secret_file=None)
    monkeypatch.setenv("A2A_TEST_SECRET", "secret")
    assert cli._server_secret(secret_env="A2A_TEST_SECRET", secret_file=None) == "secret"
    monkeypatch.setenv("KEDI_A2A_SECRET", "direct")
    assert cli._server_secret(secret_env=None, secret_file=None) == "direct"
    with pytest.raises(click.UsageError, match="Choose one"):
        cli._server_secret(secret_env="A2A_TEST_SECRET", secret_file=None)
    monkeypatch.delenv("KEDI_A2A_SECRET")
    path = tmp_path / "secret"
    with pytest.raises(click.ClickException, match="Could not read"):
        cli._server_secret(secret_env=None, secret_file=path)
    path.write_bytes(b"x" * 65537)
    with pytest.raises(click.ClickException, match="exceeds"):
        cli._server_secret(secret_env=None, secret_file=path)
    path.write_text("file-secret\n")
    assert cli._server_secret(secret_env=None, secret_file=path) == "file-secret"
    path.write_text("")
    with pytest.raises(click.ClickException, match="empty"):
        cli._server_secret(secret_env=None, secret_file=path)
    monkeypatch.setattr(cli.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(click, "prompt", lambda *a, **kw: "prompt-secret")
    assert cli._server_secret(secret_env=None, secret_file=None) == "prompt-secret"
    assert "serve" in cli.a2a_help()
    assert cli._loopback_public_url("::1", 8000) == "http://[::1]:8000"
    with pytest.raises(click.UsageError, match="required"):
        cli._loopback_public_url("0.0.0.0", 8000)


@pytest.mark.parametrize("failure", [None, ImportError("missing"), ValueError("invalid")])
def test_cli_server_startup_and_cleanup(tmp_path, monkeypatch, failure):
    import uvicorn
    from kedi.a2a import server

    source = tmp_path / "agent.kedi"
    source.write_text("")
    closed = []
    started = []

    async def close():
        closed.append(True)

    monkeypatch.setenv("A2A_TEST_SECRET", "secret")
    monkeypatch.delenv("KEDI_A2A_SECRET", raising=False)
    monkeypatch.setattr(cli, "prepare_a2a_program", lambda *a, **kw: SimpleNamespace(aclose=close))

    def create(*args, **kwargs):
        if failure:
            raise failure
        return "app"

    monkeypatch.setattr(server, "create_a2a_app", create)
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: started.append(app))
    response = CliRunner().invoke(
        main, ["a2a", "serve", str(source), "--entry", "remote", "--secret-env", "A2A_TEST_SECRET"]
    )
    assert response.exit_code == (1 if failure else 0), response.output
    assert closed == [True]
    assert started == ([] if failure else ["app"])
    response = CliRunner().invoke(
        main,
        [
            "a2a",
            "serve",
            str(source),
            "--entry",
            "remote",
            "--allow-unauthenticated",
            "--secret-env",
            "A2A_TEST_SECRET",
        ],
    )
    assert response.exit_code == 2


def test_result_assembly_handles_messages_chunks_and_failed_tasks():
    from a2a.types import (
        Artifact,
        Message,
        Part,
        Role,
        StreamResponse,
        Task,
        TaskState,
        TaskStatus,
        TaskArtifactUpdateEvent,
        TaskStatusUpdateEvent,
    )

    message = Message(role=Role.ROLE_AGENT, parts=[Part(text="answer")])
    acc = result._ResponseAccumulator(max_bytes=4096)
    with pytest.raises(RuntimeError, match="task ID"):
        acc.snapshot()
    acc.add(StreamResponse(message=message))
    assert acc.is_complete and acc.result(structured=False).text == "answer"
    acc.add(StreamResponse())
    acc.add_task(Task(id="task", history=[message, Message(role=Role.ROLE_USER)]))
    with pytest.raises(RuntimeError, match="terminal"):
        acc.result(structured=False)
    acc.add(
        StreamResponse(
            status_update=TaskStatusUpdateEvent(
                task_id="task",
                status=TaskStatus(state=TaskState.TASK_STATE_FAILED, message=message),
            )
        )
    )
    with pytest.raises(RuntimeError, match="answer"):
        acc.result(structured=False)
    acc.add(
        StreamResponse(
            status_update=TaskStatusUpdateEvent(
                task_id="task", status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED)
            )
        )
    )
    with pytest.raises(RuntimeError, match="structured-output"):
        acc.result(structured=True)
    for text, append in (("head", False), ("tail", True)):
        acc.add(
            StreamResponse(
                artifact_update=TaskArtifactUpdateEvent(
                    task_id="task",
                    artifact=Artifact(artifact_id="a", parts=[Part(text=text)]),
                    append=append,
                )
            )
        )
    assert acc.result(structured=False).text == "head\ntail"
    acc.add_task(
        Task(id="task", status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED, message=message))
    )
    acc._capture_message(Message(role=Role.ROLE_USER))
    acc._capture_message(Message(role=Role.ROLE_AGENT))
    with pytest.raises(RuntimeError, match="exceeded"):
        result._ResponseAccumulator(max_bytes=1).add(StreamResponse(message=message))
    assert result._task_state_name(999) == "999"


def test_result_optional_sdk_import_fallbacks(monkeypatch):
    monkeypatch.setitem(sys.modules, "a2a.types", None)
    assert result._is_agent_message(SimpleNamespace(role=2)) is False
    assert result._task_state_name(999) == "999"


def test_auth_sources_are_exclusive_and_api_key_header_is_configurable():
    with pytest.raises(ValueError, match="exactly one"):
        A2AAuthConfig(scheme="bearer")
    auth = A2AAuthConfig(scheme="api_key", secret_env="KEY", header="X-Test")
    assert auth.headers(env={"KEY": "secret"}) == {"X-Test": "secret"}
    assert A2AAuthConfig(scheme="bearer", provider=lambda: "secret").identity()[4] is not None


def test_structured_result_skips_unrelated_artifacts_and_text_parts():
    from a2a.types import Artifact, Part, Task, TaskState, TaskStatus
    from kedi.a2a.protocol import KEDI_STRUCTURED_OUTPUT_ARTIFACT_NAME

    accumulator = result._ResponseAccumulator(max_bytes=4096)
    accumulator.add_task(
        Task(
            id="t",
            status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
            artifacts=[
                Artifact(artifact_id="unrelated", name="other", parts=[Part(text="not data")]),
                Artifact(
                    artifact_id="typed",
                    name=KEDI_STRUCTURED_OUTPUT_ARTIFACT_NAME,
                    parts=[Part(text="still not data")],
                ),
            ],
        )
    )
    with pytest.raises(RuntimeError, match="structured-output artifact"):
        accumulator.result(structured=True)


@pytest.mark.parametrize(
    "body,message",
    [
        ("auth: `42`", "nested block"),
        ("task_timeout: invalid", "positive number"),
        ("auth:\n        unknown: value", "unsupported field"),
        ("auth:\n        scheme: `42`", "string 'scheme'"),
        ("auth:\n        scheme: invalid", "must be basic"),
        ("auth:\n        scheme: bearer\n        token_env: `42`", "must be a string"),
        (
            "auth:\n        scheme: basic\n        password_env: KEY\n        username: `42`",
            "username must be a string",
        ),
        (
            "auth:\n        scheme: api_key\n        api_key_env: KEY\n        header: `42`",
            "header must be a string",
        ),
    ],
)
def test_connection_directives_reject_invalid_values(body, message):
    from kedi.core import KediRuntime
    from kedi.lang import parse_program
    from kedi.agent_profile.directives import eval_agent_directive

    stmt = parse_program(
        "> agent: a2a:\n    endpoint: https://agent.example\n    " + body + "\n"
    ).ordered_toplevel[0]
    with pytest.raises(Exception, match=message):
        eval_agent_directive(KediRuntime(), stmt, {})


def test_connection_ast_validation_and_profile_hover(monkeypatch):
    from kedi.agent_profile import directives
    from kedi.core import KediRuntime
    from kedi.lang.ast import AgentConnectionField, AgentDirectiveStmt
    from kedi.lsp.features.directive_help import _profile_agent_entry_markdown

    rt = KediRuntime()
    field = AgentConnectionField(name="command", plain_value="run")
    with pytest.raises(Exception, match="legacy command"):
        directives.eval_agent_directive(
            rt,
            AgentDirectiveStmt(
                plain_value="acp", command_plain_value="run", connection_fields=[field]
            ),
            {},
        )
    with pytest.raises(Exception, match="no value"):
        directives._agent_connection_values(rt, [AgentConnectionField(name="command")], env={})
    assert directives._normalize_agent_command(["run", "worker"], directive_name="command") == (
        "run",
        "worker",
    )
    for value in (42, [42]):
        with pytest.raises(TypeError):
            directives._normalize_agent_command(value, directive_name="command")
    monkeypatch.setattr(directives, "_agent_connection_values", lambda *args, **kwargs: {})
    with pytest.raises(Exception, match="requires 'command'"):
        directives.eval_agent_directive(
            rt, AgentDirectiveStmt(plain_value="acp", connection_fields=[field]), {}
        )
    entry = SimpleNamespace(
        plain_value="acp",
        backtick_code=None,
        command_plain_value=None,
        command_backtick_code=None,
        connection_fields=[field],
    )
    assert "stored in this profile" in _profile_agent_entry_markdown(entry)
