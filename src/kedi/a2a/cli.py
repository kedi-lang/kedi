from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal, cast
from urllib.parse import urlunsplit

import click

from kedi.agent_adapter.execution import run_sync

from .auth import A2AServerAuth
from .protocol import (
    DEFAULT_A2A_MAX_REQUEST_BYTES,
    DEFAULT_A2A_MAX_RESPONSE_BYTES,
    DEFAULT_A2A_MAX_TASKS_PER_PRINCIPAL,
)
from .runtime import (
    DEFAULT_A2A_MAX_INPUT_CHARS,
    DEFAULT_A2A_MAX_PENDING_TASKS,
    DEFAULT_A2A_MAX_SESSIONS,
    prepare_a2a_program,
)


@click.group(name="a2a")
def _a2a_group() -> None:
    """Serve exported Kedi agent profiles over A2A."""


@_a2a_group.command(name="serve")
@click.argument("source", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option("--entry", required=True, help="Exported agent profile to serve")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", type=click.IntRange(1, 65535), default=8000, show_default=True)
@click.option("--public-url", envvar="KEDI_A2A_PUBLIC_URL")
@click.option("--rpc-path", default="/a2a", show_default=True)
@click.option(
    "--auth",
    "auth_scheme",
    type=click.Choice(["basic", "bearer", "api-key"]),
    envvar="KEDI_A2A_AUTH_SCHEME",
    default="basic",
    show_default=True,
)
@click.option("--username", envvar="KEDI_A2A_USERNAME", default="kedi", show_default=True)
@click.option("--secret-env", help="Read the server credential from this environment variable")
@click.option("--secret-file", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--api-key-header", default="X-API-Key", show_default=True)
@click.option(
    "--allow-unauthenticated",
    is_flag=True,
    help="Allow an unauthenticated loopback-only development server",
)
@click.option("--model", envvar="KEDI_A2A_MODEL")
@click.option("--max-concurrency", type=click.IntRange(min=1), default=4, show_default=True)
@click.option(
    "--max-pending-tasks",
    type=click.IntRange(min=1),
    default=DEFAULT_A2A_MAX_PENDING_TASKS,
    show_default=True,
)
@click.option(
    "--max-sessions",
    type=click.IntRange(min=1),
    default=DEFAULT_A2A_MAX_SESSIONS,
    show_default=True,
)
@click.option(
    "--max-request-bytes",
    type=click.IntRange(min=1),
    default=DEFAULT_A2A_MAX_REQUEST_BYTES,
    show_default=True,
)
@click.option(
    "--max-input-chars",
    type=click.IntRange(min=1),
    default=DEFAULT_A2A_MAX_INPUT_CHARS,
    show_default=True,
)
@click.option(
    "--max-response-bytes",
    type=click.IntRange(min=1),
    default=DEFAULT_A2A_MAX_RESPONSE_BYTES,
    show_default=True,
)
@click.option(
    "--max-tasks-per-principal",
    type=click.IntRange(min=1),
    default=DEFAULT_A2A_MAX_TASKS_PER_PRINCIPAL,
    show_default=True,
)
@click.pass_context
def _serve(
    ctx: click.Context,
    source: Path,
    entry: str,
    host: str,
    port: int,
    public_url: str | None,
    rpc_path: str,
    auth_scheme: str,
    username: str,
    secret_env: str | None,
    secret_file: Path | None,
    api_key_header: str,
    allow_unauthenticated: bool,
    model: str | None,
    max_concurrency: int,
    max_pending_tasks: int,
    max_sessions: int,
    max_request_bytes: int,
    max_input_chars: int,
    max_response_bytes: int,
    max_tasks_per_principal: int,
) -> None:
    defaults = ctx.find_root().obj or {}
    adapter_shortname = defaults.get("adapter", "pydantic")
    selected_model = model or defaults.get("model")
    if allow_unauthenticated and host.strip().strip("[]") not in {"localhost", "127.0.0.1", "::1"}:
        raise click.UsageError("Unauthenticated A2A serving requires a loopback bind host")
    endpoint = public_url or _loopback_public_url(host, port)
    auth = None
    if not allow_unauthenticated:
        secret = _server_secret(secret_env=secret_env, secret_file=secret_file)
        auth = A2AServerAuth(
            scheme=cast(
                Literal["basic", "bearer", "api_key"],
                auth_scheme.replace("-", "_"),
            ),
            secret=secret,
            username=username if auth_scheme == "basic" else None,
            header=api_key_header,
        )
    elif secret_env is not None or secret_file is not None:
        raise click.UsageError(
            "--allow-unauthenticated cannot be combined with a credential source"
        )
    served = None
    try:
        import uvicorn

        from .server import create_a2a_app

        served = prepare_a2a_program(
            source,
            entry=entry,
            adapter_shortname=adapter_shortname,
            model=selected_model,
            max_concurrency=max_concurrency,
            max_pending_tasks=max_pending_tasks,
            max_sessions=max_sessions,
            max_input_chars=max_input_chars,
            max_response_bytes=max_response_bytes,
        )
        app = create_a2a_app(
            served,
            public_url=endpoint,
            rpc_path=rpc_path,
            auth=auth,
            allow_unauthenticated_loopback=allow_unauthenticated,
            max_request_bytes=max_request_bytes,
            max_tasks_per_principal=max_tasks_per_principal,
        )
    except ImportError as exc:
        if served is not None:
            run_sync(served.aclose())
        raise click.ClickException(
            "A2A server support is not installed. Install it with: uv add 'kedi[a2a]'"
        ) from exc
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        if served is not None:
            run_sync(served.aclose())
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Serving exported Kedi agent {entry!r} at {endpoint.rstrip('/')}{rpc_path}")
    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            proxy_headers=False,
            server_header=False,
        )
    finally:
        run_sync(served.aclose())


def run_a2a_cli(
    arguments: list[str],
    *,
    adapter: str,
    model: str | None,
) -> int:
    result = _a2a_group.main(
        args=arguments,
        prog_name="kedi a2a",
        standalone_mode=False,
        obj={"adapter": adapter, "model": model},
    )
    return int(result or 0)


def a2a_help() -> str:
    context = click.Context(_a2a_group, info_name="kedi a2a")
    return _a2a_group.get_help(context)


def _server_secret(*, secret_env: str | None, secret_file: Path | None) -> str:
    direct = os.getenv("KEDI_A2A_SECRET")
    sources = sum(value is not None for value in (secret_env, secret_file, direct))
    if sources > 1:
        raise click.UsageError(
            "Choose one credential source: --secret-env, --secret-file, or KEDI_A2A_SECRET"
        )
    if secret_env is not None:
        value = os.getenv(secret_env)
        if value is None:
            raise click.ClickException(f"Credential environment variable {secret_env!r} is not set")
    elif secret_file is not None:
        try:
            if secret_file.stat().st_size > 64 * 1024:
                raise click.ClickException("Credential file exceeds 65536 bytes")
            value = secret_file.read_text(encoding="utf-8").rstrip("\r\n")
        except click.ClickException:
            raise
        except (OSError, UnicodeError) as exc:
            raise click.ClickException(f"Could not read credential file: {exc}") from exc
    elif direct is not None:
        value = direct
    elif sys.stdin.isatty():
        value = click.prompt("A2A server credential", hide_input=True, confirmation_prompt=True)
    else:
        raise click.ClickException(
            "A2A authentication requires --secret-env, --secret-file, or KEDI_A2A_SECRET"
        )
    if not value:
        raise click.ClickException("A2A server credential cannot be empty")
    return value


def _loopback_public_url(host: str, port: int) -> str:
    normalized = host.strip().strip("[]")
    if normalized not in {"localhost", "127.0.0.1", "::1"}:
        raise click.UsageError("--public-url is required when binding beyond loopback")
    netloc = f"[{normalized}]:{port}" if ":" in normalized else f"{normalized}:{port}"
    return urlunsplit(("http", netloc, "", "", ""))


__all__ = ["a2a_help", "run_a2a_cli"]
