from __future__ import annotations

import asyncio
import hashlib
import ipaddress
from collections import OrderedDict
from contextlib import asynccontextmanager
from functools import partial
from threading import RLock
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from google.protobuf import struct_pb2
from google.protobuf.json_format import ParseDict
from starlette.authentication import AuthenticationBackend

from kedi.agent_adapter.subagents.runs import safe_subagent_error_message

from .auth import A2AServerAuth
from .protocol import (
    DEFAULT_A2A_MAX_REQUEST_BYTES,
    DEFAULT_A2A_MAX_SCHEMA_BYTES,
    DEFAULT_A2A_MAX_TASKS_PER_PRINCIPAL,
    KEDI_STRUCTURED_OUTPUT_ARTIFACT_NAME,
    KEDI_STRUCTURED_OUTPUT_EXTENSION_URI,
    KEDI_STRUCTURED_OUTPUT_VERSION,
    KEDI_TEXT_ARTIFACT_NAME,
    response_size,
    structured_output_schema,
)
from .runtime import A2AServedProgram


def create_a2a_app(
    program: A2AServedProgram,
    *,
    public_url: str,
    rpc_path: str = "/a2a",
    auth: A2AServerAuth | None,
    allow_unauthenticated_loopback: bool = False,
    max_request_bytes: int = DEFAULT_A2A_MAX_REQUEST_BYTES,
    max_tasks_per_principal: int = DEFAULT_A2A_MAX_TASKS_PER_PRINCIPAL,
    state_dir: Path | None = None,
) -> Any:
    """Build a Starlette A2A v1 application around one exported Kedi profile."""

    try:
        from a2a.server.request_handlers import DefaultRequestHandler
        from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
        from a2a.server.tasks import InMemoryTaskStore
        from starlette.applications import Starlette
        from starlette.middleware import Middleware
        from starlette.middleware.authentication import AuthenticationMiddleware
    except ImportError as exc:
        raise RuntimeError(
            "A2A server support is not installed. Install it with: uv add 'kedi[a2a]'"
        ) from exc

    if not rpc_path.startswith("/") or "?" in rpc_path or "#" in rpc_path:
        raise ValueError("A2A RPC path must be an absolute URL path")
    if isinstance(max_request_bytes, bool) or max_request_bytes < 1:
        raise ValueError("A2A request size limit must be positive")
    if isinstance(max_tasks_per_principal, bool) or max_tasks_per_principal < 1:
        raise ValueError("A2A task retention limit must be positive")
    endpoint = _validated_public_url(public_url)
    loopback = _is_loopback(endpoint)
    if auth is None and not (allow_unauthenticated_loopback and loopback):
        raise ValueError(
            "A2A authentication is required; unauthenticated serving must be explicitly "
            "enabled on loopback"
        )

    card = _agent_card(program, endpoint=endpoint, rpc_path=rpc_path, auth=auth)
    state = None
    if state_dir is not None:
        if not program.source_fingerprint or not program.model_identity:
            raise ValueError("Durable A2A serving requires a source fingerprint and model identity")
        try:
            from .persistence import A2APersistence
        except ImportError as exc:
            raise RuntimeError(
                "A2A persistence support is not installed. Install kedi[a2a-persistence]"
            ) from exc

        identity = hashlib.sha256(
            "\0".join(
                (
                    program.entry,
                    program.source_fingerprint,
                    program.adapter.shortname,
                    program.model_identity,
                    str(program.profile.history),
                )
            ).encode("utf-8")
        ).hexdigest()
        state = A2APersistence(
            state_dir,
            identity=identity,
            max_tasks_per_principal=max_tasks_per_principal,
            max_sessions=program.max_sessions,
            adapter_shortname=program.adapter.shortname,
            history_enabled=program.profile.history is True,
        )
        program.configure_persistence(state)
    executor = _KediAgentExecutor(program, state=state)
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=(
            state
            if state is not None
            else _BoundedTaskStore(
                InMemoryTaskStore(),
                max_tasks_per_principal=max_tasks_per_principal,
            )
        ),
        agent_card=card,
    )
    routes = [
        *create_agent_card_routes(card),
        *create_jsonrpc_routes(handler, rpc_path),
    ]
    middleware: list[Any] = []
    if auth is not None:
        backend = _A2AAuthenticationBackend(auth)
        middleware.append(
            Middleware(
                AuthenticationMiddleware,
                backend=backend,
                on_error=partial(_authentication_error, auth=auth),
            )
        )
    middleware.append(Middleware(_RequestSizeLimitMiddleware, limit=max_request_bytes))

    @asynccontextmanager
    async def lifespan(_app: Any):
        try:
            if state is not None:
                await state.open()
            yield
        finally:
            try:
                await handler.aclose()
            finally:
                try:
                    await program.aclose()
                finally:
                    if state is not None:
                        await state.close()

    return Starlette(routes=routes, middleware=middleware, lifespan=lifespan)


class _KediAgentExecutor:
    def __init__(self, program: A2AServedProgram, *, state: Any = None) -> None:
        self._program = program
        self._state = state

    async def execute(self, context: Any, event_queue: Any) -> None:
        from a2a.server.tasks import TaskUpdater
        from a2a.types import Part, Task, TaskState, TaskStatus

        if context.task_id is None or context.context_id is None:
            raise RuntimeError("A2A SDK did not assign task and context identifiers")
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        admission = (
            self._state.expect_admission(context.task_id) if self._state is not None else None
        )
        try:
            if context.current_task is None:
                await event_queue.enqueue_event(
                    Task(
                        id=context.task_id,
                        context_id=context.context_id,
                        status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
                        history=[context.message] if context.message is not None else [],
                    )
                )
            else:
                await updater.start_work()
            if admission is not None:
                await admission
            if context.current_task is None:
                await updater.start_work()
        except BaseException:
            if self._state is not None:
                self._state.abort_admission(context.task_id)
            raise
        try:
            schema = _requested_output_schema(context)

            async def publish_result(result: Any) -> None:
                if schema is None:
                    await updater.add_artifact(
                        [Part(text=str(result))],
                        name=KEDI_TEXT_ARTIFACT_NAME,
                        last_chunk=True,
                    )
                else:
                    value = getattr(struct_pb2, "Value")()
                    ParseDict(result, value)
                    await updater.add_artifact(
                        [Part(data=value, media_type="application/json")],
                        name=KEDI_STRUCTURED_OUTPUT_ARTIFACT_NAME,
                        last_chunk=True,
                        extensions=[KEDI_STRUCTURED_OUTPUT_EXTENSION_URI],
                    )
                await updater.complete()

            if self._state is None:
                result = await self._program.invoke(
                    principal=context.call_context.user.user_name,
                    context_id=context.context_id,
                    prompt=context.get_user_input(),
                    output_schema=schema,
                )
                await publish_result(result)
            else:

                async def commit(result: Any, checkpoint: dict[str, Any]) -> None:
                    future = self._state.stage_completion(
                        task_id=context.task_id,
                        owner=context.call_context.user.user_name,
                        context_id=context.context_id,
                        checkpoint=checkpoint,
                    )
                    try:
                        await publish_result(result)
                        await asyncio.shield(future)
                    except BaseException:
                        await self._state.abort_completion(context.task_id)
                        if future.done() and not future.cancelled():
                            future.exception()
                        raise

                await self._program.invoke(
                    principal=context.call_context.user.user_name,
                    context_id=context.context_id,
                    prompt=context.get_user_input(),
                    output_schema=schema,
                    on_complete=commit,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            message = updater.new_agent_message([Part(text=_safe_failure_message(exc))])
            await updater.failed(message)

    async def cancel(self, context: Any, event_queue: Any) -> None:
        from a2a.server.tasks import TaskUpdater

        if context.task_id is None or context.context_id is None:
            return
        await TaskUpdater(event_queue, context.task_id, context.context_id).cancel()


def _requested_output_schema(context: Any) -> dict[str, Any] | None:
    requested = KEDI_STRUCTURED_OUTPUT_EXTENSION_URI in context.requested_extensions
    schema = structured_output_schema(context.metadata)
    if requested and schema is None:
        raise ValueError("Kedi structured output v1 requires a valid JSON Schema request")
    if schema is not None and not requested:
        raise ValueError("Kedi structured output metadata requires extension negotiation")
    if schema is not None and response_size(schema) > DEFAULT_A2A_MAX_SCHEMA_BYTES:
        raise ValueError(
            f"Kedi structured output schema exceeds {DEFAULT_A2A_MAX_SCHEMA_BYTES} bytes"
        )
    return schema


def _agent_card(
    program: A2AServedProgram,
    *,
    endpoint: str,
    rpc_path: str,
    auth: A2AServerAuth | None,
) -> Any:
    from a2a.types import (
        AgentCapabilities,
        AgentCard,
        AgentExtension,
        AgentInterface,
        AgentSkill,
        APIKeySecurityScheme,
        HTTPAuthSecurityScheme,
        SecurityRequirement,
        SecurityScheme,
        StringList,
    )

    security_schemes: dict[str, Any] = {}
    security_requirements: list[Any] = []
    if auth is not None:
        if auth.scheme == "api_key":
            scheme = SecurityScheme(
                api_key_security_scheme=APIKeySecurityScheme(
                    description="Kedi A2A API key",
                    location="header",
                    name=auth.header,
                )
            )
        else:
            scheme = SecurityScheme(
                http_auth_security_scheme=HTTPAuthSecurityScheme(
                    description="Kedi A2A authentication",
                    scheme=auth.scheme,
                    bearer_format="opaque" if auth.scheme == "bearer" else "",
                )
            )
        security_schemes[auth.scheme] = scheme
        security_requirements.append(
            SecurityRequirement(schemes={auth.scheme: StringList(list=[])})
        )
    interface_url = f"{endpoint.rstrip('/')}{rpc_path}"
    extension_params = getattr(struct_pb2, "Struct")()
    ParseDict(
        {
            "version": KEDI_STRUCTURED_OUTPUT_VERSION,
            "mode": "request-schema",
            "schemaDialect": "https://json-schema.org/draft/2020-12/schema",
            "maxSchemaBytes": DEFAULT_A2A_MAX_SCHEMA_BYTES,
            "resultArtifactName": KEDI_STRUCTURED_OUTPUT_ARTIFACT_NAME,
        },
        extension_params,
    )
    return AgentCard(
        name=program.entry,
        description=program.description,
        supported_interfaces=[
            AgentInterface(
                url=interface_url,
                protocol_binding="JSONRPC",
                protocol_version="1.0",
            )
        ],
        version="1.0",
        capabilities=AgentCapabilities(
            streaming=True,
            push_notifications=False,
            extensions=[
                AgentExtension(
                    uri=KEDI_STRUCTURED_OUTPUT_EXTENSION_URI,
                    description=(
                        "Versioned Kedi JSON-Schema request and structured-result artifact"
                    ),
                    required=False,
                    params=extension_params,
                )
            ],
        ),
        security_schemes=security_schemes,
        security_requirements=security_requirements,
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain", "application/json"],
        skills=[
            AgentSkill(
                id=program.entry,
                name=program.entry,
                description=program.description,
                tags=["kedi", "agent"],
                input_modes=["text/plain"],
                output_modes=["text/plain", "application/json"],
            )
        ],
    )


class _A2AAuthenticationBackend(AuthenticationBackend):
    def __init__(self, auth: A2AServerAuth) -> None:
        self._auth = auth

    async def authenticate(self, conn: Any) -> tuple[Any, Any] | None:
        from starlette.authentication import (
            AuthCredentials,
            AuthenticationError,
            SimpleUser,
        )

        principal = self._auth.authenticate(conn.headers)
        if principal is None:
            raise AuthenticationError("Invalid A2A credentials")
        return AuthCredentials(["authenticated"]), SimpleUser(principal)


def _authentication_error(
    _connection: Any,
    _exc: Exception,
    *,
    auth: A2AServerAuth,
) -> Any:
    from starlette.responses import JSONResponse

    headers = {"Cache-Control": "no-store"}
    if auth.scheme == "basic":
        headers["WWW-Authenticate"] = 'Basic realm="kedi-a2a", charset="UTF-8"'
    elif auth.scheme == "bearer":
        headers["WWW-Authenticate"] = 'Bearer realm="kedi-a2a"'
    return JSONResponse(
        {"error": "authentication_required"},
        status_code=401,
        headers=headers,
    )


class _RequestTooLarge(Exception):
    pass


class _BoundedTaskStore:
    """Principal-scoped retention wrapper for the SDK's in-memory task store."""

    def __init__(self, store: Any, *, max_tasks_per_principal: int) -> None:
        self._store = store
        self._max_tasks_per_principal = max_tasks_per_principal
        self._tasks: dict[str, OrderedDict[str, str]] = {}
        self._lock = RLock()

    async def save(self, task: Any, context: Any) -> None:
        owner = context.user.user_name
        evicted: str | None = None
        reserved = False
        with self._lock:
            tasks = self._tasks.setdefault(owner, OrderedDict())
            if task.id not in tasks:
                if len(tasks) >= self._max_tasks_per_principal:
                    evicted = next(
                        (
                            task_id
                            for task_id, state in tasks.items()
                            if state in _TERMINAL_STORED_TASK_STATES
                        ),
                        None,
                    )
                    if evicted is None:
                        raise RuntimeError("A2A task retention capacity is exhausted")
                    del tasks[evicted]
                tasks[task.id] = "reserving"
                reserved = True
        try:
            if evicted is not None:
                await self._store.delete(evicted, context)
            await self._store.save(task, context)
        except Exception:
            if reserved:
                with self._lock:
                    tasks = self._tasks.get(owner)
                    if tasks is not None:
                        tasks.pop(task.id, None)
                        if not tasks:
                            self._tasks.pop(owner, None)
            raise
        with self._lock:
            tasks = self._tasks.setdefault(owner, OrderedDict())
            tasks[task.id] = _stored_task_state(task)
            tasks.move_to_end(task.id)

    async def get(self, task_id: str, context: Any) -> Any:
        return await self._store.get(task_id, context)

    async def list(self, params: Any, context: Any) -> Any:
        return await self._store.list(params, context)

    async def delete(self, task_id: str, context: Any) -> None:
        await self._store.delete(task_id, context)
        owner = context.user.user_name
        with self._lock:
            tasks = self._tasks.get(owner)
            if tasks is None:
                return
            tasks.pop(task_id, None)
            if not tasks:
                del self._tasks[owner]


_TERMINAL_STORED_TASK_STATES = frozenset(
    {"completed", "failed", "canceled", "rejected", "input_required", "auth_required"}
)


def _stored_task_state(task: Any) -> str:
    if not task.HasField("status"):
        return "unspecified"
    try:
        from a2a.types import TaskState

        return TaskState.Name(task.status.state).removeprefix("TASK_STATE_").lower()
    except (ImportError, ValueError):
        return str(task.status.state)


class _RequestSizeLimitMiddleware:
    def __init__(self, app: Any, *, limit: int) -> None:
        self.app = app
        self.limit = limit

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", ()))
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                if int(content_length) > self.limit:
                    await _send_too_large(send)
                    return
            except ValueError:
                pass
        received = 0
        response_started = False

        async def limited_receive() -> dict[str, Any]:
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > self.limit:
                    raise _RequestTooLarge
            return message

        async def tracked_send(message: dict[str, Any]) -> None:
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except _RequestTooLarge:
            if not response_started:
                await _send_too_large(send)


async def _send_too_large(send: Any) -> None:
    body = b'{"error":"request_too_large"}'
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def _validated_public_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("A2A public URL must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("A2A public URL cannot contain credentials, query, or fragment")
    if parsed.scheme == "http" and not _is_loopback(value):
        raise ValueError("Non-loopback A2A servers require an HTTPS public URL")
    return value.strip().rstrip("/")


def _is_loopback(value: str) -> bool:
    host = urlsplit(value).hostname
    if host is None:
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _safe_failure_message(exc: Exception) -> str:
    detail = safe_subagent_error_message(exc) or "no detail"
    return f"Kedi A2A execution failed ({type(exc).__name__}): {detail}"


__all__ = [
    "DEFAULT_A2A_MAX_REQUEST_BYTES",
    "DEFAULT_A2A_MAX_TASKS_PER_PRINCIPAL",
    "create_a2a_app",
]
