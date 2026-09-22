from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, asynccontextmanager, contextmanager
from contextvars import ContextVar
from threading import Lock
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit
from uuid import uuid4

from pydantic import TypeAdapter

from kedi.a2a.protocol import (
    DEFAULT_A2A_MAX_RESPONSE_BYTES,
    DEFAULT_A2A_MAX_SCHEMA_BYTES,
    KEDI_STRUCTURED_OUTPUT_EXTENSION_URI,
    A2AAmbiguousSendError,
    A2AError,
    A2ARemoteResult,
    A2ARemoteTaskError,
    A2ARemoteTaskSnapshot,
    A2ATaskTimeoutError,
    A2ATransportError,
    response_size,
    structured_output_request,
)
from kedi.a2a.result import _ResponseAccumulator
from kedi.agent_adapter.conversation import AdapterConversationContext
from kedi.agent_adapter.events import agent_event_run
from kedi.agent_adapter.execution import run_sync
from kedi.agent_adapter.hooks import HookRegistrationMixin, hooked_agent_run
from kedi.agent_adapter.invocation import InvocationUsage, report_invocation
from kedi.agent_adapter.meta import AdapterCapabilities, AgentAdapterKind, HookCapabilities
from kedi.agent_adapter.output_schema import build_output_model
from kedi.agent_adapter.subagents import SubagentConversationContext, SubagentExecutionRequest
from kedi.agent_profile import (
    A2AAuthConfig,
    A2AConnection,
    AgentProfile,
    AgentProfileOverrideUnsupported,
)
from kedi.resolved_types import ResolvedKediType

if TYPE_CHECKING:
    from kedi.agent_adapter.hooks import HookHandler
    from kedi.agent_adapter.subagents.limits import SubagentUsageLimits


_CONTINUABLE_TASK_STATES = frozenset({"input_required", "auth_required"})
_MAX_AGENT_CARD_CACHE_ENTRIES = 64


class A2AAdapter(HookRegistrationMixin):
    """Remote agent harness backed by the official A2A Python SDK."""

    kind: AgentAdapterKind = "agent-harness"
    shortname = "a2a"
    capabilities = AdapterCapabilities(
        # Structured output is negotiated per peer through Kedi's versioned
        # A2A extension, so it is intentionally not claimed globally.
        supports_structured_output=None,
        negotiated_capabilities=frozenset({"structured_output"}),
        supports_tool_registration=False,
        supports_mcp=False,
        supports_profile_override=True,
        supports_model_override=False,
        supports_effort=False,
        supports_settings=False,
        supports_codemode=False,
        supports_native_approvals=False,
        supports_native_approval_handler=False,
        supports_subagents=True,
        supports_background_subagents=True,
        supports_artifacts=False,
        supports_native_artifacts=False,
        supports_stateful_history=True,
        supports_history_replay=False,
        supports_native_compaction=False,
        supports_stream_events=True,
        hooks=HookCapabilities.prompt_only(),
    )
    supports_native_subagent_resume = True

    def __init__(
        self,
        model: str | None = None,
        *,
        connection: A2AConnection | None = None,
        max_response_bytes: int = DEFAULT_A2A_MAX_RESPONSE_BYTES,
        hook_handler: HookHandler | None = None,
        **kwargs: Any,
    ) -> None:
        self._init_hook_registration(hook_handler)
        if model is not None:
            raise TypeError("A2AAdapter leaves model selection to the remote agent")
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported A2A adapter setting(s): {names}")
        if isinstance(max_response_bytes, bool) or max_response_bytes < 1:
            raise ValueError("max_response_bytes must be a positive integer")
        self._connection = connection
        self._max_response_bytes = max_response_bytes
        self._profile_stack_var: ContextVar[tuple[AgentProfile, ...]] = ContextVar(
            f"kedi_a2a_profile_stack_{id(self)}",
            default=(),
        )
        self._conversation_var: ContextVar[
            AdapterConversationContext | SubagentConversationContext | None
        ] = ContextVar(f"kedi_a2a_conversation_{id(self)}", default=None)
        self._card_cache: OrderedDict[tuple[Any, ...], bytes] = OrderedDict()
        self._card_cache_lock = Lock()

    @property
    def active_profile(self) -> AgentProfile:
        stack = self._profile_stack_var.get()
        return stack[-1] if stack else AgentProfile.empty()

    @staticmethod
    def type_builder(output_schema: dict[str, ResolvedKediType]) -> type[Any]:
        return build_output_model(output_schema)

    def default_model(self) -> None:
        return None

    def telemetry_model_name(self) -> None:
        return None

    @contextmanager
    def profile_override(self, profile: AgentProfile) -> Iterator[None]:
        self._validate_profile(profile)
        token = self._profile_stack_var.set((*self._profile_stack_var.get(), profile))
        try:
            yield
        finally:
            self._profile_stack_var.reset(token)

    @contextmanager
    def conversation_scope(self, conversation: AdapterConversationContext) -> Iterator[None]:
        token = self._conversation_var.set(conversation)
        try:
            yield
        finally:
            self._conversation_var.reset(token)

    @contextmanager
    def subagent_scope(self, request: SubagentExecutionRequest) -> Iterator[None]:
        if request.tools:
            raise AgentProfileOverrideUnsupported(
                "A2A child profiles cannot forward local Kedi tools to a remote agent"
            )
        conversation_token = self._conversation_var.set(request.conversation)
        try:
            with self.profile_override(request.profile):
                yield
        finally:
            self._conversation_var.reset(conversation_token)

    @staticmethod
    def subagent_run_kwargs(_limits: SubagentUsageLimits) -> dict[str, Any]:
        return {}

    def tool_scope(self, tools: Sequence[Any] = ()) -> AbstractContextManager[None]:
        if tools:
            raise AgentProfileOverrideUnsupported(
                "A2AAdapter cannot forward local Kedi tools to a remote agent"
            )
        return _empty_scope()

    def register_tool(self, tool: Any) -> None:
        raise AgentProfileOverrideUnsupported(
            "A2AAdapter cannot register local tools on a remote agent"
        )

    @property
    def scoped_tools(self) -> tuple[Any, ...]:
        return ()

    @hooked_agent_run("prompt")
    async def invoke(self, *, prompt: str, **kwargs: Any) -> str:
        return await self._invoke(prompt=prompt, kwargs=kwargs)

    async def _invoke(self, *, prompt: str, kwargs: Mapping[str, Any]) -> str:
        result = await self._send(prompt, output_schema=None, kwargs=kwargs)
        if result.text is None:
            raise A2ARemoteTaskError(
                "Remote A2A agent completed without a text result",
                task_id=result.task_id,
                context_id=result.context_id,
                state=result.state,
            )
        return result.text

    @hooked_agent_run("prompt")
    def invoke_sync(self, *, prompt: str, **kwargs: Any) -> str:
        return run_sync(self._invoke(prompt=prompt, kwargs=kwargs))

    @hooked_agent_run("template")
    async def produce(
        self,
        template: str,
        output_schema: dict[str, ResolvedKediType] | None = None,
        output_type: type[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        return await self._produce(
            template,
            output_schema=output_schema,
            output_type=output_type,
            kwargs=kwargs,
        )

    async def _produce(
        self,
        template: str,
        *,
        output_schema: dict[str, ResolvedKediType] | None,
        output_type: type[Any] | None,
        kwargs: Mapping[str, Any],
    ) -> Any:
        if output_type is None:
            if not output_schema:
                raise ValueError("A2A structured output requires an output schema or type")
            output_type = self.type_builder(output_schema)
        adapter = TypeAdapter(output_type)
        result = await self._send(template, output_schema=adapter.json_schema(), kwargs=kwargs)
        return adapter.validate_python(result.data)

    @hooked_agent_run("template")
    def produce_sync(
        self,
        template: str,
        output_schema: dict[str, ResolvedKediType] | None = None,
        output_type: type[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        return run_sync(
            self._produce(
                template,
                output_schema=output_schema,
                output_type=output_type,
                kwargs=kwargs,
            )
        )

    @hooked_agent_run("template")
    async def produce_subagent(
        self,
        *,
        template: str,
        output_schema: dict[str, Any],
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        result = await self._send(template, output_schema=output_schema, kwargs=kwargs)
        if not isinstance(result.data, Mapping):
            raise A2ARemoteTaskError(
                "Remote Kedi A2A agent returned a non-object structured result",
                task_id=result.task_id,
                context_id=result.context_id,
                state=result.state,
            )
        return cast(Mapping[str, Any], result.data)

    async def continue_task(
        self,
        *,
        task_id: str,
        prompt: str,
        context_id: str | None = None,
    ) -> str:
        """Continue a suspended remote task without replaying its original request."""

        connection = self._active_connection()
        result = await self._send_official(
            connection,
            prompt,
            output_schema=None,
            task_id=_required_identifier(task_id, "task_id"),
            context_id=_optional_identifier(context_id, "context_id"),
        )
        if result.text is None:
            raise A2ARemoteTaskError(
                "Remote A2A task completed without a text result",
                task_id=result.task_id,
                context_id=result.context_id,
                state=result.state,
            )
        return result.text

    def continue_task_sync(
        self,
        *,
        task_id: str,
        prompt: str,
        context_id: str | None = None,
    ) -> str:
        return run_sync(
            self.continue_task(
                task_id=task_id,
                prompt=prompt,
                context_id=context_id,
            )
        )

    async def get_task(
        self,
        task_id: str,
        *,
        history_length: int = 1,
    ) -> A2ARemoteTaskSnapshot:
        """Read an existing remote task without starting or resending work."""

        if isinstance(history_length, bool) or history_length < 0:
            raise ValueError("history_length must be a non-negative integer")
        connection = self._active_connection()
        try:
            from a2a.types import GetTaskRequest
        except ImportError as exc:
            raise _missing_a2a_dependency() from exc
        async with self._client_scope(connection, structured=False) as (client, call_context):
            task = await _safe_status_read(
                lambda: client.get_task(
                    GetTaskRequest(
                        id=_required_identifier(task_id, "task_id"), history_length=history_length
                    ),
                    context=call_context,
                ),
                connection=connection,
                task_id=task_id,
            )
        return _task_snapshot(task, max_bytes=self._max_response_bytes)

    def get_task_sync(
        self,
        task_id: str,
        *,
        history_length: int = 1,
    ) -> A2ARemoteTaskSnapshot:
        return run_sync(self.get_task(task_id, history_length=history_length))

    async def cancel_task(self, task_id: str) -> A2ARemoteTaskSnapshot:
        """Request cancellation and return the state acknowledged by the peer."""

        connection = self._active_connection()
        task_id = _required_identifier(task_id, "task_id")
        try:
            from a2a.client.errors import A2AClientError
            from a2a.types import CancelTaskRequest
        except ImportError as exc:
            raise _missing_a2a_dependency() from exc
        async with self._client_scope(connection, structured=False) as (client, call_context):
            try:
                task = await client.cancel_task(
                    CancelTaskRequest(id=task_id),
                    context=call_context,
                )
            except A2AClientError as exc:
                raise A2ATransportError(
                    "Remote A2A cancellation could not be confirmed",
                    task_id=task_id,
                ) from exc
        return _task_snapshot(task, max_bytes=self._max_response_bytes)

    def cancel_task_sync(self, task_id: str) -> A2ARemoteTaskSnapshot:
        return run_sync(self.cancel_task(task_id))

    async def _send(
        self,
        prompt: str,
        *,
        output_schema: dict[str, Any] | None,
        kwargs: Mapping[str, Any],
    ) -> A2ARemoteResult:
        if (
            output_schema is not None
            and response_size(output_schema) > DEFAULT_A2A_MAX_SCHEMA_BYTES
        ):
            raise ValueError(
                f"A2A structured output schema exceeds {DEFAULT_A2A_MAX_SCHEMA_BYTES} bytes"
            )
        instructions = kwargs.get("instructions")
        unknown = set(kwargs).difference({"instructions"})
        if unknown:
            names = ", ".join(sorted(unknown))
            raise TypeError(f"Unsupported A2A invocation setting(s): {names}")
        if instructions is not None:
            if not isinstance(instructions, str):
                raise TypeError("A2A instructions must be a string")
            prompt = f"{instructions.rstrip()}\n\n{prompt}" if instructions.strip() else prompt
        connection = self._active_connection()
        conversation = self._conversation_var.get()
        resume = _resume_coordinates(conversation, connection.endpoint)
        task_id = resume.get("task_id") if resume.get("state") in _CONTINUABLE_TASK_STATES else None
        try:
            result = await self._send_official(
                connection,
                prompt,
                output_schema=output_schema,
                task_id=task_id,
                context_id=resume.get("context_id"),
            )
        except A2AError as exc:
            _record_error_coordinates(conversation, connection.endpoint, exc)
            raise
        _record_coordinates(conversation, connection.endpoint, result)
        report_invocation(InvocationUsage(requests=1))
        return result

    def _active_connection(self) -> A2AConnection:
        connection = self.active_profile.a2a_connection or self._connection
        if connection is None:
            raise RuntimeError(
                "A2AAdapter requires an endpoint in '> agent: a2a:' or A2AAdapter(connection=...)"
            )
        return connection

    async def _send_official(
        self,
        connection: A2AConnection,
        prompt: str,
        *,
        output_schema: dict[str, Any] | None,
        task_id: str | None,
        context_id: str | None,
    ) -> A2ARemoteResult:
        try:
            from a2a.client.errors import A2AClientError
            from a2a.types import (
                Message,
                Part,
                Role,
                SendMessageConfiguration,
                SendMessageRequest,
            )
        except ImportError as exc:
            raise _missing_a2a_dependency() from exc

        extension_requested = output_schema is not None
        async with self._client_scope(connection, structured=extension_requested) as (
            client,
            call_context,
        ):
            message = Message(
                message_id=uuid4().hex,
                context_id=context_id or "",
                task_id=task_id or "",
                role=Role.ROLE_USER,
                parts=[Part(text=prompt)],
                extensions=([KEDI_STRUCTURED_OUTPUT_EXTENSION_URI] if extension_requested else []),
            )
            request = SendMessageRequest(
                message=message,
                configuration=SendMessageConfiguration(
                    accepted_output_modes=(
                        ["application/json"] if extension_requested else ["text/plain"]
                    ),
                    history_length=1,
                    return_immediately=False,
                ),
                metadata=(structured_output_request(output_schema) if extension_requested else {}),
            )
            accumulator = _ResponseAccumulator(max_bytes=self._max_response_bytes)
            deadline = asyncio.get_running_loop().time() + connection.task_timeout
            with agent_event_run(self.shortname) as resolver:
                try:
                    await _consume_stream(
                        client.send_message(request, context=call_context),
                        accumulator=accumulator,
                        deadline=deadline,
                        event_resolver=resolver,
                        fallback_message_id=message.message_id,
                    )
                except A2AClientError as exc:
                    if accumulator.task_id is None:
                        raise A2AAmbiguousSendError(
                            "The A2A send may have reached the peer, but no remote task ID was observed; "
                            "Kedi will not resend it automatically",
                            context_id=accumulator.context_id or context_id,
                            state=accumulator.state,
                        ) from exc
                    await _recover_observation(
                        client,
                        connection=connection,
                        call_context=call_context,
                        accumulator=accumulator,
                        deadline=deadline,
                        event_resolver=resolver,
                        cause=exc,
                    )
                except asyncio.TimeoutError as exc:
                    raise _task_timeout(accumulator) from exc
                if not accumulator.is_complete:
                    if accumulator.task_id is None:
                        raise A2ATransportError(
                            "Remote A2A response ended without a message or task handle"
                        )
                    await _recover_observation(
                        client,
                        connection=connection,
                        call_context=call_context,
                        accumulator=accumulator,
                        deadline=deadline,
                        event_resolver=resolver,
                    )
                result = accumulator.result(structured=extension_requested)
                if resolver is not None and result.text is not None:
                    resolver.complete_run(final_text=result.text)
            return result

    @asynccontextmanager
    async def _client_scope(
        self,
        connection: A2AConnection,
        *,
        structured: bool,
    ) -> AsyncIterator[tuple[Any, Any]]:
        try:
            import httpx
            from a2a.client import A2ACardResolver, ClientCallContext, ClientConfig, create_client
        except ImportError as exc:
            raise _missing_a2a_dependency() from exc

        headers = {} if connection.auth is None else connection.auth.headers()
        timeout = httpx.Timeout(connection.request_timeout)
        async with httpx.AsyncClient(
            headers=headers,
            timeout=timeout,
            follow_redirects=False,
        ) as http_client:
            card = await self._agent_card(connection, http_client, A2ACardResolver)
            _validate_agent_card(connection.endpoint, card)
            _validate_card_auth(card, connection.auth)
            require_structured = (
                structured or "structured_output" in self.active_profile.requirements
            )
            if require_structured and not _supports_kedi_structured_output(card):
                raise A2ARemoteTaskError(
                    "Remote A2A agent does not advertise Kedi structured output v1"
                )
            accepted_output_modes = ["application/json"] if structured else ["text/plain"]
            client = await create_client(
                card,
                ClientConfig(
                    streaming=True,
                    polling=False,
                    httpx_client=http_client,
                    accepted_output_modes=accepted_output_modes,
                ),
            )
            call_context = ClientCallContext(
                timeout=connection.request_timeout,
                service_parameters=(
                    {"A2A-Extensions": KEDI_STRUCTURED_OUTPUT_EXTENSION_URI} if structured else None
                ),
            )
            yield client, call_context

    async def _agent_card(
        self, connection: A2AConnection, http_client: Any, resolver_type: Any
    ) -> Any:
        identity = (
            connection.endpoint,
            None if connection.auth is None else connection.auth.identity(),
        )
        with self._card_cache_lock:
            encoded = self._card_cache.get(identity)
            if encoded is not None:
                self._card_cache.move_to_end(identity)
        if encoded is not None:
            from a2a.types import AgentCard

            card = AgentCard()
            card.ParseFromString(encoded)
            return card
        from a2a.client.errors import AgentCardResolutionError

        resolver = resolver_type(http_client, connection.endpoint)
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                card = await resolver.get_agent_card(
                    http_kwargs={"timeout": connection.request_timeout}
                )
                break
            except AgentCardResolutionError as exc:
                last_error = exc
                status_code = getattr(exc, "status_code", None)
                retryable = status_code is None or status_code in {408, 429} or status_code >= 500
                if not retryable or attempt == 2:
                    raise A2ATransportError("Remote A2A Agent Card could not be resolved") from exc
                await asyncio.sleep(connection.poll_interval * (2**attempt))
        else:  # pragma: no cover - the loop either breaks or raises
            raise A2ATransportError("Remote A2A Agent Card could not be resolved") from last_error
        with self._card_cache_lock:
            self._card_cache[identity] = card.SerializeToString()
            self._card_cache.move_to_end(identity)
            while len(self._card_cache) > _MAX_AGENT_CARD_CACHE_ENTRIES:
                self._card_cache.popitem(last=False)
        return card

    @staticmethod
    def _validate_profile(profile: AgentProfile) -> None:
        unsupported: list[str] = []
        if profile.model is not None:
            unsupported.append("model")
        if profile.effort is not None:
            unsupported.append("effort")
        if profile.mcp_servers:
            unsupported.append("mcp")
        if profile.settings:
            unsupported.append("settings")
        codemode = profile.codemode
        if codemode is True or (
            codemode is not None and not isinstance(codemode, bool) and codemode.enabled
        ):
            unsupported.append("codemode")
        if profile.compaction is not None:
            unsupported.append("compaction")
        if unsupported:
            raise AgentProfileOverrideUnsupported(
                "A2A leaves remote runtime configuration to the remote agent; unsupported local "
                f"profile field(s): {', '.join(unsupported)}"
            )


async def _consume_stream(
    stream: AsyncIterator[Any],
    *,
    accumulator: _ResponseAccumulator,
    deadline: float,
    event_resolver: Any,
    fallback_message_id: str,
) -> None:
    async def consume() -> None:
        async for response in stream:
            accumulator.add(response)
            if event_resolver is not None and accumulator.latest_message_text:
                event_resolver.complete_message(
                    f"a2a:{accumulator.task_id or fallback_message_id}",
                    accumulator.latest_message_text,
                )

    await asyncio.wait_for(consume(), timeout=_remaining(deadline))


async def _recover_observation(
    client: Any,
    *,
    connection: A2AConnection,
    call_context: Any,
    accumulator: _ResponseAccumulator,
    deadline: float,
    event_resolver: Any,
    cause: Exception | None = None,
) -> None:
    try:
        from a2a.client.errors import A2AClientError
        from a2a.types import GetTaskRequest, SubscribeToTaskRequest
    except ImportError as exc:
        raise _missing_a2a_dependency() from exc

    task_id = accumulator.task_id
    if task_id is None:
        raise A2ATransportError("Cannot recover an A2A task without its task ID")

    subscribe_error: Exception | None = None
    try:
        await _consume_stream(
            client.subscribe(
                SubscribeToTaskRequest(id=task_id),
                context=call_context,
            ),
            accumulator=accumulator,
            deadline=deadline,
            event_resolver=event_resolver,
            fallback_message_id=task_id,
        )
    except asyncio.TimeoutError as exc:
        raise _task_timeout(accumulator) from exc
    except A2AClientError as exc:
        subscribe_error = exc
    if accumulator.is_complete:
        return

    consecutive_failures = 0
    while True:
        try:
            task = await asyncio.wait_for(
                client.get_task(
                    GetTaskRequest(id=task_id, history_length=1),
                    context=call_context,
                ),
                timeout=min(connection.request_timeout, _remaining(deadline)),
            )
        except asyncio.TimeoutError as exc:
            if deadline <= asyncio.get_running_loop().time():
                raise _task_timeout(accumulator) from exc
            consecutive_failures += 1
        except A2AClientError as exc:
            consecutive_failures += 1
            subscribe_error = exc
        else:
            consecutive_failures = 0
            accumulator.add_task(task)
            if event_resolver is not None and accumulator.latest_message_text:
                event_resolver.complete_message(
                    f"a2a:{task_id}",
                    accumulator.latest_message_text,
                )
        if accumulator.is_complete:
            return
        if consecutive_failures >= 3:
            error = A2ATransportError(
                "Remote A2A task exists, but subscription and status recovery failed",
                task_id=task_id,
                context_id=accumulator.context_id,
                state=accumulator.state,
            )
            raise error from (subscribe_error or cause)
        try:
            delay = min(connection.poll_interval, _remaining(deadline))
        except asyncio.TimeoutError as exc:
            raise _task_timeout(accumulator) from exc
        await asyncio.sleep(delay)


async def _safe_status_read(
    read: Callable[[], Awaitable[Any]],
    *,
    connection: A2AConnection,
    task_id: str,
) -> Any:
    try:
        from a2a.client.errors import A2AClientError
    except ImportError as exc:
        raise _missing_a2a_dependency() from exc

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            return await read()
        except A2AClientError as exc:
            last_error = exc
            if attempt < 2:
                await asyncio.sleep(connection.poll_interval * (2**attempt))
    raise A2ATransportError(
        "Remote A2A task status could not be read",
        task_id=task_id,
    ) from last_error


def _supports_kedi_structured_output(card: Any) -> bool:
    from google.protobuf.json_format import MessageToDict

    for extension in card.capabilities.extensions:
        if extension.uri != KEDI_STRUCTURED_OUTPUT_EXTENSION_URI:
            continue
        params = MessageToDict(extension.params) if extension.HasField("params") else {}
        return params.get("version") == 1 and params.get("mode") in {
            "fixed",
            "request-schema",
        }
    return False


def _validate_agent_card(endpoint: str, card: Any) -> None:
    endpoint_url = urlsplit(endpoint)
    candidates = [
        interface
        for interface in card.supported_interfaces
        if interface.protocol_binding.upper() == "JSONRPC"
    ]
    if not candidates:
        raise A2ARemoteTaskError("Remote agent card has no JSON-RPC interface")
    for interface in candidates:
        target = urlsplit(interface.url)
        if target.scheme not in {"http", "https"} or not target.hostname:
            raise A2ARemoteTaskError("Remote agent card advertises an invalid interface URL")
        if _url_origin(target) != _url_origin(endpoint_url):
            raise A2ARemoteTaskError(
                "Remote agent card redirects A2A traffic to a different origin"
            )


def _validate_card_auth(card: Any, auth: A2AAuthConfig | None) -> None:
    requirements = tuple(card.security_requirements)
    if not requirements:
        return
    for requirement in requirements:
        names = tuple(requirement.schemes)
        if not names:
            return
        if len(names) != 1 or auth is None:
            continue
        scheme = card.security_schemes.get(names[0])
        if scheme is not None and _auth_matches_scheme(auth, scheme):
            return
    configured = "none" if auth is None else auth.scheme
    raise A2ATransportError(
        "Remote A2A Agent Card requires an authentication scheme that the configured "
        f"credential cannot satisfy (configured: {configured})"
    )


def _auth_matches_scheme(auth: A2AAuthConfig, scheme: Any) -> bool:
    selected = scheme.WhichOneof("scheme")
    if selected == "http_auth_security_scheme":
        advertised = scheme.http_auth_security_scheme.scheme.lower()
        return (auth.scheme == "basic" and advertised == "basic") or (
            auth.scheme == "bearer" and advertised == "bearer"
        )
    if selected == "api_key_security_scheme" and auth.scheme == "api_key":
        advertised = scheme.api_key_security_scheme
        return (
            advertised.location.lower() == "header"
            and auth.header is not None
            and advertised.name.lower() == auth.header.lower()
        )
    return False


def _url_origin(value: Any) -> tuple[str, str | None, int | None]:
    port = value.port
    if port is None:
        port = 443 if value.scheme == "https" else 80
    return value.scheme, value.hostname, port


def _resume_coordinates(
    conversation: AdapterConversationContext | SubagentConversationContext | None,
    endpoint: str,
) -> dict[str, str]:
    if conversation is None:
        return {}
    value = conversation.resume_state.get("a2a")
    if not isinstance(value, Mapping) or value.get("endpoint") != endpoint:
        return {}
    return {
        key: item
        for key in ("task_id", "context_id", "state")
        if isinstance((item := value.get(key)), str) and item
    }


def _record_coordinates(
    conversation: AdapterConversationContext | SubagentConversationContext | None,
    endpoint: str,
    result: A2ARemoteResult,
) -> None:
    if conversation is None:
        return
    conversation.result_state["a2a"] = {
        "endpoint": endpoint,
        "task_id": result.task_id,
        "context_id": result.context_id,
        "state": result.state,
    }
    mark_completed = getattr(conversation, "mark_completed", None)
    if callable(mark_completed):
        mark_completed()


def _record_error_coordinates(
    conversation: AdapterConversationContext | SubagentConversationContext | None,
    endpoint: str,
    error: A2AError,
) -> None:
    if conversation is None or error.task_id is None:
        return
    state = {
        "endpoint": endpoint,
        "task_id": error.task_id,
        "context_id": error.context_id,
        "state": error.state,
    }
    conversation.result_state["a2a"] = state
    # A suspended task is a valid continuation checkpoint even though its
    # invocation did not produce a successful Kedi value.
    if error.state in _CONTINUABLE_TASK_STATES:
        conversation.resume_state["a2a"] = dict(state)


def _task_snapshot(task: Any, *, max_bytes: int) -> A2ARemoteTaskSnapshot:
    accumulator = _ResponseAccumulator(max_bytes=max_bytes)
    accumulator.add_task(task)
    return accumulator.snapshot()


def _task_timeout(accumulator: _ResponseAccumulator) -> A2ATaskTimeoutError:
    return A2ATaskTimeoutError(
        "Remote A2A task exceeded its local observation deadline; the remote task was not cancelled",
        task_id=accumulator.task_id,
        context_id=accumulator.context_id,
        state=accumulator.state,
    )


def _remaining(deadline: float) -> float:
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        raise asyncio.TimeoutError
    return remaining


def _required_identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _optional_identifier(value: str | None, name: str) -> str | None:
    return None if value is None else _required_identifier(value, name)


def _missing_a2a_dependency() -> RuntimeError:
    return RuntimeError("A2A support is not installed. Install it with: uv add 'kedi[a2a]'")


@contextmanager
def _empty_scope() -> Iterator[None]:
    yield
