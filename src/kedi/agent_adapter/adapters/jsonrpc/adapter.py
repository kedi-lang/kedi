"""``JsonRpcAdapter``: Kedi's adapter contract over JSON-RPC to a bridge in any language.

The adapter launches the bridge, completes ``initialize`` before it exists, and takes its
capabilities from that handshake. Each call forwards ``current_request().render()``; the
bridge runs its framework and calls Kedi tools back with ``tool/call``. Kedi validates the
answer and, when it is wrong, sends its own correction as a continuation of the run.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
from collections.abc import Coroutine, Generator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self
from uuid import uuid4

from pydantic import TypeAdapter, ValidationError
from pydantic_core import to_jsonable_python

from ....agent_profile import AgentProfile, AgentProfileOverrideUnsupported
from ....agent_profile.settings import KEDI_RUNTIME_SETTING_KEYS
from ...approvals.policy import ToolApprovalDenied
from ...execution import run_sync
from ...hooks import HookDeniedError
from ...invocation import InvocationUsage, report_invocation
from ...meta import AdapterCapabilities, AgentAdapterKind
from ...request import (
    OutputContract,
    SchemaOutput,
    TextOutput,
    TypedOutput,
    call_request,
    output_validator,
)
from ...request.prompts import output_correction, protocol_correction, required_tools_correction
from ...tools.errors import correctable_tool_error
from ...tools.execution import validate_tool_arguments
from ...tools.specs import ToolSpec
from .connection import Connection
from .protocol import (
    CANCELLED,
    CAPABILITIES,
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    PROTOCOL_VERSION,
    RUN_FAILED,
    UNKNOWN_RUN,
    UNSUPPORTED,
    JsonObject,
    JsonRpcDisconnected,
    JsonRpcError,
)

REFERENCE_BRIDGE = Path(__file__).with_name("openai_chat_bridge.mjs")
"""A bridge in JavaScript (Node.js 18+) for any OpenAI-compatible chat endpoint."""


@dataclass
class _Run:
    """A run in progress: its tools, where their callbacks execute, and their tasks."""

    run_id: str
    loop: asyncio.AbstractEventLoop
    context: contextvars.Context
    tools: Mapping[str, ToolSpec]
    called: set[str] = field(default_factory=set)
    attempted: set[str] = field(default_factory=set)
    tasks: set[asyncio.Task[None]] = field(default_factory=set)
    failure: BaseException | None = None

    def start(self, coroutine: Coroutine[Any, Any, None]) -> None:
        """Start a callback on the run's loop; called in the run's context."""

        task = self.loop.create_task(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)


class JsonRpcAdapter:
    """An adapter whose backend is a bridge process speaking Kedi's JSON-RPC protocol."""

    shortname = "jsonrpc"

    def __init__(
        self,
        command: Sequence[str],
        *,
        model: str | None = None,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        request_timeout: float | None = None,
        initialize_timeout: float = 30,
        output_retries: int = 1,
        max_inbound_requests: int = 64,
    ) -> None:
        """Launch ``command`` and complete the handshake; raises if the bridge is not ready.

        The adapter owns the process and stops it on :meth:`close`. ``request_timeout``
        bounds each run; ``output_retries`` is how many corrections Kedi sends per call.
        """

        self._model = model
        self._request_timeout = request_timeout
        self._output_retries = output_retries
        self._runs: dict[str, _Run] = {}
        self._profiles: ContextVar[tuple[AgentProfile, ...]] = ContextVar(
            f"jsonrpc_profiles_{id(self)}", default=()
        )
        self._tools: ContextVar[tuple[ToolSpec, ...]] = ContextVar(
            f"jsonrpc_tools_{id(self)}", default=()
        )
        self._connection = Connection(
            command,
            handler=self._on_request,
            env=env,
            cwd=cwd,
            max_inbound_requests=max_inbound_requests,
        )
        try:
            handshake = self._connection.request_sync(
                "initialize",
                {"protocolVersion": PROTOCOL_VERSION, "client": {"name": "kedi"}},
                timeout=initialize_timeout,
            )
            self.kind, self.backend, self.capabilities = _negotiate(handshake)
        except BaseException:
            self._connection.close(timeout=1)
            raise

    @classmethod
    async def launch(cls, command: Sequence[str], **options: Any) -> Self:
        """Launch the bridge from async code without blocking the event loop."""

        return await asyncio.to_thread(cls, command, **options)

    @classmethod
    def for_conformance(cls, base_url: str) -> Self:
        """The reference bridge on the conformance kit's endpoint (``kedi adapter check``)."""

        return cls(reference_bridge_command(f"{base_url}/v1"), model="conformance")

    def default_model(self) -> str | None:
        return self._model

    @property
    def active_profile(self) -> AgentProfile:
        stack = self._profiles.get()
        return stack[-1] if stack else AgentProfile.empty()

    @contextmanager
    def profile_override(self, profile: AgentProfile) -> Generator[None, None, None]:
        unsupported = [
            name
            for name, present in (
                ("effort", profile.effort is not None),
                ("mcp", bool(profile.mcp_servers)),
            )
            if present
        ]
        unsupported += sorted(set(profile.settings) - KEDI_RUNTIME_SETTING_KEYS)
        if unsupported:
            raise AgentProfileOverrideUnsupported(
                f"{self.shortname} does not support profile fields: {', '.join(unsupported)}"
            )
        token = self._profiles.set((*self._profiles.get(), profile))
        try:
            yield
        finally:
            self._profiles.reset(token)

    @contextmanager
    def tool_scope(self, tools: Sequence[object] = ()) -> Generator[None, None, None]:
        token = self._tools.set(tuple(tool for tool in tools if isinstance(tool, ToolSpec)))
        try:
            yield
        finally:
            self._tools.reset(token)

    def register_tool(self, tool: ToolSpec) -> None:
        self._tools.set((*self._tools.get(), tool))

    async def invoke(self, *, prompt: str, instructions: str | None = None, **_: Any) -> str:
        return await self._run(prompt, TextOutput(), instructions)

    async def produce(
        self,
        *,
        template: str,
        output_schema: dict[str, Any] | None = None,
        output_type: Any = None,
        instructions: str | None = None,
        **_: Any,
    ) -> Any:
        if not self.capabilities.supports_structured_output:
            raise JsonRpcError(
                UNSUPPORTED, f"bridge {self.backend['name']!r} does not support structured output"
            )
        if output_schema is not None:
            output: OutputContract = SchemaOutput(schema=output_schema)
        elif output_type is not None:
            output = TypedOutput(output_type=output_type)
        else:
            raise ValueError("produce needs output_schema or output_type")
        return await self._run(template, output, instructions)

    def invoke_sync(self, **kwargs: Any) -> str:
        return run_sync(self.invoke(**kwargs))

    def produce_sync(self, **kwargs: Any) -> Any:
        return run_sync(self.produce(**kwargs))

    def close(self, *, timeout: float = 5) -> None:
        """Shut the bridge down and stop its process, waiting at most ``timeout`` per step."""

        self._connection.close(timeout=timeout)

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    async def _run(self, prompt: str, output: OutputContract, instructions: str | None) -> Any:
        disconnected = self._connection.disconnected
        if disconnected is not None:
            raise disconnected
        tools = self._tools.get()
        request = call_request(
            prompt=prompt,
            output=output,
            instructions=instructions,
            profile=self.active_profile,
            capabilities=self.capabilities,
            tools=tools,
        )
        rendered = request.render()
        validator = output_validator(request.output, request=request)
        run_id = uuid4().hex
        run = _Run(
            run_id=run_id,
            loop=asyncio.get_running_loop(),
            context=contextvars.copy_context(),
            tools={tool.name: tool for tool in tools},
        )
        required = {tool.name for tool in rendered.tools if tool.required_before_output}
        params: JsonObject = {
            "runId": run_id,
            "callKind": rendered.call_kind,
            "promptKind": rendered.prompt_kind,
            "model": self.active_profile.model or self._model,
            "system": rendered.system,
            "instructionParts": [
                {"identity": part.identity, "source": part.source, "content": part.content}
                for part in rendered.instruction_parts
            ],
            "user": rendered.user,
            "output": None
            if rendered.output_json_schema is None
            else {"schema": rendered.output_json_schema, "toolName": rendered.output_tool_name},
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.parameters,
                    "outputSchema": tool.returns,
                    "requiredBeforeOutput": tool.required_before_output,
                }
                for tool in rendered.tools
            ],
            "continuation": None,
        }
        self._runs[run_id] = run
        correction: str | None = None
        try:
            for _ in range(self._output_retries + 1):
                try:
                    result = await self._request_run(params)
                finally:
                    if run.failure is not None:
                        raise run.failure
                answer, correction = _check_answer(result, validator, required, run)
                if correction is None:
                    return answer
                params = {**params, "continuation": {"answer": result, "message": correction}}
            raise JsonRpcError(
                INVALID_PARAMS,
                f"bridge {self.backend['name']!r} gave no valid answer after "
                f"{self._output_retries} corrections; the last: {correction}",
            )
        finally:
            self._runs.pop(run_id, None)
            for task in run.tasks:
                task.cancel()

    async def _request_run(self, params: JsonObject) -> JsonObject:
        try:
            result = await self._connection.request("run", params, timeout=self._request_timeout)
        except (asyncio.CancelledError, TimeoutError):
            # Cancellation is a request: the bridge stopping is not confirmed by it.
            self._connection.notify("run/cancel", {"runId": params["runId"]})
            raise
        usage = result.get("usage")
        if isinstance(usage, dict):
            report_invocation(
                InvocationUsage(
                    requests=int(usage.get("requests", 1)),
                    input_tokens=int(usage.get("inputTokens", 0)),
                    output_tokens=int(usage.get("outputTokens", 0)),
                    total_tokens=int(
                        usage.get(
                            "totalTokens",
                            usage.get("inputTokens", 0) + usage.get("outputTokens", 0),
                        )
                    ),
                )
            )
        return result

    def _on_request(self, request_id: int | str, method: str, params: JsonObject) -> None:
        """A bridge request, on the reader thread: start it as a task on the run's loop."""

        if method != "tool/call":
            error = JsonRpcError(METHOD_NOT_FOUND, f"Kedi does not handle {method!r}")
            self._connection.respond(request_id, error=error)
            return
        run_id, name = params.get("runId"), params.get("name")
        arguments = params.get("arguments", {})
        if (
            not isinstance(run_id, str)
            or not isinstance(name, str)
            or not isinstance(arguments, dict)
        ):
            error = JsonRpcError(
                INVALID_PARAMS, "tool/call needs a string runId and name, and object arguments"
            )
            self._connection.respond(request_id, error=error)
            return
        run = self._runs.get(run_id)
        if run is None:
            error = JsonRpcError(UNKNOWN_RUN, "tool/call names no run in progress")
            self._connection.respond(request_id, error=error)
            return
        coroutine = self._call_tool(request_id, run, name, arguments)
        try:
            run.loop.call_soon_threadsafe(lambda: run.context.run(run.start, coroutine))
        except RuntimeError:  # the run's loop closed: the run is over
            coroutine.close()
            error = JsonRpcError(UNKNOWN_RUN, "tool/call names a run that has ended")
            self._connection.respond(request_id, error=error)

    async def _call_tool(
        self, request_id: int | str, run: _Run, name: str, arguments: JsonObject
    ) -> None:
        """Run one Kedi tool as the reference adapter does, and answer the bridge."""

        tool = run.tools.get(name)
        if tool is None:
            self._connection.respond(
                request_id, result={"ok": False, "error": f"Unknown Kedi tool: {name}"}
            )
            return
        run.attempted.add(name)
        try:
            # Sync tools run off the loop; Kedi's wrapper applies approval and budgets.
            value = await asyncio.to_thread(_call, tool, arguments)
            if inspect.isawaitable(value):
                value = await value
            reply = {"ok": True, "result": to_jsonable_python(value, fallback=str)}
            run.called.add(name)
        except asyncio.CancelledError:
            self._connection.respond(request_id, error=JsonRpcError(CANCELLED, "the run ended"))
            raise
        except (ToolApprovalDenied, HookDeniedError) as exc:
            reply = {"ok": False, "error": exc.message}
        except Exception as exc:
            message = correctable_tool_error(tool, exc)
            if message is None:
                # Not the model's to correct (a budget, a bug): Kedi stops the run with it.
                run.failure = exc
                error = JsonRpcError(RUN_FAILED, f"Kedi stopped the run: {type(exc).__name__}")
                self._connection.respond(request_id, error=error)
                self._connection.notify("run/cancel", {"runId": run.run_id})
                return
            reply = {"ok": False, "error": message}
        self._connection.respond(request_id, result=reply)


def _call(tool: ToolSpec, arguments: JsonObject) -> Any:
    return tool.fn(**validate_tool_arguments(tool, arguments))


def _negotiate(handshake: JsonObject) -> tuple[AgentAdapterKind, JsonObject, AdapterCapabilities]:
    """The bridge's kind, identity and capabilities, or an error naming what is wrong."""

    version = handshake.get("protocolVersion")
    if version != PROTOCOL_VERSION:
        raise JsonRpcError(
            UNSUPPORTED,
            f"the bridge speaks protocol version {version!r}; Kedi speaks {PROTOCOL_VERSION}",
        )
    kind = handshake.get("kind")
    backend = handshake.get("backend")
    advertised = handshake.get("capabilities", {})
    if kind not in ("agent-framework", "agent-harness"):
        raise JsonRpcError(INVALID_PARAMS, f"unknown bridge kind {kind!r}")
    if not isinstance(backend, dict) or not isinstance(backend.get("name"), str):
        raise JsonRpcError(INVALID_PARAMS, "initialize must name the backend")
    if not isinstance(advertised, dict):
        raise JsonRpcError(INVALID_PARAMS, "capabilities must be an object")
    flags = {flag: advertised.get(name) is True for name, flag in CAPABILITIES.items()}
    tools = flags["supports_tool_registration"]
    capabilities = AdapterCapabilities(
        supports_structured_output=flags["supports_structured_output"],
        supports_tool_registration=tools,
        supports_artifacts=tools,
        supports_native_structured_output=flags["supports_native_structured_output"],
        supports_native_tool_calling=flags["supports_native_tool_calling"],
        supports_profile_override=True,
        supports_model_override=True,
        supports_effort=False,
        supports_settings=False,
        supports_mcp=False,
    )
    return kind, backend, capabilities


def _check_answer(
    result: JsonObject, validator: TypeAdapter[Any] | None, required: set[str], run: _Run
) -> tuple[Any, str | None]:
    """The validated answer, or Kedi's correction for the next turn."""

    if "invalid" in result:
        # The model's answer was not one protocol action; the bridge quotes its parse error.
        return None, protocol_correction(str(result.get("error") or "") or None)
    missing = required - run.called
    if missing:
        return None, required_tools_correction(missing, attempted=bool(missing & run.attempted))
    if validator is None:
        text = result.get("text")
        if not isinstance(text, str):
            raise JsonRpcError(INVALID_PARAMS, "a text run must answer with a string 'text'")
        return text, None
    if "data" not in result:
        raise JsonRpcError(INVALID_PARAMS, "a structured run must answer with 'data'")
    try:
        return validator.validate_python(result["data"]), None
    except ValidationError as exc:
        return None, output_correction(exc)


def reference_bridge_command(base_url: str, *, mode: str = "native") -> list[str]:
    """Run the reference bridge with Node.js against an OpenAI-compatible ``base_url``.

    ``mode="protocol"`` treats the endpoint as having no native structured output or tool
    calling, so Kedi sends its protocol parts.
    """

    return ["node", str(REFERENCE_BRIDGE), "--base-url", base_url, "--mode", mode]


__all__ = ["REFERENCE_BRIDGE", "JsonRpcAdapter", "JsonRpcDisconnected", "reference_bridge_command"]
