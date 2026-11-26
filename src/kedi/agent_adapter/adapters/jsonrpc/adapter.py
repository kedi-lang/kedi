"""``JsonRpcAdapter``: Kedi's adapter contract over JSON-RPC to a bridge in any language.

The adapter launches the bridge, completes ``initialize`` before it exists, and takes its
capabilities from that handshake. Each call forwards ``current_request().render()``; the
bridge runs its framework and calls Kedi tools back with ``tool/call``. Kedi validates the
answer and, when it is wrong, sends its own correction as a continuation of the run.
"""

from __future__ import annotations

import asyncio
import base64
import contextvars
import copy
import inspect
import shlex
from collections.abc import Callable, Coroutine, Generator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Self, cast
from uuid import uuid4

from pydantic import TypeAdapter, ValidationError
from pydantic_core import to_jsonable_python

from ....agent_profile import AgentProfile, AgentProfileOverrideUnsupported
from ....agent_profile.settings import KEDI_RUNTIME_SETTING_KEYS
from ....media import MediaError
from ....run_budget import BudgetDimension, RunBudgetExceeded, admit_run_operation
from ...approvals.policy import (
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
    ToolApprovalDenied,
    ToolRisk,
    resolve_tool_approval_async,
)
from ...conversation import AdapterConversationContext
from ...events import AgentMessageResolver, agent_event_run
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
from ...request.context import take_prompt_media
from ...request.media import MediaAttachment, attachment_content
from ...request.prompts import output_correction, protocol_correction, required_tools_correction
from ...tools.errors import correctable_tool_error
from ...tools.execution import validate_tool_arguments
from ...tools.specs import ToolSpec
from .connection import Connection
from .protocol import (
    BUDGET_DIMENSIONS,
    CANCELLED,
    CAPABILITIES,
    INVALID_PARAMS,
    MEDIA_KINDS,
    METHOD_NOT_FOUND,
    NATIVE_TOOLS,
    PROTOCOL_VERSION,
    RUN_FAILED,
    TOOL_RISKS,
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
    events: AgentMessageResolver | None = None

    def start(self, coroutine: Coroutine[Any, Any, None]) -> None:
        """Start a callback on the run's loop; called in the run's context."""

        task = self.loop.create_task(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    def stop(self, failure: BaseException) -> None:
        """Kedi ends the run with ``failure``: a budget, a cancelled approval, a fatal tool."""

        if self.failure is None:
            self.failure = failure


_Handler = Callable[[int | str, _Run], Coroutine[Any, Any, None]]
"""Serves one bridge request for a run; started on the run's loop, in its context."""


class JsonRpcAdapter:
    """An adapter whose backend is a bridge process speaking Kedi's JSON-RPC protocol."""

    shortname = "rpc"
    kind: AgentAdapterKind = "agent-harness"
    capabilities = AdapterCapabilities(
        # Negotiated per bridge in initialize; an instance replaces these with the bridge's.
        supports_structured_output=None,
        supports_tool_registration=None,
        supports_artifacts=None,
        supports_stream_events=None,
        supports_media_input=None,
        supports_stateful_history=None,
        supports_request_budget=None,
        supports_tool_attempt_budget=None,
        negotiated_capabilities=frozenset(
            {
                "structured_output",
                "tool_registration",
                "artifacts",
                "stream_events",
                "media_input",
                "stateful_history",
                "request_budget",
                "tool_attempt_budget",
            }
        ),
        supports_profile_override=True,
        supports_model_override=True,
        supports_effort=False,
        supports_settings=False,
        supports_mcp=False,
    )

    def __init__(
        self,
        command: str | Sequence[str],
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

        self.command = tuple(shlex.split(command) if isinstance(command, str) else command)
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
        self._conversation: ContextVar[AdapterConversationContext | None] = ContextVar(
            f"jsonrpc_conversation_{id(self)}", default=None
        )
        self._connection = Connection(
            self.command,
            handler=self._on_request,
            notifications=self._on_notification,
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
            peer = _negotiate(handshake)
        except BaseException:
            self._connection.close(timeout=1)
            raise
        self.kind, self.backend, self.capabilities = peer.kind, peer.backend, peer.capabilities
        self.media_kinds, self.native_tools = peer.media_kinds, peer.native_tools

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

    @contextmanager
    def conversation_scope(
        self, conversation: AdapterConversationContext
    ) -> Generator[None, None, None]:
        """Kedi owns the conversation; the bridge's ``state`` travels with each turn."""

        token = self._conversation.set(conversation)
        try:
            yield
        finally:
            self._conversation.reset(token)

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
        conversation = self._conversation.get()
        # The first native run in a Kedi call takes the prompt's media; nested runs get none.
        attachments = take_prompt_media()
        user_content = (
            _user_content(rendered.user, attachments, self.media_kinds, self.backend["name"])
            if attachments
            else None
        )
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
            "userContent": user_content,
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
            "session": None
            if conversation is None
            else {
                "id": conversation.session_id,
                "turn": conversation.turn_sequence,
                "state": conversation.resume_state.get("bridge"),
            },
        }
        self._runs[run_id] = run
        try:
            with agent_event_run(self.shortname) as run.events:
                answer = await self._attempts(run, params, validator, required)
                if run.events is not None and isinstance(answer, str):
                    run.events.complete_run(final_text=answer)
                return answer
        finally:
            self._runs.pop(run_id, None)
            for task in run.tasks:
                task.cancel()

    async def _attempts(
        self,
        run: _Run,
        params: JsonObject,
        validator: TypeAdapter[Any] | None,
        required: set[str],
    ) -> Any:
        """Ask the bridge, then send Kedi's corrections, until an answer is valid."""

        correction: str | None = None
        for _ in range(self._output_retries + 1):
            try:
                result = await self._request_run(params)
            finally:
                if run.failure is not None:
                    raise run.failure
            answer, correction = _check_answer(result, validator, required, run)
            if correction is None:
                conversation = self._conversation.get()
                if conversation is not None and "state" in result:
                    # Kedi commits it only after accepting this turn's result.
                    conversation.result_state["bridge"] = result["state"]
                return answer
            params = {**params, "continuation": {"answer": result, "message": correction}}
        raise JsonRpcError(
            INVALID_PARAMS,
            f"bridge {self.backend['name']!r} gave no valid answer after "
            f"{self._output_retries} corrections; the last: {correction}",
        )

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
        """A bridge request, on the reader thread: start it as a task on the run's loop.

        The task runs in the run's context, so Kedi's run budget, approval policies and
        observers are the ones of the Kedi call that started the run.
        """

        parse = {
            "tool/call": self._parse_tool_call,
            "budget/admit": self._parse_admission,
            "approval/request": self._parse_approval,
        }.get(method)
        try:
            if parse is None:
                raise JsonRpcError(METHOD_NOT_FOUND, f"Kedi does not handle {method!r}")
            handle = parse(params)
            run = self._runs.get(params["runId"])
            if run is None:
                raise JsonRpcError(UNKNOWN_RUN, f"{method} names no run in progress")
        except JsonRpcError as error:
            self._connection.respond(request_id, error=error)
            return
        coroutine = handle(request_id, run)
        try:
            run.loop.call_soon_threadsafe(lambda: run.context.run(run.start, coroutine))
        except RuntimeError:  # the run's loop closed: the run is over
            coroutine.close()
            error = JsonRpcError(UNKNOWN_RUN, f"{method} names a run that has ended")
            self._connection.respond(request_id, error=error)

    def _parse_tool_call(self, params: JsonObject) -> _Handler:
        run_id, name = params.get("runId"), params.get("name")
        arguments = params.get("arguments", {})
        if (
            not isinstance(run_id, str)
            or not isinstance(name, str)
            or not isinstance(arguments, dict)
        ):
            raise JsonRpcError(
                INVALID_PARAMS, "tool/call needs a string runId and name, and object arguments"
            )
        return lambda request_id, run: self._call_tool(request_id, run, name, arguments)

    def _parse_admission(self, params: JsonObject) -> _Handler:
        if not self.capabilities.supports_request_budget:
            raise JsonRpcError(UNSUPPORTED, "budget/admit needs the governed capability")
        dimension = params.get("dimension")
        if not isinstance(params.get("runId"), str) or dimension not in BUDGET_DIMENSIONS:
            raise JsonRpcError(
                INVALID_PARAMS,
                "budget/admit needs a string runId and a dimension: request or toolAttempt",
            )
        if dimension == "toolAttempt" and not self.native_tools:
            raise JsonRpcError(
                UNSUPPORTED,
                "toolAttempt admission is for the bridge's own tools (nativeTools); "
                "Kedi admits its tools at tool/call",
            )
        return lambda request_id, run: self._admit(request_id, run, dimension)

    def _parse_approval(self, params: JsonObject) -> _Handler:
        if not self.native_tools:
            raise JsonRpcError(UNSUPPORTED, "approval/request needs the nativeTools capability")
        tool, arguments = params.get("tool"), params.get("arguments")
        risk, description = params.get("risk"), params.get("description")
        if (
            not isinstance(params.get("runId"), str)
            or not isinstance(tool, str)
            or not isinstance(arguments, dict)
            or risk not in TOOL_RISKS
            or not isinstance(description, str | None)
        ):
            raise JsonRpcError(
                INVALID_PARAMS,
                "approval/request needs a string runId and tool, object arguments, a risk "
                f"of {', '.join(sorted(TOOL_RISKS))}, and an optional string description",
            )
        request = ApprovalRequest.create(
            tool_name=tool,
            arguments=arguments,
            risk=cast(ToolRisk, risk),
            adapter_shortname=self.shortname,
            description=description,
            metadata={"source": "jsonrpc-bridge", "backend": self.backend["name"]},
        )
        return lambda request_id, run: self._approve(request_id, run, request)

    def _on_notification(self, method: str, params: JsonObject) -> None:
        """A bridge notification, on the reader thread: a run event, in stream order."""

        if method != "run/event" or not self.capabilities.supports_stream_events:
            return
        run_id = params.get("runId")
        run = self._runs.get(run_id) if isinstance(run_id, str) else None
        events = run.events if run is not None else None
        if events is None:
            return  # nobody observes events, or the run is over
        kind, message_id, text = params.get("type"), params.get("messageId"), params.get("text")
        phase = params.get("phase") if params.get("phase") in ("commentary", "final") else None
        if kind == "tool.boundary":
            events.mark_tool_boundary()
        elif kind == "message.delta" and isinstance(message_id, str) and isinstance(text, str):
            events.append_delta(message_id, text, explicit_phase=phase)
        elif kind == "message.completed" and isinstance(message_id, str):
            events.complete_message(
                message_id, text if isinstance(text, str) else None, explicit_phase=phase
            )

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
                self._stop(request_id, run, exc)
                return
            reply = {"ok": False, "error": message}
        self._connection.respond(request_id, result=reply)

    async def _admit(self, request_id: int | str, run: _Run, dimension: str) -> None:
        """Charge a step the bridge is about to take to the run's budget, as in-process."""

        try:
            admit_run_operation(cast(BudgetDimension, BUDGET_DIMENSIONS[dimension]))
        except RunBudgetExceeded as exc:
            run.stop(exc)
            self._connection.respond(
                request_id,
                result={
                    "admitted": False,
                    "dimension": dimension,
                    "limit": exc.limit,
                    "consumed": exc.consumed,
                    "scopeDepth": exc.scope_depth,
                },
            )
            # The bridge must skip the step and end the run; cancel it in case it does not.
            self._connection.notify("run/cancel", {"runId": run.run_id})
            return
        self._connection.respond(request_id, result={"admitted": True})

    async def _approve(self, request_id: int | str, run: _Run, request: ApprovalRequest) -> None:
        """Decide a call of the bridge's own tool with the policies Kedi's tools get."""

        try:
            decision = await _decide_approval(self._approval_policies(), request)
            reply: JsonObject = {"decision": decision.decision}
            if decision.reason is not None:
                reply["reason"] = decision.reason
            if decision.edit_arguments is not None:
                reply["arguments"] = to_jsonable_python(dict(decision.edit_arguments), fallback=str)
        except asyncio.CancelledError:
            self._connection.respond(request_id, error=JsonRpcError(CANCELLED, "the run ended"))
            raise
        except Exception as exc:  # a cancelled approval or a failing handler ends the run
            self._stop(request_id, run, exc)
            return
        self._connection.respond(request_id, result=reply)

    def _approval_policies(self) -> tuple[ApprovalPolicy | None, ...]:
        return (self.active_profile.approval,)

    def _stop(self, request_id: int | str, run: _Run, failure: BaseException) -> None:
        """End the run with Kedi's ``failure`` and tell the bridge why its request failed."""

        run.stop(failure)
        error = JsonRpcError(RUN_FAILED, f"Kedi stopped the run: {type(failure).__name__}")
        self._connection.respond(request_id, error=error)
        self._connection.notify("run/cancel", {"runId": run.run_id})


async def _decide_approval(
    policies: Sequence[ApprovalPolicy | None], request: ApprovalRequest
) -> ApprovalDecision:
    """Every policy decides in turn, as ``with_approval_policies`` chains Kedi's tools.

    An edit passes the edited arguments to the next policy. The bridge declared the risk, so
    a riskier edit is the bridge's to ask about again.
    """

    arguments = dict(request.arguments)
    reasons: list[str] = []
    for policy in policies:
        current = replace(request, arguments=MappingProxyType(copy.deepcopy(arguments)))
        decision = await resolve_tool_approval_async(policy, current)
        if decision.decision == "deny":
            return decision
        if decision.reason is not None:
            reasons.append(decision.reason)
        if decision.edit_arguments is not None:
            arguments = dict(decision.edit_arguments)
    reason = "; ".join(reasons) or None
    if arguments != dict(request.arguments):
        return ApprovalDecision.edit(arguments, reason=reason)
    return ApprovalDecision.allow(reason=reason)


def _call(tool: ToolSpec, arguments: JsonObject) -> Any:
    return tool.fn(**validate_tool_arguments(tool, arguments))


@dataclass(frozen=True)
class _Peer:
    kind: AgentAdapterKind
    backend: JsonObject
    capabilities: AdapterCapabilities
    media_kinds: frozenset[str]
    native_tools: bool


def _negotiate(handshake: JsonObject) -> _Peer:
    """What the bridge is and what it does, from its ``initialize`` answer, or what is wrong."""

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
    kinds = handshake.get("mediaKinds", [])
    if not isinstance(kinds, list) or not set(kinds) <= MEDIA_KINDS:
        raise JsonRpcError(INVALID_PARAMS, f"mediaKinds must list kinds of {sorted(MEDIA_KINDS)}")
    if flags["supports_media_input"] and not kinds:
        raise JsonRpcError(INVALID_PARAMS, "mediaInput needs mediaKinds")
    tools = flags["supports_tool_registration"]
    governed = flags["supports_request_budget"]
    native_tools = advertised.get(NATIVE_TOOLS) is True
    capabilities = AdapterCapabilities(
        supports_structured_output=flags["supports_structured_output"],
        supports_tool_registration=tools,
        supports_artifacts=tools,
        supports_native_structured_output=flags["supports_native_structured_output"],
        supports_native_tool_calling=flags["supports_native_tool_calling"],
        supports_stream_events=flags["supports_stream_events"],
        supports_media_input=flags["supports_media_input"],
        supports_stateful_history=flags["supports_stateful_history"],
        # Kedi admits each attempt of its own tools at tool/call; the bridge's own tools only
        # count when it asks first.
        supports_request_budget=governed,
        supports_tool_attempt_budget=governed or not native_tools,
        supports_profile_override=True,
        supports_model_override=True,
        supports_effort=False,
        supports_settings=False,
        supports_mcp=False,
    )
    return _Peer(kind, backend, capabilities, frozenset(kinds), native_tools)


def _user_content(
    prompt: str, attachments: Sequence[MediaAttachment], kinds: frozenset[str], bridge: str
) -> list[JsonObject]:
    """The user turn as Kedi lays it out, each attachment where its label puts it."""

    parts: list[JsonObject] = []
    for piece in attachment_content(prompt, attachments):
        if isinstance(piece, str):
            parts.append({"type": "text", "text": piece})
            continue
        media = piece.media
        if media.kind not in kinds:
            raise MediaError(f"{piece.name}: bridge {bridge!r} does not take {media.kind} input")
        part: JsonObject = {
            "type": "media",
            "name": piece.name,
            "kind": media.kind,
            "mediaType": media.media_type,
        }
        if media.data is not None:
            part["data"] = base64.b64encode(media.data).decode("ascii")
        elif media.url is not None:
            part["url"] = media.url
        else:
            raise MediaError(f"{piece.name}: a provider file id cannot cross a JSON-RPC bridge")
        parts.append(part)
    return parts


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
