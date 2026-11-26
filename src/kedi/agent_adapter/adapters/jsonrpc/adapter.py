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
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Any, Self, cast
from uuid import uuid4

from jsonschema.validators import validator_for
from pydantic import TypeAdapter, ValidationError
from pydantic_core import to_jsonable_python

from ....agent_profile import AgentProfile, AgentProfileOverrideUnsupported
from ....agent_profile.settings import KEDI_RUNTIME_SETTING_KEYS, speech_voice
from ....effort import EFFORT_VALUES
from ....errors import KediExecutionError
from ....media import Media, MediaError
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
from ...hooks import HookDeniedError, HookHandler, HookRegistrationMixin, hooked_agent_run
from ...hooks.dispatcher import current_hook_dispatcher
from ...hooks.models import HookEventName
from ...hooks.tool_wrapper import (
    dispatch_post_tool_use,
    dispatch_post_tool_use_failure,
    dispatch_pre_tool_use,
    with_hook_lifecycle,
)
from ...invocation import InvocationUsage, report_invocation
from ...meta import AdapterCapabilities, AgentAdapterKind, HookCapabilities, HookToolOrigin
from ...output_schema import OutputCorrection
from ...request import (
    JsonSchemaOutput,
    OutputContract,
    SchemaOutput,
    TextOutput,
    TypedOutput,
    call_request,
    output_validator,
)
from ...request.context import take_prompt_media
from ...request.media import (
    MediaAttachment,
    OutputMediaKind,
    attachment_content,
    media_validation_context,
    output_media_kind,
    tools_may_return_media,
)
from ...request.models import RenderedRequest
from ...request.prompts import output_correction, protocol_correction, required_tools_correction
from ...subagents import SubagentExecutionRequest, SubagentUsageLimits, accept_subagent_output
from ...subagents.context import current_subagent_execution
from ...subagents.control import AgentInput, LiveInput, controlled_attempt, current_live_input
from ...tools.errors import correctable_tool_error
from ...tools.execution import validate_tool_arguments
from ...tools.specs import ToolSpec
from .connection import Connection
from .protocol import (
    BUDGET_DIMENSIONS,
    CANCELLED,
    CAPABILITIES,
    HOOK_EVENTS,
    INTERRUPTED,
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
    media: OutputMediaKind | None = None
    """The media the run's model produces: ``[img: Image] <<`` or ``[speech: Audio] <<``."""
    live: LiveInput | None = None
    """The live input of the Kedi call; the run's prompt may carry some of it."""
    prompt: str = ""

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


class JsonRpcAdapter(HookRegistrationMixin):
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
        supports_subagents=None,
        supports_background_subagents=None,
        supports_live_subagent_input=None,
        supports_subagent_interrupt=None,
        supports_token_usage=None,
        supports_image_output=None,
        supports_audio_output=None,
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
                "subagents",
                "background_subagents",
                "live_subagent_input",
                "subagent_interrupt",
                "token_usage",
                "effort",
                "settings",
                "image_output",
                "audio_output",
            }
        ),
        supports_profile_override=True,
        supports_model_override=True,
        # Kedi runs the prompt hooks and its tools' hooks at tool/call; a bridge reports its own
        # tools' events with hook/invoke.
        hooks=HookCapabilities.full(tool_bridge="kedi", tool_origins=frozenset({"kedi"})),
        supports_effort=None,
        supports_settings=None,
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
        hook_handler: HookHandler | None = None,
    ) -> None:
        """Launch ``command`` and complete the handshake; raises if the bridge is not ready.

        The adapter owns the process and stops it on :meth:`close`. ``request_timeout``
        bounds each run; ``output_retries`` is how many corrections Kedi sends per call.
        """

        self.command = tuple(shlex.split(command) if isinstance(command, str) else command)
        self._init_hook_registration(hook_handler)
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
        self._child: ContextVar[SubagentExecutionRequest | None] = ContextVar(
            f"jsonrpc_child_{id(self)}", default=None
        )
        # The bridge run whose callbacks are executing: a child started there names it.
        self._calling_run: ContextVar[str | None] = ContextVar(
            f"jsonrpc_calling_run_{id(self)}", default=None
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
        self.effort_levels = peer.effort_levels
        """The ``> effort:`` levels the bridge maps to its backend."""
        self.hook_origins = peer.hook_origins
        """The sources of the bridge's own tools: it reports their calls with hook/invoke."""

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

    def telemetry_model_name(self) -> str | None:
        return self.active_profile.model or self._model

    @property
    def supports_native_subagent_resume(self) -> bool:
        """With sessions, a child's continuation carries the bridge's own state."""

        return bool(self.capabilities.supports_stateful_history)

    @property
    def active_profile(self) -> AgentProfile:
        stack = self._profiles.get()
        return stack[-1] if stack else AgentProfile.empty()

    @contextmanager
    def profile_override(self, profile: AgentProfile) -> Generator[None, None, None]:
        unsupported = [
            name
            for name, present in (
                ("effort", profile.effort is not None and profile.effort not in self.effort_levels),
                ("mcp", bool(profile.mcp_servers)),
            )
            if present
        ]
        # Kedi applies its own settings; the bridge applies the names it listed.
        unsupported += sorted(
            set(profile.settings) - KEDI_RUNTIME_SETTING_KEYS - self.capabilities.settings
        )
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
        # A Kedi call made by one of a child's tools is an ordinary call, not the child's turn.
        child = self._child.set(None)
        try:
            yield
        finally:
            self._child.reset(child)
            self._tools.reset(token)

    @contextmanager
    def subagent_scope(self, request: SubagentExecutionRequest) -> Generator[None, None, None]:
        """The child's profile, exactly its tools, and its own conversation."""

        with self.tool_scope(request.tools), self.profile_override(request.profile):
            token = self._child.set(request)
            try:
                yield
            finally:
                self._child.reset(token)

    def subagent_run_kwargs(self, limits: SubagentUsageLimits) -> Mapping[str, Any]:
        return {"limits": limits}

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

    @hooked_agent_run("prompt")
    async def invoke(
        self,
        *,
        prompt: str,
        instructions: str | None = None,
        limits: SubagentUsageLimits | None = None,
        **_: Any,
    ) -> str:
        return await self._run(prompt, TextOutput(), instructions, limits=limits)

    @hooked_agent_run("template")
    async def produce_subagent(
        self,
        *,
        template: str,
        output_schema: dict[str, Any],
        instructions: str | None = None,
        limits: SubagentUsageLimits | None = None,
        **_: Any,
    ) -> Mapping[str, Any]:
        """A child's JSON result, corrected with Kedi's feedback until the schema accepts it."""

        child = self._child.get()
        if child is None:
            raise KediExecutionError("produce_subagent runs only inside subagent_scope")
        schema_validator = validator_for(output_schema)(output_schema)

        def accept(data: Any) -> Any:
            return accept_subagent_output(data, schema_validator, child.output_state)

        output = JsonSchemaOutput(schema=output_schema)
        return await self._run(template, output, instructions, limits=limits, accept=accept)

    @hooked_agent_run("template")
    async def produce(
        self,
        *,
        template: str,
        output_schema: dict[str, Any] | None = None,
        output_type: Any = None,
        instructions: str | None = None,
        **_: Any,
    ) -> Any:
        if not (self.capabilities.supports_structured_output or self._media_output(output_type)):
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

    async def _run(
        self,
        prompt: str,
        output: OutputContract,
        instructions: str | None,
        *,
        limits: SubagentUsageLimits | None = None,
        accept: Callable[[Any], Any] | None = None,
    ) -> Any:
        disconnected = self._connection.disconnected
        if disconnected is not None:
            raise disconnected
        media = self._media_output(output.output_type if isinstance(output, TypedOutput) else None)
        # A speech request sends text only: the model speaks, and calls no tools.
        tools = () if media == "audio" else self._tools.get()
        request = call_request(
            prompt=prompt,
            output=output,
            instructions=instructions,
            profile=self.active_profile,
            capabilities=self.capabilities,
            tools=tools,
        )
        rendered = request.render()
        child = self._child.get()
        # A child keeps its own conversation: the parent's history is not the child's.
        conversation = None if child is not None else self._conversation.get()
        # The first native run in a Kedi call takes the prompt's media; nested runs get none.
        attachments = take_prompt_media()
        user_content = (
            _user_content(rendered.user, attachments, self.media_kinds, self.backend["name"])
            if attachments
            else None
        )
        # A child's result is checked against its JSON Schema by Kedi's subagent validator.
        validator = (
            None if accept is not None else output_validator(request.output, request=request)
        )
        run_id = uuid4().hex
        parent_run_id = self._calling_run.get()
        calling = self._calling_run.set(run_id)
        try:
            context = contextvars.copy_context()
        finally:
            self._calling_run.reset(calling)
        run = _Run(
            run_id=run_id,
            loop=asyncio.get_running_loop(),
            context=context,
            tools={tool.name: tool for tool in tools},
            live=current_live_input(),
            prompt=rendered.user,
            media=media,
        )
        required = {tool.name for tool in rendered.tools if tool.required_before_output}
        params: JsonObject = {
            "runId": run_id,
            "callKind": rendered.call_kind,
            "promptKind": rendered.prompt_kind,
            "model": self.active_profile.model or self._model,
            "fallbackModels": list(self.active_profile.fallback_models or ()),
            "settings": {
                name: to_jsonable_python(value, fallback=str)
                for name, value in self.active_profile.settings.items()
                if name in self.capabilities.settings
            },
            "effort": self.active_profile.effort,
            "system": rendered.system,
            "instructionParts": [
                {"identity": part.identity, "source": part.source, "content": part.content}
                for part in rendered.instruction_parts
            ],
            "user": rendered.user,
            "userContent": user_content,
            "output": self._output(rendered, media, attachments, tools),
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
            "session": None,
            "subagent": None if child is None else _lineage(child, parent_run_id, limits),
            "hooks": self._hook_events(),
        }
        commit: Callable[[Any], None] | None = None
        if child is not None and self.capabilities.supports_stateful_history:
            # A continuation resumes from the bridge state of the child's last committed run.
            history = child.conversation
            params["session"] = {
                "id": history.conversation_id,
                "turn": len(history.turns),
                "state": history.resume_state.get("bridge"),
            }
            commit = partial(history.result_state.__setitem__, "bridge")
        elif conversation is not None:
            params["session"] = {
                "id": conversation.session_id,
                "turn": conversation.turn_sequence,
                "state": conversation.resume_state.get("bridge"),
            }
            commit = partial(conversation.result_state.__setitem__, "bridge")
        self._runs[run_id] = run
        try:
            with agent_event_run(self.shortname) as run.events:
                attempt = self._attempts(run, params, validator, required, accept, commit)
                control = current_live_input()
                if control is None or not self.capabilities.supports_live_subagent_input:
                    answer = await attempt
                else:
                    # Input for this run goes to the bridge while it runs; an interrupt stops
                    # the attempt, and Kedi's coordinator starts the next one.
                    interrupt = self.capabilities.supports_subagent_interrupt
                    answer = await controlled_attempt(
                        control,
                        attempt,
                        deliver=partial(self._deliver, run_id),
                        stop=partial(self._interrupt, run_id) if interrupt else None,
                    )
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
        accept: Callable[[Any], Any] | None,
        commit: Callable[[Any], None] | None,
    ) -> Any:
        """Ask the bridge, then send Kedi's corrections, until an answer is valid."""

        correction: str | None = None
        for _ in range(self._output_retries + 1):
            try:
                result = await self._request_run(params)
            except JsonRpcError as error:
                # An interrupted attempt keeps the conversation so far for the next one.
                data = error.data
                if (
                    error.code == INTERRUPTED
                    and commit is not None
                    and isinstance(data, dict)
                    and "state" in data
                ):
                    commit(data["state"])
                raise
            finally:
                if run.failure is not None:
                    raise run.failure
            answer, correction = _check_answer(result, validator, required, run, accept)
            if correction is None:
                if commit is not None and "state" in result:
                    # Kedi commits it only after accepting this turn's result.
                    commit(result["state"])
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
        if not self.capabilities.supports_token_usage:
            # A governed bridge's requests were counted as Kedi admitted them; Kedi counts
            # tool calls. With usage, each request's counts arrived in run/usage.
            governed = bool(self.capabilities.supports_request_budget)
            report_invocation(_usage(result.get("usage"), requests_counted=governed))
        return result

    def _hook_events(self) -> list[str]:
        """The tool events with handlers: the bridge sends hook/invoke for these only."""

        dispatcher = current_hook_dispatcher()
        if not self.hook_origins or dispatcher is None:
            return []
        return sorted(
            event for event in HOOK_EVENTS if dispatcher.handlers_for(cast("HookEventName", event))
        )

    def _media_output(self, output_type: Any) -> OutputMediaKind | None:
        """The media the bridge's model produces for ``output_type``, when it can."""

        kind = output_media_kind(output_type)
        produces = {
            "image": self.capabilities.supports_image_output,
            "audio": self.capabilities.supports_audio_output,
        }
        return kind if kind is not None and produces[kind] else None

    def _output(
        self,
        rendered: RenderedRequest,
        media: OutputMediaKind | None,
        attachments: Sequence[MediaAttachment],
        tools: Sequence[ToolSpec],
    ) -> JsonObject | None:
        """The run's ``output``: text, data, or the media its model produces."""

        data = (
            None
            if rendered.output_json_schema is None
            else {"schema": rendered.output_json_schema, "toolName": rendered.output_tool_name}
        )
        if media == "audio":
            return {"media": "audio", "voice": speech_voice(self.active_profile.settings)}
        if media == "image":
            # The model draws, or names an image the request or a tool result holds.
            may_name = tools_may_return_media(tools) or any(
                attachment.media.kind == "image" for attachment in attachments
            )
            return {"media": "image", **(data if may_name and data else {})}
        return data

    async def _deliver(self, run_id: str, items: tuple[AgentInput, ...]) -> bool:
        """Hand live input to the running bridge run; true once its model has it."""

        payload = [{"id": item.message_id, "text": item.render()} for item in items]
        try:
            reply = await self._connection.request("run/input", {"runId": run_id, "items": payload})
        except JsonRpcError:
            return False  # the run ended or the bridge refused: the next attempt gets them
        return reply.get("delivered") is True

    async def _interrupt(self, run_id: str) -> None:
        """Ask the bridge to stop the attempt; the run's INTERRUPTED answer confirms it."""

        self._connection.notify("run/interrupt", {"runId": run_id})

    def _on_request(self, request_id: int | str, method: str, params: JsonObject) -> None:
        """A bridge request, on the reader thread: start it as a task on the run's loop.

        The task runs in the run's context, so Kedi's run budget, approval policies and
        observers are the ones of the Kedi call that started the run.
        """

        parse = {
            "tool/call": self._parse_tool_call,
            "budget/admit": self._parse_admission,
            "approval/request": self._parse_approval,
            "hook/invoke": self._parse_hook,
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
        call_id = params.get("callId")
        call_id = call_id if isinstance(call_id, str) else uuid4().hex
        return lambda request_id, run: self._call_tool(request_id, run, name, arguments, call_id)

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

    def _parse_hook(self, params: JsonObject) -> _Handler:
        event, origin, tool = params.get("event"), params.get("origin"), params.get("tool")
        call_id, arguments = params.get("callId"), params.get("arguments")
        if origin not in self.hook_origins:
            raise JsonRpcError(
                UNSUPPORTED,
                f"hook/invoke for {origin!r} tools: the bridge runs no such tools "
                f"(it reports {', '.join(sorted(self.hook_origins)) or 'none'})",
            )
        if (
            event not in HOOK_EVENTS
            or not isinstance(tool, str)
            or not isinstance(call_id, str)
            or not isinstance(arguments, dict)
        ):
            raise JsonRpcError(
                INVALID_PARAMS,
                f"hook/invoke needs an event of {', '.join(sorted(HOOK_EVENTS))}, a string tool "
                "and callId, and object arguments",
            )
        spec = ToolSpec(
            fn=_bridge_tool,
            name=tool,
            json_schema=params.get("inputSchema")
            if isinstance(params.get("inputSchema"), dict)
            else None,
            metadata={"source": origin, "backend": self.backend["name"]},
        )
        call = _HookCall(
            event=cast(str, event),
            spec=spec,
            arguments=arguments,
            call_id=call_id,
            origin=cast("HookToolOrigin", origin),
            result=params.get("result"),
            error=params.get("error"),
            interrupted=params.get("interrupted") is True,
        )
        return lambda request_id, run: self._hook(request_id, run, call)

    async def _hook(self, request_id: int | str, run: _Run, call: _HookCall) -> None:
        """Run the program's hooks for a call of the bridge's own tool, as Kedi's tools get."""

        fields: dict[str, Any] = {
            "adapter_shortname": self.shortname,
            "tool_call_id": call.call_id,
            "origin": call.origin,
            "native_tool_name": call.spec.name,
        }
        try:
            reply: JsonObject = {}
            if call.event == "pre_tool_use":
                arguments = await dispatch_pre_tool_use(call.spec, call.arguments, **fields)
                reply = {"decision": "continue"}
                if arguments != call.arguments:
                    reply = {"decision": "edit", "arguments": to_jsonable_python(arguments)}
            elif call.event == "post_tool_use":
                await dispatch_post_tool_use(call.spec, call.arguments, call.result, **fields)
            else:
                error = call.error if isinstance(call.error, dict) else {}
                await dispatch_post_tool_use_failure(
                    call.spec,
                    call.arguments,
                    _bridge_error(error),
                    interrupted=call.interrupted,
                    **fields,
                )
        except HookDeniedError as exc:
            reply = {"decision": "deny", "reason": exc.message}
        except asyncio.CancelledError:
            self._connection.respond(request_id, error=JsonRpcError(CANCELLED, "the run ended"))
            raise
        except Exception as exc:  # a failing or timed-out hook is a failing control: stop
            self._stop(request_id, run, exc)
            return
        self._connection.respond(request_id, result=reply)

    def _on_notification(self, method: str, params: JsonObject) -> None:
        """A bridge notification, on the reader thread: a run event, in stream order."""

        if method == "run/usage" and self.capabilities.supports_token_usage:
            run_id = params.get("runId")
            run = self._runs.get(run_id) if isinstance(run_id, str) else None
            if run is not None:
                governed = bool(self.capabilities.supports_request_budget)
                self._schedule_usage(run, _request_usage(params, requests_counted=governed))
            return
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
        self, request_id: int | str, run: _Run, name: str, arguments: JsonObject, call_id: str
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
            # Kedi counts the call before it runs, so a child's tool-call limit can stop it.
            report_invocation(InvocationUsage(tool_calls=1))
        except KediExecutionError as exc:
            self._stop(request_id, run, exc)
            return
        # The program's tool hooks run around the call, before Kedi's approval and budgets.
        hooked = with_hook_lifecycle(tool, adapter_shortname=self.shortname, tool_call_id=call_id)
        try:
            # Sync tools run off the loop; Kedi's wrapper applies approval and budgets.
            value = await asyncio.to_thread(_call, hooked, arguments)
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
        """Admit a step the bridge is about to take, as in-process adapters do.

        The run budget charges it, then Kedi counts it, so a child's request or tool-call
        limit stops it before the bridge takes it.
        """

        denial: JsonObject = {"admitted": False, "dimension": dimension}
        try:
            admit_run_operation(cast(BudgetDimension, BUDGET_DIMENSIONS[dimension]))
            step = InvocationUsage(requests=1) if dimension == "request" else None
            report_invocation(step or InvocationUsage(tool_calls=1))
            if step is not None and run.live is not None:
                # The model request about to go out carries the prompt, and any live input
                # Kedi put in it: that input has reached the model.
                run.live.confirm_contents((run.prompt,))
        except RunBudgetExceeded as exc:
            denial |= {"limit": exc.limit, "consumed": exc.consumed, "scopeDepth": exc.scope_depth}
            failure: KediExecutionError = exc
        except KediExecutionError as exc:  # a child's usage limit
            failure = exc
        else:
            self._connection.respond(request_id, result={"admitted": True})
            return
        run.stop(failure)
        self._connection.respond(request_id, result={**denial, "reason": failure.message})
        # The bridge must skip the step and end the run; cancel it in case it does not.
        self._connection.notify("run/cancel", {"runId": run.run_id})

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
        """A child's ancestors' policies first, then its own, as for Kedi tools."""

        child = self._child.get()
        ceiling = () if child is None else tuple(child.approval_ceiling)
        local = self.active_profile.approval
        return (*ceiling, local) if local is not None or not ceiling else ceiling

    def _schedule_usage(self, run: _Run, usage: InvocationUsage) -> None:
        # After the run's earlier callbacks and before its answer: same queue, in order.
        run.loop.call_soon_threadsafe(lambda: run.context.run(self._report_usage, run, usage))

    def _report_usage(self, run: _Run, usage: InvocationUsage) -> None:
        """One model request's counts; a child's token limit that they break ends the run."""

        try:
            report_invocation(usage)
        except KediExecutionError as exc:
            run.stop(exc)
            self._connection.notify("run/cancel", {"runId": run.run_id})

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
    reason = "; ".join(dict.fromkeys(reasons)) or None
    if arguments != dict(request.arguments):
        return ApprovalDecision.edit(arguments, reason=reason)
    return ApprovalDecision.allow(reason=reason)


@dataclass(frozen=True)
class _HookCall:
    """One ``hook/invoke`` for a call of the bridge's own tool."""

    event: str
    spec: ToolSpec
    arguments: JsonObject
    call_id: str
    origin: HookToolOrigin
    result: Any = None
    error: Any = None
    interrupted: bool = False


def _bridge_error(error: JsonObject) -> Exception:
    """A bridge tool's failure as an exception of the type the bridge named.

    ``post_tool_use_failure`` reports the error's type name and message.
    """

    name = error.get("type")
    kind = type(name if isinstance(name, str) and name else "Error", (Exception,), {})
    return kind(str(error.get("message") or ""))


def _bridge_tool(**_: Any) -> Any:
    """A bridge's own tool runs in the bridge; Kedi only describes it for hooks."""

    raise RuntimeError("a bridge tool runs in the bridge")


def _lineage(
    child: SubagentExecutionRequest, parent_run_id: str | None, limits: SubagentUsageLimits | None
) -> JsonObject:
    """Who the child is: Kedi's run, the bridge run that started it, its remaining limits."""

    execution = current_subagent_execution()
    remaining = child.usage_limits if limits is None else limits
    return {
        "id": None if execution is None else execution.run_id,
        "parentRunId": parent_run_id,
        "conversationId": child.conversation.conversation_id,
        "name": child.profile.name,
        "limits": {
            "requests": remaining.request_limit,
            "toolCalls": remaining.tool_calls_limit,
            "inputTokens": remaining.input_tokens_limit,
            "outputTokens": remaining.output_tokens_limit,
            "totalTokens": remaining.total_tokens_limit,
        },
    }


def _request_usage(params: JsonObject, *, requests_counted: bool) -> InvocationUsage:
    """One model request's counts from ``run/usage``; a count left out is unknown."""

    input_tokens, output_tokens = _count(params, "inputTokens"), _count(params, "outputTokens")
    total = _count(params, "totalTokens")
    if total is None and input_tokens is not None and output_tokens is not None:
        total = input_tokens + output_tokens
    return InvocationUsage(
        requests=0 if requests_counted else 1,
        input_tokens=input_tokens,
        cache_read_tokens=_count(params, "cacheReadTokens"),
        cache_write_tokens=_count(params, "cacheWriteTokens"),
        output_tokens=output_tokens,
        total_tokens=total,
    )


def _usage(usage: Any, *, requests_counted: bool = False) -> InvocationUsage:
    """The run's usage as the bridge reported it; a count it left out is unknown.

    ``requests_counted``: Kedi already counted each request when it admitted it.
    """

    if not isinstance(usage, dict):
        return InvocationUsage.unmeasured(requests=0 if requests_counted else 1)
    requests = 0 if requests_counted else _count(usage, "requests")
    input_tokens, output_tokens = _count(usage, "inputTokens"), _count(usage, "outputTokens")
    total = _count(usage, "totalTokens")
    if total is None and input_tokens is not None and output_tokens is not None:
        total = input_tokens + output_tokens
    return InvocationUsage(
        requests=1 if requests is None else requests,
        input_tokens=input_tokens,
        cache_read_tokens=None,
        cache_write_tokens=None,
        output_tokens=output_tokens,
        total_tokens=total,
    )


def _count(usage: JsonObject, name: str) -> int | None:
    value = usage.get(name)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _call(tool: ToolSpec, arguments: JsonObject) -> Any:
    return tool.fn(**validate_tool_arguments(tool, arguments))


@dataclass(frozen=True)
class _Peer:
    kind: AgentAdapterKind
    backend: JsonObject
    capabilities: AdapterCapabilities
    media_kinds: frozenset[str]
    native_tools: bool
    effort_levels: frozenset[str]
    hook_origins: frozenset[HookToolOrigin]


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
    settings, effort = handshake.get("settings", []), handshake.get("effort", [])
    if not isinstance(settings, list) or not all(isinstance(name, str) for name in settings):
        raise JsonRpcError(INVALID_PARAMS, "settings must list setting names")
    if own := sorted(set(settings) & KEDI_RUNTIME_SETTING_KEYS):
        raise JsonRpcError(INVALID_PARAMS, f"settings lists Kedi's own settings: {', '.join(own)}")
    if not isinstance(effort, list) or not set(effort) <= EFFORT_VALUES:
        raise JsonRpcError(INVALID_PARAMS, f"effort must list levels of {sorted(EFFORT_VALUES)}")
    tools = flags["supports_tool_registration"]
    governed = flags["supports_request_budget"]
    native_tools = advertised.get(NATIVE_TOOLS) is True
    # A bridge that runs tools of its own reports their calls to Kedi's hooks.
    origins: frozenset[HookToolOrigin] = (
        frozenset({"adapter_native", "provider_builtin"}) if native_tools else frozenset()
    )
    subagents = flags["supports_subagents"]
    live_input, interrupt = (
        flags["supports_live_subagent_input"],
        flags["supports_subagent_interrupt"],
    )
    for unmet, problem in (
        (subagents and not governed, "subagents need governed: Kedi admits each child request"),
        (live_input and not subagents, "liveInput needs subagents"),
        (interrupt and not live_input, "interrupt needs liveInput"),
    ):
        if unmet:
            raise JsonRpcError(INVALID_PARAMS, problem)
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
        supports_token_usage=flags["supports_token_usage"],
        supports_subagents=subagents,
        # Kedi runs a background child as another run on the same connection.
        supports_background_subagents=subagents,
        supports_live_subagent_input=live_input,
        supports_subagent_interrupt=interrupt,
        supports_profile_override=True,
        supports_model_override=True,
        supports_effort=bool(effort),
        supports_settings=bool(settings),
        settings=frozenset(settings),
        supports_model_fallback=flags["supports_model_fallback"],
        supports_image_output=flags["supports_image_output"],
        supports_audio_output=flags["supports_audio_output"],
        hooks=HookCapabilities.full(
            tool_bridge="native" if origins else "kedi",
            tool_origins=frozenset({"kedi", *origins}),
        ),
        supports_mcp=False,
    )
    return _Peer(
        kind,
        backend,
        capabilities,
        frozenset(kinds),
        native_tools,
        frozenset(effort),
        origins,
    )


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
    result: JsonObject,
    validator: TypeAdapter[Any] | None,
    required: set[str],
    run: _Run,
    accept: Callable[[Any], Any] | None = None,
) -> tuple[Any, str | None]:
    """The validated answer, or Kedi's correction for the next turn."""

    if "invalid" in result:
        # The model's answer was not one protocol action; the bridge quotes its parse error.
        return None, protocol_correction(str(result.get("error") or "") or None)
    missing = required - run.called
    if missing:
        return None, required_tools_correction(missing, attempted=bool(missing & run.attempted))
    if run.media is not None and "content" in result:
        return _produced(result["content"], run.media), None
    if validator is None and accept is None:
        text = result.get("text")
        if not isinstance(text, str):
            raise JsonRpcError(INVALID_PARAMS, "a text run must answer with a string 'text'")
        return text, None
    if "data" not in result:
        raise JsonRpcError(INVALID_PARAMS, "a structured run must answer with 'data'")
    if accept is not None:
        try:
            return accept(result["data"]), None
        except OutputCorrection as correction:
            return None, str(correction)
    assert validator is not None
    try:
        # A media-typed value names an image or audio of the request (MEDIA_NAMES).
        return validator.validate_python(result["data"], context=media_validation_context()), None
    except ValidationError as exc:
        return None, output_correction(exc)


def _produced(content: Any, kind: OutputMediaKind) -> Media:
    """The one image or audio among a run's ``content`` parts, as Kedi media."""

    from ....media import Audio, Image

    if not isinstance(content, list):
        raise JsonRpcError(INVALID_PARAMS, "'content' must list text and media parts")
    parts = [
        part
        for part in content
        if isinstance(part, dict) and part.get("type") == "media" and part.get("kind") == kind
    ]
    if len(parts) != 1:
        raise JsonRpcError(
            INVALID_PARAMS, f"the run asked for one {kind}; its 'content' holds {len(parts)}"
        )
    (part,) = parts
    media_type = part.get("mediaType")
    cls = Image if kind == "image" else Audio
    if isinstance(part.get("url"), str):
        return cls.from_url(part["url"], media_type=media_type)
    try:
        data = base64.b64decode(part["data"], validate=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise JsonRpcError(INVALID_PARAMS, "a media part needs base64 'data' or a 'url'") from exc
    return cls.from_bytes(data, media_type)


def reference_bridge_command(
    base_url: str,
    *,
    mode: str = "native",
    image_output: bool = False,
    audio_output: bool = False,
) -> list[str]:
    """Run the reference bridge with Node.js against an OpenAI-compatible ``base_url``.

    ``mode="protocol"`` treats the endpoint as having no native structured output or tool
    calling, so Kedi sends its protocol parts. ``image_output`` and ``audio_output`` say the
    endpoint's models draw (OpenRouter's image output) or speak (OpenAI's chat audio).
    """

    flags = [
        flag
        for flag, enabled in (("--image-output", image_output), ("--audio-output", audio_output))
        if enabled
    ]
    return ["node", str(REFERENCE_BRIDGE), "--base-url", base_url, "--mode", mode, *flags]


__all__ = ["REFERENCE_BRIDGE", "JsonRpcAdapter", "JsonRpcDisconnected", "reference_bridge_command"]
