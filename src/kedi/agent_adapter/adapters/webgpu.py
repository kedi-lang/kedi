from __future__ import annotations

import asyncio
import inspect
import json
import urllib.request
from contextlib import (
    AbstractContextManager,
    AsyncExitStack,
    contextmanager,
    nullcontext,
)
from contextvars import ContextVar
from dataclasses import dataclass
from time import monotonic
from typing import Any, Callable, Collection, Generator, Mapping, Protocol, Sequence
from uuid import uuid4

from pydantic import ValidationError
from pydantic_core import to_jsonable_python
from typing_extensions import Self

from ...agent_profile import AgentProfile, McpTransport
from ...errors import KediExecutionError
from ...resolved_types import ResolvedKediType
from ..approvals.policy import ToolApprovalDenied
from ..codemode import materialize_mcp_tool_specs
from ..hooks import (
    HookDeniedError,
    HookExecutionError,
    HookHandler,
    HookRegistrationMixin,
    ToolOrigin,
    dispatch_post_tool_use,
    dispatch_post_tool_use_failure,
    dispatch_pre_tool_use,
    hooked_agent_run,
    is_tool_execution_interrupted,
)
from ..meta import AdapterCapabilities, AgentAdapterKind, HookCapabilities
from ..request import (
    SchemaOutput,
    TextOutput,
    TypedOutput,
    call_request,
    output_validator,
    tool_definition,
)
from ..request.models import OutputContract
from ..request.prompts import output_correction, protocol_correction, required_tools_correction
from ..tools.execution import with_approval
from ..tools.media import refuse_tool_media
from ..tools.scope import ToolScope
from ..tools.specs import ToolSpec, tool_approval_policy


class BrowserModelBridge(Protocol):
    """Synchronous transport used by WebGPUAdapter to reach a browser model."""

    def request(self, payload: Mapping[str, Any], *, timeout: float) -> Mapping[str, Any]: ...


class HttpBridge:
    """A bridge that posts each payload as JSON to ``url`` and returns the JSON reply."""

    def __init__(self, url: str) -> None:
        self.url = url

    def request(self, payload: Mapping[str, Any], *, timeout: float) -> Mapping[str, Any]:
        data = json.dumps(to_jsonable_python(payload)).encode()
        request = urllib.request.Request(
            self.url, data=data, headers={"content-type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())


class WebGPUInstrumentation(Protocol):
    """Optional instrumentation hooks for browser-backed adapter operations."""

    def tool_span(
        self,
        name: str,
        call_id: str,
        arguments: Mapping[str, Any],
    ) -> AbstractContextManager[Callable[[Any], None]]: ...


@dataclass(frozen=True)
class _RuntimeTool:
    spec: ToolSpec
    origin: ToolOrigin

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def description(self) -> str:
        return self.spec.description or ""

    @property
    def input_schema(self) -> dict[str, Any]:
        return self.spec.json_schema or {"type": "object", "properties": {}}

    @property
    def return_schema(self) -> dict[str, Any] | None:
        return self.spec.return_schema

    async def call(self, arguments: dict[str, Any]) -> Any:
        result = self.spec.fn(**arguments)
        if inspect.isawaitable(result):
            result = await result
        return result


WEBGPU_SETTING_KEYS = frozenset(
    {
        "max_tokens",
        "temperature",
        "top_p",
        "seed",
        "stop_sequences",
        "timeout",
    }
)


class WebGPUAdapter(HookRegistrationMixin):
    kind: AgentAdapterKind = "agent-framework"
    shortname = "webgpu"
    capabilities = AdapterCapabilities(
        supports_structured_output=True,
        supports_tool_registration=True,
        supports_profile_override=True,
        supports_model_override=True,
        supports_effort=True,
        supports_settings=True,
        supports_artifacts=True,
        supports_native_artifacts=False,
        supports_stateful_history=False,
        supports_token_usage=False,
        supports_history_processing=False,
        supports_history_replay=False,
        supports_stream_events=False,
        supports_media_input=False,
        supports_image_output=False,
        supports_audio_output=False,
        # The browser model answers in text: Kedi renders the output schema and the tool
        # catalog as protocol parts, and the bridge parses the model's JSON actions.
        supports_native_structured_output=False,
        supports_native_tool_calling=False,
        hooks=HookCapabilities.full(
            tool_bridge="kedi",
            tool_origins=frozenset({"kedi", "mcp"}),
        ),
        settings=WEBGPU_SETTING_KEYS,
    )

    """Route Kedi agent calls to a browser-hosted model through a small bridge."""

    _SETTING_KEYS = WEBGPU_SETTING_KEYS

    @classmethod
    def for_conformance(cls, base_url: str) -> Self:
        """An adapter whose bridge posts each payload to the conformance kit's endpoint."""

        return cls(HttpBridge(f"{base_url}/kedi/webgpu"), model="conformance")

    def __init__(
        self,
        bridge: BrowserModelBridge,
        *,
        model: str,
        supported_models: Sequence[str] = (),
        model_settings: Mapping[str, Any] | None = None,
        timeout: float = 300,
        max_tool_steps: int = 8,
        allowed_mcp_transports: Collection[McpTransport] | None = None,
        instrumentation: WebGPUInstrumentation | None = None,
        hook_handler: HookHandler | None = None,
    ) -> None:
        self._init_hook_registration(hook_handler)
        if max_tool_steps < 1:
            raise ValueError("max_tool_steps must be at least 1")
        self._bridge = bridge
        self._model = model
        self._supported_models = frozenset(supported_models)
        self._model_settings = dict(model_settings or {})
        self._timeout = timeout
        self._max_tool_steps = max_tool_steps
        self._allowed_mcp_transports = (
            frozenset(allowed_mcp_transports) if allowed_mcp_transports is not None else None
        )
        self._instrumentation = instrumentation
        self._tool_scope = ToolScope()
        self._profile_stack: ContextVar[tuple[AgentProfile, ...]] = ContextVar(
            f"kedi_webgpu_profile_stack_{id(self)}",
            default=(),
        )
        self._validate_model(model)

    def default_model(self) -> str:
        return self._model

    @property
    def active_profile(self) -> AgentProfile:
        stack = self._profile_stack.get()
        return stack[-1] if stack else AgentProfile.empty()

    @contextmanager
    def profile_override(self, profile: AgentProfile) -> Generator[None, None, None]:
        if profile.model is not None:
            self._validate_model(profile.model)
        token = self._profile_stack.set((*self._profile_stack.get(), profile))
        try:
            yield
        finally:
            self._profile_stack.reset(token)

    def tool_scope(self, tools: Sequence[Any] = ()) -> AbstractContextManager[None]:
        return self._tool_scope.tool_scope(tools)

    def register_tool(self, tool: ToolSpec) -> None:
        self._tool_scope.register_tool(tool)

    @property
    def scoped_tools(self) -> tuple[Any, ...]:
        return self._tool_scope.scoped_tools

    @hooked_agent_run("template")
    async def produce(
        self,
        *,
        template: str,
        output_schema: dict[str, ResolvedKediType] | None = None,
        output_type: type[Any] | None = None,
        instructions: str | None = None,
        **_: Any,
    ) -> Any:
        if output_schema:
            output: OutputContract = SchemaOutput(schema=output_schema)
        elif output_type is not None:
            output = TypedOutput(output_type=output_type)
        else:
            raise ValueError("Output spec must be provided.")
        return await self._run(prompt=template, output=output, instructions=instructions)

    @hooked_agent_run("prompt")
    async def invoke(self, *, prompt: str, instructions: str | None = None, **_: Any) -> str:
        return await self._run(prompt=prompt, output=TextOutput(), instructions=instructions)

    @hooked_agent_run("prompt")
    def invoke_sync(self, *, prompt: str, instructions: str | None = None, **kwargs: Any) -> str:
        from ..execution import run_sync

        return run_sync(self.invoke(prompt=prompt, instructions=instructions, **kwargs))

    async def _run(
        self,
        *,
        prompt: str,
        output: OutputContract,
        instructions: str | None,
    ) -> Any:
        profile = self.active_profile
        settings = self._settings(profile)
        timeout = float(settings.pop("timeout", self._timeout))

        async with AsyncExitStack() as stack:
            tools = await self._runtime_tools(stack, profile)
            request = call_request(
                prompt=prompt,
                output=output,
                instructions=instructions,
                profile=profile,
                capabilities=self.capabilities,
            )
            # Kedi renders the browser model's whole task text, MCP tools included.
            rendered = request.render(tools=tuple(tool_definition(tool.spec) for tool in tools))
            validator = output_validator(request.output, request=request)
            messages: list[dict[str, Any]] = [{"role": "user", "content": rendered.user}]
            tool_contracts = [_tool_contract(tool) for tool in tools]
            required = {tool.name for tool in rendered.tools if tool.required_before_output}
            called: set[str] = set()
            attempted: set[str] = set()
            for step in range(self._max_tool_steps):
                payload: dict[str, Any] = {
                    "model": profile.model or self._model,
                    "messages": [dict(message) for message in messages],
                    "instructions": rendered.system,
                    "effort": profile.effort,
                    "settings": settings,
                    "tools": tool_contracts,
                    "step": step,
                }
                if rendered.output_json_schema is not None:
                    payload["outputSchema"] = rendered.output_json_schema
                else:
                    payload["responseMode"] = "text"
                response = await asyncio.to_thread(
                    self._bridge.request,
                    payload,
                    timeout=timeout,
                )
                kind = response.get("kind", "final")
                if kind == "final":
                    missing = required - called
                    if missing:
                        messages.append(
                            {
                                "role": "user",
                                "content": required_tools_correction(
                                    missing, attempted=bool(missing & attempted)
                                ),
                            }
                        )
                        continue
                    if validator is None:
                        return _text_response(response)
                    data = response.get("data")
                    try:
                        return validator.validate_python(data)
                    except ValidationError as exc:
                        messages.append(_final_action(data))
                        messages.append({"role": "user", "content": output_correction(exc)})
                        continue
                if kind == "error":
                    raise RuntimeError(str(response.get("error") or "Browser model request failed"))
                if kind == "retry":
                    telemetry = response.get("telemetry")
                    if isinstance(telemetry, Mapping):
                        output_text = telemetry.get("outputText")
                        if isinstance(output_text, str) and output_text:
                            messages.append({"role": "assistant", "content": output_text})
                    detail = response.get("error")
                    messages.append(
                        {
                            "role": "user",
                            "content": protocol_correction(
                                detail if isinstance(detail, str) else None
                            ),
                        }
                    )
                    continue
                if kind != "tool_call":
                    raise ValueError(f"Unknown browser model response kind: {kind!r}")
                name, succeeded, tool_messages = await self._execute_tool_call(response, tools)
                (called if succeeded else attempted).add(name)
                messages.extend(tool_messages)

        raise RuntimeError(f"Browser model exceeded the {self._max_tool_steps}-step tool limit")

    def produce_sync(
        self,
        *,
        template: str,
        output_schema: dict[str, ResolvedKediType] | None = None,
        output_type: type[Any] | None = None,
        instructions: str | None = None,
        **kwargs: Any,
    ) -> Any:
        from ..execution import run_sync

        return run_sync(
            self.produce(
                template=template,
                output_schema=output_schema,
                output_type=output_type,
                instructions=instructions,
                **kwargs,
            )
        )

    def _validate_model(self, model: str) -> None:
        if self._supported_models and model not in self._supported_models:
            raise ValueError(f"Unknown browser model: {model!r}")

    def _settings(self, profile: AgentProfile) -> dict[str, Any]:
        settings = {**self._model_settings, **profile.settings}
        unsupported = set(settings) - self._SETTING_KEYS
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(f"Unsupported WebGPU model settings: {names}")
        return settings

    async def _runtime_tools(
        self,
        stack: AsyncExitStack,
        profile: AgentProfile,
    ) -> list[_RuntimeTool]:
        if self._allowed_mcp_transports is not None:
            disallowed = {
                spec.transport
                for spec in profile.mcp_servers
                if spec.transport not in self._allowed_mcp_transports
            }
            if disallowed:
                names = ", ".join(sorted(disallowed))
                raise ValueError(f"WebGPUAdapter does not allow {names} MCP transport")
        tools = [
            _RuntimeTool(tool, "kedi") for tool in self.scoped_tools if isinstance(tool, ToolSpec)
        ]
        # MCP tools bypass Kedi's tool wrappers, so they get the same approval boundary here.
        for spec in await materialize_mcp_tool_specs(
            stack, profile.mcp_servers, adapter_shortname=self.shortname
        ):
            policy = tool_approval_policy(spec, profile.approval)
            spec = with_approval(spec, policy=policy, adapter_shortname=self.shortname)
            tools.append(_RuntimeTool(spec, "mcp"))
        names = [tool.name for tool in tools]
        if len(names) != len(set(names)):
            raise ValueError("Tool names must be unique within one WebGPU agent run")
        return tools

    async def _execute_tool_call(
        self,
        response: Mapping[str, Any],
        tools: Sequence[_RuntimeTool],
    ) -> tuple[str, bool, list[dict[str, Any]]]:
        name = response.get("name")
        arguments = response.get("arguments", {})
        call_id = response.get("callId") or f"tool-{uuid4().hex}"
        if not isinstance(name, str):
            raise TypeError("Browser tool call name must be a string")
        if not isinstance(arguments, dict):
            raise TypeError("Browser tool call arguments must be an object")
        tool = next((candidate for candidate in tools if candidate.name == name), None)
        if tool is None:
            raise ValueError(f"Browser model requested unknown tool: {name!r}")
        spec, origin = tool.spec, tool.origin
        final_arguments = arguments
        try:
            final_arguments = await dispatch_pre_tool_use(
                spec,
                arguments,
                adapter_shortname=self.shortname,
                tool_call_id=str(call_id),
                origin=origin,
                native_tool_name=name,
            )
        except (HookDeniedError, ToolApprovalDenied) as exc:
            content = {"ok": False, "error": _tool_error(exc)}
        except HookExecutionError:
            raise
        except Exception as exc:  # argument validation is not tool execution
            content = {"ok": False, "error": _tool_error(exc)}
        else:
            started = monotonic()
            try:
                span = (
                    self._instrumentation.tool_span(name, str(call_id), final_arguments)
                    if self._instrumentation is not None
                    else nullcontext(lambda _: None)
                )
                with span as finish:
                    result = await tool.call(final_arguments)
                    refuse_tool_media(result)
                    result = to_jsonable_python(result, fallback=str)
                    finish(result)
            except BaseException as exc:
                await dispatch_post_tool_use_failure(
                    spec,
                    final_arguments,
                    exc,
                    adapter_shortname=self.shortname,
                    tool_call_id=str(call_id),
                    origin=origin,
                    native_tool_name=name,
                    duration_seconds=monotonic() - started,
                    interrupted=is_tool_execution_interrupted(exc),
                )
                if not isinstance(exc, Exception):
                    raise
                content = {"ok": False, "error": _tool_error(exc)}
            else:
                await dispatch_post_tool_use(
                    spec,
                    final_arguments,
                    result,
                    adapter_shortname=self.shortname,
                    tool_call_id=str(call_id),
                    origin=origin,
                    native_tool_name=name,
                )
                content = {"ok": True, "result": result}
        return (
            name,
            content["ok"] is True,
            [
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "action": "call_tool",
                            "callId": call_id,
                            "name": name,
                            "arguments": final_arguments,
                        },
                        ensure_ascii=False,
                    ),
                },
                {
                    "role": "tool",
                    "name": name,
                    "toolCallId": call_id,
                    "content": json.dumps(content, ensure_ascii=False),
                },
            ],
        )


def _tool_contract(tool: _RuntimeTool) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "inputSchema": tool.input_schema,
        "returnSchema": tool.return_schema,
    }


def _text_response(response: Mapping[str, Any]) -> str:
    text = response.get("text")
    if isinstance(text, str):
        return text
    data = response.get("data")
    if isinstance(data, str):
        return data
    if isinstance(data, Mapping):
        value = data.get("text")
        if isinstance(value, str):
            return value
    raise TypeError("Browser model final text response must be a string")


def _tool_error(exc: BaseException) -> str:
    # str() of a KediExecutionError renders a Kedi stack trace; the model needs the message.
    message = exc.message if isinstance(exc, KediExecutionError) else str(exc)
    return f"{type(exc).__name__}: {message}"


def _final_action(data: Any) -> dict[str, Any]:
    """The model's rejected final answer, replayed before Kedi's correction."""

    return {
        "role": "assistant",
        "content": json.dumps(
            {"action": "final", "data": to_jsonable_python(data, fallback=str)},
            ensure_ascii=False,
        ),
    }


__all__ = ["BrowserModelBridge", "HttpBridge", "WebGPUAdapter", "WebGPUInstrumentation"]
