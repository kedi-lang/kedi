from __future__ import annotations

import inspect
import json
from contextlib import AbstractContextManager, asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import replace
from functools import wraps
from time import monotonic
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    Iterator,
    Mapping,
    NoReturn,
    Sequence,
    cast,
)

from pydantic import BaseModel
from pydantic_ai.output import OutputDataT
from pydantic_ai import UsageLimits

from ...agent_profile import AgentProfile, CodeModeSettings, normalize_codemode_settings
from ...agent_profile.compaction import (
    CompactionSettingsError,
)
from ...agent_profile.settings import (
    LANGCHAIN_MODEL_SETTING_KEYS,
    langchain_model_settings,
)
from ...effort import pydantic_effort_value
from ...model_normalization import normalize_for_langchain
from ...resolved_types import ResolvedKediType
from .._schema import build_output_model
from ..approval import ApprovalPolicy, ToolApprovalDenied, ToolRisk
from ..codemode import CodeModeError, CodeModeLimits
from ..codemode.tool_specs import (
    CodeModeProjection,
    append_codemode_instructions,
    approval_policies,
    codemode_projection,
)
from ..conversation import AdapterConversationContext
from ..compaction.archive import HistoryArchiveSettings
from ..events import AgentMessageResolver, agent_event_run
from ..hooks import (
    HookDeniedError,
    HookHandler,
    HookRegistrationMixin,
    ToolOrigin,
    current_hook_dispatcher,
    dispatch_post_tool_use,
    dispatch_post_tool_use_failure,
    dispatch_pre_tool_use,
    hooked_agent_run,
    is_tool_execution_interrupted,
    new_tool_call_id,
)
from ..invocation import InvocationUsage, invocation_observer, report_invocation
from ..meta import AdapterCapabilities, AgentAdapterKind, HookCapabilities
from ..native_compaction import langchain_settings_with_native_compaction
from ..prefix_cache import langchain_cache_settings, langchain_provider_cache_policy
from ..responses_session import model_responses_session, model_responses_turn
from ..subagents import SubagentExecutionRequest
from ..subagents.attempts import DelegationFailurePolicy, delegation_scope
from ..subagents.limits import SubagentUsageLimits
from ..telemetry import mcp_initialization_span
from ..tool_errors import correctable_tool_error
from ..tool_override import (
    ToolScope,
    ToolSpec,
    admit_current_artifact_result,
    arguments_target_secret_file,
    current_artifact_admission,
    resolve_tool_arguments_async,
    tool_has_artifact_boundary,
)


class LangChainAdapter(HookRegistrationMixin):
    """Adapter to use LangChain agents through Kedi's uniform adapter contract."""

    kind: AgentAdapterKind = "agent-framework"
    shortname = "langchain"
    capabilities = AdapterCapabilities(
        supports_structured_output=True,
        supports_tool_registration=True,
        supports_mcp=True,
        supports_profile_override=True,
        supports_model_override=True,
        supports_effort=True,
        supports_settings=True,
        supports_codemode=True,
        supports_native_approvals=True,
        supports_native_approval_handler=True,
        supports_subagents=True,
        supports_background_subagents=True,
        supports_artifacts=True,
        supports_native_artifacts=True,
        supports_stateful_history=True,
        supports_history_replay=True,
        supports_native_compaction=True,
        supports_stream_events=True,
        hooks=HookCapabilities.full(
            tool_bridge="native",
            tool_origins=frozenset({"kedi", "mcp", "adapter_native"}),
        ),
        settings=LANGCHAIN_MODEL_SETTING_KEYS,
    )

    def __init__(
        self,
        model: Any = None,
        *,
        model_settings: dict[str, Any] | None = None,
        subagent_failure_policy: DelegationFailurePolicy = "fail_closed",
        codemode: bool | CodeModeSettings = False,
        hook_handler: HookHandler | None = None,
        default_usage_limits: UsageLimits | None = None,
        history_archive: HistoryArchiveSettings | None = None,
        tool_retries: int = 3,
        **kwargs: Any,
    ) -> None:
        self._init_hook_registration(hook_handler)
        if isinstance(tool_retries, bool) or not isinstance(tool_retries, int) or tool_retries < 0:
            raise ValueError("tool_retries must be a non-negative integer")
        self._default_usage_limits = default_usage_limits
        self._history_archive = history_archive
        self._tool_retries = tool_retries
        if subagent_failure_policy not in {"recover", "fail_closed"}:
            raise ValueError(f"Unknown subagent failure policy: {subagent_failure_policy!r}")
        self._model = normalize_for_langchain(model) if isinstance(model, str) else model
        self._model_settings = dict(model_settings or {})
        self._model_extra_kwargs = dict(kwargs)
        self._subagent_failure_policy = subagent_failure_policy
        self.codemode = normalize_codemode_settings(codemode)
        self._tool_scope = ToolScope()
        self._tool_spec_scope = ToolScope()
        self._profile_stack_var: ContextVar[tuple[AgentProfile, ...]] = ContextVar(
            f"kedi_langchain_profile_stack_{id(self)}",
            default=(),
        )
        self._approval_ceiling_var: ContextVar[tuple[ApprovalPolicy | None, ...]] = ContextVar(
            f"kedi_langchain_approval_ceiling_{id(self)}",
            default=(),
        )
        self._conversation_var: ContextVar[AdapterConversationContext | None] = ContextVar(
            f"kedi_langchain_conversation_{id(self)}",
            default=None,
        )

    def tool_scope(self, tools: Sequence[Any] = ()) -> AbstractContextManager[None]:
        @contextmanager
        def scoped() -> Iterator[None]:
            with self._tool_scope.tool_scope(), self._tool_spec_scope.tool_scope():
                for tool in tools:
                    if isinstance(tool, ToolSpec):
                        self.register_tool(tool)
                    else:
                        self._tool_scope.register_tool(tool)
                yield

        return scoped()

    def register_tool(self, tool: ToolSpec) -> None:
        self._tool_scope.register_tool(_langchain_tool_from_spec(tool))
        self._tool_spec_scope.register_tool(tool)

    @property
    def scoped_tools(self) -> tuple[Any, ...]:
        return self._tool_scope.scoped_tools

    @property
    def scoped_tool_specs(self) -> tuple[ToolSpec, ...]:
        return cast(tuple[ToolSpec, ...], self._tool_spec_scope.scoped_tools)

    @property
    def active_profile(self) -> AgentProfile:
        stack = self._profile_stack_var.get()
        return stack[-1] if stack else AgentProfile.empty()

    @staticmethod
    def type_builder(output_schema: dict[str, ResolvedKediType]) -> type[BaseModel]:
        return build_output_model(output_schema)

    @hooked_agent_run("template")
    async def produce(
        self,
        template: str,
        output_schema: dict[str, ResolvedKediType] | None = None,
        output_type: type[OutputDataT] | None = None,
        **kwargs: Any,
    ) -> OutputDataT:
        resolved_output_type = output_type
        if output_schema:
            resolved_output_type = cast(type[OutputDataT], self.type_builder(output_schema))
        if resolved_output_type is None:
            raise ValueError("Output spec must be provided.")

        instructions = cast(str | None, kwargs.pop("instructions", None))
        if instructions is None:
            instructions = self.active_profile.system
        model = self._chat_model_for_run()
        async with self._projected_tools_for_run() as projection:
            system_prompt = self._cache_aware_system_prompt(
                append_codemode_instructions(instructions, enabled=projection.enabled),
                model=model,
            )
            tools = _langchain_projection_tools(projection)
            agent = self._create_agent(
                model=model,
                tools=tools,
                response_format=resolved_output_type,
                system_prompt=system_prompt,
                middleware=self._middleware_for_run(
                    model,
                    tools,
                    codemode_enabled=projection.enabled,
                ),
            )
            payload = {"messages": self._messages_for_prompt(template, model=model), **kwargs}
            async with model_responses_turn(model):
                result = await self._invoke_agent(agent, payload, emit_final=False)
            return cast(OutputDataT, _langchain_structured_response(result))

    @hooked_agent_run("template")
    async def produce_subagent(
        self,
        *,
        template: str,
        output_schema: dict[str, Any],
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        from langchain.agents.structured_output import ToolStrategy

        instructions = cast(str | None, kwargs.pop("instructions", None))
        if instructions is None:
            instructions = self.active_profile.system
        model = self._chat_model_for_run()
        async with self._projected_tools_for_run() as projection:
            system_prompt = self._cache_aware_system_prompt(
                append_codemode_instructions(instructions, enabled=projection.enabled),
                model=model,
            )
            tools = _langchain_projection_tools(projection)
            agent = self._create_agent(
                model=model,
                tools=tools,
                response_format=ToolStrategy(output_schema),
                system_prompt=system_prompt,
                middleware=self._middleware_for_run(
                    model,
                    tools,
                    codemode_enabled=projection.enabled,
                ),
            )
            payload = {"messages": self._messages_for_prompt(template, model=model), **kwargs}
            async with model_responses_turn(model):
                result = _langchain_structured_response(
                    await self._invoke_agent(agent, payload, emit_final=False)
                )
            if not isinstance(result, Mapping):
                raise TypeError("LangChain subagent structured output must be an object")
            return result

    @hooked_agent_run("prompt")
    async def invoke(self, *, prompt: str, **kwargs: Any) -> str:
        instructions = cast(str | None, kwargs.pop("instructions", None))
        model = self._chat_model_for_run()
        async with self._projected_tools_for_run() as projection:
            system_prompt = self._cache_aware_system_prompt(
                append_codemode_instructions(
                    instructions if instructions is not None else self.active_profile.system,
                    enabled=projection.enabled,
                ),
                model=model,
            )
            tools = _langchain_projection_tools(projection)
            agent = self._create_agent(
                model=model,
                tools=tools,
                response_format=None,
                system_prompt=system_prompt,
                middleware=self._middleware_for_run(
                    model,
                    tools,
                    codemode_enabled=projection.enabled,
                ),
            )
            payload = {"messages": self._messages_for_prompt(prompt, model=model), **kwargs}
            async with model_responses_turn(model):
                return _langchain_text_response(
                    await self._invoke_agent(agent, payload, emit_final=True)
                )

    @asynccontextmanager
    async def responses_session(self) -> AsyncIterator[None]:
        """Share a supported model transport across sequential adapter calls."""
        model = self._model_for_run()
        if not callable(getattr(model, "responses_session", None)):
            raise TypeError("The selected model does not support Responses sessions")
        async with model_responses_session(model):
            yield

    def invoke_sync(self, *, prompt: str, **kwargs: Any) -> str:
        from ..utils import run_sync

        return run_sync(self.invoke(prompt=prompt, **kwargs))

    def produce_sync(
        self,
        template: str,
        output_schema: dict[str, ResolvedKediType] | None = None,
        output_type: type[OutputDataT] | None = None,
        **kwargs: Any,
    ) -> OutputDataT:
        from ..utils import run_sync

        return run_sync(
            self.produce(template, output_schema=output_schema, output_type=output_type, **kwargs)
        )

    def default_model(self) -> str | None:
        model = self._model
        return model if isinstance(model, str) else None

    def telemetry_model_name(self) -> str | None:
        return self.default_model()

    @contextmanager
    def profile_override(self, profile: AgentProfile) -> Iterator[None]:
        stack_var = self._profile_stack_var
        token = stack_var.set((*stack_var.get(), profile))
        try:
            yield
        finally:
            stack_var.reset(token)

    @contextmanager
    def conversation_scope(
        self,
        conversation: AdapterConversationContext,
    ) -> Iterator[None]:
        token = self._conversation_var.set(conversation)
        try:
            yield
        finally:
            self._conversation_var.reset(token)

    @contextmanager
    def approval_ceiling(
        self,
        policies: Sequence[ApprovalPolicy | None],
    ) -> Iterator[None]:
        token = self._approval_ceiling_var.set(tuple(policies))
        try:
            yield
        finally:
            self._approval_ceiling_var.reset(token)

    @contextmanager
    def subagent_scope(self, request: SubagentExecutionRequest) -> Iterator[None]:
        with (
            self.tool_scope(request.tools),
            self.profile_override(request.profile),
            self.approval_ceiling(request.approval_ceiling),
        ):
            yield

    @staticmethod
    def subagent_run_kwargs(limits: SubagentUsageLimits) -> dict[str, Any]:
        return {
            "usage_limits": UsageLimits(
                request_limit=limits.request_limit,
                tool_calls_limit=limits.tool_calls_limit,
                input_tokens_limit=limits.input_tokens_limit,
                output_tokens_limit=limits.output_tokens_limit,
                total_tokens_limit=limits.total_tokens_limit,
            )
        }

    async def _invoke_agent(
        self,
        agent: Any,
        payload: Mapping[str, Any],
        *,
        emit_final: bool,
    ) -> Any:
        from .langchain_runtime import langchain_run_scope

        payload = dict(payload)
        limits = payload.pop("usage_limits", self._default_usage_limits)
        config = dict(payload.pop("config", None) or {})
        recursion_limit = payload.pop("recursion_limit", None)
        if recursion_limit is not None:
            config["recursion_limit"] = recursion_limit
        elif "recursion_limit" not in config:
            request_limit = limits.request_limit if limits is not None else 50
            if request_limit is not None:
                config["recursion_limit"] = request_limit * 3 + 5
        with (
            langchain_run_scope(limits) as run_state,
            delegation_scope(self._subagent_failure_policy),
            agent_event_run(self.shortname) as resolver,
        ):
            if resolver is not None:
                callbacks = config.get("callbacks")
                callback = _langchain_event_handler(resolver)
                if callbacks is None or isinstance(callbacks, list):
                    config["callbacks"] = [*(callbacks or []), callback]
                else:
                    callbacks = callbacks.copy()
                    callbacks.add_handler(callback)
                    config["callbacks"] = callbacks
            try:
                result = await agent.ainvoke(payload, config=config)
            finally:
                if invocation_observer() is not None and run_state.usage.requests:
                    report_invocation(run_state.usage)
            self._capture_conversation_messages(result)
            if invocation_observer() is not None and not run_state.usage.requests:
                report_invocation(
                    _langchain_invocation_usage(
                        result,
                        input_messages=payload.get("messages"),
                    )
                )
            if resolver is not None:
                resolver.complete_run(
                    final_text=_langchain_text_response(result) if emit_final else None
                )
            return result

    def _model_for_run(self) -> str | Any:
        profile = self.active_profile
        if profile.model is not None:
            return normalize_for_langchain(profile.model)
        if self._model is None:
            raise RuntimeError("LangChain adapter requires a model.")
        return self._model

    def _settings_for_run(self, model: str | Any) -> dict[str, Any]:
        settings: dict[str, Any] = {}
        native_kwargs = getattr(model, "model_kwargs", None)
        if isinstance(native_kwargs, Mapping):
            settings["model_kwargs"] = dict(native_kwargs)
        for layer in (
            self._model_extra_kwargs,
            self._model_settings,
            langchain_model_settings(self.active_profile.settings),
        ):
            for key, value in layer.items():
                if key == "model_kwargs" and isinstance(value, Mapping):
                    settings[key] = {**settings.get(key, {}), **value}
                else:
                    settings[key] = value
        _apply_langchain_effort_settings(settings, model, self.active_profile.effort)
        compaction = self.active_profile.compaction
        if self.active_profile.history is True and compaction is not None:
            settings = langchain_settings_with_native_compaction(
                settings,
                model=model,
                settings=compaction,
            )
        conversation = self._conversation_var.get()
        settings = langchain_cache_settings(
            settings,
            model=model,
            conversation=conversation,
        )
        return settings

    def _chat_model_for_run(self) -> Any:
        model = self._model_for_run()
        settings = self._settings_for_run(model)
        if not isinstance(model, str):
            if not settings:
                return model
            copy_model = getattr(model, "model_copy", None)
            if not callable(copy_model):
                raise CompactionSettingsError(
                    "LangChain model settings require a model that supports model_copy"
                )
            fields = getattr(type(model), "model_fields", {})
            aliases = {field.alias: name for name, field in fields.items() if field.alias}
            updates: dict[str, Any] = {}
            for key, value in settings.items():
                target = aliases.get(key, key)
                if (
                    getattr(model, "root_async_client", None) is not None
                    and target
                    in {
                        "openai_api_key",
                        "openai_api_base",
                        "openai_organization",
                        "openai_proxy",
                        "http_client",
                        "http_async_client",
                        "request_timeout",
                        "max_retries",
                        "default_headers",
                        "default_query",
                    }
                    and value != getattr(model, target, None)
                ):
                    raise ValueError(
                        f"Configure transport setting {key!r} on the native model constructor; "
                        "copying an initialized model does not rebuild its clients."
                    )
                if fields and target not in fields:
                    if "model_kwargs" not in fields:
                        raise ValueError(f"Unsupported LangChain model setting: {key!r}")
                    updates.setdefault("model_kwargs", dict(settings.get("model_kwargs", {})))[
                        key
                    ] = value
                elif target == "model_kwargs":
                    updates[target] = {**value, **updates.get(target, {})}
                else:
                    updates[target] = value
            return copy_model(update=updates)
        try:
            from langchain.chat_models import init_chat_model
        except ImportError as exc:
            raise ModuleNotFoundError(
                "LangChainAdapter requires langchain optional dependencies. "
                "Install with: uv add 'kedi[langchain]'"
            ) from exc
        return init_chat_model(model, **settings)

    def _messages_for_prompt(self, prompt: str, *, model: Any | None = None) -> list[Any]:
        conversation = self._conversation_var.get()
        if conversation is None:
            return [{"role": "user", "content": prompt}]
        from langchain_core.messages import HumanMessage, messages_from_dict

        history: list[Any] = []
        serialized = conversation.resume_state.get("langchain_messages")
        if isinstance(serialized, list):
            history = list(messages_from_dict(serialized))
        history = [_langchain_without_cache_control(message) for message in history]
        current = HumanMessage(content=prompt)
        if _langchain_uses_explicit_cache_blocks(model):
            current = cast(HumanMessage, _langchain_with_cache_control(current))
        return [*history, current]

    def _cache_aware_system_prompt(self, prompt: str | None, *, model: Any) -> Any | None:
        if prompt is None or self._conversation_var.get() is None:
            return prompt
        if not _langchain_uses_explicit_cache_blocks(model):
            return prompt
        from langchain_core.messages import SystemMessage

        return _langchain_with_cache_control(SystemMessage(content=prompt))

    def _capture_conversation_messages(self, result: Any) -> None:
        conversation = self._conversation_var.get()
        if conversation is None or not isinstance(result, Mapping):
            return
        messages = result.get("messages")
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
            return
        from langchain_core.messages import BaseMessage, message_to_dict

        if all(isinstance(message, BaseMessage) for message in messages):
            conversation.result_state["langchain_messages"] = [
                message_to_dict(_langchain_without_cache_control(message)) for message in messages
            ]
            conversation.mark_completed()

    def _create_agent(
        self,
        *,
        model: Any,
        tools: Sequence[Any],
        response_format: Any | None,
        system_prompt: Any | None,
        middleware: Sequence[Any] = (),
    ) -> Any:
        try:
            from langchain.agents import create_agent
        except ImportError as exc:
            raise ModuleNotFoundError(
                "LangChainAdapter requires langchain optional dependencies. "
                "Install with: uv add 'kedi[langchain]'"
            ) from exc
        kwargs: dict[str, Any] = {
            "model": model,
            "tools": list(tools),
            "system_prompt": system_prompt,
            "middleware": list(middleware),
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        return create_agent(**kwargs)

    async def _tools_for_run(self) -> list[Any]:
        return [
            *self.scoped_tools,
            *await self._mcp_tools_for_profile(self.active_profile),
        ]

    def _approval_middleware_for_run(
        self,
        tools: Sequence[Any],
        *,
        approval_enabled: bool = True,
    ) -> list[Any]:
        if not tools:
            return []
        dispatcher = current_hook_dispatcher()
        has_tool_hooks = dispatcher is not None and any(
            dispatcher.handlers_for(event)
            for event in ("pre_tool_use", "post_tool_use", "post_tool_use_failure")
        )
        if not approval_enabled and not has_tool_hooks and current_artifact_admission() is None:
            return []
        known_specs = {tool.name: tool for tool in self.scoped_tool_specs}
        tool_specs = list(known_specs.values())
        for tool in tools:
            name = getattr(tool, "name", None)
            if isinstance(name, str) and name not in known_specs:
                tool_specs.append(_langchain_tool_spec_from_native(tool))
        return [
            _langchain_approval_middleware(
                profile=self.active_profile,
                tools=tool_specs,
                approval_ceiling=self._approval_ceiling_var.get(),
                approval_enabled=approval_enabled,
            )
        ]

    def _middleware_for_run(
        self,
        model: Any,
        tools: Sequence[Any],
        *,
        codemode_enabled: bool = False,
    ) -> list[Any]:
        from .langchain_runtime import LangChainRunMiddleware

        return [
            LangChainRunMiddleware(archive=self._history_archive, tool_retries=self._tool_retries),
            *_langchain_cache_middleware(
                model,
                enabled=self._conversation_var.get() is not None,
            ),
            *self._approval_middleware_for_run(
                tools,
                approval_enabled=not codemode_enabled,
            ),
        ]

    @asynccontextmanager
    async def _projected_tools_for_run(self) -> AsyncIterator[CodeModeProjection]:
        native_tools = await self._tools_for_run()
        settings = normalize_codemode_settings(
            self.active_profile.codemode or getattr(self, "codemode", None)
        )
        limits = (
            CodeModeLimits(**settings.limit_values())
            if settings is not None and settings.enabled
            else None
        )
        if limits is None:
            yield CodeModeProjection(tools=tuple(native_tools))
            return

        known_specs = {tool.name: tool for tool in self.scoped_tool_specs}
        specs = tuple(
            known_specs.get(tool.name) or _langchain_tool_spec_from_native(tool)
            for tool in native_tools
        )
        projection = codemode_projection(
            specs,
            limits=limits,
            policies=approval_policies(
                self.active_profile.approval,
                self._approval_ceiling_var.get(),
            ),
            adapter_shortname=self.shortname,
        )
        try:
            yield projection
        finally:
            await projection.close()

    async def _mcp_tools_for_profile(self, profile: AgentProfile) -> list[Any]:
        if not profile.mcp_servers:
            return []
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except ImportError as exc:
            raise ModuleNotFoundError(
                "LangChainAdapter MCP support requires langchain-mcp-adapters. "
                "Install with: uv add 'kedi[langchain]'"
            ) from exc
        client = MultiServerMCPClient(cast(Any, _langchain_mcp_server_configs(profile)))
        transports = {server.transport for server in profile.mcp_servers}
        transport = next(iter(transports)) if len(transports) == 1 else "mixed"
        with mcp_initialization_span(
            adapter_shortname=self.shortname,
            server_name="langchain_mcp",
            transport=transport,
            server_count=len(profile.mcp_servers),
        ) as active:
            tools = list(await client.get_tools())
            for tool in tools:
                if not hasattr(tool, "metadata"):
                    continue
                metadata = getattr(tool, "metadata", None)
                tool.metadata = {
                    **(dict(metadata) if isinstance(metadata, Mapping) else {}),
                    "kedi_origin": "mcp",
                }
            active.set_attribute("kedi.mcp.tool_count", len(tools))
            active.add_event(
                "mcp tools discovered",
                {"kedi.mcp.tool_count": len(tools)},
            )
            return tools


def _langchain_uses_explicit_cache_blocks(model: Any | None) -> bool:
    if model is None:
        return False
    return langchain_provider_cache_policy(model).backend in {"anthropic", "openrouter"}


def _langchain_cache_middleware(model: Any, *, enabled: bool) -> list[Any]:
    if not enabled or langchain_provider_cache_policy(model).backend != "bedrock":
        return []
    try:
        from langchain_aws.middleware.prompt_caching import BedrockPromptCachingMiddleware
    except ImportError as exc:
        raise ModuleNotFoundError(
            "LangChain Bedrock support requires its optional AWS dependencies. "
            "Install with: uv add 'kedi[langchain-aws]'"
        ) from exc
    return [BedrockPromptCachingMiddleware()]


def _langchain_with_cache_control(message: Any) -> Any:
    content = message.content
    if isinstance(content, str):
        blocks: list[Any] = [
            {
                "type": "text",
                "text": content,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    elif isinstance(content, list):
        blocks = [dict(block) if isinstance(block, Mapping) else block for block in content]
        for index in range(len(blocks) - 1, -1, -1):
            block = blocks[index]
            if isinstance(block, dict) and block.get("type") == "text":
                block.setdefault("cache_control", {"type": "ephemeral"})
                break
        else:
            return message
    else:
        return message
    return message.model_copy(update={"content": blocks})


def _langchain_without_cache_control(message: Any) -> Any:
    content = message.content
    if not isinstance(content, list):
        return message
    changed = False
    blocks: list[Any] = []
    for block in content:
        if isinstance(block, Mapping) and "cache_control" in block:
            normalized = dict(block)
            normalized.pop("cache_control", None)
            blocks.append(normalized)
            changed = True
        else:
            blocks.append(block)
    if not changed:
        return message
    return message.model_copy(update={"content": blocks})


def _langchain_tool_from_spec(tool: ToolSpec) -> Any:
    try:
        from langchain_core.tools import StructuredTool
    except ImportError as exc:
        raise ModuleNotFoundError(
            "LangChainAdapter tool support requires langchain-core. "
            "Install with: uv add 'kedi[langchain]'"
        ) from exc

    description = tool.description or tool.name
    kwargs: dict[str, Any] = {
        "name": tool.name,
        "description": description,
        "args_schema": tool.json_schema,
    }
    metadata = tool.metadata or {}
    if (metadata.get("kedi_tool_kind") or metadata.get("kind")) in {
        "subagent",
        "codemode_control",
        "artifact_management",
        "filesystem",
        "terminal",
        "dynamic_workflow",
    }:
        kwargs["handle_tool_error"] = True
    if _is_async_callable(tool.fn):
        kwargs["coroutine"] = _langchain_async_tool_function(tool)
    else:
        kwargs["func"] = _langchain_sync_tool_function(tool)
    native_tool = StructuredTool.from_function(**kwargs)
    native_tool.metadata = tool.metadata
    return native_tool


def _langchain_projection_tools(projection: CodeModeProjection) -> list[Any]:
    if not projection.enabled:
        return list(projection.tools)
    return [_langchain_tool_from_spec(cast(ToolSpec, tool)) for tool in projection.tools]


def _langchain_tool_spec_from_native(tool: Any) -> ToolSpec:
    name = getattr(tool, "name", None)
    if not isinstance(name, str) or not name:
        raise TypeError("LangChain CodeMode tools must expose a non-empty name")
    args_schema = getattr(tool, "args_schema", None)
    if isinstance(args_schema, Mapping):
        json_schema = dict(args_schema)
    else:
        model_json_schema = getattr(args_schema, "model_json_schema", None)
        json_schema = (
            model_json_schema()
            if callable(model_json_schema)
            else {"type": "object", "properties": {}}
        )

    async def invoke(**arguments: Any) -> Any:
        ainvoke = getattr(tool, "ainvoke", None)
        if callable(ainvoke):
            result = ainvoke(arguments)
            if inspect.isawaitable(result):
                return await result
            return result
        invoke_sync = getattr(tool, "invoke", None)
        if not callable(invoke_sync):
            raise TypeError(f"LangChain tool {name!r} is not invokable")
        result = invoke_sync(arguments)
        if inspect.isawaitable(result):
            return await result
        return result

    invoke.__name__ = name
    description = getattr(tool, "description", None)
    metadata = getattr(tool, "metadata", None)
    return ToolSpec(
        fn=invoke,
        name=name,
        description=description if isinstance(description, str) else name,
        json_schema=json_schema,
        metadata=dict(metadata) if isinstance(metadata, Mapping) else {"source": "langchain"},
        risk="mutating",
        risk_resolver=_langchain_external_tool_risk,
    )


def _langchain_sync_tool_function(tool: ToolSpec) -> Callable[..., Any]:
    @wraps(tool.fn)
    def _tool(*args: Any, **kwargs: Any) -> Any:
        from ..utils import run_sync

        try:
            result = tool.fn(*args, **_validated_tool_arguments(tool, kwargs))
            if inspect.isawaitable(result):
                return run_sync(cast(Coroutine[Any, Any, Any], result))
            return result
        except CodeModeError as exc:
            _raise_langchain_codemode_error(exc)
        except Exception as exc:
            _raise_langchain_tool_error(tool, exc)
            raise

    return _tool


def _langchain_async_tool_function(tool: ToolSpec) -> Callable[..., Any]:
    @wraps(tool.fn)
    async def _tool(*args: Any, **kwargs: Any) -> Any:
        try:
            result = tool.fn(*args, **_validated_tool_arguments(tool, kwargs))
            if inspect.isawaitable(result):
                return await result
            return result
        except CodeModeError as exc:
            _raise_langchain_codemode_error(exc)
        except Exception as exc:
            _raise_langchain_tool_error(tool, exc)
            raise

    return _tool


def _raise_langchain_tool_error(tool: ToolSpec, exc: Exception) -> None:
    message = correctable_tool_error(tool, exc)
    if message is not None:
        from langchain_core.tools import ToolException

        raise ToolException(message) from exc


def _raise_langchain_codemode_error(exc: CodeModeError) -> NoReturn:
    from langchain_core.tools import ToolException

    raise ToolException(str(exc)) from exc


def _langchain_approval_middleware(
    *,
    profile: AgentProfile,
    tools: Sequence[ToolSpec],
    approval_ceiling: Sequence[ApprovalPolicy | None] = (),
    approval_enabled: bool = True,
) -> Any:
    """Bridge Kedi hook and approval semantics into LangChain's tool lifecycle."""

    try:
        from langchain.agents.middleware import AgentMiddleware
    except ImportError as exc:
        raise ModuleNotFoundError(
            "LangChainAdapter approval support requires langchain. "
            "Install with: uv add 'kedi[langchain]'"
        ) from exc

    known_tools = {tool.name: tool for tool in tools}

    class KediToolLifecycleMiddleware(AgentMiddleware):
        async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
            tool_call = request.tool_call
            tool_name = tool_call.get("name") if isinstance(tool_call, Mapping) else None
            arguments = tool_call.get("args") if isinstance(tool_call, Mapping) else None
            tool_call_id = tool_call.get("id") if isinstance(tool_call, Mapping) else None
            if not isinstance(tool_name, str) or not isinstance(arguments, Mapping):
                return _langchain_tool_error(
                    tool_name if isinstance(tool_name, str) else "unknown",
                    tool_call_id,
                    "LangChain emitted an incomplete tool call.",
                )

            tool = known_tools.get(tool_name, _langchain_external_tool_spec(tool_name))
            call_id = str(tool_call_id or new_tool_call_id())
            origin = _langchain_hook_origin(tool, getattr(request, "tool", None))
            try:
                final_arguments = await dispatch_pre_tool_use(
                    tool,
                    arguments,
                    adapter_shortname="langchain",
                    tool_call_id=call_id,
                    origin=origin,
                    native_tool_name=tool_name,
                )
                if approval_enabled:
                    policies = tuple(approval_ceiling)
                    if profile.approval is not None or not policies:
                        policies = (*policies, profile.approval)
                    for policy in policies:
                        final_arguments = await resolve_tool_arguments_async(
                            tool,
                            final_arguments,
                            policy=policy,
                            adapter_shortname="langchain",
                        )
            except (HookDeniedError, ToolApprovalDenied) as exc:
                return _langchain_tool_error(tool_name, tool_call_id, str(exc))
            except Exception as exc:
                return _langchain_tool_error(tool_name, tool_call_id, str(exc))

            final_request = request.override(
                tool_call={
                    **dict(tool_call),
                    "args": final_arguments,
                }
            )
            started = monotonic()
            try:
                result = await handler(final_request)
            except BaseException as exc:
                await dispatch_post_tool_use_failure(
                    tool,
                    final_arguments,
                    exc,
                    adapter_shortname="langchain",
                    tool_call_id=call_id,
                    origin=origin,
                    native_tool_name=tool_name,
                    duration_seconds=monotonic() - started,
                    interrupted=is_tool_execution_interrupted(exc),
                )
                raise
            result_message = _langchain_tool_message_for_call(result, call_id)
            if result_message is not None and getattr(result_message, "status", None) == "error":
                error = RuntimeError(
                    str(getattr(result_message, "content", "Tool execution failed"))
                )
                await dispatch_post_tool_use_failure(
                    tool,
                    final_arguments,
                    error,
                    adapter_shortname="langchain",
                    tool_call_id=call_id,
                    origin=origin,
                    native_tool_name=tool_name,
                    duration_seconds=monotonic() - started,
                )
                return result
            if (
                tool.artifact_result != "never"
                and not tool_has_artifact_boundary(tool)
                and (tool.metadata or {}).get("kedi_codemode_control") is not True
            ):
                result = await _langchain_admit_native_tool_result(
                    result,
                    tool_name=tool_name,
                    call_id=call_id,
                )
            await dispatch_post_tool_use(
                tool,
                final_arguments,
                result,
                adapter_shortname="langchain",
                tool_call_id=call_id,
                origin=origin,
                native_tool_name=tool_name,
            )
            return result

    return KediToolLifecycleMiddleware()


async def _langchain_admit_native_tool_result(
    result: Any,
    *,
    tool_name: str,
    call_id: str,
) -> Any:
    """Replace causal ToolMessage content while preserving LangGraph commands."""

    try:
        from langchain_core.messages import ToolMessage
        from langgraph.types import Command
    except ImportError:  # pragma: no cover - adapter dependency guard
        return result

    async def admit_message(message: Any) -> tuple[Any, bool]:
        if not isinstance(message, ToolMessage) or message.tool_call_id != call_id:
            return message, False
        content = message.content
        admitted = await admit_current_artifact_result(
            content,
            tool_name=tool_name,
            call_id=call_id,
        )
        if admitted is content:
            return message, False
        model_content = (
            json.dumps(admitted, ensure_ascii=True, separators=(",", ":"))
            if isinstance(admitted, Mapping)
            else admitted
        )
        return message.model_copy(update={"content": model_content}), True

    async def admit_messages(value: Any) -> tuple[Any, bool]:
        if isinstance(value, ToolMessage):
            return await admit_message(value)
        if isinstance(value, list):
            changed = False
            items: list[Any] = []
            for item in value:
                admitted, item_changed = await admit_messages(item)
                items.append(admitted)
                changed = changed or item_changed
            return (items if changed else value), changed
        if isinstance(value, tuple):
            changed = False
            items = []
            for item in value:
                admitted, item_changed = await admit_messages(item)
                items.append(admitted)
                changed = changed or item_changed
            return (tuple(items) if changed else value), changed
        return value, False

    if isinstance(result, ToolMessage):
        admitted, _ = await admit_message(result)
        return admitted
    if isinstance(result, Command):
        update = result.update
        if isinstance(update, Mapping):
            if "messages" not in update:
                return result
            messages, changed = await admit_messages(update["messages"])
            if not changed:
                return result
            return replace(result, update={**update, "messages": messages})

        admitted_update, changed = await admit_messages(update)
        return replace(result, update=admitted_update) if changed else result

    # Keep the existing helper contract for direct/raw results used by adapter-native bridges.
    content = getattr(result, "content", result)
    admitted = await admit_current_artifact_result(
        content,
        tool_name=tool_name,
        call_id=call_id,
    )
    if admitted is content:
        return result
    model_copy = getattr(result, "model_copy", None)
    if not callable(model_copy):
        return admitted
    model_content = (
        json.dumps(admitted, ensure_ascii=True, separators=(",", ":"))
        if isinstance(admitted, Mapping)
        else admitted
    )
    return model_copy(update={"content": model_content})


def _langchain_tool_message_for_call(result: Any, call_id: str) -> Any | None:
    """Find the causal ToolMessage in a direct result or LangGraph Command update."""

    try:
        from langchain_core.messages import ToolMessage
        from langgraph.types import Command
    except ImportError:  # pragma: no cover - adapter dependency guard
        return None

    def find(value: Any) -> Any | None:
        if isinstance(value, ToolMessage):
            return value if value.tool_call_id == call_id else None
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                found = find(item)
                if found is not None:
                    return found
        return None

    if isinstance(result, ToolMessage):
        return find(result)
    if not isinstance(result, Command):
        return None
    update = result.update
    return find(update.get("messages")) if isinstance(update, Mapping) else find(update)


def _langchain_hook_origin(tool: ToolSpec, native_tool: Any) -> ToolOrigin:
    if (tool.metadata or {}).get("source") != "langchain-external":
        return "kedi"
    metadata = getattr(native_tool, "metadata", None)
    if isinstance(metadata, Mapping) and metadata.get("kedi_origin") == "mcp":
        return "mcp"
    return "adapter_native"


def _langchain_event_handler(resolver: AgentMessageResolver) -> Any:
    from langchain_core.callbacks import AsyncCallbackHandler

    class KediAgentEventHandler(AsyncCallbackHandler):
        async def on_llm_end(
            self,
            response: Any,
            *,
            run_id: Any,
            **_kwargs: Any,
        ) -> None:
            generations = getattr(response, "generations", ())
            if not isinstance(generations, Sequence):
                return
            for group_index, group in enumerate(generations):
                if not isinstance(group, Sequence):
                    continue
                for generation_index, generation in enumerate(group):
                    message = getattr(generation, "message", None)
                    text = _langchain_completed_message_text(message)
                    if text:
                        resolver.complete_message(
                            f"langchain:{run_id}:{group_index}:{generation_index}",
                            text,
                        )
                        resolver.mark_message_boundary()
                    tool_calls = getattr(message, "tool_calls", None)
                    if isinstance(tool_calls, Sequence) and tool_calls:
                        resolver.mark_tool_boundary()

        async def on_tool_start(
            self,
            serialized: dict[str, Any],
            input_str: str,
            *,
            run_id: Any,
            **_kwargs: Any,
        ) -> None:
            del serialized, input_str, run_id
            resolver.mark_tool_boundary()

    return KediAgentEventHandler()


def _langchain_completed_message_text(message: Any) -> str:
    return (_langchain_content_text(getattr(message, "content", None)) or "").strip()


def _langchain_invocation_usage(
    result: Any,
    *,
    input_messages: object = None,
) -> InvocationUsage:
    if not isinstance(result, Mapping):
        return InvocationUsage(requests=1)
    messages = result.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        return InvocationUsage(requests=1)
    messages = _langchain_current_messages(messages, input_messages)

    input_tokens = cache_read_tokens = cache_write_tokens = 0
    output_tokens = total_tokens = tool_calls = requests = 0
    for message in messages:
        usage = getattr(message, "usage_metadata", None)
        if isinstance(usage, Mapping):
            input_tokens += _usage_int(usage, "input_tokens", "inputTokens")
            input_details = usage.get("input_token_details")
            if isinstance(input_details, Mapping):
                cache_read_tokens += _usage_int(
                    input_details,
                    "cache_read",
                    "cached_tokens",
                )
                cache_write_tokens += _usage_int(
                    input_details,
                    "cache_creation",
                    "cache_write_tokens",
                )
            output_tokens += _usage_int(usage, "output_tokens", "outputTokens")
            total_tokens += _usage_int(usage, "total_tokens", "totalTokens")
            requests += 1
        calls = getattr(message, "tool_calls", None)
        if isinstance(calls, Sequence) and not isinstance(calls, (str, bytes)):
            tool_calls += len(calls)
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    return InvocationUsage(
        requests=max(requests, 1),
        tool_calls=tool_calls,
        input_tokens=input_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _langchain_current_messages(
    messages: Sequence[Any],
    input_messages: object,
) -> Sequence[Any]:
    if not isinstance(input_messages, Sequence) or isinstance(input_messages, (str, bytes)):
        return messages
    if len(messages) < len(input_messages):
        return messages
    if all(
        _same_langchain_message(actual, expected)
        for actual, expected in zip(messages, input_messages, strict=False)
    ):
        return messages[len(input_messages) :]
    return messages


def _same_langchain_message(actual: Any, expected: Any) -> bool:
    if actual is expected or actual == expected:
        return True
    return _langchain_message_identity(actual) == _langchain_message_identity(expected)


def _langchain_message_identity(message: Any) -> tuple[object, object]:
    if isinstance(message, Mapping):
        role = message.get("role", message.get("type"))
        content = message.get("content")
    else:
        role = getattr(message, "type", getattr(message, "role", None))
        content = getattr(message, "content", None)
    if role == "human":
        role = "user"
    elif role == "ai":
        role = "assistant"
    return role, content


def _usage_int(values: Mapping[str, Any], *names: str) -> int:
    for name in names:
        value = values.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return 0


def _langchain_tool_error(tool_name: str, tool_call_id: object, message: str) -> Any:
    from langchain.messages import ToolMessage

    return ToolMessage(
        content=message,
        name=tool_name,
        tool_call_id=str(tool_call_id or "kedi-approval"),
        status="error",
    )


def _validated_tool_arguments(tool: ToolSpec, arguments: Mapping[str, Any]) -> dict[str, Any]:
    if tool.argument_validator is None:
        return dict(arguments)
    return tool.argument_validator(arguments)


def _is_async_callable(fn: Callable[..., Any]) -> bool:
    return inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(
        getattr(fn, "__call__", None)
    )


def _langchain_external_tool_spec(name: str) -> ToolSpec:
    return ToolSpec(
        fn=lambda **_kwargs: None,
        name=name,
        description="External LangChain or MCP tool",
        metadata={"source": "langchain-external"},
        risk="mutating",
        risk_resolver=_langchain_external_tool_risk,
    )


def _langchain_external_tool_risk(arguments: Mapping[str, Any]) -> ToolRisk:
    return "sensitive" if arguments_target_secret_file(arguments) else "mutating"


def _langchain_mcp_server_configs(profile: AgentProfile) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for index, spec in enumerate(profile.mcp_servers):
        name = f"kedi_mcp_{index}"
        config: dict[str, Any]
        if spec.transport == "stdio":
            if spec.command is None:
                raise ValueError("MCP stdio transport requires a command.")
            config = {
                "transport": "stdio",
                "command": spec.command,
                "args": list(spec.args),
            }
            if spec.env is not None:
                config["env"] = spec.env
        elif spec.transport == "sse":
            if spec.url is None:
                raise ValueError("MCP sse transport requires a url.")
            config = {"transport": "sse", "url": spec.url}
            if spec.headers is not None:
                config["headers"] = spec.headers
        elif spec.transport == "http":
            if spec.url is None:
                raise ValueError("MCP http transport requires a url.")
            config = {"transport": "streamable_http", "url": spec.url}
            if spec.headers is not None:
                config["headers"] = spec.headers
        else:
            raise ValueError(f"Unsupported MCP transport: {spec.transport}")
        configs[name] = config
    return configs


def _apply_langchain_effort_settings(
    settings: dict[str, Any],
    model: str | Any,
    effort: str | None,
) -> None:
    if effort is None:
        return
    mapped = pydantic_effort_value(effort)
    existing = settings.get("reasoning", getattr(model, "reasoning", None))
    if (isinstance(model, str) and model.startswith("openrouter:")) or isinstance(
        existing, Mapping
    ):
        reasoning = dict(existing) if isinstance(existing, Mapping) else {}
        reasoning["effort"] = mapped
        settings["reasoning"] = reasoning
        return
    settings["reasoning_effort"] = mapped


def _langchain_structured_response(result: Any) -> Any:
    if isinstance(result, Mapping):
        if "structured_response" in result:
            return result["structured_response"]
        raise ValueError("LangChain agent did not return structured_response.")
    try:
        return result.structured_response
    except AttributeError:
        raise ValueError("LangChain agent did not return structured_response.") from None


def _langchain_text_response(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, Mapping):
        if isinstance(result.get("content"), str):
            return result["content"]
        if isinstance(result.get("output"), str):
            return result["output"]
        messages = result.get("messages")
        if isinstance(messages, (list, tuple)) and messages:
            return _langchain_message_text(messages[-1])
    return _langchain_message_text(result)


def _langchain_message_text(message: Any) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, Mapping):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)
    text = _langchain_content_text(content)
    if text is not None:
        return text
    raise ValueError("LangChain agent did not return a text response.")


def _langchain_content_text(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
        return None
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, Mapping):
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)
