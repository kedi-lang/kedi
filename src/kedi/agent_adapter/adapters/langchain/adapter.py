from __future__ import annotations

from contextlib import AbstractContextManager, asynccontextmanager, contextmanager, nullcontext
from contextvars import ContextVar
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Iterator,
    Mapping,
    Sequence,
    cast,
)

from pydantic import BaseModel, create_model
from pydantic_ai import UsageLimits
from pydantic_ai.output import OutputDataT

from kedi.agent_adapter.adapters.langchain.messages import (
    _apply_langchain_effort_settings,
    _capture_langchain_typesafe_decision_evidence,
    _langchain_event_handler,
    _langchain_invocation_usage,
    _langchain_structured_response,
    _langchain_text_response,
    _langchain_uses_explicit_cache_blocks,
    _langchain_with_cache_control,
    _langchain_without_cache_control,
)
from kedi.agent_adapter.adapters.langchain.middleware import (
    _langchain_approval_middleware,
    _langchain_cache_middleware,
)
from kedi.agent_adapter.adapters.langchain.tools import (
    _langchain_mcp_server_configs,
    _langchain_projection_tools,
    _langchain_tool_from_spec,
    _langchain_tool_spec_from_native,
)
from kedi.agent_adapter.approvals.policy import ApprovalPolicy
from kedi.agent_adapter.codemode import CodeModeLimits
from kedi.agent_adapter.codemode.tool_specs import (
    CodeModeProjection,
    append_codemode_instructions,
    approval_policies,
    codemode_projection,
)
from kedi.agent_adapter.compaction.archive import HistoryArchiveSettings
from kedi.agent_adapter.compaction.native_settings import langchain_settings_with_native_compaction
from kedi.agent_adapter.conversation import AdapterConversationContext
from kedi.agent_adapter.events import agent_event_run
from kedi.agent_adapter.history_processing import (
    HistoryProcessorCallback,
    HistoryProcessorRunner,
)
from kedi.agent_adapter.hooks import (
    HookHandler,
    HookRegistrationMixin,
    current_hook_dispatcher,
    hooked_agent_run,
)
from kedi.agent_adapter.invocation import invocation_observer, report_invocation
from kedi.agent_adapter.meta import AdapterCapabilities, AgentAdapterKind, HookCapabilities
from kedi.agent_adapter.models import (
    close_typesafe_model_current,
    is_typesafe_model_id,
    typesafe_langchain_model,
)
from kedi.agent_adapter.output_schema import build_output_model
from kedi.agent_adapter.prefix_cache import langchain_cache_settings
from kedi.agent_adapter.responses_session import model_responses_session, model_responses_turn
from kedi.agent_adapter.subagents import SubagentExecutionRequest
from kedi.agent_adapter.subagents.attempts import DelegationFailurePolicy, delegation_scope
from kedi.agent_adapter.subagents.limits import SubagentUsageLimits
from kedi.agent_adapter.telemetry import mcp_initialization_span
from kedi.agent_adapter.tools.scope import ToolScope, current_artifact_admission
from kedi.agent_adapter.tools.specs import ToolSpec
from kedi.agent_profile import AgentProfile, CodeModeSettings, normalize_codemode_settings
from kedi.agent_profile.compaction import CompactionSettingsError
from kedi.agent_profile.settings import (
    KEDI_RUNTIME_SETTING_KEYS,
    LANGCHAIN_MODEL_SETTING_KEYS,
    langchain_model_settings,
    tool_reason_enabled,
)
from kedi.model_normalization import normalize_for_langchain
from kedi.resolved_types import ResolvedKediType


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
        settings=LANGCHAIN_MODEL_SETTING_KEYS | KEDI_RUNTIME_SETTING_KEYS,
    )

    def supports_artifact_contract(self, profile: AgentProfile) -> bool:
        """Return whether the active model can receive artifact management tools."""

        if profile.model is not None:
            return not is_typesafe_model_id(profile.model)
        return getattr(self._model, "_llm_type", None) != "typesafe-jev"

    def __init__(
        self,
        model: Any = None,
        *,
        model_settings: dict[str, Any] | None = None,
        subagent_failure_policy: DelegationFailurePolicy = "fail_closed",
        codemode: bool | CodeModeSettings = False,
        query_artifact: bool = False,
        tool_reason: bool = False,
        hook_handler: HookHandler | None = None,
        default_usage_limits: UsageLimits | None = None,
        history_archive: HistoryArchiveSettings | None = None,
        history_processor: HistoryProcessorCallback[Any] | None = None,
        tool_retries: int = 3,
        **kwargs: Any,
    ) -> None:
        self._init_hook_registration(hook_handler)
        if not isinstance(query_artifact, bool):
            raise TypeError("query_artifact must be a boolean")
        self.query_artifact = query_artifact
        if not isinstance(tool_reason, bool):
            raise TypeError("tool_reason must be a boolean")
        self.tool_reason = tool_reason
        if isinstance(tool_retries, bool) or not isinstance(tool_retries, int) or tool_retries < 0:
            raise ValueError("tool_retries must be a non-negative integer")
        self._default_usage_limits = default_usage_limits
        self._history_archive = history_archive
        self._history_processor_runner: HistoryProcessorRunner[Any] | None = None
        if history_processor is not None:
            from kedi.agent_adapter.adapters.langchain.history_processing import (
                LangChainHistoryMessageCodec,
            )

            self._history_processor_runner = HistoryProcessorRunner(
                history_processor,
                LangChainHistoryMessageCodec(),
            )
        self._tool_retries = tool_retries
        if subagent_failure_policy not in {"recover", "fail_closed"}:
            raise ValueError(f"Unknown subagent failure policy: {subagent_failure_policy!r}")
        self._model_overrides: dict[str, Any] = {}
        self._default_model = model if isinstance(model, str) else None
        self._model = self._resolve_model(model) if isinstance(model, str) else model
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

    @property
    def active_tool_reason(self) -> bool:
        return tool_reason_enabled(self.active_profile.settings, default=self.tool_reason)

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

        response_format, unwrap_response = _langchain_typed_output(resolved_output_type)
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
                response_format=response_format,
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
            _capture_langchain_typesafe_decision_evidence(
                result,
                input_messages=payload.get("messages"),
            )
            structured = _langchain_structured_response(result)
            return cast(OutputDataT, unwrap_response(structured))

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
        from kedi.agent_adapter.execution import run_sync

        model = self._model_for_run()

        async def invoke_and_close() -> str:
            try:
                return await self.invoke(prompt=prompt, **kwargs)
            finally:
                await close_typesafe_model_current(model)

        return run_sync(invoke_and_close())

    def produce_sync(
        self,
        template: str,
        output_schema: dict[str, ResolvedKediType] | None = None,
        output_type: type[OutputDataT] | None = None,
        **kwargs: Any,
    ) -> OutputDataT:
        from kedi.agent_adapter.execution import run_sync

        model = self._model_for_run()

        async def produce_and_close() -> OutputDataT:
            try:
                return await self.produce(
                    template,
                    output_schema=output_schema,
                    output_type=output_type,
                    **kwargs,
                )
            finally:
                await close_typesafe_model_current(model)

        return run_sync(produce_and_close())

    def default_model(self) -> str | None:
        return self._default_model

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
        from kedi.agent_adapter.adapters.langchain.runtime import langchain_run_scope

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
            (
                self._history_processor_runner.run_scope(self._conversation_var.get())
                if self._history_processor_runner is not None
                else nullcontext()
            ),
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
            return self._resolve_model(profile.model)
        if self._model is None:
            raise RuntimeError("LangChain adapter requires a model.")
        return self._model

    def _resolve_model(self, model: str) -> str | Any:
        if not is_typesafe_model_id(model):
            return normalize_for_langchain(model)
        resolved = self._model_overrides.get(model)
        if resolved is None:
            resolved = typesafe_langchain_model(model)
            self._model_overrides[model] = resolved
        return resolved

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
        if self.active_tool_reason:
            from kedi.agent_adapter.approvals.reason import TOOL_REASON_INSTRUCTIONS

            prompt = (
                f"{prompt.rstrip()}\n\n{TOOL_REASON_INSTRUCTIONS}"
                if prompt
                else TOOL_REASON_INSTRUCTIONS
            )
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
                tool_reason=self.active_tool_reason,
            )
        ]

    def _middleware_for_run(
        self,
        model: Any,
        tools: Sequence[Any],
        *,
        codemode_enabled: bool = False,
    ) -> list[Any]:
        from kedi.agent_adapter.adapters.langchain.runtime import LangChainRunMiddleware

        return [
            LangChainRunMiddleware(
                archive=self._history_archive,
                history_processor=self._history_processor_runner,
                model_id=_history_processor_model_id(model),
                tool_retries=self._tool_retries,
            ),
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
            if self.active_tool_reason:
                from .reason import tools_with_reason

                native_tools = tools_with_reason(native_tools, self.scoped_tool_specs)
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
            tool_reason=self.active_tool_reason,
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


def _langchain_typed_output(
    output_type: type[Any],
) -> tuple[type[BaseModel], Callable[[Any], Any]]:
    if isinstance(output_type, type) and issubclass(output_type, BaseModel):
        return output_type, lambda value: value
    wrapper = create_model("KediTypedOutput", response=(output_type, ...))
    return wrapper, lambda value: value.response


def _history_processor_model_id(model: Any) -> str | None:
    from kedi.agent_adapter.adapters.langchain.history_processing import langchain_model_id

    return langchain_model_id(model)
