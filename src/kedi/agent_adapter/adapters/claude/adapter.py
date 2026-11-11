from __future__ import annotations

from contextlib import AbstractContextManager, AsyncExitStack, contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Generator, Mapping, Sequence, cast

from kedi.agent_adapter.adapters.claude.hooks import _with_kedi_approval_hook
from kedi.agent_adapter.adapters.claude.messages import (
    _active_claude_hook_identities,
    _claude_invocation_usage,
    _claude_non_retryable_system_error,
    _claude_session_fingerprint,
    _claude_text_response,
    _is_result_message,
    _observe_claude_message,
    _output_json_schema,
    _set_claude_system_prompt,
    _structured_output_value,
)
from kedi.agent_adapter.adapters.claude.tools import (
    _claude_external_mcp_servers,
    _claude_sdk,
    _claude_sdk_mcp_server,
)
from kedi.agent_adapter.approvals.policy import ApprovalPolicy
from kedi.agent_adapter.codemode import CodeModeLimits, materialize_mcp_tool_specs
from kedi.agent_adapter.codemode.tool_specs import (
    CONTROL_TOOL_NAMES,
    CodeModeProjection,
    append_codemode_instructions,
    approval_policies,
    codemode_projection,
)
from kedi.agent_adapter.compaction.native_settings import validate_claude_native_compaction
from kedi.agent_adapter.conversation import AdapterConversationContext
from kedi.agent_adapter.events import agent_event_run
from kedi.agent_adapter.hooks import HookHandler, HookRegistrationMixin, hooked_agent_run
from kedi.agent_adapter.invocation import invocation_observer, report_invocation
from kedi.agent_adapter.meta import AdapterCapabilities, AgentAdapterKind, HookCapabilities
from kedi.agent_adapter.output_schema import build_output_model
from kedi.agent_adapter.subagents import SubagentConversationContext, SubagentExecutionRequest
from kedi.agent_adapter.subagents.attempts import DelegationFailurePolicy, delegation_scope
from kedi.agent_adapter.subagents.limits import SubagentUsageLimits
from kedi.agent_adapter.tools.scope import ToolScope
from kedi.agent_adapter.tools.specs import ToolSpec
from kedi.agent_profile import AgentProfile, CodeModeSettings, normalize_codemode_settings
from kedi.agent_profile.settings import CLAUDE_AGENT_SETTING_KEYS, claude_agent_settings
from kedi.effort import pydantic_effort_value
from kedi.resolved_types import ResolvedKediType

from .tools import _CLAUDE_KEDI_MCP_TOOL_PREFIX

if TYPE_CHECKING:
    from claude_agent_sdk import Message, ResultMessage

OutputDataT = Any


CLAUDE_CODE_TOOLS_PRESET: Mapping[str, str] = {
    "type": "preset",
    "preset": "claude_code",
}


CLAUDE_CODE_PERMISSION_MODE = "acceptEdits"


class ClaudeAdapter(HookRegistrationMixin):
    """Adapter for Anthropic's Claude Agent SDK harness."""

    kind: AgentAdapterKind = "agent-harness"
    shortname = "claude"
    supports_native_subagent_resume = True
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
        supports_live_subagent_input=True,
        supports_subagent_interrupt=True,
        supports_artifacts=True,
        supports_native_artifacts=False,
        supports_stateful_history=True,
        supports_history_processing=False,
        supports_history_replay=False,
        supports_native_compaction=True,
        supports_stream_events=True,
        supports_media_input=False,
        hooks=HookCapabilities.full(
            tool_bridge="native",
            tool_origins=frozenset({"kedi", "mcp", "provider_builtin"}),
        ),
        settings=CLAUDE_AGENT_SETTING_KEYS,
    )

    def __init__(
        self,
        model: str | None = None,
        *,
        model_settings: dict[str, Any] | None = None,
        subagent_failure_policy: DelegationFailurePolicy = "fail_closed",
        codemode: bool | CodeModeSettings = False,
        query_artifact: bool = False,
        hook_handler: HookHandler | None = None,
        **kwargs: Any,
    ) -> None:
        self._init_hook_registration(hook_handler)
        if not isinstance(query_artifact, bool):
            raise TypeError("query_artifact must be a boolean")
        self.query_artifact = query_artifact
        if subagent_failure_policy not in {"recover", "fail_closed"}:
            raise ValueError(f"Unknown subagent failure policy: {subagent_failure_policy!r}")
        self._model = model
        self._settings = claude_agent_settings(model_settings or {})
        self._extra_kwargs = dict(kwargs)
        self._subagent_failure_policy: DelegationFailurePolicy = subagent_failure_policy
        self.codemode = normalize_codemode_settings(codemode)
        self._tool_scope = ToolScope()
        self._profile_stack_var: ContextVar[tuple[AgentProfile, ...]] = ContextVar(
            f"kedi_claude_profile_stack_{id(self)}",
            default=(),
        )
        self._approval_ceiling_var: ContextVar[tuple[ApprovalPolicy | None, ...]] = ContextVar(
            f"kedi_claude_approval_ceiling_{id(self)}",
            default=(),
        )
        self._conversation_var: ContextVar[
            AdapterConversationContext | SubagentConversationContext | None
        ] = ContextVar(
            f"kedi_claude_conversation_{id(self)}",
            default=None,
        )

    def tool_scope(self, tools: Sequence[Any] = ()) -> AbstractContextManager[None]:
        return self._tool_scope.tool_scope(tools)

    def register_tool(self, tool: ToolSpec) -> None:
        self._tool_scope.register_tool(tool)

    @property
    def scoped_tools(self) -> tuple[ToolSpec, ...]:
        return cast(tuple[ToolSpec, ...], self._tool_scope.scoped_tools)

    @property
    def active_profile(self) -> AgentProfile:
        stack = self._profile_stack_var.get()
        return stack[-1] if stack else AgentProfile.empty()

    @staticmethod
    def type_builder(output_schema: dict[str, ResolvedKediType]) -> type:
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

        result = await self._query_result(
            prompt=template,
            output_type=resolved_output_type,
            output_schema=None,
            kwargs=kwargs,
            emit_final=False,
        )
        structured_output = getattr(result, "structured_output", None)
        if structured_output is None:
            raise ValueError("Claude Agent SDK did not return structured_output.")
        return _structured_output_value(structured_output, resolved_output_type)

    @hooked_agent_run("template")
    async def produce_subagent(
        self,
        *,
        template: str,
        output_schema: dict[str, Any],
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        result = await self._query_result(
            prompt=template,
            output_type=None,
            output_schema=output_schema,
            kwargs=kwargs,
            emit_final=False,
        )
        structured_output = getattr(result, "structured_output", None)
        if not isinstance(structured_output, Mapping):
            raise TypeError("Claude subagent structured output must be an object")
        return structured_output

    @hooked_agent_run("prompt")
    async def invoke(self, *, prompt: str, **kwargs: Any) -> str:
        result = await self._query_result(
            prompt=prompt,
            output_type=None,
            output_schema=None,
            kwargs=kwargs,
            emit_final=True,
        )
        return _claude_text_response(result)

    async def _query_result(
        self,
        *,
        prompt: str,
        output_type: type[Any] | None,
        output_schema: dict[str, Any] | None,
        kwargs: Mapping[str, Any],
        emit_final: bool,
    ) -> ResultMessage:
        sdk = _claude_sdk()
        run_kwargs = dict(kwargs)
        instructions = cast(str | None, run_kwargs.pop("instructions", None))
        async with AsyncExitStack() as stack:
            projection, codemode_settings = await self._projected_tools_for_run(stack)
            options, session_fingerprint = self._options_for_run(
                sdk,
                output_type=output_type,
                output_schema=output_schema,
                instructions=instructions,
                run_kwargs=run_kwargs,
                tools=cast(Sequence[ToolSpec], projection.tools),
                codemode_settings=codemode_settings,
                codemode_preloaded_tools=projection.preloaded_tools,
            )
            result: ResultMessage | None = None
            message_index = 0
            with (
                delegation_scope(self._subagent_failure_policy),
                agent_event_run(self.shortname) as resolver,
            ):

                def observe_message(message: Message) -> None:
                    nonlocal result, message_index
                    system_error = _claude_non_retryable_system_error(sdk, message)
                    if system_error is not None:
                        raise RuntimeError(system_error)
                    if resolver is not None:
                        message_index = _observe_claude_message(
                            sdk,
                            message,
                            resolver,
                            message_index,
                        )
                    if _is_result_message(sdk, message):
                        result = cast("ResultMessage", message)
                        conversation = self._conversation_var.get()
                        if conversation is not None:
                            conversation.result_state.update(
                                {
                                    "claude_session_id": result.session_id,
                                    "claude_session_fingerprint": session_fingerprint,
                                }
                            )
                        if invocation_observer() is not None:
                            report_invocation(_claude_invocation_usage(message))

                from kedi.agent_adapter.subagents.control import current_live_input

                control = current_live_input()
                if control is None:
                    async for message in sdk.query(prompt=prompt, options=options):
                        observe_message(message)
                else:
                    from .live import run_query

                    result = await run_query(options, prompt, control, observe_message)

                if result is None:
                    raise RuntimeError("Claude Agent SDK did not return a result message.")
                if bool(getattr(result, "is_error", False)):
                    raise RuntimeError(
                        str(getattr(result, "result", "Claude Agent SDK run failed."))
                    )
                conversation = self._conversation_var.get()
                if conversation is not None:
                    conversation.result_state.update(
                        {
                            "claude_session_id": result.session_id,
                            "claude_session_fingerprint": session_fingerprint,
                        }
                    )
                    if isinstance(conversation, AdapterConversationContext):
                        conversation.mark_completed()
                if resolver is not None:
                    resolver.complete_run(
                        final_text=_claude_text_response(result) if emit_final else None
                    )
                return result

    def invoke_sync(self, *, prompt: str, **kwargs: Any) -> str:
        from kedi.agent_adapter.execution import run_sync

        return run_sync(self.invoke(prompt=prompt, **kwargs))

    def produce_sync(
        self,
        template: str,
        output_schema: dict[str, ResolvedKediType] | None = None,
        output_type: type[OutputDataT] | None = None,
        **kwargs: Any,
    ) -> OutputDataT:
        from kedi.agent_adapter.execution import run_sync

        return run_sync(
            self.produce(template, output_schema=output_schema, output_type=output_type, **kwargs)
        )

    def default_model(self) -> str | None:
        return self._model

    def telemetry_model_name(self) -> str | None:
        return self._model

    @contextmanager
    def profile_override(self, profile: AgentProfile) -> Generator[None, None, None]:
        stack_var = self._profile_stack_var
        token = stack_var.set((*stack_var.get(), profile))
        try:
            yield
        finally:
            stack_var.reset(token)

    @contextmanager
    def approval_ceiling(
        self,
        policies: Sequence[ApprovalPolicy | None],
    ) -> Generator[None, None, None]:
        token = self._approval_ceiling_var.set(tuple(policies))
        try:
            yield
        finally:
            self._approval_ceiling_var.reset(token)

    @contextmanager
    def subagent_scope(self, request: SubagentExecutionRequest) -> Generator[None, None, None]:
        token = self._conversation_var.set(request.conversation)
        try:
            with (
                self.tool_scope(request.tools),
                self.profile_override(request.profile),
                self.approval_ceiling(request.approval_ceiling),
            ):
                yield
        finally:
            self._conversation_var.reset(token)

    @contextmanager
    def conversation_scope(
        self,
        conversation: AdapterConversationContext,
    ) -> Generator[None, None, None]:
        token = self._conversation_var.set(conversation)
        try:
            yield
        finally:
            self._conversation_var.reset(token)

    @staticmethod
    def subagent_run_kwargs(limits: SubagentUsageLimits) -> dict[str, Any]:
        return {"max_turns": limits.request_limit}

    def _model_for_run(self) -> str | None:
        return self.active_profile.model or self._model

    def _settings_for_run(self) -> dict[str, Any]:
        settings = {
            **dict(self._extra_kwargs),
            **dict(self._settings),
            **claude_agent_settings(self.active_profile.settings),
        }
        if self.active_profile.effort is not None:
            settings["effort"] = pydantic_effort_value(self.active_profile.effort)
        return settings

    def _options_for_run(
        self,
        sdk: Any,
        *,
        output_type: type[Any] | None,
        output_schema: dict[str, Any] | None,
        instructions: str | None,
        run_kwargs: Mapping[str, Any],
        tools: Sequence[ToolSpec] | None = None,
        codemode_settings: CodeModeSettings | None = None,
        codemode_preloaded_tools: Sequence[Mapping[str, Any]] = (),
    ) -> tuple[Any, str]:
        run_tools = tuple(self.scoped_tools if tools is None else tools)
        codemode_enabled = codemode_settings is not None and codemode_settings.enabled
        options_kwargs = {
            **self._settings_for_run(),
            **dict(run_kwargs),
        }
        compaction = self.active_profile.compaction
        if self.active_profile.history is True and compaction is not None:
            validate_claude_native_compaction(compaction)
        options_kwargs.setdefault("tools", dict(CLAUDE_CODE_TOOLS_PRESET))
        options_kwargs.setdefault("permission_mode", CLAUDE_CODE_PERMISSION_MODE)
        model = self._model_for_run()
        if model is not None:
            options_kwargs["model"] = model
        system_prompt = append_codemode_instructions(
            instructions or self.active_profile.system,
            enabled=codemode_enabled,
            preloaded_tools=codemode_preloaded_tools,
        )
        _set_claude_system_prompt(
            options_kwargs,
            system_prompt,
            codemode_enabled=codemode_enabled,
            codemode_preloaded_tools=codemode_preloaded_tools,
        )

        mcp_servers, allowed_tools = self._mcp_servers_for_run(
            sdk,
            tools=run_tools,
            include_external=not codemode_enabled,
        )
        if mcp_servers:
            existing_servers = options_kwargs.get("mcp_servers")
            if isinstance(existing_servers, Mapping):
                options_kwargs["mcp_servers"] = {
                    **dict(existing_servers),
                    **mcp_servers,
                }
            else:
                options_kwargs["mcp_servers"] = mcp_servers
        if allowed_tools:
            existing_allowed = options_kwargs.get("allowed_tools")
            if isinstance(existing_allowed, (list, tuple)):
                options_kwargs["allowed_tools"] = [*existing_allowed, *allowed_tools]
            else:
                options_kwargs["allowed_tools"] = allowed_tools
        session_fingerprint = _claude_session_fingerprint(
            options_kwargs,
            tools=run_tools,
            codemode_settings=codemode_settings,
            hook_identities=_active_claude_hook_identities(),
        )
        conversation = self._conversation_var.get()
        if conversation is not None:
            session_id = conversation.resume_state.get("claude_session_id")
            previous_fingerprint = conversation.resume_state.get("claude_session_fingerprint")
            if isinstance(session_id, str) and previous_fingerprint in {
                None,
                session_fingerprint,
            }:
                options_kwargs.setdefault("resume", session_id)
            elif isinstance(session_id, str) and isinstance(
                conversation, AdapterConversationContext
            ):
                conversation.cache_prefix_changed = True
        if output_schema is not None:
            options_kwargs["output_format"] = {
                "type": "json_schema",
                "schema": output_schema,
            }
        elif output_type is not None:
            options_kwargs["output_format"] = {
                "type": "json_schema",
                "schema": _output_json_schema(output_type),
            }
        options_kwargs["hooks"] = _with_kedi_approval_hook(
            sdk,
            options_kwargs.get("hooks"),
            profile=self.active_profile,
            tools=run_tools,
            approval_ceiling=self._approval_ceiling_var.get(),
            passthrough_tool_names=(
                frozenset(f"{_CLAUDE_KEDI_MCP_TOOL_PREFIX}{name}" for name in CONTROL_TOOL_NAMES)
                if codemode_enabled
                else frozenset()
            ),
        )
        options_cls = getattr(sdk, "ClaudeAgentOptions")
        return options_cls(**options_kwargs), session_fingerprint

    def _mcp_servers_for_run(
        self,
        sdk: Any,
        *,
        tools: Sequence[ToolSpec] | None = None,
        include_external: bool = True,
    ) -> tuple[dict[str, Any], list[str]]:
        run_tools = tuple(self.scoped_tools if tools is None else tools)
        mcp_servers = _claude_external_mcp_servers(self.active_profile) if include_external else {}
        allowed_tools: list[str] = []
        if run_tools:
            mcp_servers["kedi"] = _claude_sdk_mcp_server(sdk, run_tools)
            allowed_tools.extend(f"mcp__kedi__{tool.name}" for tool in run_tools)
        return mcp_servers, allowed_tools

    async def _projected_tools_for_run(
        self,
        stack: AsyncExitStack,
    ) -> tuple[CodeModeProjection, CodeModeSettings | None]:
        settings = normalize_codemode_settings(
            self.active_profile.codemode or getattr(self, "codemode", None)
        )
        if settings is None or not settings.enabled:
            return CodeModeProjection(tools=tuple(self.scoped_tools)), None

        mcp_tools = await materialize_mcp_tool_specs(
            stack,
            self.active_profile.mcp_servers,
            adapter_shortname=self.shortname,
        )
        projection = codemode_projection(
            (*self.scoped_tools, *mcp_tools),
            limits=CodeModeLimits(**settings.limit_values()),
            policies=approval_policies(
                self.active_profile.approval,
                self._approval_ceiling_var.get(),
            ),
            adapter_shortname=self.shortname,
            preload_tools=settings.preload_tools,
        )
        stack.push_async_callback(projection.close)
        return projection, settings
