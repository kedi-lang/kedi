from __future__ import annotations

import inspect
from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager, nullcontext
from contextvars import ContextVar
from typing import (
    TYPE_CHECKING,
    Any,
    Iterator,
    Mapping,
    Sequence,
    cast,
    overload,
)

from pydantic_ai import (
    Agent,
    AgentModelSettings,
    AgentRetries,
    AgentSpec,
    CancellationToken,
)
from pydantic_ai.agent.abstract import AgentInstructions, AgentMetadata, RunOutputDataT
from pydantic_ai.capabilities import (
    AgentCapability,
)
from pydantic_ai.messages import (
    ModelMessage,
    UserContent,
)
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.output import OutputDataT, OutputSpec
from pydantic_ai.run import AgentRun
from pydantic_ai.tools import (
    AgentDepsT,
    DeferredToolResults,
    ToolDefinition,
)
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import RunUsage, UsageLimits

from kedi.agent_adapter.adapters.pydantic import hooks as _hooks
from kedi.agent_adapter.adapters.pydantic import messages as _messages
from kedi.agent_adapter.adapters.pydantic import run_config as _run_config
from kedi.agent_adapter.adapters.pydantic import tools as _tools
from kedi.agent_adapter.adapters.pydantic.capabilities import (
    _KediAgentEventCapability,
    _KediDeferredApprovalCapability,
    _KediExternalApprovalCapability,
)
from kedi.agent_adapter.adapters.pydantic.compat import (
    application_toolset_count,
    has_application_tools,
    infer_agent_name,
)
from kedi.agent_adapter.adapters.pydantic.run_config import (
    _responses_session_model,
    _with_capability,
)
from kedi.agent_adapter.adapters.pydantic.tools import (
    _DEFAULT_TOOL_RETRIES,
    PydanticApprovalResolution,
    _external_tool_spec,
    pydantic_tool_from_spec,
)
from kedi.agent_adapter.approvals.policy import ApprovalPolicy
from kedi.agent_adapter.compaction.archive import HistoryArchiveSettings
from kedi.agent_adapter.conversation import AdapterConversationContext
from kedi.agent_adapter.events import agent_event_run, current_agent_run_identity
from kedi.agent_adapter.model_resolution import resolve_pydantic_model
from kedi.agent_adapter.history_processing import (
    HistoryProcessorCallback,
    HistoryProcessorRunner,
)
from kedi.agent_adapter.hooks import (
    HookHandler,
    HookRegistrationMixin,
    ToolOrigin,
    hooked_agent_run,
    native_agent_hook_run,
)
from kedi.agent_adapter.invocation import InvocationUsage, invocation_observer, report_invocation
from kedi.agent_adapter.meta import AdapterCapabilities, AgentAdapterKind, HookCapabilities
from kedi.agent_adapter.models import (
    close_typesafe_model_current,
    codex_responses_model,
    is_codex_model_id,
    is_typesafe_model_id,
    typesafe_pydantic_model,
)
from kedi.agent_adapter.output_schema import build_output_model
from kedi.agent_adapter.responses_session import model_responses_session as _model_responses_session
from kedi.agent_adapter.responses_session import model_responses_turn as _model_responses_turn
from kedi.agent_adapter.subagents import SubagentConversationContext, SubagentExecutionRequest
from kedi.agent_adapter.subagents.attempts import DelegationFailurePolicy, delegation_scope
from kedi.agent_adapter.subagents.limits import SubagentUsageLimits
from kedi.agent_adapter.tools.execution import (
    with_required_before_output,
)
from kedi.agent_adapter.tools.scope import (
    ToolScope,
    required_tool_call_scope,
)
from kedi.agent_adapter.tools.specs import ToolSpec
from kedi.agent_profile import AgentProfile, CodeModeSettings, normalize_codemode_settings
from kedi.agent_profile.settings import (
    KEDI_RUNTIME_SETTING_KEYS,
    PYDANTIC_MODEL_SETTING_KEYS,
    tool_reason_enabled,
)
from kedi.model_normalization import normalize_for_pydantic_ai
from kedi.resolved_types import ResolvedKediType

if TYPE_CHECKING:
    from pydantic_ai.tools import Tool as PydanticTool


class PydanticAdapter(HookRegistrationMixin, Agent[AgentDepsT, OutputDataT]):
    """
    Adapter to use Pydantic AI agents.
    """

    kind: AgentAdapterKind = "agent-framework"
    shortname = "pydantic"
    supports_native_subagent_resume = True
    approval_resolution: PydanticApprovalResolution
    subagent_approval_resolution: PydanticApprovalResolution | None
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
        settings=PYDANTIC_MODEL_SETTING_KEYS | KEDI_RUNTIME_SETTING_KEYS,
    )

    def native_telemetry_enabled(self, operation: str) -> bool:
        """Report per-agent native ownership without relying on process-global state."""

        return operation in {"run_agent", "chat", "call_tool"} and self.instrument is not False

    def supports_artifact_contract(self, profile: AgentProfile) -> bool:
        """Return whether the active model can receive artifact management tools."""

        if profile.model is not None:
            return not is_typesafe_model_id(profile.model)
        model = getattr(self, "_agent_model_arg", None)
        return getattr(model, "system", None) != "typesafe"

    def __init__(
        self,
        model: str | Model | None = None,
        *,
        model_settings: dict[str, Any] | None = None,
        approval_resolution: PydanticApprovalResolution = "kedi",
        subagent_approval_resolution: PydanticApprovalResolution | None = None,
        subagent_failure_policy: DelegationFailurePolicy = "fail_closed",
        codemode: bool | CodeModeSettings = False,
        query_artifact: bool = False,
        tool_reason: bool = False,
        hook_handler: HookHandler | None = None,
        default_usage_limits: UsageLimits | None = None,
        history_archive: HistoryArchiveSettings | None = None,
        history_processor: HistoryProcessorCallback[ModelMessage] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the native Pydantic AI adapter.

        ``model`` may be LiteLLM-style (``vendor/a/b``) or Pydantic-style
        (``vendor:a/b``). ``approval_resolution='kedi'`` resolves deferred tool
        calls with the active Kedi policy. ``'external'`` keeps those calls
        deferred for an outer controller such as an ACP runtime.
        """
        self._init_hook_registration(hook_handler)
        if not isinstance(query_artifact, bool):
            raise TypeError("query_artifact must be a boolean")
        self.query_artifact = query_artifact
        if not isinstance(tool_reason, bool):
            raise TypeError("tool_reason must be a boolean")
        self.tool_reason = tool_reason
        self._tool_reasons_var: ContextVar[dict[str, str | None]] = ContextVar(
            f"kedi_pydantic_tool_reasons_{id(self)}", default={}
        )
        self._tool_scope = ToolScope()
        self._tool_spec_scope = ToolScope()
        self._profile_stack_var: ContextVar[tuple[AgentProfile, ...]] = ContextVar(
            f"kedi_pydantic_profile_stack_{id(self)}",
            default=(),
        )
        self._isolated_run_depth: ContextVar[int] = ContextVar(
            f"kedi_pydantic_isolated_run_depth_{id(self)}",
            default=0,
        )
        self._approval_ceiling_var: ContextVar[tuple[ApprovalPolicy | None, ...]] = ContextVar(
            f"kedi_pydantic_approval_ceiling_{id(self)}",
            default=(),
        )
        self._approval_resolution_var: ContextVar[PydanticApprovalResolution | None] = ContextVar(
            f"kedi_pydantic_approval_resolution_{id(self)}",
            default=None,
        )
        self._run_tool_specs_var: ContextVar[tuple[ToolSpec, ...] | None] = ContextVar(
            f"kedi_pydantic_run_tool_specs_{id(self)}",
            default=None,
        )
        self._hook_arguments_var: ContextVar[dict[str, dict[str, Any]]] = ContextVar(
            f"kedi_pydantic_hook_arguments_{id(self)}",
            default={},
        )
        self._hook_tools_var: ContextVar[dict[str, tuple[ToolSpec, ToolOrigin]]] = ContextVar(
            f"kedi_pydantic_hook_tools_{id(self)}",
            default={},
        )
        self._hook_started_var: ContextVar[dict[str, float]] = ContextVar(
            f"kedi_pydantic_hook_started_{id(self)}",
            default={},
        )
        self._subagent_conversation_var: ContextVar[
            AdapterConversationContext | SubagentConversationContext | None
        ] = ContextVar(
            f"kedi_pydantic_subagent_conversation_{id(self)}",
            default=None,
        )
        if subagent_failure_policy not in {"recover", "fail_closed"}:
            raise ValueError(f"Unknown subagent failure policy: {subagent_failure_policy!r}")
        if approval_resolution not in {"kedi", "external"}:
            raise ValueError(f"Unknown approval resolution owner: {approval_resolution!r}")
        if subagent_approval_resolution not in {None, "kedi", "external"}:
            raise ValueError(
                f"Unknown subagent approval resolution owner: {subagent_approval_resolution!r}"
            )
        self.approval_resolution = approval_resolution
        self.subagent_approval_resolution = subagent_approval_resolution
        self._subagent_failure_policy = subagent_failure_policy
        self._default_usage_limits = default_usage_limits
        self.codemode = normalize_codemode_settings(codemode)
        self._model_overrides: dict[str, Model] = {}
        logical_model = model if isinstance(model, str) else None
        if isinstance(model, str):
            model = self._resolve_model(model)
        self._agent_model_arg = model
        self._default_model = logical_model
        if model_settings:
            # Native constructor settings include provider-specific keys, unlike DSL settings.
            kwargs["model_settings"] = dict(model_settings)
        kwargs.setdefault("retries", {"tools": _DEFAULT_TOOL_RETRIES})
        self._history_processor_runner: HistoryProcessorRunner[ModelMessage] | None = None
        if history_processor is not None:
            from kedi.agent_adapter.adapters.pydantic.history_processing import (
                PydanticHistoryMessageCodec,
                reject_conflicting_history_capabilities,
            )

            reject_conflicting_history_capabilities(kwargs.get("capabilities"))
            self._history_processor_runner = HistoryProcessorRunner(
                history_processor,
                PydanticHistoryMessageCodec(),
            )
        if history_archive is not None:
            from kedi.agent_adapter.compaction.pydantic_archive import (
                PydanticArtifactHistoryCapability,
            )

            kwargs["capabilities"] = _with_capability(
                kwargs.get("capabilities"),
                PydanticArtifactHistoryCapability,
                lambda: PydanticArtifactHistoryCapability(history_archive),
            )
        if self._history_processor_runner is not None:
            from kedi.agent_adapter.adapters.pydantic.history_processing import (
                PydanticHistoryProcessingCapability,
            )

            kwargs["capabilities"] = _with_capability(
                kwargs.get("capabilities"),
                PydanticHistoryProcessingCapability,
                lambda: PydanticHistoryProcessingCapability(
                    cast(
                        "HistoryProcessorRunner[ModelMessage]",
                        self._history_processor_runner,
                    )
                ),
            )
        self._agent_init_kwargs = dict(kwargs)
        super().__init__(model, **kwargs)

    @staticmethod
    def _validate_required_tool_calls(output: Any) -> Any:
        return _tools._validate_required_tool_calls(output)

    @contextmanager
    def tool_scope(self, tools: Sequence[Any] = ()) -> Iterator[None]:
        """Register Pydantic tools and their Kedi metadata in one lexical scope."""
        with (
            self._tool_scope.tool_scope(),
            self._tool_spec_scope_for_current_adapter().tool_scope(),
        ):
            for tool in tools:
                self.register_tool(cast(ToolSpec, tool))
            yield

    def register_tool(self, tool: ToolSpec) -> None:
        tool = with_required_before_output(tool)
        native_tool = pydantic_tool_from_spec(tool)
        self._tool_scope.register_tool(native_tool)
        self._tool_spec_scope_for_current_adapter().register_tool(tool)

    @property
    def scoped_tools(self) -> tuple[PydanticTool[Any], ...]:
        return cast("tuple[PydanticTool[Any], ...]", self._tool_scope.scoped_tools)

    @property
    def scoped_tool_specs(self) -> tuple[ToolSpec, ...]:
        return cast(
            "tuple[ToolSpec, ...]", self._tool_spec_scope_for_current_adapter().scoped_tools
        )

    @contextmanager
    def isolated_run(self) -> Iterator[None]:
        """Run a nested agent with only the tools registered in the current scope."""
        depth = self._isolated_run_depth_for_current_adapter()
        token = depth.set(depth.get() + 1)
        try:
            yield
        finally:
            depth.reset(token)

    @contextmanager
    def subagent_scope(self, request: SubagentExecutionRequest) -> Iterator[None]:
        token = self._subagent_conversation_var.set(request.conversation)
        approval_token = self._approval_resolution_var.set(
            self.subagent_approval_resolution or self.approval_resolution
        )
        try:
            with (
                self.tool_scope(request.tools),
                self.profile_override(request.profile),
                self.approval_ceiling(request.approval_ceiling),
                self.isolated_run(),
            ):
                yield
        finally:
            self._approval_resolution_var.reset(approval_token)
            self._subagent_conversation_var.reset(token)

    @property
    def active_approval_resolution(self) -> PydanticApprovalResolution:
        """Return the task-local approval owner for the active agent run."""

        resolution_var = getattr(self, "_approval_resolution_var", None)
        active = resolution_var.get() if resolution_var is not None else None
        return active or getattr(self, "approval_resolution", "kedi")

    @contextmanager
    def conversation_scope(
        self,
        conversation: AdapterConversationContext,
    ) -> Iterator[None]:
        token = self._subagent_conversation_var.set(conversation)
        try:
            yield
        finally:
            self._subagent_conversation_var.reset(token)

    @asynccontextmanager
    async def responses_session(self) -> AsyncGenerator[None]:
        """Own one Responses connection across sequential runs in the current task."""
        model = _responses_session_model(self, None)
        if model is None:
            raise TypeError("The active model does not support Responses sessions.")
        async with _model_responses_session(model):
            yield

    @staticmethod
    def subagent_run_kwargs(limits: SubagentUsageLimits) -> dict[str, Any]:
        return {"usage_limits": limits.to_pydantic()}

    @hooked_agent_run("template")
    async def produce_subagent(
        self,
        *,
        template: str,
        output_schema: dict[str, Any],
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        from pydantic_ai.output import StructuredDict

        result = await self.produce(
            template=template,
            output_type=cast(Any, StructuredDict(output_schema)),
            **kwargs,
        )
        if not isinstance(result, Mapping):
            raise TypeError("Pydantic subagent structured output must be an object")
        return cast(Mapping[str, Any], result)

    @contextmanager
    def approval_ceiling(
        self,
        policies: Sequence[ApprovalPolicy | None],
    ) -> Iterator[None]:
        stack_var = self._approval_ceiling_for_current_adapter()
        token = stack_var.set(tuple(policies))
        try:
            yield
        finally:
            stack_var.reset(token)

    @property
    def active_profile(self) -> AgentProfile:
        stack_var = getattr(self, "_profile_stack_var", None)
        stack = stack_var.get() if stack_var is not None else ()
        return stack[-1] if stack else AgentProfile.empty()

    @property
    def active_tool_reason(self) -> bool:
        return tool_reason_enabled(
            self.active_profile.settings, default=getattr(self, "tool_reason", False)
        )

    @staticmethod
    def type_builder(output_schema: dict[str, ResolvedKediType]) -> type[OutputDataT]:
        """
        Type builder to create a Pydantic model from a given output schema.
        Args:
            output_schema (dict): A dictionary defining the output schema.
        Returns:
            type: A Pydantic model class based on the provided schema.
        """
        return cast(
            type[OutputDataT],
            build_output_model(output_schema),
        )

    @hooked_agent_run("template")
    async def produce(
        self,
        template: str,
        output_schema: dict[str, ResolvedKediType] | None = None,
        output_type: type[OutputDataT] | None = None,
        **kwargs,
    ) -> OutputDataT:
        """
        Invoke the Pydantic AI agent with the provided template and output schema.
        """
        if not output_schema and output_type is None:
            raise ValueError("Output spec must be provided.")
        if output_schema:
            output_type = self.type_builder(output_schema)
        agent = self._agent_for_produce()
        run_kwargs: dict[str, object] = {"output_type": output_type, **kwargs}
        self._apply_default_usage_limits(run_kwargs)
        tool_specs = self._tool_specs_for_run(run_kwargs)
        if agent is not self:
            run_kwargs = self._run_kwargs(run_kwargs, tool_specs=tool_specs)
            run_kwargs.update(self._profile_override_kwargs(self.active_profile))
            self._add_nested_native_toolsets(run_kwargs)
        result = await self._run_agent(
            agent,
            template,
            run_kwargs,
            tool_specs=tool_specs,
            emit_final=False,
        )
        _messages._capture_typesafe_decision_evidence(result)
        return result.output

    @hooked_agent_run("prompt")
    async def invoke(self, *, prompt: str, **kwargs: Any) -> str:
        run_kwargs: dict[str, object] = dict(kwargs)
        self._apply_default_usage_limits(run_kwargs)
        agent = self._agent_for_produce()
        tool_specs = self._tool_specs_for_run(run_kwargs)
        if agent is not self:
            run_kwargs = self._run_kwargs(run_kwargs, tool_specs=tool_specs)
            run_kwargs.update(self._profile_override_kwargs(self.active_profile))
            self._add_nested_native_toolsets(run_kwargs)
        return str(
            (
                await self._run_agent(
                    agent,
                    prompt,
                    run_kwargs,
                    tool_specs=tool_specs,
                    emit_final=True,
                )
            ).output
        )

    @overload
    def iter(
        self,
        user_prompt: str | Sequence[UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: Sequence[ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        run_id: str | None = None,
        model: Model | KnownModelName | str | None = None,
        instructions: AgentInstructions[AgentDepsT] = None,
        deps: AgentDepsT | None = None,
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        usage_limits: UsageLimits | None = None,
        cancellation_token: CancellationToken | None = None,
        usage: RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        retries: int | AgentRetries | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> AbstractAsyncContextManager[AgentRun[AgentDepsT, OutputDataT]]: ...

    @overload
    def iter(
        self,
        user_prompt: str | Sequence[UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: Sequence[ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        run_id: str | None = None,
        model: Model | KnownModelName | str | None = None,
        instructions: AgentInstructions[AgentDepsT] = None,
        deps: AgentDepsT | None = None,
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        usage_limits: UsageLimits | None = None,
        cancellation_token: CancellationToken | None = None,
        usage: RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        retries: int | AgentRetries | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> AbstractAsyncContextManager[AgentRun[AgentDepsT, RunOutputDataT]]: ...

    def iter(
        self,
        user_prompt: str | Sequence[UserContent] | None = None,
        *,
        output_type: OutputSpec[Any] | None = None,
        message_history: Sequence[ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        run_id: str | None = None,
        model: Model | KnownModelName | str | None = None,
        instructions: AgentInstructions[AgentDepsT] = None,
        deps: AgentDepsT | None = None,
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        usage_limits: UsageLimits | None = None,
        cancellation_token: CancellationToken | None = None,
        usage: RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        retries: int | AgentRetries | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> AbstractAsyncContextManager[AgentRun[AgentDepsT, Any]]:
        """Apply active Kedi state to every native Pydantic AI run surface."""

        if infer_name and self.name is None:
            infer_agent_name(self, inspect.currentframe())
        return self._iter_with_kedi_state(
            user_prompt,
            {
                "output_type": output_type,
                "message_history": message_history,
                "deferred_tool_results": deferred_tool_results,
                "conversation_id": conversation_id,
                "run_id": run_id,
                "model": model,
                "instructions": instructions,
                "deps": deps,
                "model_settings": model_settings,
                "usage_limits": usage_limits,
                "cancellation_token": cancellation_token,
                "usage": usage,
                "metadata": metadata,
                "retries": retries,
                "infer_name": False,
                "toolsets": toolsets,
                "capabilities": capabilities,
                "spec": spec,
            },
        )

    @asynccontextmanager
    async def _iter_with_kedi_state(
        self,
        user_prompt: str | Sequence[UserContent] | None,
        kwargs: Mapping[str, Any],
    ) -> AsyncGenerator[AgentRun[AgentDepsT, Any]]:
        async with native_agent_hook_run(self, user_prompt) as hooked_prompt:
            tool_specs = self._tool_specs_for_run(kwargs)
            run_kwargs = self._run_kwargs(kwargs, tool_specs=tool_specs)
            conversation = self._subagent_conversation_var.get()
            event_scope = (
                nullcontext(None)
                if current_agent_run_identity() is not None
                else agent_event_run(self.shortname)
            )
            with (
                (
                    self._history_processor_runner.run_scope(
                        conversation
                        if isinstance(conversation, AdapterConversationContext)
                        else None,
                        reuse_existing=True,
                    )
                    if self._history_processor_runner is not None
                    else nullcontext()
                ),
                self._run_tool_spec_scope(tool_specs),
                delegation_scope(self._subagent_failure_policy),
                event_scope as event_resolver,
            ):
                if event_resolver is not None:
                    capabilities = run_kwargs.get("capabilities")
                    capability_list = list(cast("Sequence[Any]", capabilities or ()))
                    capability_list.append(_KediAgentEventCapability(event_resolver))
                    run_kwargs["capabilities"] = capability_list
                selected_model = _responses_session_model(
                    self,
                    cast("Model | str | None", run_kwargs.get("model")),
                )
                async with (
                    _model_responses_turn(selected_model),
                    super().iter(hooked_prompt, **cast(Any, run_kwargs)) as agent_run,
                ):
                    yield agent_run
                    if agent_run.result is not None:
                        self._capture_conversation_messages(conversation, agent_run)
                        if event_resolver is not None:
                            event_resolver.complete_run(final_text=str(agent_run.result.output))

    async def _run_agent(
        self,
        agent: Agent[Any, Any],
        prompt: str,
        run_kwargs: dict[str, object],
        *,
        tool_specs: Sequence[ToolSpec] | None = None,
        emit_final: bool = False,
    ) -> Any:
        conversation = self._subagent_conversation_var.get()
        self._apply_conversation_history(run_kwargs, conversation)
        if tool_specs is None:
            tool_specs = self._tool_specs_for_run(run_kwargs)
        with (
            (
                self._history_processor_runner.run_scope(
                    conversation if isinstance(conversation, AdapterConversationContext) else None
                )
                if self._history_processor_runner is not None
                else nullcontext()
            ),
            self._run_tool_spec_scope(tool_specs),
            delegation_scope(self._subagent_failure_policy),
            agent_event_run(self.shortname) as event_resolver,
        ):
            if event_resolver is not None:
                capabilities = run_kwargs.get("capabilities")
                if capabilities is None:
                    capability_list: list[Any] = []
                elif isinstance(capabilities, (list, tuple)):
                    capability_list = list(capabilities)
                else:
                    raise TypeError("Pydantic AI capabilities must be a list or tuple.")
                capability_list.append(_KediAgentEventCapability(event_resolver))
                run_kwargs["capabilities"] = capability_list

            observer = invocation_observer()
            usage: RunUsage | None = None
            if observer is not None:
                usage_value = run_kwargs.get("usage")
                if isinstance(usage_value, RunUsage):
                    usage = usage_value
                else:
                    usage = RunUsage()
                    run_kwargs["usage"] = usage
            try:
                selected_model = _responses_session_model(
                    agent,
                    cast("Model | str | None", run_kwargs.get("model")),
                )
                # This adapter's native iter() owns the turn when agent is self.
                model_scope = _model_responses_session if agent is self else _model_responses_turn
                async with model_scope(selected_model):
                    result = await agent.run(prompt, **cast(Any, run_kwargs))
                self._capture_conversation_messages(conversation, result)
            finally:
                if usage is not None:
                    report_invocation(
                        InvocationUsage(
                            requests=usage.requests,
                            tool_calls=usage.tool_calls,
                            input_tokens=usage.input_tokens,
                            cache_read_tokens=usage.cache_read_tokens,
                            cache_write_tokens=usage.cache_write_tokens,
                            output_tokens=usage.output_tokens,
                            total_tokens=usage.total_tokens,
                            cost_usd=usage.cost,
                        )
                    )
            if event_resolver is not None:
                event_resolver.complete_run(final_text=str(result.output) if emit_final else None)
            return result

    @contextmanager
    def _required_tool_call_scope(self) -> Iterator[None]:
        tool_specs = self._run_tool_specs_var.get()
        with required_tool_call_scope(
            self.scoped_tool_specs if tool_specs is None else tool_specs,
            inherit=True,
        ):
            yield

    @contextmanager
    def _run_tool_spec_scope(self, tool_specs: Sequence[ToolSpec]) -> Iterator[None]:
        yield from _hooks._run_tool_spec_scope(self, tool_specs)

    @staticmethod
    def _capture_conversation_messages(
        conversation: AdapterConversationContext | SubagentConversationContext | None,
        result: Any,
    ) -> None:
        return _messages._capture_conversation_messages(conversation, result)

    @staticmethod
    def _apply_conversation_history(
        run_kwargs: dict[str, object],
        conversation: AdapterConversationContext | SubagentConversationContext | None,
    ) -> None:
        return _messages._apply_conversation_history(run_kwargs, conversation)

    def invoke_sync(self, *, prompt: str, **kwargs: Any) -> str:
        from kedi.agent_adapter.execution import run_sync

        model = self._model_for_sync_cleanup()

        async def invoke_and_close() -> str:
            try:
                return await self.invoke(prompt=prompt, **kwargs)
            finally:
                await close_typesafe_model_current(model)

        return run_sync(invoke_and_close())

    def _run_kwargs(
        self,
        kwargs: Mapping[str, Any],
        *,
        tool_specs: Sequence[ToolSpec],
    ) -> dict[str, object]:
        return _run_config._run_kwargs(self, kwargs, tool_specs=tool_specs)

    def _apply_default_usage_limits(self, run_kwargs: dict[str, object]) -> None:
        return _run_config._apply_default_usage_limits(self, run_kwargs)

    def _with_telemetry_capability(self, existing: object | None) -> list[Any]:
        return _run_config._with_telemetry_capability(self, existing)

    def _agent_for_produce(self) -> Agent[Any, Any]:
        return _run_config._agent_for_produce(self)

    def _add_nested_native_toolsets(self, run_kwargs: dict[str, object]) -> None:
        return _tools._add_nested_native_toolsets(self, run_kwargs)

    def _build_toolset_list(
        self, cap_toolsets: Sequence[Any] | None = None, *, ignore_overrides: bool = False
    ) -> list[Any]:
        """Require native approval for every application-provided Pydantic toolset."""
        return _tools._build_toolset_list(self, cap_toolsets, ignore_overrides=ignore_overrides)

    def _application_toolset_count(self) -> int:
        """Count function, user, and dynamic toolsets without capability toolsets."""

        return application_toolset_count(self)

    def _has_native_approval_tools(self) -> bool:
        """Whether constructor or inherited Pydantic toolsets can issue a call."""

        return has_application_tools(self)

    def _toolsets_for_run(self, *, include_scoped_tools: bool = True) -> list[Any] | None:
        return _tools._toolsets_for_run(self, include_scoped_tools=include_scoped_tools)

    def _tool_specs_for_run(self, run_kwargs: Mapping[str, Any]) -> tuple[ToolSpec, ...]:
        return _tools._tool_specs_for_run(self, run_kwargs)

    def _with_deferred_approval_capability(self, existing: object | None) -> list[Any]:
        return _with_capability(
            existing,
            _KediDeferredApprovalCapability,
            lambda: _KediDeferredApprovalCapability(handler=self._handle_deferred_tool_calls),
        )

    def _with_external_approval_capability(self, existing: object | None) -> list[Any]:
        return _with_capability(
            existing,
            _KediExternalApprovalCapability,
            lambda: _KediExternalApprovalCapability(self),
        )

    def _tool_spec_for_call(self, tool_name: str) -> ToolSpec:
        active_tool_specs = self._run_tool_specs_var.get()
        known_tools = {
            tool.name: tool
            for tool in (self.scoped_tool_specs if active_tool_specs is None else active_tool_specs)
        }
        return known_tools.get(tool_name, _external_tool_spec(tool_name))

    def _hook_tool_for_call(
        self,
        tool_name: str,
        tool_def: ToolDefinition,
    ) -> tuple[ToolSpec, ToolOrigin]:
        return _hooks._hook_tool_for_call(self, tool_name, tool_def)

    def _remember_hook_tool(
        self,
        tool_call_id: str,
        tool: ToolSpec,
        origin: ToolOrigin,
    ) -> None:
        return _hooks._remember_hook_tool(self, tool_call_id, tool, origin)

    def _hook_tool_for_call_id(
        self,
        tool_call_id: str,
        fallback: tuple[ToolSpec, ToolOrigin],
        *,
        consume: bool = False,
    ) -> tuple[ToolSpec, ToolOrigin]:
        return _hooks._hook_tool_for_call_id(self, tool_call_id, fallback, consume=consume)

    def _hook_tools_for_current_adapter(
        self,
    ) -> ContextVar[dict[str, tuple[ToolSpec, ToolOrigin]]]:
        return _hooks._hook_tools_for_current_adapter(self)

    def _remember_hook_arguments(
        self,
        tool_call_id: str,
        arguments: Mapping[str, Any],
    ) -> None:
        return _hooks._remember_hook_arguments(self, tool_call_id, arguments)

    def _hook_arguments_for_call(
        self,
        tool_call_id: str,
        fallback: Mapping[str, Any],
        *,
        consume: bool = False,
    ) -> dict[str, Any]:
        return _hooks._hook_arguments_for_call(self, tool_call_id, fallback, consume=consume)

    def _hook_arguments_for_current_adapter(
        self,
    ) -> ContextVar[dict[str, dict[str, Any]]]:
        return _hooks._hook_arguments_for_current_adapter(self)

    def _remember_hook_started(self, tool_call_id: str) -> None:
        return _hooks._remember_hook_started(self, tool_call_id)

    def _consume_hook_started(self, tool_call_id: str) -> float | None:
        return _hooks._consume_hook_started(self, tool_call_id)

    def _hook_started_for_current_adapter(self) -> ContextVar[dict[str, float]]:
        return _hooks._hook_started_for_current_adapter(self)

    async def _handle_deferred_tool_calls(self, _context: Any, requests: Any) -> Any:
        return await _tools._handle_deferred_tool_calls(self, _context, requests)

    def _tool_spec_scope_for_current_adapter(self) -> ToolScope:
        try:
            return self._tool_spec_scope
        except AttributeError:
            scope = ToolScope()
            self._tool_spec_scope = scope
            return scope

    def _isolated_run_depth_for_current_adapter(self) -> ContextVar[int]:
        try:
            return self._isolated_run_depth
        except AttributeError:
            depth = ContextVar(
                f"kedi_pydantic_isolated_run_depth_{id(self)}",
                default=0,
            )
            self._isolated_run_depth = depth
            return depth

    def _approval_ceiling_for_current_adapter(
        self,
    ) -> ContextVar[tuple[ApprovalPolicy | None, ...]]:
        try:
            return self._approval_ceiling_var
        except AttributeError:
            ceiling: ContextVar[tuple[ApprovalPolicy | None, ...]] = ContextVar(
                f"kedi_pydantic_approval_ceiling_{id(self)}",
                default=(),
            )
            self._approval_ceiling_var = ceiling
            return ceiling

    def _mcp_toolsets_for_profile(self, profile: AgentProfile) -> list[Any]:
        return _tools._mcp_toolsets_for_profile(self, profile)

    def produce_sync(
        self,
        template: str,
        output_schema: dict[str, ResolvedKediType] | None = None,
        output_type: type[OutputDataT] | None = None,
        **kwargs,
    ) -> OutputDataT:
        """
        Synchronous version of produce method.
        """
        from kedi.agent_adapter.execution import run_sync

        model = self._model_for_sync_cleanup()

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

    def _model_for_sync_cleanup(self) -> object:
        profile = self.active_profile
        if profile.model is not None:
            return self._resolve_model(profile.model)
        return self._agent_model_arg

    def default_model(self) -> str | None:
        default_model = getattr(self, "_default_model", None)
        if isinstance(default_model, str):
            return default_model
        model = getattr(self, "_model", None)
        return model if isinstance(model, str) else None

    def telemetry_model_name(self) -> str | None:
        if isinstance(self._agent_model_arg, str):
            return self._agent_model_arg
        if isinstance(self._agent_model_arg, Model):
            return self._agent_model_arg.model_name
        return None

    @contextmanager
    def profile_override(self, profile: AgentProfile) -> Iterator[None]:
        override_kwargs = self._profile_override_kwargs(profile)
        stack_var = self._profile_stack_var
        token = stack_var.set((*stack_var.get(), profile))
        try:
            if not override_kwargs:
                yield
                return
            with self.override(**override_kwargs):
                yield
        finally:
            stack_var.reset(token)

    def _profile_override_kwargs(self, profile: AgentProfile) -> dict[str, Any]:
        override_kwargs: dict[str, Any] = {}
        if profile.needs_model_override(self.default_model()):
            if profile.model is None:
                raise RuntimeError("Profile requires a model override but no model is set.")
            override_kwargs["model"] = self._resolve_model(profile.model)
        return override_kwargs

    def _resolve_model(self, model: str) -> str | Model:
        if is_typesafe_model_id(model):
            resolved = self._model_overrides.get(model)
            if resolved is None:
                resolved = typesafe_pydantic_model(model)
                self._model_overrides[model] = resolved
            return resolved
        if not is_codex_model_id(model):
            return resolve_pydantic_model(normalize_for_pydantic_ai(model))
        resolved = self._model_overrides.get(model)
        if resolved is None:
            resolved = cast(Model, codex_responses_model(model, adapter="pydantic"))
            self._model_overrides[model] = resolved
        return resolved
