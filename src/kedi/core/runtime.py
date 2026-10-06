from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from pathlib import Path
from threading import Lock, get_ident
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    cast,
)

from kedi.core import conversation as _conversation
from kedi.core import expressions as _expressions
from kedi.core import model_calls as _model_calls
from kedi.core import tool_binding as _tool_binding
from kedi.core import tracing as _tracing
from kedi.core.engines import _default_engine
from kedi.core.expressions import CallExpr, Expression, InputExpr, OutputExpr, ValueEnv, c, i, o
from kedi.core.scopes import ProcessingScope
from kedi.errors import KediExecutionError, KediPythonOrigin, KediTraceFrame
from kedi.executors.base import DefaultExecutor, Executor
from kedi.promise import KediPromise, force
from kedi.run_budget import RunBudget

if TYPE_CHECKING:
    from kedi.agent_adapter import AgentAdapter
    from kedi.agent_adapter.conversation import ConversationState, ConversationTurnTicket
    from kedi.agent_adapter.request import (
        AgentRunRequest,
        InstructionPart,
        OutputContract,
        PromptDraft,
    )
    from kedi.agent_adapter.subagents import DelegationResult
    from kedi.agent_adapter.subagents.budget import DelegationBudget, DelegationConcurrency
    from kedi.agent_adapter.subagents.limits import SubagentUsageLimits
    from kedi.agent_adapter.tools.specs import ToolSpec
    from kedi.agent_profile import AgentProfile
    from kedi.agent_surface import AgentSurface
    from kedi.artifacts import ArtifactManager
    from kedi.lang.ast import EvalSuite, TestSuite

DEFAULT_LOOP_ITERATION_LIMIT = 10_000


class KediRuntime:
    """Isolated kedi runtime with its own registry and environment."""

    def __init__(
        self,
        *,
        adapter: AgentAdapter[Any] | None = None,
        executor: Executor | None = None,
        engine: Any | None = None,
        default_agent_profile: Any | None = None,
        subagent_max_depth: int = 5,
        subagent_max_concurrency: int = 4,
        subagent_timeout_seconds: float | None = 120.0,
        subagent_usage_limits: "SubagentUsageLimits | None" = None,
        subagent_state_path: str | Path | None = None,
        conversation_state: "ConversationState | None" = None,
        loop_iteration_limit: int = DEFAULT_LOOP_ITERATION_LIMIT,
        run_budget: RunBudget | None = None,
    ) -> None:
        from kedi.agent_profile import AgentProfile
        from kedi.artifacts import ArtifactManager

        self.adapter = adapter
        self.run_budget = run_budget
        self.executor = executor or DefaultExecutor()
        self._engine = engine if engine is not None else _default_engine()
        self._selected_adapters: dict[tuple[str, str | None], AgentAdapter[Any]] = {}
        # Outstanding produce Futures are owned by one outer invocation. Concurrent runs on the
        # same runtime must never drain or surface one another's work.
        self._outstanding: dict[object, list[Any]] = {}
        self._outstanding_lock = Lock()
        self._origin_counter_lock = Lock()
        self._conversation_state_lock = Lock()
        self._lifecycle_lock = Lock()
        self._closed = False
        self._processing_scope: ContextVar[ProcessingScope | None] = ContextVar(
            "kedi_processing_scope", default=None
        )
        self._root_procedures: Dict[str, Callable[..., Any]] = {}
        self._main: Optional[Callable[..., Any]] = None
        self._import_env: ValueEnv = {}
        self._current_env: ContextVar[Optional[ValueEnv]] = ContextVar(
            "kedi_current_env", default=None
        )
        self._procedure_env_observer: ContextVar[Callable[[ValueEnv], None] | None] = ContextVar(
            "kedi_procedure_env_observer",
            default=None,
        )
        self._trace_stack: ContextVar[Tuple[KediTraceFrame, ...]] = ContextVar(
            "kedi_trace_stack", default=()
        )
        self._subagent_stack: ContextVar[tuple[tuple[str, str], ...]] = ContextVar(
            "kedi_subagent_stack", default=()
        )
        # Tracks procedure-call nesting so the end-of-run drain fires exactly once, at the
        # outermost invocation, regardless of entry point (run_main / query / direct invoke).
        self._invoke_depth: ContextVar[int] = ContextVar("kedi_invoke_depth", default=0)
        self._invocation_id: ContextVar[object | None] = ContextVar(
            "kedi_invocation_id",
            default=None,
        )
        self._native_subagent_owner: ContextVar[Any | None] = ContextVar(
            "kedi_native_subagent_owner", default=None
        )
        from kedi.agent_adapter.request_budget import ModelRequestBudget

        self._request_budget: ContextVar[ModelRequestBudget | None] = ContextVar(
            "kedi_model_request_budget",
            default=None,
        )
        self._initial_globals_env: ValueEnv = {}
        self._globals_env: ValueEnv = {}
        self._python_origin_counter = 0
        self._python_origins: Dict[str, KediPythonOrigin] = {}
        self._python_origin_keys: Dict[tuple[Any, ...], str] = {}
        self._program: Any | None = None
        self._source_map: Any | None = None
        self._source_text: str | None = None
        self._root_type_env_stack: list[dict[str, Any]] = []
        self._profile_registry: dict[str, Any] = {}
        self.default_agent_profile = (
            default_agent_profile
            if isinstance(default_agent_profile, AgentProfile)
            else AgentProfile.empty()
        )
        self._prelude_env: ValueEnv | None = None
        self._validation_tests: list[TestSuite] = []
        self._validation_evals: list[EvalSuite] = []
        self.conversation_state = conversation_state
        if isinstance(loop_iteration_limit, bool) or loop_iteration_limit <= 0:
            raise ValueError("loop_iteration_limit must be a positive integer")
        self.loop_iteration_limit = loop_iteration_limit
        self._implicit_conversation_state: ConversationState | None = None
        self.artifacts = (
            conversation_state.artifacts if conversation_state is not None else ArtifactManager()
        )
        from kedi.agent_adapter.subagents.coordinator import (
            SubagentCoordinator,
            SubagentCoordinatorDependencies,
        )

        self._subagent_coordinator = SubagentCoordinator(
            SubagentCoordinatorDependencies(
                profile_registry=lambda: self._profile_registry,
                evaluate_profile=lambda definition: self._evaluate_profile_definition(definition),
                resolve_adapter=lambda profile: self._resolve_adapter(profile),
                assert_profile_override_supported=lambda profile, adapter: (
                    self._assert_profile_override_supported(profile, adapter)
                ),
                resolve_tools=lambda profile, adapter: self._resolve_agent_tool_specs(
                    profile, adapter
                ),
                instructions_for_profile=lambda profile, adapter: self._agent_instructions(
                    profile, adapter
                ),
                trace_frame=lambda frame: self.trace_frame(frame),
                active_stack=self._subagent_stack,
                default_cwd=Path.cwd,
            ),
            max_depth=subagent_max_depth,
            max_concurrency=subagent_max_concurrency,
            timeout_seconds=subagent_timeout_seconds,
            usage_limits=subagent_usage_limits,
            state_path=subagent_state_path,
        )

    # Expressions
    def i(self, key: str) -> InputExpr:
        return i(key)

    def o(
        self,
        key: str,
        pytype: Optional[Any] = None,
        *,
        binding_contract: Optional[Any] = None,
    ) -> OutputExpr:
        return o(key, pytype, binding_contract=binding_contract)

    def c(self, func_name: str, *args: Any) -> CallExpr:
        return c(func_name, *args)

    def bind_agent(
        self,
        *,
        adapter: AgentAdapter[Any] | None = None,
        profile: AgentProfile | None = None,
        close_conversation_state: bool = False,
    ) -> AgentSurface:
        """Assemble Kedi runtime services for a native embedded-agent lifecycle."""

        from kedi.agent_surface import AgentSurface

        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("KediRuntime is closed")
        selected_profile = (profile or self.default_agent_profile).frozen_copy()
        selected_adapter = self._concrete_adapter(
            adapter or self._resolve_adapter(selected_profile)
        )
        self._assert_profile_override_supported(selected_profile, selected_adapter)
        tool_specs = cast(
            "tuple[ToolSpec, ...]",
            self._resolve_agent_tool_specs(selected_profile, selected_adapter),
        )
        instructions = self._agent_instructions(selected_profile, selected_adapter)
        effective_profile = replace(selected_profile, system=instructions)
        conversation_state = self._conversation_state_for_profile(selected_profile)
        return AgentSurface(
            agent=selected_adapter,
            profile=effective_profile,
            tool_specs=tool_specs,
            conversation_state=conversation_state,
            artifact_manager=self.artifacts,
            close_conversation_state=close_conversation_state,
        )

    async def aclose(self) -> None:
        """Cancel owned child runs and close runtime-owned session resources."""

        with self._lifecycle_lock:
            self._closed = True
        await self._subagent_coordinator.aclose()
        if self._implicit_conversation_state is not None:
            self._implicit_conversation_state.close()
        if self.conversation_state is None:
            self.artifacts.close()

    # Decorators
    def procedure(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        # Allow re-registration for AI-generated procedures from cache
        from kedi.lang.procedure_introspection import callable_display_name

        name = callable_display_name(fn)
        if name in self._procedures:
            pass  # Silently replace
        self._procedures[name] = fn
        return fn

    def main(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        # Allow overriding main for test isolation
        self._main = fn
        return fn

    # Public API
    def m(self, expressions: Sequence[Expression]) -> Dict[str, Any]:
        env = self._current_env.get()
        if env is None:
            raise RuntimeError("m() called outside of a kedi procedure context")
        return self._kedi_run(expressions, env)

    def invoke(self, expressions: Sequence[Expression], *, capture: str | None = None) -> Any:
        env = self._current_env.get()
        if env is None:
            raise RuntimeError("invoke() called outside of a kedi procedure context")

        parts: List[Any] = []
        deps: List[Any] = []
        for expr in expressions:
            text, out = self._evaluate_expression(expr, env)
            if out is not None:
                raise ValueError("Raw invoke prompts cannot contain output fields.")
            if text is not None:
                parts.append(text)
                if isinstance(text, KediPromise):
                    deps.append(text._future)

        result = self._invoke_text_parts(parts, deps, env, capture_key=capture)
        if capture is None:
            return None
        return result[capture]

    def run_main(self) -> Any:
        if self._main is None:
            return None
        result = self._invoke_procedure(self._main, [])
        # Drain awaits every scheduled template (parallel mode) so side effects fire and errors
        # surface; force resolves a promise result before handing it to the caller.
        self.drain()
        from kedi.lang.compiler.environment import native_value

        return native_value(force(result))

    def set_initial_globals(self, values: ValueEnv) -> None:
        self._initial_globals_env = dict(values)
        self._globals_env = dict(values)

    # Internal helpers
    def _get_proc(self, name: str) -> Callable[..., Any]:
        try:
            return self._procedures[name]
        except KeyError as e:
            raise KeyError(f"Unknown procedure: {name}") from e

    def _lookup_value(self, key: str, env: ValueEnv) -> Any:
        return _expressions._lookup_value(self, key, env)

    @staticmethod
    def _validate_adapter_output(value: Any, expected_type: Any, *, explicit_type: bool) -> Any:
        return _expressions._validate_adapter_output(
            value, expected_type, explicit_type=explicit_type
        )

    def _evaluate_expression(
        self, expr: Expression, env: ValueEnv
    ) -> Tuple[Any, Optional[OutputExpr]]:
        return _expressions._evaluate_expression(self, expr, env)

    def _evaluate_value(self, value: Any, env: ValueEnv) -> Any:
        # Procedure arguments are forced to concrete values at the call boundary.
        return _expressions._evaluate_value(self, value, env)

    def _invoke_procedure(self, fn: Callable[..., Any], args: Sequence[Any]) -> Any:
        return _expressions._invoke_procedure(self, fn, args)

    def _drain_outermost(self, primary_exc: BaseException | None) -> None:
        try:
            self.drain()
        except BaseException:
            if primary_exc is None:
                raise
            import logging

            logging.getLogger("kedi").error(
                "Template failure during drain after a primary error", exc_info=True
            )

    @contextmanager
    def trace_frame(self, frame: KediTraceFrame):
        yield from _tracing.trace_frame(self, frame)

    def current_trace_frames(self) -> tuple[KediTraceFrame, ...]:
        return _tracing.current_trace_frames(self)

    def build_execution_error(self, exc: BaseException) -> KediExecutionError:
        return _tracing.build_execution_error(self, exc)

    def _last_python_origin_traceback(self, exc: BaseException) -> TracebackType | None:
        return _tracing._last_python_origin_traceback(self, exc)

    def register_python_origin(
        self,
        *,
        kind: str,
        code: str,
        span: Any,
        compile_line_offset: int = 0,
        wrapper_line_offset: int = 0,
        kedi_line_offset: int = 0,
        procedure_name: str | None = None,
        traceback_code: str | None = None,
    ) -> str:
        return _tracing.register_python_origin(
            self,
            kind=kind,
            code=code,
            span=span,
            compile_line_offset=compile_line_offset,
            wrapper_line_offset=wrapper_line_offset,
            kedi_line_offset=kedi_line_offset,
            procedure_name=procedure_name,
            traceback_code=traceback_code,
        )

    def _hydrate_frame(self, frame: KediTraceFrame) -> KediTraceFrame:
        return _tracing._hydrate_frame(self, frame)

    def _invoke_text_parts(
        self,
        parts: Sequence[Any],
        deps: Sequence[Any],
        env: ValueEnv,
        *,
        capture_key: str | None,
    ) -> Dict[str, Any]:
        return _model_calls._invoke_text_parts(self, parts, deps, env, capture_key=capture_key)

    def _evaluate_claim(self, claim: str) -> bool:
        """Classify a rendered control-flow claim through the active agent lane."""
        return _model_calls._evaluate_claim(self, claim)

    def _kedi_run(self, expressions: Sequence[Expression], env: ValueEnv) -> Dict[str, Any]:
        return _model_calls._kedi_run(self, expressions, env)

    def _run_conversation_call(
        self,
        adapter: Any,
        prompt: PromptDraft,
        call: Callable[[AgentRunRequest], Any],
        *,
        instructions: Sequence[InstructionPart] = (),
        tools: Sequence[ToolSpec] = (),
        profile: AgentProfile,
        call_kind: Literal["produce", "invoke"],
        output: OutputContract,
        conversation_state: ConversationState | None = None,
        conversation_ticket: ConversationTurnTicket | None = None,
        lane_identity: str | None = None,
    ) -> Any:
        """Assemble and execute one adapter-neutral request transaction."""
        return _conversation._run_conversation_call(
            self,
            adapter,
            prompt,
            call,
            instructions=instructions,
            tools=tools,
            profile=profile,
            call_kind=call_kind,
            output=output,
            conversation_state=conversation_state,
            conversation_ticket=conversation_ticket,
            lane_identity=lane_identity,
        )

    def _conversation_state_for_profile(
        self,
        profile: AgentProfile,
    ) -> ConversationState | None:
        return _conversation._conversation_state_for_profile(self, profile)

    def _reserve_conversation_turn(
        self,
        profile: AgentProfile,
        adapter: Any,
        tools: Sequence[ToolSpec],
    ) -> tuple[ConversationState | None, ConversationTurnTicket | None, str | None]:
        return _conversation._reserve_conversation_turn(self, profile, adapter, tools)

    @staticmethod
    def _cancel_conversation_turn(
        state: ConversationState | None,
        ticket: ConversationTurnTicket | None,
    ) -> None:
        return _conversation._cancel_conversation_turn(state, ticket)

    @classmethod
    def _retire_conversation_turn_after_failed_future(
        cls,
        state: ConversationState | None,
        ticket: ConversationTurnTicket | None,
        completed: Any,
    ) -> None:
        """Release a reservation when an engine fails before or outside the job body."""
        return _conversation._retire_conversation_turn_after_failed_future(
            cls, state, ticket, completed
        )

    @staticmethod
    def _assert_profile_override_supported(profile: Any, adapter: Any) -> None:
        """Raise if the adapter can't honor a requested model/effort/mcp/settings override.

        Called on the main thread at the template call site so the error surfaces synchronously in
        both modes — symmetric with the tool guard in :meth:`_resolve_tool_specs`. ``_produce`` then
        only has to enter the override scope, never re-validate (which would defer the error to a
        worker in parallel mode).
        """
        from kedi.agent_adapter.hooks import HOOK_EVENT_NAMES
        from kedi.agent_adapter.meta import AdapterCapabilities, SupportsProfileOverride
        from kedi.agent_adapter.requirements import validate_adapter_requirements
        from kedi.agent_profile import AgentProfileOverrideUnsupported
        from kedi.agent_profile.compaction import CompactionSettingsError

        validate_adapter_requirements(
            adapter=adapter,
            requirements=profile.requirements,
            origins=dict(profile.requirement_origins),
        )

        if profile.history_processing is not None and (
            profile.history_processing.processor is not None
            or profile.history_processing.condition is not None
        ):
            validate_adapter_requirements(adapter=adapter, requirements=("history_processing",))

        if profile.settings.get("tool_reason") is True:
            capabilities = getattr(adapter, "capabilities", None)
            if (
                not isinstance(capabilities, AdapterCapabilities)
                or "tool_reason" not in capabilities.settings
            ):
                shortname = getattr(adapter, "shortname", type(adapter).__name__)
                raise AgentProfileOverrideUnsupported(
                    f"tool_reason is not supported by adapter {shortname!r}"
                )

        compaction = profile.compaction
        if compaction is not None and compaction.mode == "native" and profile.history is not True:
            raise CompactionSettingsError("native compaction requires history to be enabled")
        if profile.history is True and compaction is not None and compaction.mode == "native":
            capabilities = getattr(adapter, "capabilities", None)
            if not isinstance(capabilities, AdapterCapabilities) or not capabilities.supports(
                "native_compaction"
            ):
                shortname = getattr(adapter, "shortname", type(adapter).__name__)
                raise CompactionSettingsError(
                    f"native compaction is not supported by adapter {shortname!r}"
                )

        codemode = profile.codemode
        if codemode is not None and codemode.enabled:
            capabilities = getattr(adapter, "capabilities", None)
            if not isinstance(capabilities, AdapterCapabilities) or not capabilities.supports(
                "codemode"
            ):
                shortname = getattr(adapter, "shortname", type(adapter).__name__)
                raise AgentProfileOverrideUnsupported(
                    f"CodeMode is not supported by adapter {shortname!r}"
                )

        hooks = profile.hooks
        if hooks is not None and hooks.active:
            capabilities = getattr(adapter, "capabilities", None)
            shortname = getattr(adapter, "shortname", type(adapter).__name__)
            if not isinstance(capabilities, AdapterCapabilities) or capabilities.hooks is None:
                raise AgentProfileOverrideUnsupported(
                    f"Agent hooks are not supported by adapter {shortname!r}"
                )
            unsupported = capabilities.hooks.unsupported(
                tuple(event for event in HOOK_EVENT_NAMES if hooks.handlers_for(event))
            )
            if unsupported:
                names = ", ".join(unsupported)
                raise AgentProfileOverrideUnsupported(
                    f"Agent hook events {names} are not supported by adapter {shortname!r}"
                )

        if isinstance(adapter, SupportsProfileOverride):
            return
        if (
            profile.model is not None
            or profile.effort is not None
            or profile.mcp_servers
            or profile.settings
            or profile.compaction is not None
            or profile.codemode is not None
        ):
            raise AgentProfileOverrideUnsupported(
                "Active adapter does not support agent profile override "
                f"(requested model={profile.model!r}, effort={profile.effort!r}, "
                f"mcp_servers={profile.mcp_servers!r}, settings={dict(profile.settings)!r}, "
                f"compaction={profile.compaction!r}, codemode={profile.codemode!r})"
            )

    def _resolve_adapter(self, profile: Any) -> AgentAdapter[Any]:
        shortname = getattr(profile, "adapter_shortname", None)
        if shortname is None:
            if self.adapter is None:
                raise RuntimeError("No adapter provided to KediRuntime")
            return self.adapter

        adapter_shortname = (
            getattr(self.adapter, "shortname", None) if self.adapter is not None else None
        )
        if self.adapter is not None and shortname == adapter_shortname:
            return self.adapter

        model = getattr(profile, "model", None)
        if model is None and getattr(self.adapter, "kind", None) == "agent-framework":
            from kedi.agent_adapter.adapters.lazy import LazyAdapter
            from kedi.agent_adapter.meta import SupportsProfileOverride
            from kedi.agent_adapter.registry import adapter_factory

            if adapter_factory(shortname).kind == "agent-framework":
                if isinstance(self.adapter, LazyAdapter):
                    model = self.adapter.configured_model
                elif isinstance(self.adapter, SupportsProfileOverride):
                    model = self.adapter.default_model()
        cache_key = (shortname, model)
        selected = self._selected_adapters.get(cache_key)
        if selected is None:
            from kedi.agent_adapter.registry import build_adapter

            selected = build_adapter(shortname, model)
            self._selected_adapters[cache_key] = selected
        return selected

    @staticmethod
    def _concrete_adapter(adapter: AgentAdapter[Any]) -> AgentAdapter[Any]:
        from kedi.agent_adapter.adapters.lazy import LazyAdapter

        if isinstance(adapter, LazyAdapter):
            return cast("AgentAdapter[Any]", adapter.resolve())
        return adapter

    def _resolve_tool_specs(self, profile: Any, adapter: Any) -> tuple[Any, ...]:
        """Resolve a template's tools to fully-bound ToolSpecs (on the main thread, in scope).

        Returns () when the template has no tools. Raises if the adapter cannot register tools or
        a named tool is unknown — surfaced synchronously at the call site, not deferred.
        """
        return _tool_binding._resolve_tool_specs(self, profile, adapter)

    def _resolve_agent_tool_specs(
        self, profile: Any, adapter: Any, *, artifact_manager: ArtifactManager | None = None
    ) -> tuple[Any, ...]:
        """Resolve explicit Kedi tools plus the opt-in project-local skill tools."""
        return _tool_binding._resolve_agent_tool_specs(
            self, profile, adapter, artifact_manager=artifact_manager
        )

    async def _delegate_task(
        self,
        *,
        allowed_subagents: tuple[str, ...],
        subagent: str,
        task: str,
        final_schema: dict[str, Any] | None,
        parent_profile: Any,
        parent_budget: "DelegationBudget",
        parent_concurrency: "DelegationConcurrency",
        subagent_definitions: Mapping[str, Any] | None = None,
    ) -> Any:
        return await self._subagent_coordinator.delegate(
            allowed_subagents=allowed_subagents,
            subagent=subagent,
            task=task,
            final_schema=final_schema,
            parent_profile=parent_profile,
            parent_budget=parent_budget,
            parent_concurrency=parent_concurrency,
            subagent_definitions=subagent_definitions,
        )

    def _start_subagent(
        self,
        *,
        allowed_subagents: tuple[str, ...],
        subagent: str,
        task: str,
        final_schema: dict[str, Any] | None,
        parent_profile: Any,
        parent_budget: "DelegationBudget",
        parent_concurrency: "DelegationConcurrency",
        subagent_definitions: Mapping[str, Any] | None = None,
        evaluated_profile: Any | None = None,
        retain_for_native_owner: bool = False,
    ) -> Any:
        return self._subagent_coordinator.start(
            allowed_subagents=allowed_subagents,
            subagent=subagent,
            task=task,
            final_schema=final_schema,
            parent_profile=parent_profile,
            parent_budget=parent_budget,
            parent_concurrency=parent_concurrency,
            subagent_definitions=subagent_definitions,
            evaluated_profile=evaluated_profile,
            retain_for_native_owner=retain_for_native_owner,
        )

    async def _continue_subagent(
        self,
        *,
        run_id: str,
        allowed_subagents: tuple[str, ...],
        task: str,
        final_schema: dict[str, Any] | None,
        parent_profile: Any,
        parent_budget: "DelegationBudget",
        parent_concurrency: "DelegationConcurrency",
        subagent_definitions: Mapping[str, Any] | None = None,
    ) -> Any:
        return await self._subagent_coordinator.continue_run(
            run_id=run_id,
            allowed_subagents=allowed_subagents,
            task=task,
            final_schema=final_schema,
            parent_profile=parent_profile,
            parent_budget=parent_budget,
            parent_concurrency=parent_concurrency,
            subagent_definitions=subagent_definitions,
        )

    def _start_subagent_continuation(
        self,
        *,
        run_id: str,
        allowed_subagents: tuple[str, ...],
        task: str,
        final_schema: dict[str, Any] | None,
        parent_profile: Any,
        parent_budget: "DelegationBudget",
        parent_concurrency: "DelegationConcurrency",
        subagent_definitions: Mapping[str, Any] | None = None,
    ) -> Any:
        return self._subagent_coordinator.start_continuation(
            run_id=run_id,
            allowed_subagents=allowed_subagents,
            task=task,
            final_schema=final_schema,
            parent_profile=parent_profile,
            parent_budget=parent_budget,
            parent_concurrency=parent_concurrency,
            subagent_definitions=subagent_definitions,
        )

    async def _wait_subagent(
        self,
        run_id: str,
        *,
        timeout: float | None = None,
    ) -> DelegationResult:
        return await self._subagent_coordinator.wait(run_id, timeout=timeout)

    def _cancel_subagent(self, run_id: str) -> bool:
        return self._subagent_coordinator.cancel(run_id)

    def _subagent_status(self, run_id: str) -> Any:
        return self._subagent_coordinator.status(run_id)

    def _new_subagent_concurrency(self) -> "DelegationConcurrency":
        return self._subagent_coordinator.new_concurrency()

    def subagents(self, *, parent: str) -> Any:
        """Open a program-controlled subagent scope for an async Python caller."""
        from kedi.agent_adapter.subagents.native import SubagentScope

        return SubagentScope(self, parent)

    def _evaluate_profile_definition(self, definition: Any) -> Any:
        from kedi.agent_profile.compile import build_capture_env
        from kedi.agent_profile.evaluation import eval_profile_definition

        return eval_profile_definition(self, definition, build_capture_env(self))

    @staticmethod
    def _adapter_uses_native_approvals(adapter: Any) -> bool:
        return _tool_binding._adapter_uses_native_approvals(adapter)

    def _skill_base_dir(self) -> Path:
        """Resolve the project base beside the executed Kedi source."""
        return _tool_binding._skill_base_dir(self)

    @staticmethod
    def _skills_settings(profile: Any) -> Any:
        return _tool_binding._skills_settings(profile)

    def _agent_instruction_parts(self, profile: Any, adapter: Any) -> tuple[Any, ...]:
        """Resolve cache-stable instruction fragments with explicit provenance."""
        return _tool_binding._agent_instruction_parts(self, profile, adapter)

    def _agent_instructions(self, profile: Any, adapter: Any) -> str | None:
        """Append enabled runtime contracts without replacing user instructions."""
        return _tool_binding._agent_instructions(self, profile, adapter)

    def _artifact_contract_needed(
        self,
        profile: Any,
        adapter: Any,
        *,
        has_tools: bool,
    ) -> bool:
        """Choose a cache-stable artifact contract from static call capabilities."""
        return _tool_binding._artifact_contract_needed(self, profile, adapter, has_tools=has_tools)

    def _register_outstanding(self, future: Any) -> None:
        from kedi.agent_adapter.subagents.groups import register_process_future

        register_process_future(future)
        owner = self._outstanding_owner()
        with self._outstanding_lock:
            self._outstanding.setdefault(owner, []).append(future)

    def _outstanding_owner(self) -> object:
        scope = self._processing_scope.get()
        if scope is not None:
            return scope.work_owner
        invocation_id = self._invocation_id.get()
        if invocation_id is not None:
            return invocation_id
        return ("unscoped", id(self), get_ident())

    @property
    def _procedures(self) -> dict[str, Callable[..., Any]]:
        scope = self._processing_scope.get()
        return scope.procedures if scope is not None else self._root_procedures

    @_procedures.setter
    def _procedures(self, value: dict[str, Callable[..., Any]]) -> None:
        scope = self._processing_scope.get()
        if scope is not None:
            scope.procedures = value
        else:
            self._root_procedures = value

    @property
    def _type_env_stack(self) -> list[dict[str, Any]]:
        scope = self._processing_scope.get()
        return scope.types if scope is not None else self._root_type_env_stack

    @_type_env_stack.setter
    def _type_env_stack(self, value: list[dict[str, Any]]) -> None:
        scope = self._processing_scope.get()
        if scope is not None:
            scope.types = value
        else:
            self._root_type_env_stack = value

    def drain(self) -> None:
        """Await every scheduled produce Future (parallel mode) at end of run.

        Guarantees that even unconsumed templates execute (their side effects) and that any
        failure surfaces. Fail-fast on the first error (matching today's behavior); additional
        failures are logged loudly, never silently swallowed. Loops until no new work appears
        (a drained template may have scheduled more).
        """
        import logging

        owner = self._outstanding_owner()
        first_exc: BaseException | None = None
        extra: list[BaseException] = []
        while True:
            with self._outstanding_lock:
                pending = self._outstanding.pop(owner, [])
            if not pending:
                break
            for fut in pending:
                try:
                    self._engine.wait(fut)
                    exc = None
                except BaseException as failure:
                    exc = failure
                if exc is not None:
                    if first_exc is None:
                        first_exc = exc
                    else:
                        extra.append(exc)
        for exc in extra:
            logging.getLogger("kedi").error(
                "Suppressed additional template failure during drain: %r", exc
            )
        if first_exc is not None:
            raise first_exc
