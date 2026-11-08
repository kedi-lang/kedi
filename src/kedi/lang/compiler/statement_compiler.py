"""Statement compilation for Kedi compiler."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from typing import Any, Callable

from ...agent_profile import AgentProfile, block_agent_profile_scope
from ...agent_profile.directives import (
    eval_adapter_directive,
    eval_agent_directive,
    eval_approval_directive,
    eval_artifacts_directive,
    eval_budget_directive,
    eval_codemode_directive,
    eval_effort_directive,
    eval_history_directive,
    eval_hooks_directive,
    eval_mcp_directive,
    eval_model_directive,
    eval_requires_directive,
    eval_settings_directive,
    eval_skills_directive,
    eval_system_directive,
)
from ...agent_profile.scope import current_block_agent_profile, replace_block_agent_profile_scope
from ...agent_profile.telemetry import merge_profile_with_telemetry
from ...agent_profile.use_resolution import eval_use_directive
from ...core import KediRuntime
from ...core.expressions import ValueEnv
from ...debugging import checkpoint
from ...errors import KediTraceFrame
from ...groups import group_member, group_scope
from ..ast import (
    AdapterDirectiveStmt,
    AgentDirectiveStmt,
    ApprovalDirectiveStmt,
    ArtifactsDirectiveStmt,
    AssignmentBlockStmt,
    AssignmentStmt,
    AwaitStmt,
    BacktickLineStmt,
    BudgetDirectiveStmt,
    CodeModeDirectiveStmt,
    ConditionalLoopStmt,
    EffortDirectiveStmt,
    HistoryDirectiveStmt,
    HooksDirectiveStmt,
    GroupStmt,
    IfStmt,
    LoopStmt,
    McpDirectiveStmt,
    ModelDirectiveStmt,
    OutputSegment,
    ProcedureDef,
    PythonBlock,
    RawInvokeStmt,
    RequiresDirectiveStmt,
    ReturnBlockStmt,
    ReturnStmt,
    SendStmt,
    SettingsDirectiveStmt,
    ShowStmt,
    SkillsDirectiveStmt,
    Stmt,
    SystemDirectiveStmt,
    TaskGroupStmt,
    TaskStmt,
    TemplateBlock,
    TemplateLine,
    UseDirectiveStmt,
    VariableInitBlockStmt,
    VariableInitStmt,
)
from ..parser.escapes import preserve_escaped_whitespace
from .control_flow import (
    active_scope,
    assert_conditional_loop_capacity,
    branch_trace,
    child_scope,
    evaluate_condition,
    evaluate_iterable,
    group_trace,
    loop_binding,
    loop_trace,
    managed_loop_iterator,
    run_map_continuations,
)
from .evaluation import (
    NO_NATIVE_VALUE,
    evaluate_binding_rhs,
    evaluate_python_inline,
    execute_inline_side_effects,
    execute_python_block,
    merge_template_block_lines,
    prepare_segments,
    render_expression,
)
from .type_system import KediType, detect_rhs_kind, initialize_value_from_native

_NO_RETURN = object()


class StatementCompiler:
    """Compiles individual statements to executable functions."""

    def __init__(self, rt: KediRuntime):
        self.rt = rt

    def compile_template_line(
        self,
        stmt: TemplateLine,
        *,
        trace_label: str | None = None,
        source: Stmt | None = None,
    ) -> Callable:
        """Compile a template line to a function.

        If the template has an optimized_prompt_prefix, it is prepended to the
        original seed prompt (with proper type annotations) at evaluation time.
        This allows GEPA to output plain text without needing correct type names,
        while preserving the original prompt's type annotations for proper resolution.
        """
        label = trace_label or "template line"
        member = stmt if source is None else source

        def _fn(**kwargs: Any) -> Any:
            with (
                self.rt.trace_frame(KediTraceFrame(kind="template", label=label, span=stmt.loc)),
                group_member(member),
            ):
                # If an override is provided, short-circuit LLM invocation and inject values.
                if stmt.optimized_output_override is not None:
                    env = self.rt._current_env.get()
                    if env is None:
                        raise RuntimeError("Template override requires active runtime environment")
                    contracts = {
                        segment.name: KediType.from_node(
                            segment.type_expr,
                            segment.type_annotation,
                        )
                        for segment in stmt.segments
                        if isinstance(segment, OutputSegment)
                    }
                    for key, value in stmt.optimized_output_override.items():
                        contract = contracts.get(key)
                        if contract is not None and not contract.is_none():
                            value = contract.validate(self.rt, env, value)
                        self._declare_value(
                            key,
                            value,
                            env,
                            kedi_type=contract,
                            source=stmt.loc,
                        )
                        kwargs[key] = value
                    return stmt.optimized_output_override

                # If there's an optimized prompt prefix, prepend it to the segments
                # This allows GEPA to output plain text while preserving type annotations
                segments_to_use = stmt.segments
                if stmt.optimized_prompt_prefix:
                    from ..ast import TextSegment

                    # Prepend optimized text, then original seed prompt with types
                    # Use two newlines for better separation between optimized and seed prompts
                    optimized_text = TextSegment(text=stmt.optimized_prompt_prefix + "\n\n")
                    segments_to_use = [optimized_text] + stmt.segments

                expr, _ = prepare_segments(self.rt, segments_to_use, kwargs)
                return self.rt.m(expr)

        return _fn

    def compile_template_block(self, stmt: TemplateBlock) -> Callable:
        """Compile a ``>>`` template block as one newline-joined LLM run."""
        merged = merge_template_block_lines(stmt.lines, loc=stmt.loc)
        n = len(stmt.lines)
        label = f"template block ({n} line{'s' if n != 1 else ''})"
        return self.compile_template_line(merged, trace_label=label, source=stmt)

    def compile_raw_invoke(self, stmt: RawInvokeStmt) -> Callable:
        """Compile ``[name] << prompt``; a non-``str`` capture type requests typed output."""

        def _fn(**kwargs: Any) -> Any:
            with (
                self.rt.trace_frame(
                    KediTraceFrame(kind="template", label=f"raw invoke {stmt.name}", span=stmt.loc)
                ),
                group_member(stmt),
            ):
                expr, _ = prepare_segments(self.rt, stmt.segments, kwargs)
                kedi_type = KediType.from_node(stmt.type_expr, stmt.type_annotation)
                output_type = kedi_type.resolve_required(self.rt, kwargs)
                if output_type is None or output_type is str:
                    return self.rt.invoke(expr, capture=stmt.name)
                return self.rt.invoke(
                    expr,
                    capture=stmt.name,
                    output_type=output_type,
                    binding_contract=kedi_type,
                )

        return _fn

    def compile_return(self, stmt: ReturnStmt) -> Callable:
        """Compile a return statement to a function."""

        def _fn(**kwargs: Any) -> Any:
            with self.rt.trace_frame(KediTraceFrame(kind="return", label="return", span=stmt.loc)):
                rendered = render_expression(self.rt, stmt.segments, kwargs, context="return")
                return rendered

        return _fn

    def execute_show(self, stmt: ShowStmt, env: dict[str, Any]) -> None:
        """Render once and use the executor's existing output destination."""
        with self.rt.trace_frame(KediTraceFrame(kind="show", label="show", span=stmt.loc)):
            text = render_expression(self.rt, stmt.segments, env, context="show")
            self.rt.executor.evaluate_inline(
                self.rt,
                "__builtins__['print'](__kedi_display_text)",
                {"__kedi_display_text": text},
            )

    def execute_task(self, stmt: TaskStmt, env: dict[str, Any], active: AgentProfile) -> None:
        from kedi.agent_adapter.subagents.native import NativeSubagentOwner

        from .subagent_tasks import prepare_subagent_task

        with self.rt.trace_frame(KediTraceFrame(kind="task", label=stmt.subagent, span=stmt.loc)):
            owner: NativeSubagentOwner | None = self.rt._native_subagent_owner.get()
            if owner is None:
                raise RuntimeError("Subagent tasks require an active Kedi invocation")
            prompt, schema, output_adapter = prepare_subagent_task(self.rt, stmt, env)
            job = owner.start_sync(
                stmt.subagent,
                task=prompt,
                parent_profile=active,
                final_schema=schema,
                output_adapter=output_adapter,
            )
            self._declare_value(stmt.binder, job, env, source=stmt.loc)

    def execute_await(self, stmt: AwaitStmt, env: dict[str, Any]) -> None:
        from kedi.agent_adapter.subagents.native import SubagentTask

        from .evaluation import lookup_value

        with self.rt.trace_frame(KediTraceFrame(kind="await", label=stmt.handle, span=stmt.loc)):
            job = lookup_value(self.rt, stmt.handle, env)
            if not isinstance(job, SubagentTask):
                raise TypeError(f"> await expects a subagent task, got {type(job).__name__}")
            owner = self.rt._native_subagent_owner.get()
            if owner is None:
                raise RuntimeError("Subagent await requires an active Kedi invocation")
            result = owner.wait_sync(job)
            if stmt.binder is not None:
                self._declare_value(stmt.binder, result, env, source=stmt.loc)

    def execute_send(self, stmt: SendStmt, env: ValueEnv) -> None:
        from kedi.agent_adapter.subagents.native import SubagentTask

        from .evaluation import lookup_value, merge_template_block_lines

        job = lookup_value(self.rt, stmt.handle, env)
        if not isinstance(job, SubagentTask):
            raise TypeError(f"> send/interrupt expects a subagent task, got {type(job).__name__}")
        owner = self.rt._native_subagent_owner.get()
        if owner is None or job._owner is not owner:
            raise RuntimeError("Agent input requires the task's active invocation")
        merged = merge_template_block_lines(stmt.template.lines, loc=stmt.template.loc)
        with self.rt.trace_frame(KediTraceFrame(kind="send", label=stmt.handle, span=stmt.loc)):
            message = render_expression(self.rt, merged.segments, env, context="send")
            receipt = owner._run_sync(lambda: job.send(message, interrupt=stmt.interrupt))
            if receipt.status == "rejected":
                raise RuntimeError(f"Agent input rejected: {receipt.reason}")

    def compile_return_block(self, stmt: ReturnBlockStmt, kwargs: dict[str, Any]) -> Any:
        """Execute a return block and return its value."""
        code = stmt.code
        with self.rt.trace_frame(
            KediTraceFrame(kind="return_block", label="return block", span=stmt.loc)
        ):
            if "\n" in code or code.strip().startswith("return"):
                return execute_python_block(code, self.rt, kwargs, span=stmt.loc)
            return evaluate_python_inline(self.rt, code, kwargs, span=stmt.loc)

    def execute_variable_init(self, stmt: VariableInitStmt, closure_env: dict[str, Any]) -> None:
        """Initialize a variable in the active lexical frame."""
        with self.rt.trace_frame(
            KediTraceFrame(
                kind="variable_init",
                label=f"variable initialization {stmt.name}",
                span=stmt.loc,
            )
        ):
            kedi_type = KediType.from_node(stmt.type_expr, stmt.type_annotation)
            value = self._evaluate_initialization_value(stmt.expr, kedi_type, closure_env)
            self._declare_value(
                stmt.name,
                value,
                closure_env,
                kedi_type=kedi_type,
                source=stmt.loc,
            )

    def execute_variable_init_block(
        self, stmt: VariableInitBlockStmt, closure_env: dict[str, Any]
    ) -> None:
        """Initialize a variable from a fenced Python value."""
        with self.rt.trace_frame(
            KediTraceFrame(
                kind="variable_init_block",
                label=f"variable initialization {stmt.name}",
                span=stmt.loc,
            )
        ):
            result = execute_python_block(stmt.code, self.rt, closure_env, span=stmt.loc)
            kedi_type = KediType.from_node(stmt.type_expr, stmt.type_annotation)

            value = result
            if not kedi_type.is_none():
                # Strict validation with TypeAdapter
                try:
                    value = kedi_type.validate(self.rt, closure_env, value)
                except Exception:
                    raise TypeError("Native initialization value does not match annotated type")
            else:
                if not isinstance(value, str):
                    raise TypeError(
                        "Native non-string initialization requires explicit type annotation"
                    )

            self._declare_value(
                stmt.name,
                value,
                closure_env,
                kedi_type=kedi_type,
                source=stmt.loc,
            )

    def execute_assignment(self, stmt: AssignmentStmt, closure_env: dict[str, Any]) -> None:
        """Update the nearest existing binding while preserving its type contract."""
        with self.rt.trace_frame(
            KediTraceFrame(kind="assignment", label=f"assignment {stmt.name}", span=stmt.loc)
        ):
            kedi_type = self._assignment_type(stmt.name, closure_env)
            native, rendered = evaluate_binding_rhs(self.rt, stmt.expr, closure_env)
            value = native if native is not NO_NATIVE_VALUE else rendered
            if kedi_type is not None and not kedi_type.is_none():
                try:
                    value = kedi_type.validate(self.rt, closure_env, value)
                except Exception as exc:
                    raise TypeError(
                        "Assigned value does not match the binding's annotated type"
                    ) from exc
            self._assign_value(stmt.name, value, closure_env)

    def execute_assignment_block(
        self, stmt: AssignmentBlockStmt, closure_env: dict[str, Any]
    ) -> None:
        """Evaluate a fenced native value and update its nearest binding owner."""
        with self.rt.trace_frame(
            KediTraceFrame(
                kind="assignment_block",
                label=f"assignment {stmt.name}",
                span=stmt.loc,
            )
        ):
            value = execute_python_block(stmt.code, self.rt, closure_env, span=stmt.loc)
            kedi_type = self._assignment_type(stmt.name, closure_env)
            if kedi_type is not None and not kedi_type.is_none():
                try:
                    value = kedi_type.validate(self.rt, closure_env, value)
                except Exception as exc:
                    raise TypeError(
                        "Assigned value does not match the binding's annotated type"
                    ) from exc
            self._assign_value(stmt.name, value, closure_env)

    def _evaluate_initialization_value(
        self,
        segments: list[Any],
        kedi_type: KediType,
        closure_env: dict[str, Any],
    ) -> Any:
        native, rendered = evaluate_binding_rhs(self.rt, segments, closure_env)
        is_backtick, is_angle, is_call = detect_rhs_kind(segments)
        if native is not NO_NATIVE_VALUE:
            if is_backtick:
                return initialize_value_from_native(
                    native, kedi_type, self.rt, closure_env, strict_native=True
                )
            if is_angle:
                return initialize_value_from_native(
                    native, kedi_type, self.rt, closure_env, strict_native=False
                )
            if is_call:
                return (
                    kedi_type.validate(self.rt, closure_env, native)
                    if not kedi_type.is_none()
                    else native
                )
        if kedi_type.is_none():
            return rendered
        try:
            return kedi_type.validate(self.rt, closure_env, rendered)
        except Exception as exc:
            raise TypeError("Native initialization value does not match annotated type") from exc

    def execute_python_block(self, stmt: PythonBlock, closure_env: dict[str, Any]) -> None:
        """Execute a Python block for side effects."""
        execute_inline_side_effects(
            self.rt,
            stmt.code,
            closure_env,
            span=stmt.loc,
            kedi_line_offset=1,
        )

    def execute_backtick_line(self, stmt: BacktickLineStmt, closure_env: dict[str, Any]) -> None:
        """Execute a single-line Python statement for side effects."""
        execute_inline_side_effects(self.rt, stmt.code, closure_env, span=stmt.loc)

    def _declare_value(
        self,
        name: str,
        value: Any,
        closure_env: dict[str, Any],
        *,
        kedi_type: KediType | None = None,
        source: Any = None,
    ) -> None:
        """Declare a value in the active lexical frame."""
        from .environment import BindingMetadata, ScopeFrame, assert_not_cli_args_identifier

        assert_not_cli_args_identifier(name)
        if isinstance(closure_env, ScopeFrame):
            closure_env.declare(
                name,
                value,
                BindingMetadata(type_contract=kedi_type, source=source),
            )
        else:
            closure_env[name] = value

    @staticmethod
    def _assignment_type(name: str, closure_env: dict[str, Any]) -> KediType | None:
        from .environment import ScopeFrame

        if isinstance(closure_env, ScopeFrame):
            owner = closure_env.resolve_owner(name)
            if owner is None:
                raise NameError(f"Cannot assign unknown Kedi binding: {name}")
            metadata = closure_env.binding_metadata(name)
            contract = metadata.type_contract if metadata is not None else None
            return contract if isinstance(contract, KediType) else None
        if name not in closure_env:
            raise NameError(f"Cannot assign unknown Kedi binding: {name}")
        return None

    @staticmethod
    def _assign_value(name: str, value: Any, closure_env: dict[str, Any]) -> None:
        from .environment import ScopeFrame, assert_not_cli_args_identifier

        assert_not_cli_args_identifier(name)
        if isinstance(closure_env, ScopeFrame):
            closure_env.assign(name, value)
            return
        if name not in closure_env:
            raise NameError(f"Cannot assign unknown Kedi binding: {name}")
        closure_env[name] = value


class StatementExecutor:
    """Executes a sequence of statements in a given environment."""

    def __init__(self, rt: KediRuntime):
        self.rt = rt
        self.compiler = StatementCompiler(rt)

    def execute(self, proc: ProcedureDef, closure_env: dict[str, Any]) -> Any:
        """Execute procedure statements until the first evaluated return."""
        from kedi.agent_profile.capture import CapturedAgentProfile

        from .type_system import pop_type_env, push_type_env

        captured = proc.captured_agent_profile
        if isinstance(captured, CapturedAgentProfile):
            captured = captured.resolve()
        active = captured if isinstance(captured, AgentProfile) else self.rt.default_agent_profile
        # An unbound model still uses the invoking agent's model, as for imported
        # helpers. This is not permission to inherit its instructions or tools.
        if active.model is None:
            active = replace(active, model=current_block_agent_profile().model)
        last_return: Any = None
        type_scope = push_type_env(self.rt)

        # Track nested procedures for cleanup
        saved_procs: dict[str, Any] = {}
        new_proc_names: list[str] = []

        # Inherit parent type environment
        parent_stack = self.rt._type_env_stack
        if len(parent_stack) >= 2:
            type_scope.update(parent_stack[-2])

        profile_registry = self.rt._profile_registry
        form = (
            nullcontext()
            if proc.auto_form is None
            else group_scope(
                module=self.rt.module_name,
                qualified_name=proc.name,
                kind="auto",
                body=proc.body,
                canonical=proc.auto_form,
                producer="codegen",
            )
        )

        try:
            with form:
                block_result, _ = self._execute_block(
                    proc.body,
                    proc,
                    closure_env,
                    type_scope,
                    saved_procs,
                    new_proc_names,
                    active,
                    profile_registry,
                )
            if block_result is not _NO_RETURN:
                last_return = block_result

        finally:
            pop_type_env(self.rt)

            # Restore nested procedure scoping
            procs = self.rt._procedures
            for name in new_proc_names:
                if name in procs:
                    del procs[name]
            for name, old_fn in saved_procs.items():
                procs[name] = old_fn

        return last_return

    def _execute_block(
        self,
        statements: list[Stmt],
        proc: ProcedureDef,
        closure_env: dict[str, Any],
        type_scope: dict[str, Any],
        saved_procs: dict[str, Any],
        new_proc_names: list[str],
        active: AgentProfile,
        profile_registry: dict[str, Any],
    ) -> tuple[Any, AgentProfile]:
        """Execute one ordered block while keeping directive state block-local."""
        last_return: Any = _NO_RETURN
        for stmt in statements:
            from kedi.agent_adapter.subagents.groups import check_process_cancelled

            check_process_cancelled()
            checkpoint(self.rt, "statement", env=closure_env, span=stmt.loc)
            updated = self._apply_profile_directive(
                stmt,
                active,
                closure_env,
                profile_registry,
            )
            if updated is not None:
                active = updated
                continue

            with replace_block_agent_profile_scope(active):
                result = self._execute_body_statement(
                    stmt,
                    proc,
                    closure_env,
                    type_scope,
                    saved_procs,
                    new_proc_names,
                    active,
                    profile_registry,
                )
            if result is not _NO_RETURN:
                return result, active

        return last_return, active

    def _apply_profile_directive(
        self,
        stmt: Stmt,
        active: AgentProfile,
        closure_env: dict[str, Any],
        profile_registry: dict[str, Any],
    ) -> AgentProfile | None:
        if isinstance(stmt, AgentDirectiveStmt):
            return active.merge(eval_agent_directive(self.rt, stmt, closure_env))
        if isinstance(stmt, AdapterDirectiveStmt):
            return active.merge(eval_adapter_directive(self.rt, stmt, closure_env))
        if isinstance(stmt, ModelDirectiveStmt):
            return active.merge(eval_model_directive(self.rt, stmt, closure_env))
        if isinstance(stmt, EffortDirectiveStmt):
            return active.merge(eval_effort_directive(self.rt, stmt, closure_env))
        if isinstance(stmt, ApprovalDirectiveStmt):
            return active.merge(eval_approval_directive(self.rt, stmt, closure_env))
        if isinstance(stmt, HooksDirectiveStmt):
            hook_patch = eval_hooks_directive(
                self.rt,
                stmt,
                closure_env,
            ).with_hook_enforcement()
            return active.merge(hook_patch)
        if isinstance(stmt, HistoryDirectiveStmt):
            return active.merge(eval_history_directive(self.rt, stmt, closure_env))
        if isinstance(stmt, SkillsDirectiveStmt):
            return active.merge(eval_skills_directive(self.rt, stmt, closure_env))
        if isinstance(stmt, CodeModeDirectiveStmt):
            return active.merge(eval_codemode_directive(self.rt, stmt, closure_env))
        if isinstance(stmt, RequiresDirectiveStmt):
            return active.merge(eval_requires_directive(self.rt, stmt, closure_env))
        if isinstance(stmt, BudgetDirectiveStmt):
            from kedi.run_budget import current_run_budget

            patch = eval_budget_directive(self.rt, stmt, closure_env)
            ledger = current_run_budget()
            if ledger is not None and patch.run_budget is not None:
                ledger.set_budget(patch.run_budget)
            return active.merge(patch)
        if isinstance(stmt, SystemDirectiveStmt):
            return merge_profile_with_telemetry(
                active, eval_system_directive(self.rt, stmt, closure_env)
            )
        if isinstance(stmt, McpDirectiveStmt):
            return active.merge(eval_mcp_directive(self.rt, stmt, closure_env))
        if isinstance(stmt, SettingsDirectiveStmt):
            return active.merge(eval_settings_directive(self.rt, stmt, closure_env))
        if isinstance(stmt, ArtifactsDirectiveStmt):
            return active.merge(eval_artifacts_directive(self.rt, stmt, closure_env))
        if isinstance(stmt, UseDirectiveStmt):
            from kedi.run_budget import current_run_budget

            patch = eval_use_directive(
                self.rt,
                stmt,
                profile_registry,
                closure_env,
                available_procedures=set(self.rt._procedures.keys()),
            )
            ledger = current_run_budget()
            if ledger is not None and patch.run_budget is not None:
                ledger.set_budget(patch.run_budget)
            return merge_profile_with_telemetry(active, patch)
        return None

    def _execute_body_statement(
        self,
        stmt: Stmt,
        proc: Any,
        closure_env: dict[str, Any],
        type_scope: dict[str, Any],
        saved_procs: dict[str, Any],
        new_proc_names: list[str],
        active: AgentProfile,
        profile_registry: dict[str, Any] | None = None,
    ) -> Any:
        from kedi.run_budget import RunBudget, run_budget_scope

        from ..ast import ProcDefStmt, TypeDefStmt
        from .procedure_compiler import ProcedureCompiler
        from .type_system import register_type

        if profile_registry is None:
            profile_registry = self.rt._profile_registry

        if isinstance(stmt, GroupStmt):
            # Transparent: the form runs in this scope, so its bindings stay visible.
            with group_trace(self.rt, stmt, proc.name) as body:
                result, _ = self._execute_block(
                    body,
                    proc,
                    closure_env,
                    type_scope,
                    saved_procs,
                    new_proc_names,
                    active,
                    profile_registry,
                )
            return result

        if isinstance(stmt, IfStmt):
            condition = evaluate_condition(self.rt, stmt, closure_env)
            body = stmt.body if condition else stmt.else_body
            if body is None:
                return _NO_RETURN
            with branch_trace(self.rt, stmt, alternative=not condition):
                with child_scope(self.rt, closure_env) as branch_env, run_budget_scope(RunBudget()):
                    result, _ = self._execute_block(
                        body,
                        proc,
                        branch_env,
                        type_scope,
                        saved_procs,
                        new_proc_names,
                        active,
                        profile_registry,
                    )
            return result

        if isinstance(stmt, LoopStmt):
            last_return: Any = _NO_RETURN
            with loop_trace(self.rt, stmt):
                records: list[tuple[Any, AgentProfile]] = []
                values = evaluate_iterable(self.rt, stmt, closure_env)
                with managed_loop_iterator(values):
                    for value in values:
                        with (
                            loop_binding(self.rt, closure_env, stmt.binder, value) as iteration_env,
                            run_budget_scope(RunBudget()),
                        ):
                            result, iteration_active = self._execute_block(
                                stmt.body,
                                proc,
                                iteration_env,
                                type_scope,
                                saved_procs,
                                new_proc_names,
                                active,
                                profile_registry,
                            )
                            records.append((iteration_env, iteration_active))
                        if result is not _NO_RETURN:
                            return result

                if stmt.map_body is not None:
                    frames = [record[0] for record in records]
                    results: list[Any] = [_NO_RETURN] * len(frames)

                    def run_map(index: int, frame: Any) -> None:
                        _, map_active = records[index]
                        with active_scope(self.rt, frame), run_budget_scope(RunBudget()):
                            with block_agent_profile_scope(map_active):
                                results[index], _ = self._execute_block(
                                    stmt.map_body or [],
                                    proc,
                                    frame,
                                    type_scope,
                                    saved_procs,
                                    new_proc_names,
                                    map_active,
                                    profile_registry,
                                )

                    run_map_continuations(self.rt, stmt, frames, run_map)
                    for result in results:
                        if result is not _NO_RETURN:
                            return result
            return last_return

        if isinstance(stmt, ConditionalLoopStmt):
            conditional_last_return: Any = _NO_RETURN
            completed_iterations = 0
            with loop_trace(self.rt, stmt):
                while evaluate_condition(self.rt, stmt, closure_env):
                    assert_conditional_loop_capacity(self.rt, completed_iterations)
                    with (
                        child_scope(self.rt, closure_env) as iteration_env,
                        run_budget_scope(RunBudget()),
                    ):
                        result, _ = self._execute_block(
                            stmt.body,
                            proc,
                            iteration_env,
                            type_scope,
                            saved_procs,
                            new_proc_names,
                            active,
                            profile_registry,
                        )
                    completed_iterations += 1
                    if result is not _NO_RETURN:
                        return result
            return conditional_last_return

        if isinstance(stmt, ShowStmt):
            self.compiler.execute_show(stmt, closure_env)
            return _NO_RETURN

        if isinstance(stmt, TaskGroupStmt):
            from .task_groups import execute_task_group

            execute_task_group(self.rt, stmt, closure_env, active)
            return _NO_RETURN

        if isinstance(stmt, TaskStmt):
            self.compiler.execute_task(stmt, closure_env, active)
            return _NO_RETURN

        if isinstance(stmt, AwaitStmt):
            self.compiler.execute_await(stmt, closure_env)
            return _NO_RETURN
        if isinstance(stmt, SendStmt):
            self.compiler.execute_send(stmt, closure_env)
            return _NO_RETURN

        if isinstance(stmt, TemplateBlock):
            self.compiler.compile_template_block(stmt)(**closure_env)
            self._sync_env(closure_env)
            return _NO_RETURN

        if isinstance(stmt, TemplateLine):
            self.compiler.compile_template_line(stmt)(**closure_env)
            self._sync_env(closure_env)
            return _NO_RETURN

        if isinstance(stmt, RawInvokeStmt):
            self.compiler.compile_raw_invoke(stmt)(**closure_env)
            self._sync_env(closure_env)
            return _NO_RETURN

        if isinstance(stmt, TypeDefStmt):
            if stmt.typedef is not None:
                register_type(self.rt, stmt.typedef, type_scope, closure_env)
            return _NO_RETURN

        if isinstance(stmt, ProcDefStmt) and stmt.proc is not None:
            child = replace(
                stmt.proc, captured_agent_profile=active.frozen_copy(), lexical_env=closure_env
            )
            proc_compiler = ProcedureCompiler(self.rt)
            fn_nested = proc_compiler._make_procedure_impl(child, child.lexical_env)

            procs = self.rt._procedures
            if child.name in procs and child.name not in saved_procs:
                saved_procs[child.name] = procs[child.name]
            elif child.name not in procs:
                new_proc_names.append(child.name)
            procs[child.name] = fn_nested
            return _NO_RETURN

        if isinstance(stmt, PythonBlock):
            self.compiler.execute_python_block(stmt, closure_env)
            return _NO_RETURN

        if isinstance(stmt, BacktickLineStmt):
            self.compiler.execute_backtick_line(stmt, closure_env)
            return _NO_RETURN

        if isinstance(stmt, VariableInitBlockStmt):
            self.compiler.execute_variable_init_block(stmt, closure_env)
            return _NO_RETURN

        if isinstance(stmt, AssignmentBlockStmt):
            self.compiler.execute_assignment_block(stmt, closure_env)
            return _NO_RETURN

        if isinstance(stmt, VariableInitStmt):
            self.compiler.execute_variable_init(stmt, closure_env)
            return _NO_RETURN

        if isinstance(stmt, AssignmentStmt):
            self.compiler.execute_assignment(stmt, closure_env)
            return _NO_RETURN

        if isinstance(stmt, ReturnStmt):
            # Gate on the structured type too (not just the legacy string) so
            # structured-only typed returns take the enforced native path.
            return_kedi_type = KediType.from_node(
                proc.return_type_expr, proc.return_type_annotation
            )
            if not return_kedi_type.is_none():
                native, rendered = evaluate_binding_rhs(self.rt, stmt.segments, closure_env)
                return (
                    native
                    if native is not NO_NATIVE_VALUE
                    else preserve_escaped_whitespace(rendered)
                )
            return self.compiler.compile_return(stmt)(**closure_env)

        if isinstance(stmt, ReturnBlockStmt):
            return self.compiler.compile_return_block(stmt, closure_env)

        return _NO_RETURN

    def execute_single_statement(
        self, stmt: Stmt, env: dict[str, Any], is_toplevel: bool = False
    ) -> None:
        """Execute a single statement (for top-level execution)."""
        from ..ast import ProcDefStmt

        if isinstance(stmt, ShowStmt):
            self.compiler.execute_show(stmt, env)

        elif isinstance(stmt, TemplateBlock):
            self.compiler.compile_template_block(stmt)(**env)
            self._sync_env(env)

        elif isinstance(stmt, RawInvokeStmt):
            self.compiler.compile_raw_invoke(stmt)(**env)
            self._sync_env(env)

        elif isinstance(stmt, ProcDefStmt):
            if is_toplevel and stmt.proc is not None:
                # Register top-level procedure
                from .procedure_compiler import ProcedureCompiler

                proc_compiler = ProcedureCompiler(self.rt)
                fn = proc_compiler.compile_procedure(stmt.proc)
                self.rt.procedure(fn)

        elif isinstance(stmt, PythonBlock):
            self.compiler.execute_python_block(stmt, env)

        elif isinstance(stmt, BacktickLineStmt):
            self.compiler.execute_backtick_line(stmt, env)

        elif isinstance(stmt, VariableInitBlockStmt):
            self.compiler.execute_variable_init_block(stmt, env)

        elif isinstance(stmt, AssignmentBlockStmt):
            self.compiler.execute_assignment_block(stmt, env)

        elif isinstance(stmt, VariableInitStmt):
            self.compiler.execute_variable_init(stmt, env)

        elif isinstance(stmt, AssignmentStmt):
            self.compiler.execute_assignment(stmt, env)

    def _sync_env(self, closure_env: dict[str, Any]) -> None:
        """Sync only pre-existing names for legacy direct executor calls."""
        env = self.rt._current_env.get()
        if env is None or env is closure_env:
            return
        for name in list(closure_env):
            if name in env:
                closure_env[name] = env[name]
