"""Execute top-level statements and their lexical control-flow scopes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from kedi.agent_profile import AgentProfile, block_agent_profile_scope
from kedi.agent_profile.directives import (
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
from kedi.agent_profile.use_resolution import eval_use_directive
from kedi.debugging import checkpoint
from kedi.lang.ast import (
    AdapterDirectiveStmt,
    AgentDirectiveStmt,
    ApprovalDirectiveStmt,
    ArtifactsDirectiveStmt,
    AwaitStmt,
    BudgetDirectiveStmt,
    CodeModeDirectiveStmt,
    ConditionalLoopStmt,
    EffortDirectiveStmt,
    HistoryDirectiveStmt,
    HooksDirectiveStmt,
    IfStmt,
    LoopStmt,
    McpDirectiveStmt,
    ModelDirectiveStmt,
    ModuleImport,
    ProcDefStmt,
    ProcedureDef,
    ProfileDefStmt,
    RequiresDirectiveStmt,
    ReturnBlockStmt,
    ReturnStmt,
    SettingsDirectiveStmt,
    SkillsDirectiveStmt,
    SystemDirectiveStmt,
    TaskGroupStmt,
    TaskStmt,
    TemplateLine,
    TypeDef,
    TypeDefStmt,
    UseDirectiveStmt,
)
from kedi.lang.compiler.control_flow import (
    active_scope,
    assert_conditional_loop_capacity,
    branch_trace,
    child_scope,
    evaluate_condition,
    evaluate_iterable,
    loop_binding,
    loop_trace,
    managed_loop_iterator,
    run_map_continuations,
)
from kedi.lang.compiler.type_system import register_type
from kedi.run_budget import RunBudget, current_run_budget, run_budget_scope

if TYPE_CHECKING:
    from kedi.lang.compiler.program_compiler import ProgramCompiler, _PreparedProgram


@dataclass
class _TopLevelExecutionState:
    """Directive state that advances as top-level statements execute."""

    active_profile: AgentProfile


def _execute_top_level_sequence(
    self: ProgramCompiler,
    prepared: _PreparedProgram,
    env: dict[str, Any],
    state: _TopLevelExecutionState,
    sequence: tuple[Any, ...] | list[Any],
) -> Any:
    """Execute one top-level-owned block, recursively preserving lexical directives."""
    prog = prepared.program
    profile_registry = self.rt._profile_registry
    last_result: Any = _NO_TOP_LEVEL_RESULT

    for item in sequence:
        checkpoint(self.rt, "statement", env=env, span=item.loc)
        if isinstance(item, IfStmt):
            with block_agent_profile_scope(state.active_profile):
                condition = evaluate_condition(self.rt, item, env)
            body = item.body if condition else item.else_body
            if body is None:
                continue
            branch_state = replace(state)
            with branch_trace(self.rt, item, alternative=not condition):
                with child_scope(self.rt, env) as branch_env, run_budget_scope(RunBudget()):
                    result = self._execute_top_level_sequence(
                        prepared,
                        branch_env,
                        branch_state,
                        body,
                    )
            if result is not _NO_TOP_LEVEL_RESULT:
                return result
            continue
        if isinstance(item, LoopStmt):
            with loop_trace(self.rt, item):
                records: list[tuple[Any, _TopLevelExecutionState]] = []
                values = evaluate_iterable(self.rt, item, env)
                with managed_loop_iterator(values):
                    for value in values:
                        iteration_state = replace(state)
                        with (
                            loop_binding(self.rt, env, item.binder, value) as iteration_env,
                            run_budget_scope(RunBudget()),
                        ):
                            result = self._execute_top_level_sequence(
                                prepared,
                                iteration_env,
                                iteration_state,
                                item.body,
                            )
                            records.append((iteration_env, iteration_state))
                        if result is not _NO_TOP_LEVEL_RESULT:
                            return result

                if item.map_body is not None:
                    frames = [record[0] for record in records]
                    results: list[Any] = [_NO_TOP_LEVEL_RESULT] * len(frames)

                    def run_map(index: int, frame: Any) -> None:
                        _, map_state = records[index]
                        with active_scope(self.rt, frame), run_budget_scope(RunBudget()):
                            with block_agent_profile_scope(map_state.active_profile):
                                results[index] = self._execute_top_level_sequence(
                                    prepared,
                                    frame,
                                    map_state,
                                    item.map_body or [],
                                )

                    run_map_continuations(self.rt, item, frames, run_map)
                    for result in results:
                        if result is not _NO_TOP_LEVEL_RESULT:
                            return result
            continue
        if isinstance(item, ConditionalLoopStmt):
            completed_iterations = 0
            with loop_trace(self.rt, item):
                while True:
                    with block_agent_profile_scope(state.active_profile):
                        condition = evaluate_condition(self.rt, item, env)
                    if not condition:
                        break
                    assert_conditional_loop_capacity(self.rt, completed_iterations)
                    iteration_state = replace(state)
                    with child_scope(self.rt, env) as iteration_env, run_budget_scope(RunBudget()):
                        result = self._execute_top_level_sequence(
                            prepared,
                            iteration_env,
                            iteration_state,
                            item.body,
                        )
                    completed_iterations += 1
                    if result is not _NO_TOP_LEVEL_RESULT:
                        return result
            continue
        if isinstance(item, ProcedureDef):
            if not item.auto_spec and not item.ai_prompt:
                self.rt.procedure(self.proc_compiler.compile_procedure(item))
            continue
        if isinstance(item, ProcDefStmt) and item.proc is not None:
            child = item.proc
            child.captured_agent_profile = state.active_profile.frozen_copy()
            child.lexical_env = env
            self.rt.procedure(self.proc_compiler._make_procedure_impl(child, env))
            continue
        if isinstance(item, TypeDef):
            register_type(self.rt, item, self.rt._type_env_stack[0])
            continue
        if isinstance(item, TypeDefStmt) and item.typedef is not None:
            register_type(
                self.rt,
                item.typedef,
                self.rt._type_env_stack[0],
                env,
            )
            continue
        if isinstance(item, ModuleImport):
            module_path = self._resolve_import_path(
                prog,
                item,
                source_path=prepared.import_source_path,
            )
            loaded = self.module_loader.loaded_for(module_path)
            self._merge_exports(
                item,
                loaded.path,
                loaded.program,
                loaded.runtime,
                target_env=env,
            )
            continue
        if isinstance(item, AgentDirectiveStmt):
            state.active_profile = state.active_profile.merge(
                eval_agent_directive(self.rt, item, env)
            )
            continue
        if isinstance(item, AdapterDirectiveStmt):
            state.active_profile = state.active_profile.merge(
                eval_adapter_directive(self.rt, item, env)
            )
            continue
        if isinstance(item, ModelDirectiveStmt):
            state.active_profile = state.active_profile.merge(
                eval_model_directive(self.rt, item, env)
            )
            continue
        if isinstance(item, EffortDirectiveStmt):
            state.active_profile = state.active_profile.merge(
                eval_effort_directive(self.rt, item, env)
            )
            continue
        if isinstance(item, ApprovalDirectiveStmt):
            state.active_profile = state.active_profile.merge(
                eval_approval_directive(self.rt, item, env)
            )
            continue
        if isinstance(item, HooksDirectiveStmt):
            hook_patch = eval_hooks_directive(self.rt, item, env).with_hook_enforcement()
            state.active_profile = state.active_profile.merge(hook_patch)
            continue
        if isinstance(item, HistoryDirectiveStmt):
            state.active_profile = state.active_profile.merge(
                eval_history_directive(self.rt, item, env)
            )
            continue
        if isinstance(item, SkillsDirectiveStmt):
            state.active_profile = state.active_profile.merge(
                eval_skills_directive(self.rt, item, env)
            )
            continue
        if isinstance(item, CodeModeDirectiveStmt):
            state.active_profile = state.active_profile.merge(
                eval_codemode_directive(self.rt, item, env)
            )
            continue
        if isinstance(item, RequiresDirectiveStmt):
            state.active_profile = state.active_profile.merge(
                eval_requires_directive(self.rt, item, env)
            )
            continue
        if isinstance(item, BudgetDirectiveStmt):
            patch = eval_budget_directive(self.rt, item, env)
            state.active_profile = state.active_profile.merge(patch)
            ledger = current_run_budget()
            if ledger is not None and patch.run_budget is not None:
                ledger.set_budget(patch.run_budget)
            continue
        if isinstance(item, SystemDirectiveStmt):
            state.active_profile = state.active_profile.merge(
                eval_system_directive(self.rt, item, env)
            )
            continue
        if isinstance(item, McpDirectiveStmt):
            state.active_profile = state.active_profile.merge(
                eval_mcp_directive(self.rt, item, env)
            )
            continue
        if isinstance(item, SettingsDirectiveStmt):
            state.active_profile = state.active_profile.merge(
                eval_settings_directive(self.rt, item, env)
            )
            continue
        if isinstance(item, ArtifactsDirectiveStmt):
            state.active_profile = state.active_profile.merge(
                eval_artifacts_directive(self.rt, item, env)
            )
            continue
        if isinstance(item, UseDirectiveStmt):
            patch = eval_use_directive(
                self.rt,
                item,
                profile_registry,
                env,
                available_procedures=set(self.rt._procedures.keys()),
            )
            state.active_profile = state.active_profile.merge(patch)
            ledger = current_run_budget()
            if ledger is not None and patch.run_budget is not None:
                ledger.set_budget(patch.run_budget)
            continue
        if isinstance(item, ProfileDefStmt):
            profile_registry[item.name] = prepared.local_profiles[item.name]
            continue
        if isinstance(item, TemplateLine):
            raise ValueError(
                "Top-level TemplateLine is not a valid program node; "
                "use '>>' for a top-level template prompt."
            )
        if isinstance(item, TaskGroupStmt):
            from .task_groups import execute_task_group

            with block_agent_profile_scope(state.active_profile):
                execute_task_group(self.rt, item, env, state.active_profile)
            continue
        if isinstance(item, TaskStmt):
            with block_agent_profile_scope(state.active_profile):
                self.stmt_compiler.execute_task(item, env, state.active_profile)
            continue
        if isinstance(item, AwaitStmt):
            with block_agent_profile_scope(state.active_profile):
                self.stmt_compiler.execute_await(item, env)
            continue
        if isinstance(item, ReturnStmt):
            with block_agent_profile_scope(state.active_profile):
                return self.stmt_compiler.compile_return(item)(**env)
        if isinstance(item, ReturnBlockStmt):
            with block_agent_profile_scope(state.active_profile):
                return self.stmt_compiler.compile_return_block(item, env)
        with block_agent_profile_scope(state.active_profile):
            self.executor.execute_single_statement(item, env, is_toplevel=True)

    return last_result


_NO_TOP_LEVEL_RESULT = object()
