"""Top-level program compilation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kedi.agent_profile import AgentProfile
from kedi.agent_profile.compile import (
    build_profile_registry,
    capture_procedure_profiles,
    validate_subagent_graph,
)
from kedi.agent_profile.definitions import ProfileDefinition
from kedi.core import KediRuntime
from kedi.errors import KediTraceFrame
from kedi.executors.base import SupportsIncrementalPrelude
from kedi.lang.ast import ModuleImport, Program, ProgramWithValidation, TemplateLine
from kedi.lang.compiler.procedure_compiler import ProcedureCompiler
from kedi.lang.compiler.statement_compiler import StatementCompiler, StatementExecutor
from kedi.lang.compiler.type_system import register_type
from kedi.module_registry import resolve_module_path
from kedi.run_budget import RunBudget
from kedi.source_loading import CompositeSourceMap, load_kedi_source_bundle
from kedi.telemetry import capture_enabled, runtime_span
from kedi.telemetry.naming import program_name

from .top_level import _NO_TOP_LEVEL_RESULT

if TYPE_CHECKING:
    from kedi.agent_adapter.conversation import ConversationState
    from kedi.agent_adapter.meta import AgentAdapter
    from kedi.agent_adapter.subagents.limits import SubagentUsageLimits
    from kedi.executors.base import Executor

from kedi.lang.compiler import import_exports as _import_exports
from kedi.lang.compiler import top_level as _top_level
from kedi.lang.compiler.import_exports import _StaticExportSurface
from kedi.lang.compiler.module_loader import _ModuleLoader, _program_source_path
from kedi.lang.compiler.top_level import _TopLevelExecutionState


@dataclass(frozen=True)
class _PreparedProgram:
    """A program whose imports and declarations are ready for top-level execution."""

    program: Program
    sequence: tuple[Any, ...]
    toplevel_return: Any | None
    local_profiles: dict[str, ProfileDefinition]
    import_source_path: Path | None = None


def compile_program(
    prog: Program,
    *,
    adapter: AgentAdapter[Any] | None = None,
    executor: Executor | None = None,
    runtime_globals: dict[str, Any] | None = None,
    engine: Any | None = None,
    default_agent_profile: AgentProfile | None = None,
    subagent_max_depth: int = 5,
    subagent_max_concurrency: int = 4,
    subagent_timeout_seconds: float | None = 120.0,
    subagent_usage_limits: SubagentUsageLimits | None = None,
    subagent_state_path: str | Path | None = None,
    conversation_state: ConversationState | None = None,
    loop_iteration_limit: int = 10_000,
    run_budget: RunBudget | None = None,
) -> KediRuntime:
    """Compile a Kedi program to an executable runtime.

    This is the main entry point for the compiler.
    """
    source_path = _program_source_path(prog)
    display_name = program_name(str(source_path) if source_path is not None else None)
    validation_case_count = 0
    if isinstance(prog, ProgramWithValidation):
        validation_case_count = sum(len(suite.cases) for suite in prog.tests) + sum(
            len(suite.metrics) for suite in prog.evals
        )
    attributes: dict[str, bool | str | int] = {
        "kedi.program.name": display_name,
        "kedi.compile.type_count": len(prog.type_defs),
        "kedi.compile.profile_count": len(prog.profile_defs),
        "kedi.compile.procedure_count": len(prog.procedures),
        "kedi.compile.import_count": len(prog.imports),
        "kedi.compile.validation_case_count": validation_case_count,
        "kedi.compile.cache_hit": False,
        "kedi.executor.name": type(executor).__name__
        if executor is not None
        else "DefaultExecutor",
    }
    if source_path is not None and capture_enabled("source_path"):
        attributes["code.file.path"] = str(source_path)

    compiler = ProgramCompiler(
        adapter,
        executor=executor,
        runtime_globals=runtime_globals,
        engine=engine,
        default_agent_profile=default_agent_profile,
        subagent_max_depth=subagent_max_depth,
        subagent_max_concurrency=subagent_max_concurrency,
        subagent_timeout_seconds=subagent_timeout_seconds,
        subagent_usage_limits=subagent_usage_limits,
        subagent_state_path=subagent_state_path,
        conversation_state=conversation_state,
        loop_iteration_limit=loop_iteration_limit,
        run_budget=run_budget,
    )
    with runtime_span(
        f"compile {display_name}",
        "compile_program",
        attributes=attributes,
        metric_attributes={"cache_hit": False},
    ) as active:
        try:
            return compiler.compile(prog)
        except BaseException as exc:
            active.set_attribute("error.type", type(exc).__name__)
            raise


class ProgramCompiler:
    """Orchestrates program compilation."""

    def __init__(
        self,
        adapter: AgentAdapter[Any] | None,
        *,
        executor: Executor | None = None,
        module_stack: tuple[Path, ...] = (),
        module_loader: _ModuleLoader | None = None,
        runtime_globals: dict[str, Any] | None = None,
        engine: Any | None = None,
        default_agent_profile: AgentProfile | None = None,
        subagent_max_depth: int = 5,
        subagent_max_concurrency: int = 4,
        subagent_timeout_seconds: float | None = 120.0,
        subagent_usage_limits: SubagentUsageLimits | None = None,
        subagent_state_path: str | Path | None = None,
        conversation_state: ConversationState | None = None,
        loop_iteration_limit: int = 10_000,
        run_budget: RunBudget | None = None,
    ):
        self.adapter = adapter
        self.module_stack = module_stack
        self.module_loader = module_loader or _ModuleLoader()
        self.engine = engine
        self.default_agent_profile = default_agent_profile
        self.subagent_max_depth = subagent_max_depth
        self.subagent_max_concurrency = subagent_max_concurrency
        self.subagent_timeout_seconds = subagent_timeout_seconds
        self.subagent_usage_limits = subagent_usage_limits
        self.subagent_state_path = subagent_state_path
        self.conversation_state = conversation_state
        self.loop_iteration_limit = loop_iteration_limit
        self.rt = KediRuntime(
            adapter=adapter,
            executor=executor,
            engine=engine,
            default_agent_profile=default_agent_profile,
            subagent_max_depth=subagent_max_depth,
            subagent_max_concurrency=subagent_max_concurrency,
            subagent_timeout_seconds=subagent_timeout_seconds,
            subagent_usage_limits=subagent_usage_limits,
            subagent_state_path=subagent_state_path,
            conversation_state=conversation_state,
            loop_iteration_limit=loop_iteration_limit,
            run_budget=run_budget,
        )
        if runtime_globals:
            self.rt.set_initial_globals(runtime_globals)
        self.proc_compiler = ProcedureCompiler(self.rt)
        self.executor = StatementExecutor(self.rt)
        self.stmt_compiler = StatementCompiler(self.rt)

    def compile(self, prog: Program) -> KediRuntime:
        """Compile program and return configured runtime."""
        from ..parser.control_flow_validation import validate_control_flow

        validate_control_flow(prog)
        # Initialize type environment
        self.rt._type_env_stack = [{}]
        self.rt._program = prog
        self.rt._source_map = prog.source_map
        self.rt._source_text = prog.source_text
        current_path = self._program_path(prog)
        if prog.package is not None and (
            current_path is None or current_path.name != "package.kedi"
        ):
            raise ValueError("package metadata is only allowed in package.kedi")
        active_stack = self.module_stack
        if current_path is not None:
            if current_path in active_stack:
                cycle = " -> ".join(str(path) for path in (*active_stack, current_path))
                raise RuntimeError(f"Cyclic Kedi import detected: {cycle}")
            active_stack = (*active_stack, current_path)

        self.rt._profile_registry = {}

        prepared = self._prepare_program_contents(prog, active_stack=active_stack)

        # Setup main entry point when there is executable top-level code.
        if prog.ordered_toplevel or prog.toplevel_body or prog.toplevel_return is not None:
            self._setup_main(prepared)

        # Attach validation suites
        self._attach_validation(prog)

        return self.rt

    def prepare_fragment(
        self,
        prog: Program,
        *,
        import_source_path: Path,
    ) -> _PreparedProgram:
        """Prepare one incremental fragment without resetting existing runtime state."""
        from ..parser.control_flow_validation import validate_control_flow

        validate_control_flow(prog, interactive=True)
        if prog.package is not None:
            raise ValueError("package metadata is not supported in interactive fragments")
        if prog.exports:
            raise ValueError("export directives are not supported in interactive fragments")
        if isinstance(prog, ProgramWithValidation) and (prog.tests or prog.evals):
            raise ValueError("validation suites are not supported in interactive fragments")

        self._merge_fragment_source(prog)
        root_path = import_source_path.resolve()
        active_stack = (*self.module_stack, root_path)
        return self._prepare_program_contents(
            prog,
            active_stack=active_stack,
            import_source_path=root_path,
            incremental=True,
        )

    def new_top_level_state(self) -> _TopLevelExecutionState:
        """Create mutable directive state for one top-level execution stream."""
        active = self.rt.default_agent_profile
        return _TopLevelExecutionState(active, active.adapter_kind)

    def initialize_fragments(self) -> dict[str, Any]:
        """Initialize persistent state for a new incremental execution stream."""
        self.rt._type_env_stack = [{}]
        self.rt._profile_registry = {}
        from kedi.lang.compiler.environment import ScopeFrame

        env = ScopeFrame(self.rt._initial_globals_env)
        self.rt._globals_env = env
        return env

    def execute_fragment(
        self,
        prepared: _PreparedProgram,
        env: dict[str, Any],
        state: _TopLevelExecutionState,
    ) -> Any:
        """Execute one prepared fragment exactly once against ``env``."""
        from kedi.lang.compiler.environment import native_value
        from kedi.promise import force

        self.rt._globals_env = env

        def fragment() -> Any:
            token = self.rt._current_env.set(env)
            try:
                return self._execute_top_level(prepared, env, state)
            finally:
                self.rt._current_env.reset(token)

        return native_value(force(self.rt._invoke_procedure(fragment, [])))

    def _prepare_program_contents(
        self,
        prog: Program,
        *,
        active_stack: tuple[Path, ...],
        import_source_path: Path | None = None,
        incremental: bool = False,
    ) -> _PreparedProgram:
        self._compile_imports(
            prog,
            active_stack,
            import_source_path=import_source_path,
        )

        if prog.prelude is not None:
            self._setup_prelude(
                prog.prelude,
                prog.prelude_loc,
                incremental=incremental,
            )

        local_profiles = build_profile_registry(prog)
        self.rt._profile_registry.update(local_profiles)
        validate_subagent_graph(self.rt._profile_registry)
        procedure_bindings = capture_procedure_profiles(self.rt, prog)

        type_env = self.rt._type_env_stack[0]
        for typedef in prog.type_defs:
            register_type(self.rt, typedef, type_env)

        for proc in prog.procedures:
            if proc.auto_spec or proc.ai_prompt:
                continue
            compiled = self.proc_compiler.compile_procedure(proc)
            procedure_bindings[id(proc)] = compiled
            self.rt.procedure(compiled)

        sequence = list(prog.ordered_toplevel or prog.toplevel_body)
        if prog.toplevel_return is not None and not any(
            item is prog.toplevel_return for item in sequence
        ):
            sequence.append(prog.toplevel_return)
        return _PreparedProgram(
            program=prog,
            sequence=tuple(sequence),
            toplevel_return=prog.toplevel_return,
            local_profiles=local_profiles,
            import_source_path=import_source_path,
        )

    def _merge_fragment_source(self, prog: Program) -> None:
        self.rt._program = prog
        self.rt._source_text = prog.source_text
        current = self.rt._source_map
        incoming = prog.source_map
        if incoming is None:
            return
        if current is None:
            self.rt._source_map = incoming
            return
        self.rt._source_map = CompositeSourceMap((*current.documents, *incoming.documents))

    def _compile_imports(
        self,
        prog: Program,
        active_stack: tuple[Path, ...],
        *,
        import_source_path: Path | None = None,
    ) -> None:
        for module_import in prog.imports:
            with self.rt.trace_frame(
                KediTraceFrame(
                    kind="module_import",
                    label=f"import {module_import.module}",
                    span=module_import.loc,
                )
            ):
                module_path = self._resolve_import_path(
                    prog,
                    module_import,
                    source_path=import_source_path,
                )
                module_path, imported_prog = self.module_loader.program_for(
                    module_path,
                    self._load_imported_program,
                )
                self._validate_selected_import_before_execution(
                    module_import,
                    module_path,
                    imported_prog,
                )

                def compile_module(program: Program) -> KediRuntime:
                    # Share immutable CLI globals and the loader, while keeping the
                    # module's runtime state private to its canonical path.
                    return ProgramCompiler(
                        self.adapter,
                        executor=self.rt.executor,
                        module_stack=active_stack,
                        module_loader=self.module_loader,
                        runtime_globals=dict(self.rt._initial_globals_env),
                        engine=self.engine,
                        default_agent_profile=self.default_agent_profile,
                        subagent_max_depth=self.subagent_max_depth,
                        subagent_max_concurrency=self.subagent_max_concurrency,
                        subagent_timeout_seconds=self.subagent_timeout_seconds,
                        subagent_usage_limits=self.subagent_usage_limits,
                        subagent_state_path=self.subagent_state_path,
                        conversation_state=self.conversation_state,
                        loop_iteration_limit=self.loop_iteration_limit,
                    ).compile(program)

                loaded = self.module_loader.load(module_path, imported_prog, compile_module)
                self._merge_exports(module_import, loaded.path, loaded.program, loaded.runtime)

    def _validate_selected_import_before_execution(
        self,
        module_import: ModuleImport,
        module_path: Path,
        imported_prog: Program,
    ) -> None:
        """Reject provably absent selective imports before module initialization."""
        return _import_exports._validate_selected_import_before_execution(
            self, module_import, module_path, imported_prog
        )

    def _static_export_surface(
        self,
        module_path: Path,
        prog: Program,
        *,
        visiting: set[Path],
    ) -> _StaticExportSurface:
        """Inventory an export surface without running embedded code.

        Explicit export directives are authoritative even when their value is
        created dynamically. ``export: *`` additionally receives statically
        knowable local and imported names. A genuinely dynamic prelude keeps
        the surface open, preserving the historical runtime fallback only for
        cases that cannot be proven before initialization.
        """
        return _import_exports._static_export_surface(self, module_path, prog, visiting=visiting)

    def _static_local_runtime_names(self, prog: Program) -> tuple[set[str], bool]:
        return _import_exports._static_local_runtime_names(self, prog)

    def _static_python_names(self, code: str) -> tuple[set[str], bool]:
        return _import_exports._static_python_names(self, code)

    @staticmethod
    def _template_output_names(lines: list[TemplateLine]) -> set[str]:
        return _import_exports._template_output_names(lines)

    def _resolve_import_path(
        self,
        prog: Program,
        module_import: ModuleImport,
        *,
        source_path: Path | None = None,
    ) -> Path:
        try:
            return resolve_module_path(
                module_import.module,
                source_path=source_path or self._program_path(prog),
            )
        except (FileNotFoundError, ValueError) as exc:
            raise ModuleNotFoundError(str(exc)) from exc

    def _load_imported_program(self, module_path: Path) -> Program:
        from kedi.lang.parser import parse_program

        loaded = load_kedi_source_bundle(
            module_path,
            include_val=False,
            include_cache=True,
        )
        return parse_program(
            loaded.text,
            source_path=str(module_path),
            source_map=loaded.source_map,
        )

    def _merge_exports(
        self,
        module_import: ModuleImport,
        module_path: Path,
        imported_prog: Program,
        imported_rt: KediRuntime,
        *,
        target_env: dict[str, Any] | None = None,
    ) -> None:
        return _import_exports._merge_exports(
            self, module_import, module_path, imported_prog, imported_rt, target_env=target_env
        )

    def _selected_import_names(
        self,
        module_import: ModuleImport,
        export_names: list[str],
    ) -> list[str]:
        return _import_exports._selected_import_names(self, module_import, export_names)

    def _remove_imported_binding(self, name: str) -> None:
        """Make a later import authoritative even when the binding kind changes."""
        return _import_exports._remove_imported_binding(self, name)

    def _bind_imported_profile(
        self,
        profile: AgentProfile | ProfileDefinition,
        imported_rt: KediRuntime,
        *,
        bound_profiles: dict[str, ProfileDefinition] | None = None,
    ) -> AgentProfile | ProfileDefinition:
        """Attach imported private dependencies to one profile without exporting them."""
        return _import_exports._bind_imported_profile(
            self, profile, imported_rt, bound_profiles=bound_profiles
        )

    def _imported_tool_spec(self, tool_name: str, imported_rt: KediRuntime) -> Any | None:
        return _import_exports._imported_tool_spec(self, tool_name, imported_rt)

    def _profile_tool_names(self, profile: AgentProfile | ProfileDefinition) -> tuple[str, ...]:
        return _import_exports._profile_tool_names(self, profile)

    def _profile_use_names(self, profile: ProfileDefinition) -> tuple[str, ...]:
        return _import_exports._profile_use_names(self, profile)

    def _profile_subagent_names(
        self,
        profile: AgentProfile | ProfileDefinition,
    ) -> tuple[str, ...]:
        return _import_exports._profile_subagent_names(self, profile)

    def _export_names(self, prog: Program, imported_rt: KediRuntime) -> list[str]:
        return _import_exports._export_names(self, prog, imported_rt)

    def _public_runtime_names(self, rt: KediRuntime) -> list[str]:
        return _import_exports._public_runtime_names(self, rt)

    def _is_public_name(self, name: str) -> bool:
        return _import_exports._is_public_name(self, name)

    def _resolve_export(
        self,
        name: str,
        imported_rt: KediRuntime,
        module_path: Path,
    ) -> tuple[str, Any]:
        return _import_exports._resolve_export(self, name, imported_rt, module_path)

    def _wrap_imported_procedure(
        self,
        name: str,
        fn: Any,
        imported_rt: KediRuntime,
    ) -> Any:
        return _import_exports._wrap_imported_procedure(self, name, fn, imported_rt)

    def _program_path(self, prog: Program) -> Path | None:
        return _program_source_path(prog)

    def _setup_prelude(
        self,
        prelude_code: str,
        prelude_loc,
        *,
        incremental: bool = False,
    ) -> None:
        """Execute prelude and store environment."""
        with self.rt.trace_frame(KediTraceFrame(kind="prelude", label="prelude", span=prelude_loc)):
            executor = self.rt.executor
            if not incremental:
                self.rt._prelude_env = executor.execute_prelude(self.rt, prelude_code)
                return
            if not isinstance(executor, SupportsIncrementalPrelude):
                raise TypeError(
                    f"Executor {type(executor).__name__!r} does not support incremental preludes"
                )
            self.rt._prelude_env = executor.execute_prelude_incremental(
                self.rt,
                prelude_code,
                self.rt._prelude_env,
            )

    def _setup_main(self, prepared: _PreparedProgram) -> None:
        """Setup main entry point for top-level execution."""

        @self.rt.main
        def _main() -> Any:
            from kedi.lang.compiler.environment import ScopeFrame

            kwargs: dict[str, Any] = ScopeFrame(self.rt._initial_globals_env)
            self.rt._globals_env = kwargs
            token = self.rt._current_env.set(kwargs)
            try:
                return self._execute_top_level(prepared, kwargs, self.new_top_level_state())
            finally:
                self.rt._current_env.reset(token)

    def _execute_top_level(
        self,
        prepared: _PreparedProgram,
        env: dict[str, Any],
        state: _TopLevelExecutionState,
    ) -> Any:
        with self.rt.trace_frame(
            KediTraceFrame(kind="program", label="top-level program", span=None)
        ):
            block_result = self._execute_top_level_sequence(
                prepared,
                env,
                state,
                prepared.sequence,
            )

            return None if block_result is _NO_TOP_LEVEL_RESULT else block_result

    def _execute_top_level_sequence(
        self,
        prepared: _PreparedProgram,
        env: dict[str, Any],
        state: _TopLevelExecutionState,
        sequence: tuple[Any, ...] | list[Any],
    ) -> Any:
        """Execute one top-level-owned block, recursively preserving lexical directives."""
        return _top_level._execute_top_level_sequence(self, prepared, env, state, sequence)

    def _attach_validation(self, prog: Program) -> None:
        """Attach validation suites if present.

        The parser returns ``ProgramWithValidation`` when ``@test:`` or
        ``@eval:`` blocks are present; we forward the public ``.tests``
        / ``.evals`` collections onto the runtime so downstream
        consumers (``kedi-lsp``, ``tests_runner``, ``app.py``) can
        read them as normal attributes.
        """
        from kedi.lang.ast import ProgramWithValidation

        if isinstance(prog, ProgramWithValidation):
            if prog.tests:
                self.rt._validation_tests = prog.tests
            if prog.evals:
                self.rt._validation_evals = prog.evals
