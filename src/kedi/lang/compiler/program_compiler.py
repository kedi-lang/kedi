"""Top-level program compilation orchestration."""

from __future__ import annotations

import ast as py_ast
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

from ...agent_profile import AgentProfile, block_agent_profile_scope
from ...agent_profile.compile import (
    build_profile_registry,
    capture_procedure_profiles,
    validate_subagent_graph,
)
from ...agent_profile.model import (
    eval_adapter_directive,
    eval_agent_directive,
    eval_approval_directive,
    eval_artifacts_directive,
    eval_effort_directive,
    eval_history_directive,
    eval_skills_directive,
    eval_mcp_directive,
    eval_model_directive,
    eval_settings_directive,
    eval_system_directive,
)
from ...agent_profile.registry import ProfileDefinition
from ...agent_profile.tools import eval_use_directive
from ...core import KediRuntime
from ...errors import KediTraceFrame
from ...executors.base import SupportsIncrementalPrelude
from ...module_registry import resolve_module_path
from ...source_loading import CompositeSourceMap, load_kedi_source_bundle
from ...telemetry import capture_enabled, runtime_span
from ...telemetry.naming import program_name
from ..ast import (
    AdapterDirectiveStmt,
    AgentDirectiveStmt,
    ApprovalDirectiveStmt,
    ArtifactsDirectiveStmt,
    AssignBlockStmt,
    AssignStmt,
    EffortDirectiveStmt,
    HistoryDirectiveStmt,
    SkillsDirectiveStmt,
    McpDirectiveStmt,
    ModelDirectiveStmt,
    ModuleImport,
    OutputSegment,
    ProcedureDef,
    ProfileDefStmt,
    Program,
    ProgramWithValidation,
    RawInvokeStmt,
    ReturnStmt,
    SettingsDirectiveStmt,
    SystemDirectiveStmt,
    TemplateBlock,
    TemplateLine,
    TypeDef,
    UseDirectiveStmt,
)
from .evaluation import evaluate_python_inline, execute_python_block
from .procedure_compiler import ProcedureCompiler
from .statement_compiler import StatementCompiler, StatementExecutor
from .type_system import register_type

if TYPE_CHECKING:
    from ...agent_adapter.conversation import ConversationState
    from ...agent_adapter.meta import AgentAdapter
    from ...agent_adapter.subagents.limits import SubagentUsageLimits
    from ...executors.base import Executor


def _assert_adapter_selection_not_mixed(current: str | None, incoming: str) -> None:
    if current is not None and current != incoming:
        raise RuntimeError("Cannot mix '> agent:' and '> adapter:' in the same scope")


@dataclass(frozen=True)
class _LoadedModule:
    """One executable imported module for one root-program compilation."""

    path: Path
    program: Program
    runtime: KediRuntime


@dataclass(frozen=True)
class _StaticExportSurface:
    """Names provably exported without executing a module."""

    names: frozenset[str]
    complete: bool


@dataclass(frozen=True)
class _PreparedProgram:
    """A program whose imports and declarations are ready for top-level execution."""

    program: Program
    sequence: tuple[Any, ...]
    toplevel_return: Any | None
    local_profiles: dict[str, ProfileDefinition]
    import_source_path: Path | None = None


@dataclass
class _TopLevelExecutionState:
    """Directive state that advances as top-level statements execute."""

    active_profile: AgentProfile
    adapter_selection_kind: str | None


class _ModuleLoader:
    """Compilation-local imported-module cache with deterministic cycle reporting."""

    def __init__(self) -> None:
        self._programs: dict[Path, Program] = {}
        self._loaded: dict[Path, _LoadedModule] = {}
        self._loading: list[Path] = []

    def program_for(
        self,
        module_path: Path,
        load_program: Callable[[Path], Program],
    ) -> tuple[Path, Program]:
        path = module_path.resolve()
        program = self._programs.get(path)
        if program is None:
            program = load_program(path)
            self._programs[path] = program
        return path, program

    def load(
        self,
        module_path: Path,
        program: Program,
        compile_module: Callable[[Program], KediRuntime],
    ) -> _LoadedModule:
        path = module_path.resolve()
        loaded = self._loaded.get(path)
        if loaded is not None:
            return loaded
        if path in self._loading:
            cycle_start = self._loading.index(path)
            cycle = " -> ".join(str(item) for item in (*self._loading[cycle_start:], path))
            raise RuntimeError(f"Cyclic Kedi import detected: {cycle}")

        self._loading.append(path)
        try:
            attributes: dict[str, bool | str] = {
                "kedi.module.name": path.stem,
                "kedi.compile.cache_hit": False,
            }
            if capture_enabled("source_path"):
                attributes["code.file.path"] = str(path)
            with runtime_span(
                f"compile module {path.stem}",
                "compile_module",
                attributes=attributes,
            ) as active:
                try:
                    runtime = compile_module(program)
                    # Module initialization follows Python import semantics: top-level
                    # code runs once, then every importer shares the loaded runtime.
                    runtime.run_main()
                except BaseException as exc:
                    active.set_attribute("error.type", type(exc).__name__)
                    raise
                loaded = _LoadedModule(path=path, program=program, runtime=runtime)
                self._loaded[path] = loaded
                return loaded
        finally:
            self._loading.pop()

    def loaded_for(self, module_path: Path) -> _LoadedModule:
        """Return a previously initialized module for a source-order rebinding."""
        path = module_path.resolve()
        try:
            return self._loaded[path]
        except KeyError as exc:  # pragma: no cover - defensive compiler invariant.
            raise RuntimeError(
                f"Kedi module {path} was not initialized during compilation"
            ) from exc


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


def _program_source_path(prog: Program) -> Path | None:
    loc = prog.loc
    if loc is not None and loc.source_path:
        return Path(loc.source_path).resolve()
    source_map = prog.source_map
    if source_map is not None and source_map.documents:
        return Path(source_map.documents[0].path).resolve()
    return None


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
        )
        if runtime_globals:
            self.rt.set_initial_globals(runtime_globals)
        self.proc_compiler = ProcedureCompiler(self.rt)
        self.executor = StatementExecutor(self.rt)
        self.stmt_compiler = StatementCompiler(self.rt)

    def compile(self, prog: Program) -> KediRuntime:
        """Compile program and return configured runtime."""
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
        env = dict(self.rt._initial_globals_env)
        self.rt._globals_env = env
        return env

    def execute_fragment(
        self,
        prepared: _PreparedProgram,
        env: dict[str, Any],
        state: _TopLevelExecutionState,
    ) -> Any:
        """Execute one prepared fragment exactly once against ``env``."""
        from ...promise import force
        from .environment import native_value

        self.rt._globals_env = env

        def fragment() -> Any:
            return self._execute_top_level(prepared, env, state)

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
        capture_procedure_profiles(self.rt, prog)

        type_env = self.rt._type_env_stack[0]
        for typedef in prog.type_defs:
            register_type(self.rt, typedef, type_env)

        for proc in prog.procedures:
            if proc.auto_spec or proc.ai_prompt:
                continue
            self.rt.procedure(self.proc_compiler.compile_procedure(proc))

        return _PreparedProgram(
            program=prog,
            sequence=tuple(prog.ordered_toplevel or prog.toplevel_body),
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
        if not module_import.names:
            return

        surface = self._static_export_surface(module_path, imported_prog, visiting=set())
        missing = [name for name in module_import.names if name not in surface.names]
        if missing and surface.complete:
            rendered = ", ".join(repr(name) for name in missing)
            raise ImportError(f"Kedi module {module_import.module!r} does not export {rendered}")

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
        path = module_path.resolve()
        if path in visiting:
            return _StaticExportSurface(frozenset(), complete=False)

        next_visiting = {*visiting, path}
        local_names, complete = self._static_local_runtime_names(prog)
        imported_names: set[str] = set()
        for nested_import in prog.imports:
            nested_path = self._resolve_import_path(prog, nested_import)
            nested_path, nested_prog = self.module_loader.program_for(
                nested_path,
                self._load_imported_program,
            )
            nested_surface = self._static_export_surface(
                nested_path,
                nested_prog,
                visiting=next_visiting,
            )
            complete = complete and nested_surface.complete
            if nested_import.names:
                imported_names.update(
                    name for name in nested_import.names if name in nested_surface.names
                )
            else:
                imported_names.update(nested_surface.names)

        names: set[str] = set()
        for export in prog.exports:
            names.update(export.names)
            if export.export_all:
                names.update(
                    name for name in (*local_names, *imported_names) if self._is_public_name(name)
                )

        # A module with only explicit exports is fully known even if its
        # prelude dynamically creates values: the explicit list is its public
        # contract. Dynamic names only affect an export-star surface.
        if not any(export.export_all for export in prog.exports):
            complete = True
        return _StaticExportSurface(frozenset(names), complete)

    def _static_local_runtime_names(self, prog: Program) -> tuple[set[str], bool]:
        names = {proc.name for proc in prog.procedures}
        names.update(typedef.name for typedef in prog.type_defs)
        names.update(prog.profile_defs)
        complete = True

        if prog.prelude is not None:
            prelude_names, prelude_complete = self._static_python_names(prog.prelude)
            names.update(prelude_names)
            complete = prelude_complete

        for stmt in prog.toplevel_body:
            if isinstance(stmt, (AssignStmt, AssignBlockStmt, RawInvokeStmt)):
                names.add(stmt.name)
            elif isinstance(stmt, TemplateBlock):
                names.update(self._template_output_names(stmt.lines))
            elif isinstance(stmt, TemplateLine):
                names.update(self._template_output_names([stmt]))
        return names, complete

    def _static_python_names(self, code: str) -> tuple[set[str], bool]:
        try:
            module = py_ast.parse(code)
        except SyntaxError:
            return set(), False

        names: set[str] = set()
        dynamic = False
        for stmt in module.body:
            if isinstance(stmt, (py_ast.FunctionDef, py_ast.AsyncFunctionDef, py_ast.ClassDef)):
                names.add(stmt.name)
                continue
            if isinstance(stmt, py_ast.Import):
                names.update(alias.asname or alias.name.partition(".")[0] for alias in stmt.names)
                continue
            if isinstance(stmt, py_ast.ImportFrom):
                if any(alias.name == "*" for alias in stmt.names):
                    dynamic = True
                else:
                    names.update(alias.asname or alias.name for alias in stmt.names)
                continue
            names.update(
                node.id
                for node in py_ast.walk(stmt)
                if isinstance(node, py_ast.Name) and isinstance(node.ctx, py_ast.Store)
            )
            dynamic = dynamic or any(
                isinstance(node, py_ast.Call)
                and isinstance(node.func, py_ast.Name)
                and node.func.id == "exec"
                for node in py_ast.walk(stmt)
            )
        return names, not dynamic

    @staticmethod
    def _template_output_names(lines: list[TemplateLine]) -> set[str]:
        return {
            segment.name
            for line in lines
            for segment in line.segments
            if isinstance(segment, OutputSegment)
        }

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
        from ..parser import parse_program

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
        from .environment import assert_not_cli_args_identifier

        export_names = self._export_names(imported_prog, imported_rt)
        names = self._selected_import_names(module_import, export_names)
        for name in names:
            assert_not_cli_args_identifier(name)
            kind, value = self._resolve_export(name, imported_rt, module_path)
            self._remove_imported_binding(name)
            if target_env is not None:
                target_env.pop(name, None)
            if kind == "procedure":
                self.rt._procedures[name] = self._wrap_imported_procedure(name, value, imported_rt)
            elif kind == "type":
                self.rt._type_env_stack[0][name] = value
            elif kind == "profile":
                self.rt._profile_registry[name] = self._bind_imported_profile(value, imported_rt)
            else:
                self.rt._import_env[name] = value
                if target_env is not None:
                    target_env[name] = value

    def _selected_import_names(
        self,
        module_import: ModuleImport,
        export_names: list[str],
    ) -> list[str]:
        if not module_import.names:
            return export_names

        exported = set(export_names)
        missing = list(dict.fromkeys(name for name in module_import.names if name not in exported))
        if missing:
            rendered = ", ".join(repr(name) for name in missing)
            raise ImportError(f"Kedi module {module_import.module!r} does not export {rendered}")
        return list(dict.fromkeys(module_import.names))

    def _remove_imported_binding(self, name: str) -> None:
        """Make a later import authoritative even when the binding kind changes."""
        self.rt._procedures.pop(name, None)
        self.rt._type_env_stack[0].pop(name, None)
        self.rt._profile_registry.pop(name, None)
        self.rt._import_env.pop(name, None)

    def _bind_imported_profile(
        self,
        profile: AgentProfile | ProfileDefinition,
        imported_rt: KediRuntime,
        *,
        bound_profiles: dict[str, ProfileDefinition] | None = None,
    ) -> AgentProfile | ProfileDefinition:
        """Attach imported private dependencies to one profile without exporting them."""
        if isinstance(profile, AgentProfile):
            return profile

        bound_profiles = {} if bound_profiles is None else bound_profiles
        existing = bound_profiles.get(profile.name)
        if existing is not None:
            return existing

        tool_bindings = dict(profile.tool_bindings)
        for tool_name in self._profile_tool_names(profile):
            spec = self._imported_tool_spec(tool_name, imported_rt)
            if spec is not None:
                tool_bindings[tool_name] = spec

        # Store the shell before recursively binding children so cycles do not
        # accidentally publish a caller-owned profile under the same name.
        bound = replace(
            profile,
            tool_bindings=tool_bindings,
            profile_bindings={},
            subagent_bindings={},
        )
        bound_profiles[profile.name] = bound
        profile_bindings: dict[str, ProfileDefinition] = {}
        for used_name in self._profile_use_names(profile):
            if used_name in tool_bindings:
                continue
            used_profile = imported_rt._profile_registry.get(used_name)
            if isinstance(used_profile, ProfileDefinition):
                bound_profile = self._bind_imported_profile(
                    used_profile,
                    imported_rt,
                    bound_profiles=bound_profiles,
                )
                profile_bindings[used_name] = cast(ProfileDefinition, bound_profile)
        subagent_bindings: dict[str, ProfileDefinition] = {}
        for child_name in self._profile_subagent_names(profile):
            child = imported_rt._profile_registry.get(child_name)
            if isinstance(child, ProfileDefinition):
                bound_child = self._bind_imported_profile(
                    child,
                    imported_rt,
                    bound_profiles=bound_profiles,
                )
                subagent_bindings[child_name] = cast(ProfileDefinition, bound_child)
        bound = replace(
            bound,
            profile_bindings=profile_bindings,
            subagent_bindings=subagent_bindings,
        )
        bound_profiles[profile.name] = bound
        return bound

    def _imported_tool_spec(self, tool_name: str, imported_rt: KediRuntime) -> Any | None:
        from ...agent_adapter.utils import kedi_procedure_tool_spec, python_callable_tool_spec
        from ...agent_profile.tools import procedure_def_for_tool, python_callable_for_tool

        proc = procedure_def_for_tool(imported_rt, tool_name)
        if proc is not None:
            return kedi_procedure_tool_spec(imported_rt, proc)
        tool_fn = python_callable_for_tool(imported_rt, tool_name)
        return None if tool_fn is None else python_callable_tool_spec(tool_name, tool_fn)

    def _profile_tool_names(self, profile: AgentProfile | ProfileDefinition) -> tuple[str, ...]:
        if isinstance(profile, AgentProfile):
            return profile.tools
        names: list[str] = []
        for member in profile.members:
            if member.kind == "tools":
                names.extend(member.tool_names)
            elif member.kind == "use":
                names.append(member.name)
        return tuple(names)

    def _profile_use_names(self, profile: ProfileDefinition) -> tuple[str, ...]:
        return tuple(member.name for member in profile.members if member.kind == "use")

    def _profile_subagent_names(
        self,
        profile: AgentProfile | ProfileDefinition,
    ) -> tuple[str, ...]:
        if isinstance(profile, AgentProfile):
            return profile.subagents
        return tuple(member.name for member in profile.members if member.kind == "subagents")

    def _export_names(self, prog: Program, imported_rt: KediRuntime) -> list[str]:
        names: list[str] = []
        for export in prog.exports:
            if export.export_all:
                names.extend(self._public_runtime_names(imported_rt))
            names.extend(export.names)
        return list(dict.fromkeys(names))

    def _public_runtime_names(self, rt: KediRuntime) -> list[str]:
        names: list[str] = []
        names.extend(name for name in rt._procedures if self._is_public_name(name))

        for layer in rt._type_env_stack or []:
            names.extend(name for name in layer if self._is_public_name(name))

        for env in (rt._globals_env, rt._prelude_env, rt._import_env):
            if env is not None:
                names.extend(name for name in env if self._is_public_name(name))

        names.extend(name for name in rt._profile_registry if self._is_public_name(name))

        return list(dict.fromkeys(names))

    def _is_public_name(self, name: str) -> bool:
        return bool(name) and not name.startswith("_")

    def _resolve_export(
        self,
        name: str,
        imported_rt: KediRuntime,
        module_path: Path,
    ) -> tuple[str, Any]:
        if name in imported_rt._procedures:
            return "procedure", imported_rt._procedures[name]

        for layer in reversed(imported_rt._type_env_stack or []):
            if name in layer:
                return "type", layer[name]

        for env in (
            imported_rt._globals_env,
            imported_rt._prelude_env,
            imported_rt._import_env,
        ):
            if env is not None and name in env:
                return "value", env[name]

        if name in imported_rt._profile_registry:
            return "profile", imported_rt._profile_registry[name]

        raise RuntimeError(f"Kedi export {name!r} is not defined in {module_path}")

    def _wrap_imported_procedure(
        self,
        name: str,
        fn: Any,
        imported_rt: KediRuntime,
    ) -> Any:
        import inspect

        from ..procedure_introspection import get_procedure_def, get_procedure_metadata

        signature = inspect.signature(fn)
        params = list(signature.parameters)
        defaults = {
            param_name: parameter.default
            for param_name, parameter in signature.parameters.items()
            if parameter.default is not inspect.Parameter.empty
        }

        def _body(**kwargs: Any) -> Any:
            return imported_rt._invoke_procedure(fn, [kwargs[param] for param in params])

        wrapper = self.rt.executor.create_dynamic_function(name, params, _body, defaults)
        metadata = get_procedure_metadata(fn)
        if metadata is not None:
            setattr(wrapper, "__signature__", metadata.introspection_signature())
            setattr(wrapper, "__kedi_procedure_metadata__", metadata)
            setattr(
                wrapper,
                "__kedi_procedure_def__",
                get_procedure_def(fn),
            )
            wrapper.__doc__ = metadata.docstring
        return wrapper

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
            kwargs: dict[str, Any] = dict(self.rt._initial_globals_env)
            self.rt._globals_env = kwargs
            return self._execute_top_level(prepared, kwargs, self.new_top_level_state())

    def _execute_top_level(
        self,
        prepared: _PreparedProgram,
        env: dict[str, Any],
        state: _TopLevelExecutionState,
    ) -> Any:
        prog = prepared.program
        profile_registry = self.rt._profile_registry

        with self.rt.trace_frame(
            KediTraceFrame(kind="program", label="top-level program", span=None)
        ):
            for item in prepared.sequence:
                if isinstance(item, ProcedureDef):
                    if not item.auto_spec and not item.ai_prompt:
                        self.rt.procedure(self.proc_compiler.compile_procedure(item))
                    continue
                if isinstance(item, TypeDef):
                    register_type(self.rt, item, self.rt._type_env_stack[0])
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
                    _assert_adapter_selection_not_mixed(
                        state.adapter_selection_kind, "agent-harness"
                    )
                    state.adapter_selection_kind = "agent-harness"
                    state.active_profile = state.active_profile.merge(
                        eval_agent_directive(self.rt, item, env)
                    )
                    continue
                if isinstance(item, AdapterDirectiveStmt):
                    _assert_adapter_selection_not_mixed(
                        state.adapter_selection_kind, "agent-framework"
                    )
                    state.adapter_selection_kind = "agent-framework"
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
                    state.active_profile = state.active_profile.merge(
                        eval_use_directive(
                            self.rt,
                            item,
                            profile_registry,
                            env,
                            available_procedures=set(self.rt._procedures.keys()),
                        )
                    )
                    continue
                if isinstance(item, ProfileDefStmt):
                    profile_registry[item.name] = prepared.local_profiles[item.name]
                    continue
                if isinstance(item, TemplateLine):
                    raise ValueError(
                        "Top-level TemplateLine is not a valid program node; "
                        "use '>>' for a top-level template prompt."
                    )
                with block_agent_profile_scope(state.active_profile):
                    self.executor.execute_single_statement(item, env, is_toplevel=True)

            if prepared.toplevel_return is None:
                return None

            with block_agent_profile_scope(state.active_profile):
                if isinstance(prepared.toplevel_return, ReturnStmt):
                    return self.stmt_compiler.compile_return(prepared.toplevel_return)(**env)
                code = prepared.toplevel_return.code
                if "\n" in code or code.strip().startswith("return"):
                    return execute_python_block(
                        code,
                        self.rt,
                        env,
                        span=prepared.toplevel_return.loc,
                    )
                return evaluate_python_inline(
                    self.rt,
                    code,
                    env,
                    span=prepared.toplevel_return.loc,
                )

    def _attach_validation(self, prog: Program) -> None:
        """Attach validation suites if present.

        The parser returns ``ProgramWithValidation`` when ``@test:`` or
        ``@eval:`` blocks are present; we forward the public ``.tests``
        / ``.evals`` collections onto the runtime so downstream
        consumers (``kedi-lsp``, ``tests_runner``, ``app.py``) can
        read them as normal attributes.
        """
        from ..ast import ProgramWithValidation

        if isinstance(prog, ProgramWithValidation):
            if prog.tests:
                self.rt._validation_tests = prog.tests
            if prog.evals:
                self.rt._validation_evals = prog.evals
