"""Top-level program compilation orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...agent_profile import AgentProfile, block_agent_profile_scope
from ...agent_profile.compile import build_profile_registry, capture_procedure_profiles
from ...agent_profile.model import (
    eval_adapter_directive,
    eval_agent_directive,
    eval_effort_directive,
    eval_mcp_directive,
    eval_model_directive,
    eval_settings_directive,
    eval_system_directive,
)
from ...agent_profile.registry import ProfileDefinition
from ...agent_profile.tools import eval_use_directive
from ...core import KediRuntime
from ...errors import KediTraceFrame
from ...module_registry import resolve_module_path
from ...source_loading import load_kedi_source_bundle
from ..ast import (
    AdapterDirectiveStmt,
    AgentDirectiveStmt,
    EffortDirectiveStmt,
    McpDirectiveStmt,
    ModelDirectiveStmt,
    ModuleImport,
    ProcedureDef,
    ProfileDefStmt,
    Program,
    ReturnStmt,
    SettingsDirectiveStmt,
    SystemDirectiveStmt,
    TemplateLine,
    UseDirectiveStmt,
)
from .evaluation import evaluate_python_inline, execute_python_block
from .procedure_compiler import ProcedureCompiler
from .statement_compiler import StatementCompiler, StatementExecutor
from .type_system import register_type

if TYPE_CHECKING:
    from ...agent_adapter.meta import AgentAdapter
    from ...executors.base import Executor


def _assert_adapter_selection_not_mixed(current: str | None, incoming: str) -> None:
    if current is not None and current != incoming:
        raise RuntimeError("Cannot mix '> agent:' and '> adapter:' in the same scope")


def compile_program(
    prog: Program,
    *,
    adapter: AgentAdapter[Any] | None = None,
    executor: Executor | None = None,
    runtime_globals: dict[str, Any] | None = None,
    engine: Any | None = None,
) -> KediRuntime:
    """Compile a Kedi program to an executable runtime.

    This is the main entry point for the compiler.
    """
    compiler = ProgramCompiler(
        adapter, executor=executor, runtime_globals=runtime_globals, engine=engine
    )
    return compiler.compile(prog)


class ProgramCompiler:
    """Orchestrates program compilation."""

    def __init__(
        self,
        adapter: AgentAdapter[Any] | None,
        *,
        executor: Executor | None = None,
        module_stack: tuple[Path, ...] = (),
        runtime_globals: dict[str, Any] | None = None,
        engine: Any | None = None,
    ):
        self.adapter = adapter
        self.module_stack = module_stack
        self.engine = engine
        self.rt = KediRuntime(adapter=adapter, executor=executor, engine=engine)
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
        active_stack = self.module_stack
        if current_path is not None:
            if current_path in active_stack:
                cycle = " -> ".join(str(path) for path in (*active_stack, current_path))
                raise RuntimeError(f"Cyclic Kedi import detected: {cycle}")
            active_stack = (*active_stack, current_path)

        self.rt._profile_registry = {}

        self._compile_imports(prog, active_stack)

        # Execute prelude
        if prog.prelude is not None:
            self._setup_prelude(prog.prelude, prog.prelude_loc)

        self.rt._profile_registry.update(build_profile_registry(prog))
        capture_procedure_profiles(self.rt, prog)

        # Register top-level types
        type_env = self.rt._type_env_stack[0]
        for typedef in prog.type_defs:
            register_type(self.rt, typedef, type_env)

        # Register procedures (skip AI-prompted placeholders)
        for proc in prog.procedures:
            # Skip procedures with auto_spec or legacy ai_prompt
            if proc.auto_spec or proc.ai_prompt:
                continue
            fn = self.proc_compiler.compile_procedure(proc)
            self.rt.procedure(fn)

        # Setup main entry point when there is executable top-level code.
        if prog.ordered_toplevel or prog.toplevel_body or prog.toplevel_return is not None:
            self._setup_main(prog)

        # Attach validation suites
        self._attach_validation(prog)

        return self.rt

    def _compile_imports(self, prog: Program, active_stack: tuple[Path, ...]) -> None:
        for module_import in prog.imports:
            with self.rt.trace_frame(
                KediTraceFrame(
                    kind="module_import",
                    label=f"import {module_import.module}",
                    span=module_import.loc,
                )
            ):
                module_path = self._resolve_import_path(prog, module_import)
                imported_prog = self._load_imported_program(module_path)
                # Share the importer's globals (immutable CLI ``args``) with the module.
                imported_rt = ProgramCompiler(
                    self.adapter,
                    executor=self.rt.executor,
                    module_stack=active_stack,
                    runtime_globals=dict(self.rt._initial_globals_env),
                    engine=self.engine,
                ).compile(imported_prog)
                imported_rt.run_main()
                self._merge_exports(module_path, imported_prog, imported_rt)

    def _resolve_import_path(self, prog: Program, module_import: ModuleImport) -> Path:
        try:
            return resolve_module_path(
                module_import.module,
                source_path=self._program_path(prog),
            )
        except FileNotFoundError as exc:
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
        module_path: Path,
        imported_prog: Program,
        imported_rt: KediRuntime,
    ) -> None:
        for name in self._export_names(imported_prog, imported_rt):
            kind, value = self._resolve_export(name, imported_rt, module_path)
            if kind == "procedure":
                self.rt._procedures[name] = self._wrap_imported_procedure(name, value, imported_rt)
            elif kind == "type":
                self.rt._type_env_stack[0][name] = value
            elif kind == "profile":
                self.rt._profile_registry[name] = value
                self._merge_profile_tool_dependencies(value, imported_rt)
            else:
                self.rt._import_env[name] = value

    def _merge_profile_tool_dependencies(
        self,
        profile: AgentProfile | ProfileDefinition,
        imported_rt: KediRuntime,
    ) -> None:
        for tool_name in self._profile_tool_names(profile):
            if tool_name in self.rt._procedures or tool_name in self.rt._import_env:
                continue
            if tool_name in imported_rt._procedures:
                self.rt._procedures[tool_name] = self._wrap_imported_procedure(
                    tool_name,
                    imported_rt._procedures[tool_name],
                    imported_rt,
                )
                continue
            for env in (
                imported_rt._globals_env,
                imported_rt._prelude_env,
                imported_rt._import_env,
            ):
                if env is not None and tool_name in env:
                    self.rt._import_env[tool_name] = env[tool_name]
                    break

    def _profile_tool_names(self, profile: AgentProfile | ProfileDefinition) -> tuple[str, ...]:
        if isinstance(profile, AgentProfile):
            return profile.tools
        names: list[str] = []
        for member in profile.members:
            if member.kind == "tools":
                names.extend(member.tool_names)
        return tuple(names)

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
            setattr(wrapper, "__kedi_procedure_metadata__", metadata)
            setattr(
                wrapper,
                "__kedi_procedure_def__",
                get_procedure_def(fn),
            )
            wrapper.__doc__ = metadata.docstring
        return wrapper

    def _program_path(self, prog: Program) -> Path | None:
        loc = prog.loc
        if loc is not None and loc.source_path:
            return Path(loc.source_path).resolve()
        source_map = prog.source_map
        if source_map is not None and source_map.documents:
            return Path(source_map.documents[0].path).resolve()
        return None

    def _setup_prelude(self, prelude_code: str, prelude_loc) -> None:
        """Execute prelude and store environment."""
        with self.rt.trace_frame(KediTraceFrame(kind="prelude", label="prelude", span=prelude_loc)):
            self.rt._prelude_env = self.rt.executor.execute_prelude(self.rt, prelude_code)

    def _setup_main(self, prog: Program) -> None:
        """Setup main entry point for top-level execution."""
        toplevel_sequence = prog.ordered_toplevel or list(prog.toplevel_body)
        toplevel_return = prog.toplevel_return
        profile_registry = self.rt._profile_registry

        @self.rt.main
        def _main() -> Any:
            # Initialize top-level environment
            kwargs: dict[str, Any] = dict(self.rt._initial_globals_env)
            self.rt._globals_env = kwargs
            active = AgentProfile.empty()
            adapter_selection_kind = active.adapter_kind

            # Execute top-level statements
            with self.rt.trace_frame(
                KediTraceFrame(
                    kind="program",
                    label="top-level program",
                    span=None,
                )
            ):
                for item in toplevel_sequence:
                    if isinstance(item, ProcedureDef):
                        continue
                    if isinstance(item, AgentDirectiveStmt):
                        _assert_adapter_selection_not_mixed(adapter_selection_kind, "agent-harness")
                        adapter_selection_kind = "agent-harness"
                        active = active.merge(eval_agent_directive(self.rt, item, kwargs))
                        continue
                    if isinstance(item, AdapterDirectiveStmt):
                        _assert_adapter_selection_not_mixed(
                            adapter_selection_kind, "agent-framework"
                        )
                        adapter_selection_kind = "agent-framework"
                        active = active.merge(eval_adapter_directive(self.rt, item, kwargs))
                        continue
                    if isinstance(item, ModelDirectiveStmt):
                        active = active.merge(eval_model_directive(self.rt, item, kwargs))
                        continue
                    if isinstance(item, EffortDirectiveStmt):
                        active = active.merge(eval_effort_directive(self.rt, item, kwargs))
                        continue
                    if isinstance(item, SystemDirectiveStmt):
                        active = active.merge(eval_system_directive(self.rt, item, kwargs))
                        continue
                    if isinstance(item, McpDirectiveStmt):
                        active = active.merge(eval_mcp_directive(self.rt, item, kwargs))
                        continue
                    if isinstance(item, SettingsDirectiveStmt):
                        active = active.merge(eval_settings_directive(self.rt, item, kwargs))
                        continue
                    if isinstance(item, UseDirectiveStmt):
                        active = active.merge(
                            eval_use_directive(
                                self.rt,
                                item,
                                profile_registry,
                                kwargs,
                                available_procedures=set(self.rt._procedures.keys()),
                            )
                        )
                        continue
                    if isinstance(item, ProfileDefStmt):
                        continue
                    if isinstance(item, TemplateLine):
                        raise ValueError(
                            "Top-level TemplateLine is not a valid program node; "
                            "use '>>' for a top-level template prompt."
                        )
                    with block_agent_profile_scope(active):
                        self.executor.execute_single_statement(item, kwargs, is_toplevel=True)

                if toplevel_return is None:
                    return None

                with block_agent_profile_scope(active):
                    if isinstance(toplevel_return, ReturnStmt):
                        return self.stmt_compiler.compile_return(toplevel_return)(**kwargs)
                    code = toplevel_return.code
                    if "\n" in code or code.strip().startswith("return"):
                        result = execute_python_block(
                            code, self.rt, kwargs, span=toplevel_return.loc
                        )
                        return "" if result is None else str(result)
                    value = evaluate_python_inline(self.rt, code, kwargs, span=toplevel_return.loc)
                    return "" if value is None else str(value)

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
