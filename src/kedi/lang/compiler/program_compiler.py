"""Top-level program compilation orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...core import KediRuntime
from ...errors import KediTraceFrame
from ...source_loading import load_kedi_source_bundle
from ..ast import ModuleImport, Program, ReturnStmt
from .evaluation import evaluate_python_inline, execute_python_block
from .procedure_compiler import ProcedureCompiler
from .statement_compiler import StatementCompiler, StatementExecutor
from .type_system import register_type

if TYPE_CHECKING:
    from ...agent_adapter.meta import AgentAdapter


def compile_program(
    prog: Program,
    *,
    adapter: AgentAdapter | None = None,
    executor: Executor | None = None,
) -> KediRuntime:
    """Compile a Kedi program to an executable runtime.

    This is the main entry point for the compiler.
    """
    compiler = ProgramCompiler(adapter, executor=executor)
    return compiler.compile(prog)


class ProgramCompiler:
    """Orchestrates program compilation."""

    def __init__(
        self,
        adapter: AgentAdapter | None,
        *,
        executor: Executor | None = None,
        module_stack: tuple[Path, ...] = (),
    ):
        self.adapter = adapter
        self.module_stack = module_stack
        self.rt = KediRuntime(adapter=adapter, executor=executor)
        self.proc_compiler = ProcedureCompiler(self.rt)
        self.executor = StatementExecutor(self.rt)
        self.stmt_compiler = StatementCompiler(self.rt)

    def compile(self, prog: Program) -> KediRuntime:
        """Compile program and return configured runtime."""
        # Initialize type environment
        setattr(self.rt, "_type_env_stack", [{}])
        self.rt._program = prog
        self.rt._source_map = getattr(prog, "source_map", None)
        self.rt._source_text = getattr(prog, "source_text", None)
        current_path = self._program_path(prog)
        active_stack = self.module_stack
        if current_path is not None:
            if current_path in active_stack:
                cycle = " -> ".join(str(path) for path in (*active_stack, current_path))
                raise RuntimeError(f"Cyclic Kedi import detected: {cycle}")
            active_stack = (*active_stack, current_path)

        self._compile_imports(prog, active_stack)

        # Execute prelude
        if prog.prelude is not None:
            self._setup_prelude(prog.prelude, prog.prelude_loc)

        # Register top-level types
        type_env = getattr(self.rt, "_type_env_stack")[0]
        for typedef in prog.type_defs:
            register_type(self.rt, typedef, type_env)

        # Register procedures (skip AI-prompted placeholders)
        for proc in prog.procedures:
            # Skip procedures with auto_spec or legacy ai_prompt
            if proc.auto_spec or getattr(proc, "ai_prompt", None):
                continue
            fn = self.proc_compiler.compile_procedure(proc)
            self.rt.procedure(fn)

        # Setup main entry point when there is executable top-level code.
        if prog.toplevel_body or prog.toplevel_return is not None:
            self._setup_main(prog.toplevel_body, prog.toplevel_return)

        # Attach validation suites
        self._attach_validation(prog)

        return self.rt

    def _compile_imports(self, prog: Program, active_stack: tuple[Path, ...]) -> None:
        for module_import in prog.imports:
            module_path = self._resolve_import_path(prog, module_import)
            imported_prog = self._load_imported_program(module_path)
            imported_rt = ProgramCompiler(
                self.adapter,
                executor=self.rt.executor,
                module_stack=active_stack,
            ).compile(imported_prog)
            imported_rt.run_main()
            self._merge_exports(module_path, imported_prog, imported_rt)

    def _resolve_import_path(self, prog: Program, module_import: ModuleImport) -> Path:
        source_path = self._program_path(prog)
        if source_path is None:
            raise RuntimeError(
                f"Cannot resolve Kedi import {module_import.module!r} without a source path"
            )
        module_path = source_path.parent / f"{module_import.module}.kedi"
        if not module_path.exists():
            raise FileNotFoundError(
                f"Kedi import {module_import.module!r} not found at {module_path}"
            )
        return module_path.resolve()

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
                getattr(self.rt, "_type_env_stack")[0][name] = value
            else:
                self.rt._import_env[name] = value

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

        stack = getattr(rt, "_type_env_stack", None)
        if isinstance(stack, list):
            for layer in stack:
                names.extend(name for name in layer if self._is_public_name(name))

        for env_name in ("_globals_env", "_prelude_env", "_import_env"):
            env = getattr(rt, env_name, None)
            if isinstance(env, dict):
                names.extend(name for name in env if self._is_public_name(name))

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

        stack = getattr(imported_rt, "_type_env_stack", None)
        if isinstance(stack, list):
            for layer in reversed(stack):
                if name in layer:
                    return "type", layer[name]

        for env_name in ("_globals_env", "_prelude_env", "_import_env"):
            env = getattr(imported_rt, env_name, None)
            if isinstance(env, dict) and name in env:
                return "value", env[name]

        raise RuntimeError(f"Kedi export {name!r} is not defined in {module_path}")

    def _wrap_imported_procedure(
        self,
        name: str,
        fn: Any,
        imported_rt: KediRuntime,
    ) -> Any:
        import inspect

        from ..procedure_introspection import get_procedure_metadata

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
            setattr(wrapper, "__kedi_procedure_def__", getattr(fn, "__kedi_procedure_def__", None))
            wrapper.__doc__ = metadata.docstring
        return wrapper

    def _program_path(self, prog: Program) -> Path | None:
        loc = getattr(prog, "loc", None)
        if loc is not None and loc.source_path:
            return Path(loc.source_path).resolve()
        source_map = getattr(prog, "source_map", None)
        documents = getattr(source_map, "documents", None)
        if documents:
            return Path(documents[0].path).resolve()
        return None

    def _setup_prelude(self, prelude_code: str, prelude_loc) -> None:
        """Execute prelude and store environment."""
        with self.rt.trace_frame(KediTraceFrame(kind="prelude", label="prelude", span=prelude_loc)):
            setattr(
                self.rt,
                "_prelude_env",
                self.rt.executor.execute_prelude(self.rt, prelude_code),
            )

    def _setup_main(self, toplevel_body: list, toplevel_return) -> None:
        """Setup main entry point for top-level execution."""

        @self.rt.main
        def _main() -> Any:
            # Initialize top-level environment
            kwargs: dict[str, Any] = {}
            setattr(self.rt, "_globals_env", kwargs)

            # Execute top-level statements
            with self.rt.trace_frame(
                KediTraceFrame(
                    kind="program",
                    label="top-level program",
                    span=None,
                )
            ):
                for stmt in toplevel_body:
                    self.executor.execute_single_statement(stmt, kwargs, is_toplevel=True)

                if toplevel_return is None:
                    return None

                # Execute return
                if isinstance(toplevel_return, ReturnStmt):
                    return self.stmt_compiler.compile_return(toplevel_return)(**kwargs)
                else:  # ReturnBlockStmt
                    code = toplevel_return.code
                    # Top-level: always return string rendering for native returns
                    if "\n" in code or code.strip().startswith("return"):
                        result = execute_python_block(
                            code, self.rt, kwargs, span=toplevel_return.loc
                        )
                        return "" if result is None else str(result)
                    # Inline python native return
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
                setattr(self.rt, "_validation_tests", prog.tests)
            if prog.evals:
                setattr(self.rt, "_validation_evals", prog.evals)
