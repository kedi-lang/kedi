"""Top-level program compilation orchestration."""

from __future__ import annotations

from typing import Any

from ...core import AgentAdapter, MarkerRuntime
from ..ast import Program, ReturnBlockStmt, ReturnStmt
from .environment import build_python_env
from .evaluation import evaluate_python_inline, execute_python_block
from .procedure_compiler import ProcedureCompiler
from .statement_compiler import StatementCompiler, StatementExecutor
from .type_system import register_type


def compile_program(prog: Program, *, adapter: AgentAdapter | None = None) -> MarkerRuntime:
    """Compile a Marker program to an executable runtime.

    This is the main entry point for the compiler.
    """
    compiler = ProgramCompiler(adapter)
    return compiler.compile(prog)


class ProgramCompiler:
    """Orchestrates program compilation."""

    def __init__(self, adapter: AgentAdapter | None):
        self.adapter = adapter
        self.rt = MarkerRuntime(adapter=adapter)
        self.proc_compiler = ProcedureCompiler(self.rt)
        self.executor = StatementExecutor(self.rt)
        self.stmt_compiler = StatementCompiler(self.rt)

    def compile(self, prog: Program) -> MarkerRuntime:
        """Compile program and return configured runtime."""
        # Initialize type environment
        setattr(self.rt, "_type_env_stack", [{}])

        # Execute prelude
        if prog.prelude is not None:
            self._setup_prelude(prog.prelude)

        # Register top-level types
        type_env = getattr(self.rt, "_type_env_stack")[0]
        for typedef in prog.type_defs:
            register_type(self.rt, typedef, type_env)

        # Register procedures (skip AI-prompted placeholders)
        for proc in prog.procedures:
            if getattr(proc, "ai_prompt", None):
                continue
            fn = self.proc_compiler.compile_procedure(proc)
            self.rt.procedure(fn)

        # Setup main entry point
        if prog.toplevel_return is not None:
            self._setup_main(prog.toplevel_body, prog.toplevel_return)

        # Attach validation suites
        self._attach_validation(prog)

        return self.rt

    def _setup_prelude(self, prelude_code: str) -> None:
        """Execute prelude and store environment."""
        prelude_kwargs: dict[str, Any] = {}
        env_exec = build_python_env(self.rt, prelude_kwargs)
        try:
            exec(prelude_code, env_exec, env_exec)
        except Exception as e:
            raise RuntimeError(f"Python prelude error: {e}") from e
        setattr(self.rt, "_prelude_env", env_exec)

    def _setup_main(self, toplevel_body: list, toplevel_return) -> None:
        """Setup main entry point for top-level execution."""

        @self.rt.main
        def _main() -> Any:
            # Initialize top-level environment
            kwargs: dict[str, Any] = {}
            setattr(self.rt, "_globals_env", kwargs)

            # Execute top-level statements
            for stmt in toplevel_body:
                self.executor.execute_single_statement(stmt, kwargs, is_toplevel=True)

            # Execute return
            if isinstance(toplevel_return, ReturnStmt):
                return self.stmt_compiler.compile_return(toplevel_return)(**kwargs)
            else:  # ReturnBlockStmt
                code = toplevel_return.code
                # Top-level: always return string rendering for native returns
                if "\n" in code or code.strip().startswith("return"):
                    result = execute_python_block(code, self.rt, kwargs)
                    return "" if result is None else str(result)
                # Inline python native return
                value = evaluate_python_inline(self.rt, code, kwargs)
                return "" if value is None else str(value)

    def _attach_validation(self, prog: Program) -> None:
        """Attach validation suites if present."""
        tests = getattr(prog, "_validation_tests", None)
        evals = getattr(prog, "_validation_evals", None)
        if tests is not None:
            setattr(self.rt, "_validation_tests", tests)
        if evals is not None:
            setattr(self.rt, "_validation_evals", evals)
