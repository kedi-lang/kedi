"""Top-level program compilation orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...core import KediRuntime
from ...executors.base import Executor
from ...errors import KediTraceFrame
from ..ast import Program, ReturnStmt
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
    ):
        self.adapter = adapter
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
