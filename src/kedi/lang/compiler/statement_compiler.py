"""Statement compilation for Kedi compiler."""

from __future__ import annotations

from typing import Any, Callable

from ...core import KediRuntime
from ...errors import KediTraceFrame
from ..ast import (
    AssignBlockStmt,
    AssignStmt,
    BacktickLineStmt,
    ProcedureDef,
    PythonBlock,
    ReturnBlockStmt,
    ReturnStmt,
    Stmt,
    TemplateLine,
)
from ..parser.escapes import preserve_escaped_whitespace
from .evaluation import (
    evaluate_assignment_rhs,
    evaluate_python_inline,
    execute_inline_side_effects,
    execute_python_block,
    prepare_segments,
    render_expression,
)
from .type_system import KediType, assign_value_from_native, detect_rhs_kind


class StatementCompiler:
    """Compiles individual statements to executable functions."""

    def __init__(self, rt: KediRuntime):
        self.rt = rt

    def compile_template_line(self, stmt: TemplateLine) -> Callable:
        """Compile a template line to a function.

        If the template has an optimized_prompt_prefix, it is prepended to the
        original seed prompt (with proper type annotations) at evaluation time.
        This allows GEPA to output plain text without needing correct type names,
        while preserving the original prompt's type annotations for proper resolution.
        """

        def _fn(**kwargs: Any) -> Any:
            with self.rt.trace_frame(
                KediTraceFrame(kind="template", label="template line", span=stmt.loc)
            ):
                # If an override is provided, short-circuit LLM invocation and inject values.
                if stmt.optimized_output_override is not None:
                    env = self.rt._current_env.get()
                    if env is None:
                        raise RuntimeError("Template override requires active runtime environment")
                    for key, value in stmt.optimized_output_override.items():
                        env[key] = value
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

    def compile_return(self, stmt: ReturnStmt) -> Callable:
        """Compile a return statement to a function."""

        def _fn(**kwargs: Any) -> Any:
            with self.rt.trace_frame(KediTraceFrame(kind="return", label="return", span=stmt.loc)):
                rendered = render_expression(self.rt, stmt.segments, kwargs, context="return")
                return rendered

        return _fn

    def compile_return_block(self, stmt: ReturnBlockStmt, kwargs: dict[str, Any]) -> Any:
        """Execute a return block and return its value."""
        code = stmt.code
        with self.rt.trace_frame(
            KediTraceFrame(kind="return_block", label="return block", span=stmt.loc)
        ):
            if "\n" in code or code.strip().startswith("return"):
                return execute_python_block(code, self.rt, kwargs, span=stmt.loc)
            return evaluate_python_inline(self.rt, code, kwargs, span=stmt.loc)

    def execute_assignment(self, stmt: AssignStmt, closure_env: dict[str, Any]) -> None:
        """Execute an assignment statement."""
        with self.rt.trace_frame(
            KediTraceFrame(kind="assignment", label=f"assignment {stmt.name}", span=stmt.loc)
        ):
            native, rendered = evaluate_assignment_rhs(self.rt, stmt.expr, closure_env)
            is_backtick, is_angle, is_call = detect_rhs_kind(stmt.expr)
            kedi_type = KediType.from_node(stmt.type_expr, stmt.type_annotation)

            if native is not None:
                if is_backtick:
                    value = assign_value_from_native(
                        native, kedi_type, self.rt, closure_env, strict_native=True
                    )
                elif is_angle:
                    value = assign_value_from_native(
                        native, kedi_type, self.rt, closure_env, strict_native=False
                    )
                elif is_call:
                    value = native
                else:
                    value = rendered
                    if not kedi_type.is_none():
                        try:
                            value = kedi_type.validate(self.rt, closure_env, value)
                        except Exception:
                            raise TypeError("Native assignment value does not match annotated type")
            else:
                value = rendered
                if not kedi_type.is_none():
                    try:
                        value = kedi_type.validate(self.rt, closure_env, value)
                    except Exception:
                        raise TypeError("Native assignment value does not match annotated type")

            self._set_value(stmt.name, value, closure_env)

    def execute_assignment_block(self, stmt: AssignBlockStmt, closure_env: dict[str, Any]) -> None:
        """Execute an assignment with Python block RHS."""
        with self.rt.trace_frame(
            KediTraceFrame(
                kind="assignment_block",
                label=f"assignment {stmt.name}",
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
                    raise TypeError("Native assignment value does not match annotated type")
            else:
                if not isinstance(value, str):
                    raise TypeError(
                        "Native non-string assignment requires explicit type annotation"
                    )

            self._set_value(stmt.name, value, closure_env)

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

    def _set_value(self, name: str, value: Any, closure_env: dict[str, Any]) -> None:
        """Set a value in both closure and runtime environments."""
        closure_env[name] = value
        env = self.rt._current_env.get()
        if env is not None:
            env[name] = value


class StatementExecutor:
    """Executes a sequence of statements in a given environment."""

    def __init__(self, rt: KediRuntime):
        self.rt = rt
        self.compiler = StatementCompiler(rt)

    def execute(self, proc: Any, closure_env: dict[str, Any]) -> Any:
        """Execute procedure body statements and return the last return value."""
        from ..ast import ProcDefStmt, TypeDefStmt
        from .type_system import pop_type_env, push_type_env, register_type

        last_return: Any = None
        type_scope = push_type_env(self.rt)

        # Track nested procedures for cleanup
        saved_procs: dict[str, Any] = {}
        new_proc_names: list[str] = []

        # Inherit parent type environment
        parent_stack = getattr(self.rt, "_type_env_stack", None)
        if isinstance(parent_stack, list) and len(parent_stack) >= 2:
            type_scope.update(parent_stack[-2])

        try:
            for stmt in proc.body:
                if isinstance(stmt, TemplateLine):
                    self.compiler.compile_template_line(stmt)(**closure_env)
                    self._sync_env(closure_env)

                elif isinstance(stmt, TypeDefStmt):
                    if stmt.typedef is not None:
                        register_type(self.rt, stmt.typedef, type_scope, closure_env)

                elif isinstance(stmt, ProcDefStmt):
                    # Register nested procedure dynamically with current closure
                    from .procedure_compiler import ProcedureCompiler

                    child: ProcedureDef = stmt.proc  # type: ignore
                    child.lexical_env = dict(closure_env)
                    proc_compiler = ProcedureCompiler(self.rt)
                    fn_nested = proc_compiler._make_procedure_impl(child, child.lexical_env)

                    # Save existing procedure if present, track new ones
                    procs = self.rt._procedures
                    if child.name in procs and child.name not in saved_procs:
                        saved_procs[child.name] = procs[child.name]
                    elif child.name not in procs:
                        new_proc_names.append(child.name)
                    procs[child.name] = fn_nested

                elif isinstance(stmt, PythonBlock):
                    self.compiler.execute_python_block(stmt, closure_env)

                elif isinstance(stmt, BacktickLineStmt):
                    self.compiler.execute_backtick_line(stmt, closure_env)

                elif isinstance(stmt, AssignBlockStmt):
                    self.compiler.execute_assignment_block(stmt, closure_env)

                elif isinstance(stmt, AssignStmt):
                    self.compiler.execute_assignment(stmt, closure_env)

                elif isinstance(stmt, ReturnStmt):
                    # For typed procedures, preserve native value if resolvable; otherwise render to string
                    if getattr(proc, "return_type_annotation", None):
                        native, rendered = evaluate_assignment_rhs(
                            self.rt, stmt.segments, closure_env
                        )
                        last_return = (
                            native if native is not None else preserve_escaped_whitespace(rendered)
                        )
                    else:
                        last_return = self.compiler.compile_return(stmt)(**closure_env)

                elif isinstance(stmt, ReturnBlockStmt):
                    last_return = self.compiler.compile_return_block(stmt, closure_env)

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

    def execute_single_statement(
        self, stmt: Stmt, env: dict[str, Any], is_toplevel: bool = False
    ) -> None:
        """Execute a single statement (for top-level execution)."""
        from ..ast import ProcDefStmt

        if isinstance(stmt, TemplateLine):
            self.compiler.compile_template_line(stmt)(**env)
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

        elif isinstance(stmt, AssignBlockStmt):
            self.compiler.execute_assignment_block(stmt, env)

        elif isinstance(stmt, AssignStmt):
            self.compiler.execute_assignment(stmt, env)

    def _sync_env(self, closure_env: dict[str, Any]) -> None:
        """Sync runtime env changes back to closure."""
        env = self.rt._current_env.get()
        if env:
            for k, v in env.items():
                closure_env[k] = v
