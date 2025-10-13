"""Statement compilation for Marker compiler."""
from __future__ import annotations

from typing import Any, Callable
from ..ast import (
    Stmt, TemplateLine, ReturnStmt, ReturnBlockStmt,
    AssignStmt, AssignBlockStmt, PythonBlock, BacktickLineStmt
)
from ...core import MarkerRuntime
from .evaluation import (
    prepare_segments, render_expression, evaluate_assignment_rhs,
    execute_python_block, execute_inline_side_effects, evaluate_python_inline
)
from .type_system import (
    resolve_type, coerce_value_to_type, is_value_compatible,
    assign_value_from_native, detect_rhs_kind
)


class StatementCompiler:
    """Compiles individual statements to executable functions."""
    
    def __init__(self, rt: MarkerRuntime):
        self.rt = rt
    
    def compile_template_line(self, stmt: TemplateLine) -> Callable:
        """Compile a template line to a function."""
        def _fn(**kwargs: Any) -> Any:
            expr, _ = prepare_segments(self.rt, stmt.segments, kwargs)
            return self.rt.m(expr)
        return _fn
    
    def compile_return(self, stmt: ReturnStmt) -> Callable:
        """Compile a return statement to a function."""
        def _fn(**kwargs: Any) -> Any:
            rendered = render_expression(self.rt, stmt.segments, kwargs, context="return")
            return rendered.strip()
        return _fn
    
    def compile_return_block(self, stmt: ReturnBlockStmt, kwargs: dict[str, Any]) -> Any:
        """Execute a return block and return its value."""
        code = stmt.code
        if "\n" in code or code.strip().startswith("return"):
            return execute_python_block(code, self.rt, kwargs)
        return evaluate_python_inline(self.rt, code, kwargs)
    
    def execute_assignment(
        self,
        stmt: AssignStmt,
        closure_env: dict[str, Any]
    ) -> None:
        """Execute an assignment statement."""
        native, rendered = evaluate_assignment_rhs(self.rt, stmt.expr, closure_env)
        is_backtick, is_angle, is_call = detect_rhs_kind(stmt.expr)
        pytype = resolve_type(self.rt, stmt.type_annotation, env=closure_env)
        
        if native is not None:
            if is_backtick:
                value = assign_value_from_native(native, pytype, strict_native=True)
            elif is_angle:
                value = assign_value_from_native(native, pytype, strict_native=False)
            elif is_call:
                value = native
            else:
                value = rendered.strip()
                if pytype is not None:
                    value = coerce_value_to_type(value, pytype)
        else:
            value = rendered.strip()
            if pytype is not None:
                value = coerce_value_to_type(value, pytype)
        
        self._set_value(stmt.name, value, closure_env)
    
    def execute_assignment_block(
        self,
        stmt: AssignBlockStmt,
        closure_env: dict[str, Any]
    ) -> None:
        """Execute an assignment with Python block RHS."""
        result = execute_python_block(stmt.code, self.rt, closure_env)
        pytype = resolve_type(self.rt, stmt.type_annotation, env=closure_env)
        
        value = result
        if pytype is not None:
            value = coerce_value_to_type(value, pytype)
            if not is_value_compatible(pytype, value):
                raise TypeError("Native assignment value does not match annotated type")
        else:
            if not isinstance(value, str):
                raise TypeError("Native non-string assignment requires explicit type annotation")
        
        self._set_value(stmt.name, value, closure_env)
    
    def execute_python_block(
        self,
        stmt: PythonBlock,
        closure_env: dict[str, Any]
    ) -> None:
        """Execute a Python block for side effects."""
        from .environment import build_python_env
        
        env_exec = build_python_env(self.rt, closure_env)
        try:
            exec(stmt.code, env_exec, env_exec)
        except Exception as e:
            raise RuntimeError(f"Python block error: {e}") from e
        
        # Sync back existing variables only
        self._sync_env(env_exec, closure_env)
    
    def execute_backtick_line(
        self,
        stmt: BacktickLineStmt,
        closure_env: dict[str, Any]
    ) -> None:
        """Execute a single-line Python statement for side effects."""
        execute_inline_side_effects(self.rt, stmt.code, closure_env)
    
    def _set_value(self, name: str, value: Any, closure_env: dict[str, Any]) -> None:
        """Set a value in both closure and runtime environments."""
        closure_env[name] = value
        env = self.rt._current_env.get()
        if env is not None:
            env[name] = value
    
    def _sync_env(self, source: dict[str, Any], target: dict[str, Any]) -> None:
        """Sync variables from source to target (existing vars only)."""
        env_rt = self.rt._current_env.get()
        for var_name in list(target.keys()):
            if var_name in source:
                target[var_name] = source[var_name]
                if env_rt is not None:
                    env_rt[var_name] = source[var_name]


class StatementExecutor:
    """Executes a sequence of statements in a given environment."""
    
    def __init__(self, rt: MarkerRuntime):
        self.rt = rt
        self.compiler = StatementCompiler(rt)
    
    def execute(
        self,
        proc: Any,
        closure_env: dict[str, Any]
    ) -> Any:
        """Execute procedure body statements and return the last return value."""
        from ..ast import TypeDefStmt, ProcDefStmt
        from .type_system import register_type, push_type_env, pop_type_env
        
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
                    register_type(self.rt, stmt.typedef, type_scope)
                
                elif isinstance(stmt, ProcDefStmt):
                    # Register nested procedure dynamically with current closure
                    from .procedure_compiler import ProcedureCompiler
                    child = stmt.proc
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
        self,
        stmt: Stmt,
        env: dict[str, Any],
        is_toplevel: bool = False
    ) -> None:
        """Execute a single statement (for top-level execution)."""
        from ..ast import TypeDefStmt, ProcDefStmt
        from .type_system import register_type
        
        if isinstance(stmt, TemplateLine):
            self.compiler.compile_template_line(stmt)(**env)
            self._sync_env(env)
        
        elif isinstance(stmt, ProcDefStmt):
            if is_toplevel:
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