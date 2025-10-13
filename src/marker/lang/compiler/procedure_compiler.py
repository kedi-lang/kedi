"""Procedure compilation with lexical scoping support."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable
from ..ast import ProcedureDef, ProcDefStmt
from ...core import MarkerRuntime
from .environment import ScopedDict
from .statement_compiler import StatementExecutor
from .type_system import resolve_type, is_value_compatible, coerce_model_value


class ProcedureCompiler:
    """Compiles procedures with parameter/return type enforcement."""
    
    def __init__(self, rt: MarkerRuntime):
        self.rt = rt
        self.executor = StatementExecutor(rt)
    
    def compile_procedure(self, proc: ProcedureDef) -> Callable:
        """Compile a procedure to a callable function."""
        return self._make_procedure_impl(proc, parent_env=None)
    
    def _make_procedure_impl(
        self,
        proc: ProcedureDef,
        parent_env: dict[str, Any] | None
    ) -> Callable:
        """Create procedure implementation with scoping."""
        
        def _body(**kwargs: Any) -> Any:
            # Setup environment
            kwargs = self._prepare_kwargs(proc, kwargs, parent_env)
            self._enforce_param_types(proc, kwargs)
            
            # Setup scoped environment for execution
            closure_env = ScopedDict(
                kwargs,
                readonly=set(proc.params),
                closure=parent_env
            )
            
            # Execute body (nested procedures handled dynamically during execution)
            last_return = self.executor.execute(proc, closure_env)
            
            # Enforce return type
            return self._enforce_return_type(proc, last_return, closure_env)
        
        # Create dynamic function with proper signature
        return self._create_dynamic_function(proc.name, proc.params, _body)
    
    def _prepare_kwargs(
        self,
        proc: ProcedureDef,
        kwargs: dict[str, Any],
        parent_env: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Prepare kwargs with lexical and global environments."""
        # Merge lexical environment
        if parent_env:
            kwargs = {**parent_env, **kwargs}
        
        # Inject globals
        globals_env = getattr(self.rt, "_globals_env", None)
        if isinstance(globals_env, dict):
            for k, v in globals_env.items():
                kwargs.setdefault(k, v)
            # Also inject into current env
            current_env = self.rt._current_env.get()
            if isinstance(current_env, dict):
                for k, v in globals_env.items():
                    current_env.setdefault(k, v)
        
        return kwargs
    
    def _enforce_param_types(
        self,
        proc: ProcedureDef,
        kwargs: dict[str, Any]
    ) -> None:
        """Enforce parameter type annotations."""
        if not proc.param_type_annotations:
            return
        
        for param_name in proc.params:
            expected = proc.param_type_annotations.get(param_name)
            if expected:
                pytype = resolve_type(self.rt, expected)
            else:
                pytype = str  # default to string
            
            if param_name not in kwargs:
                raise TypeError(f"Missing parameter: {param_name}")
            
            provided = coerce_model_value(pytype, kwargs[param_name]) if pytype is not None else kwargs[param_name]
            kwargs[param_name] = provided
            
            if pytype is not None and not is_value_compatible(pytype, provided):
                raise TypeError(f"Parameter '{param_name}' does not match annotated type")
    
    def _enforce_return_type(
        self,
        proc: ProcedureDef,
        last_return: Any,
        env: dict[str, Any]
    ) -> Any:
        """Enforce return type annotation."""
        anno = resolve_type(self.rt, proc.return_type_annotation, env=env)
        
        if anno is None:
            # Default to str: non-string native returns are illegal
            if last_return is not None and not isinstance(last_return, str):
                raise TypeError("Return value must be str; annotate function to return non-string")
            return "" if last_return is None else str(last_return)
        
        # Annotated non-str: allow native returns; validate type
        if not is_value_compatible(anno, last_return):
            raise TypeError("Return value does not match annotated type")
        return last_return
    
    @contextmanager
    def _nested_procedure_scope(self, proc: ProcedureDef, closure_env: dict):
        """Context manager for nested procedure registration."""
        saved_procs: dict[str, Any] = {}
        new_names: list[str] = []
        
        try:
            # Register nested procedures
            for stmt in proc.body:
                if isinstance(stmt, ProcDefStmt):
                    child = stmt.proc
                    # Capture lexical environment
                    child.lexical_env = dict(closure_env)
                    fn_nested = self._make_procedure_impl(child, child.lexical_env)
                    
                    procs = self.rt._procedures
                    if child.name in procs and child.name not in saved_procs:
                        saved_procs[child.name] = procs[child.name]
                        procs[child.name] = fn_nested
                    elif child.name not in procs:
                        procs[child.name] = fn_nested
                        new_names.append(child.name)
            
            yield
        finally:
            # Restore scope
            procs = self.rt._procedures
            for name in new_names:
                if name in procs and name not in saved_procs:
                    try:
                        del procs[name]
                    except Exception:
                        pass
            for name, old_fn in saved_procs.items():
                procs[name] = old_fn
    
    @staticmethod
    def _create_dynamic_function(
        name: str,
        params: list[str],
        body: Callable
    ) -> Callable:
        """Create a function with specific parameter names."""
        param_list = ", ".join(params)
        args_map = ", ".join(f"{n}={n}" for n in params)
        src = f"def _impl({param_list}):\n    return _body({args_map})\n"
        
        ns: dict[str, Any] = {"_body": body}
        exec(src, ns, ns)
        fn = ns["_impl"]
        fn.__name__ = name
        return fn