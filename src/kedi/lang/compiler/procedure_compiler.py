"""Procedure compilation with lexical scoping support."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable

from ...core import KediRuntime
from ..ast import ProcDefStmt, ProcedureDef
from ..procedure_introspection import attach_procedure_metadata
from .environment import ScopedDict, build_python_env
from .statement_compiler import StatementExecutor
from .type_system import KediType


class ProcedureCompiler:
    """Compiles procedures with parameter/return type enforcement."""

    def __init__(self, rt: KediRuntime):
        self.rt = rt
        self.executor = StatementExecutor(rt)

    def compile_procedure(self, proc: ProcedureDef) -> Callable:
        """Compile a procedure to a callable function."""
        return self._make_procedure_impl(proc, parent_env=None)

    def _make_procedure_impl(
        self, proc: ProcedureDef, parent_env: dict[str, Any] | None
    ) -> Callable:
        """Create procedure implementation with scoping."""

        def _body(**kwargs: Any) -> Any:
            # Setup environment
            kwargs = self._prepare_kwargs(proc, kwargs, parent_env)
            self._enforce_param_types(proc, kwargs)

            # Setup scoped environment for execution
            closure_env = ScopedDict(kwargs, readonly=set(proc.params), closure=parent_env)

            # Execute body (nested procedures handled dynamically during execution)
            last_return = self.executor.execute(proc, closure_env)

            # Enforce return type
            return self._enforce_return_type(proc, last_return, closure_env)

        defaults = self._evaluate_param_defaults(proc, parent_env)

        # Create dynamic function with proper signature
        fn = self._create_dynamic_function(proc.name, proc.params, _body, defaults)
        return attach_procedure_metadata(fn, proc, defaults)

    def _evaluate_param_defaults(
        self, proc: ProcedureDef, parent_env: dict[str, Any] | None
    ) -> dict[str, Any]:
        if not proc.param_defaults:
            return {}
        env = build_python_env(self.rt, parent_env or {})
        return {
            name: self.rt.executor.evaluate_type_expression(self.rt, code, env)
            for name, code in proc.param_defaults.items()
        }

    def _prepare_kwargs(
        self,
        proc: ProcedureDef,
        kwargs: dict[str, Any],
        parent_env: dict[str, Any] | None,
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

    def _enforce_param_types(self, proc: ProcedureDef, kwargs: dict[str, Any]) -> None:
        """Enforce parameter type annotations."""
        if not (proc.param_type_annotations or proc.param_type_exprs):
            return

        param_exprs = proc.param_type_exprs or {}
        param_strs = proc.param_type_annotations or {}

        for param_name in proc.params:
            kedi_type = KediType.from_node(
                param_exprs.get(param_name),
                param_strs.get(param_name),
            )

            if param_name not in kwargs:
                raise TypeError(f"Missing parameter: {param_name}")

            provided = kwargs[param_name]

            # Strict validation for parameters using TypeAdapter
            if not kedi_type.is_none():
                try:
                    validated = kedi_type.validate(
                        self.rt, None, provided, allow_str_coercion=False
                    )
                except Exception:
                    raise TypeError(f"Parameter '{param_name}' does not match annotated type")
                kwargs[param_name] = validated

    def _enforce_return_type(
        self, proc: ProcedureDef, last_return: Any, env: dict[str, Any]
    ) -> Any:
        """Enforce return type annotation."""
        kedi_type = KediType.from_node(proc.return_type_expr, proc.return_type_annotation)

        if kedi_type.is_none():
            # Default to str: non-string native returns are illegal
            if last_return is not None and not isinstance(last_return, str):
                raise TypeError("Return value must be str; annotate function to return non-string")
            return "" if last_return is None else str(last_return)

        # Annotated return: strict validate and return validated value
        try:
            # Do not allow string literal coercion for returns
            return kedi_type.validate(self.rt, env, last_return, allow_str_coercion=False)
        except Exception:
            raise TypeError("Return value does not match annotated type")

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

    def _create_dynamic_function(
        self,
        name: str,
        params: list[str],
        body: Callable,
        defaults: dict[str, Any] | None = None,
    ) -> Callable:
        """Create a function with specific parameter names."""
        return self.rt.executor.create_dynamic_function(name, params, body, defaults)
