"""Procedure compilation with lexical scoping support."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable

from pydantic import ValidationError

from ...agent_adapter.tools.retry import ToolRetryPolicy, resolve_retry_on
from ...core import KediRuntime
from ...debugging import execution_frame
from ...errors import KediTraceFrame, KediTypeExecutionError
from ..ast import ProcDefStmt, ProcedureDef, TypeRef
from ..procedure_introspection import attach_procedure_metadata
from .environment import BindingMetadata, ScopeFrame, build_python_env
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

        def _validated_body(**kwargs: Any) -> Any:
            lexical_parent = parent_env if parent_env is not None else self.rt._globals_env
            closure_env = ScopeFrame(parent=lexical_parent)
            for name, value in kwargs.items():
                closure_env.declare(
                    name,
                    value,
                    BindingMetadata(kind="parameter"),
                )

            token = self.rt._current_env.set(closure_env)
            try:
                # Execute body (nested procedures handled dynamically during execution)
                with execution_frame(self.rt, proc.name, closure_env, proc.loc):
                    last_return = self.executor.execute(proc, closure_env)

                observer = self.rt._procedure_env_observer.get()
                if observer is not None:
                    observer(closure_env)

                # Enforce return type
                return self._enforce_return_type(proc, last_return, closure_env)
            finally:
                self.rt._current_env.reset(token)

        def _body(**kwargs: Any) -> Any:
            self._enforce_param_types(proc, kwargs)
            return _validated_body(**kwargs)

        defaults = self._evaluate_param_defaults(proc, parent_env)

        # Create dynamic function with proper signature
        fn = self._create_dynamic_function(proc.name, proc.params, _body, defaults)
        setattr(fn, "__kedi_validated_procedure_body__", _validated_body)
        if proc.tool is not None:
            policy = ToolRetryPolicy(
                retries=proc.tool.retries,
                retry_on=resolve_retry_on(
                    proc.tool.retry_on,
                    build_python_env(self.rt, parent_env or {}),
                ),
            )
            setattr(fn, "__kedi_tool_retry_policy__", policy)
        return attach_procedure_metadata(fn, proc, defaults)

    def _evaluate_param_defaults(
        self, proc: ProcedureDef, parent_env: dict[str, Any] | None
    ) -> dict[str, Any]:
        if not proc.param_defaults:
            return {}
        env = build_python_env(self.rt, parent_env or {})
        defaults: dict[str, Any] = {}
        for name, code in proc.param_defaults.items():
            frame = KediTraceFrame(
                kind="parameter default",
                label=f"default of {name}",
                span=proc.loc,
                procedure_name=proc.name,
            )
            with self.rt.trace_frame(frame):
                defaults[name] = self.rt.executor.evaluate_type_expression(self.rt, code, env)
        return defaults

    def _enforce_param_types(self, proc: ProcedureDef, kwargs: dict[str, Any]) -> None:
        """Enforce parameter type annotations."""
        param_exprs = proc.param_type_exprs or {}
        param_strs = proc.param_type_annotations or {}

        for param_name in proc.params:
            kedi_type = KediType.from_node(
                param_exprs.get(param_name),
                param_strs.get(param_name),
            )

            if param_name not in kwargs:
                raise KediTypeExecutionError(f"Missing parameter: {param_name}")

            provided = kwargs[param_name]

            # Untyped params are str: a str arg passes, a native non-str raises.
            if kedi_type.is_none():
                kedi_type = KediType(TypeRef(name="str"))

            try:
                validated = kedi_type.validate(self.rt, None, provided)
            except (ValidationError, OSError) as exc:
                # Only a value that fails validation is a mismatch; a media path that cannot be
                # read (OSError) is one too. Other errors, such as the recursion limit, propagate.
                from ...media import media_error_detail

                detail = media_error_detail(exc)
                raise KediTypeExecutionError(
                    f"Parameter '{param_name}' does not match annotated type"
                    + ("" if detail is None else f": {detail}")
                ) from None
            kwargs[param_name] = validated

    def _enforce_return_type(
        self, proc: ProcedureDef, last_return: Any, env: dict[str, Any]
    ) -> Any:
        """Enforce return type annotation."""
        from ...artifacts.stream import ArtifactStream, artifact_stream_matches_annotation
        from ...promise import force
        from ...tool_context import inside_kedi_tool_call

        # A procedure return is a join point: resolve a promise to a concrete value before
        # type-enforcing or stringifying it.
        last_return = force(last_return)
        kedi_type = KediType.from_node(proc.return_type_expr, proc.return_type_annotation)

        if isinstance(last_return, ArtifactStream):
            expected = str if kedi_type.is_none() else kedi_type.resolve_required(self.rt, env)
            if not inside_kedi_tool_call():
                raise KediTypeExecutionError(
                    "ArtifactStream is a tool transport and cannot be returned by a direct "
                    "procedure call"
                )
            if not artifact_stream_matches_annotation(last_return, expected):
                raise KediTypeExecutionError(
                    "ArtifactStream transport does not match the procedure return type"
                )
            return last_return

        if kedi_type.is_none():
            # Untyped return is str; a non-string native return is illegal.
            if last_return is not None and not isinstance(last_return, str):
                raise KediTypeExecutionError(
                    "Return value must be str; annotate function to return non-string"
                )
            return "" if last_return is None else str(last_return)

        # Raise a precise "Unknown type annotation X" before validating the value.
        kedi_type.resolve_required(self.rt, env)
        try:
            return kedi_type.validate(self.rt, env, last_return)
        except ValidationError:
            raise KediTypeExecutionError("Return value does not match annotated type")

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
                    if child is None:
                        continue
                    # Capture lexical environment
                    child.lexical_env = closure_env
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
