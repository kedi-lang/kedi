from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    Dict,
    Optional,
    Sequence,
    Tuple,
    Union,
    get_args,
    get_origin,
)

from pydantic import BaseModel, TypeAdapter

from kedi.errors import KediTraceFrame
from kedi.promise import KediPromise, force
from kedi.telemetry import capture_enabled
from kedi.telemetry.naming import qualified_type_name

if TYPE_CHECKING:
    from kedi.core.runtime import KediRuntime


Value = Any


ValueEnv = Dict[str, Value]


@dataclass
class InputExpr:
    key: str


@dataclass
class OutputExpr:
    key: str
    pytype: Optional[Any] = None
    binding_contract: Optional[Any] = None


@dataclass
class CallExpr:
    func_name: str
    args: Tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class ComputedExpr:
    """Already-evaluated prompt text whose input dependencies are not explicit."""

    text: str


Expression = Union[str, InputExpr, OutputExpr, CallExpr, ComputedExpr]


def i(key: str) -> InputExpr:
    return InputExpr(key)


def o(
    key: str,
    pytype: Optional[Any] = None,
    *,
    binding_contract: Optional[Any] = None,
) -> OutputExpr:
    return OutputExpr(key, pytype, binding_contract)


def c(func_name: str, *args: Any) -> CallExpr:
    return CallExpr(func_name=func_name, args=tuple(args))


def _type_display_name(value: Any) -> str:
    if get_origin(value) is Annotated:
        value = get_args(value)[0]
    return getattr(value, "__name__", repr(value))


def _lookup_value(self: KediRuntime, key: str, env: ValueEnv) -> Any:
    from kedi.lang.compiler.environment import raw_lookup_value

    try:
        return raw_lookup_value(self, key, env)
    except KeyError:
        pass
    raise NameError(f"Value not found in environment: {key}")


def _validate_adapter_output(value: Any, expected_type: Any, *, explicit_type: bool) -> Any:
    if not explicit_type:
        return value
    if (
        isinstance(value, BaseModel)
        and isinstance(expected_type, type)
        and issubclass(expected_type, BaseModel)
    ):
        # Duck-typed by design: a DSL-defined model class and the adapter's structurally-equal
        # model are different classes by identity, so any BaseModel is accepted as-is and its
        # methods/type are preserved (see tests/test_type_coercion.py). A genuinely wrong model
        # fails loudly at the use site (attribute access), not silently — so this is not the
        # "wrong class accepted" bug the general report claimed; left intentionally lenient.
        return value
    return TypeAdapter(expected_type).validate_python(value)


def _evaluate_expression(
    self: KediRuntime, expr: Expression, env: ValueEnv
) -> Tuple[Any, Optional[OutputExpr]]:
    if isinstance(expr, str):
        return expr, None
    if isinstance(expr, InputExpr):
        value = self._lookup_value(expr.key, env)
        # Defer: a promise input becomes a dependency edge; the prompt is built (in the
        # produce job) only after it resolves. ``.map(str)`` makes the deferred value resolve
        # to a STRING — matching the sequential path's eager ``str(value)`` so a non-str typed
        # output (list/model/int) chained into a prompt renders identically in both modes.
        if isinstance(value, KediPromise):
            from kedi.lang.compiler.environment import model_text

            return value.map(model_text), None
        from kedi.lang.compiler.environment import model_text

        return model_text(value), None
    if isinstance(expr, CallExpr):
        evaluated_args = [self._evaluate_value(a, env) for a in expr.args]
        callee = self._get_proc(expr.func_name)
        # _invoke_procedure force-joins its result, so a cross-procedure call never yields a
        # promise here. If one ever does, str() raising KediPromiseLeak is the correct loud
        # failure — better than silently deferring an unsupported cross-proc dataflow.
        result = self._invoke_procedure(callee, evaluated_args)
        return str(result), None
    if isinstance(expr, ComputedExpr):
        return expr.text, None
    if isinstance(expr, OutputExpr):
        return f"[{expr.key}]", expr
    return str(expr), None


def _evaluate_value(self: KediRuntime, value: Any, env: ValueEnv) -> Any:
    # Procedure arguments are forced to concrete values at the call boundary.
    if isinstance(value, InputExpr):
        from kedi.lang.compiler.environment import native_value

        return native_value(force(self._lookup_value(value.key, env)))
    if isinstance(value, CallExpr):
        evaluated_args = [self._evaluate_value(a, env) for a in value.args]
        callee = self._get_proc(value.func_name)
        return force(self._invoke_procedure(callee, evaluated_args))
    if isinstance(value, OutputExpr):
        raise ValueError("Output expressions are not allowed as arguments")
    return force(value)


def _invoke_procedure(self: KediRuntime, fn: Callable[..., Any], args: Sequence[Any]) -> Any:
    import inspect

    from kedi.lang.procedure_introspection import (
        callable_display_name,
        get_procedure_def,
        get_procedure_metadata,
    )

    sig = inspect.signature(fn)
    bound = sig.bind_partial(*args)
    bound.apply_defaults()
    local_env: ValueEnv = {name: bound.arguments[name] for name in bound.arguments}
    metadata = get_procedure_metadata(fn)
    procedure_def = get_procedure_def(fn)
    proc_span = procedure_def.loc if procedure_def is not None else None
    proc_name = metadata.name if metadata is not None else callable_display_name(fn)
    should_trace_procedure = not (fn is self._main and metadata is None)

    depth = self._invoke_depth.get()
    telemetry_name = "main" if fn is self._main else (proc_name or "procedure")
    attributes: dict[str, bool | str | int] = {
        "kedi.procedure.name": telemetry_name,
        "kedi.procedure.argument_count": len(bound.arguments),
        "kedi.procedure.depth": depth,
    }
    if proc_span is not None:
        attributes["code.line.number"] = proc_span.start_line
        if proc_span.source_path is not None and capture_enabled("source_path"):
            attributes["code.file.path"] = proc_span.source_path

    procedure_kind = "main" if fn is self._main else "procedure"
    attributes["kedi.procedure.kind"] = procedure_kind

    from kedi.telemetry import runtime_span

    with runtime_span(
        f"call {telemetry_name}",
        "call_procedure",
        attributes=attributes,
        metric_attributes={"procedure_kind": procedure_kind},
    ) as active:
        depth_token = self._invoke_depth.set(depth + 1)
        request_budget_token = None
        invocation_token = None
        if depth == 0:
            from kedi.agent_adapter.request_budget import ModelRequestBudget

            request_budget_token = self._request_budget.set(ModelRequestBudget())
            invocation_token = self._invocation_id.set(object())
        token = self._current_env.set(local_env)
        primary_exc: BaseException | None = None
        try:
            frame_ctx = (
                self.trace_frame(
                    KediTraceFrame(
                        kind="procedure",
                        label=f"procedure {proc_name}" if proc_name else "procedure",
                        span=proc_span,
                        procedure_name=proc_name,
                    )
                )
                if should_trace_procedure
                else nullcontext()
            )
            with frame_ctx:
                result = fn(*args)
            observer = self._procedure_env_observer.get()
            observed_env = self._current_env.get()
            if observer is not None and observed_env is not None:
                observer(observed_env)
            final = (
                self._evaluate_value(result, local_env)
                if isinstance(result, (InputExpr, CallExpr))
                else result
            )
            active.set_attribute("kedi.result.type", qualified_type_name(final))
        except BaseException as exc:
            primary_exc = exc
            active.set_attribute("error.type", type(exc).__name__)
            raise
        finally:
            self._current_env.reset(token)
            self._invoke_depth.reset(depth_token)
            # The outermost invocation owns every deferred template spawned by its body.
            if depth == 0:
                try:
                    self._drain_outermost(primary_exc)
                except BaseException as exc:
                    active.set_attribute("error.type", type(exc).__name__)
                    raise
                finally:
                    assert request_budget_token is not None
                    self._request_budget.reset(request_budget_token)
                    assert invocation_token is not None
                    self._invocation_id.reset(invocation_token)

        return final
