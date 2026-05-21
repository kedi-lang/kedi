from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from .executors.base import DefaultExecutor, Executor

if TYPE_CHECKING:
    from kedi.agent_adapter import AgentAdapter

# -----------------------------
# Kedi runtime implementation
# -----------------------------

# Public API types
Value = Any
ValueEnv = Dict[str, Value]


@dataclass
class InputExpr:
    key: str


@dataclass
class OutputExpr:
    key: str
    pytype: Optional[Any] = None


@dataclass
class CallExpr:
    func_name: str
    args: Tuple[Any, ...]


Expression = Union[str, InputExpr, OutputExpr, CallExpr]


def i(key: str) -> InputExpr:
    return InputExpr(key)


def o(key: str, pytype: Optional[Any] = None) -> OutputExpr:
    return OutputExpr(key, pytype)


def c(func_name: str, *args: Any) -> CallExpr:
    return CallExpr(func_name=func_name, args=tuple(args))


class KediRuntime:
    """Isolated kedi runtime with its own registry and environment."""

    def __init__(
        self,
        *,
        adapter: AgentAdapter | None = None,
        executor: Executor | None = None,
    ) -> None:
        self.adapter = adapter
        self.executor = executor or DefaultExecutor()
        self._procedures: Dict[str, Callable[..., Any]] = {}
        self._main: Optional[Callable[..., Any]] = None
        self._current_env: ContextVar[Optional[ValueEnv]] = ContextVar(
            "kedi_current_env", default=None
        )

    # Expressions
    def i(self, key: str) -> InputExpr:
        return i(key)

    def o(self, key: str, pytype: Optional[Any] = None) -> OutputExpr:
        return o(key, pytype)

    def c(self, func_name: str, *args: Any) -> CallExpr:
        return c(func_name, *args)

    # Decorators
    def procedure(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        # Allow re-registration for AI-generated procedures from cache
        if fn.__name__ in self._procedures:
            pass  # Silently replace
        self._procedures[fn.__name__] = fn
        return fn

    def main(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        # Allow overriding main for test isolation
        self._main = fn
        return fn

    # Public API
    def m(self, expressions: Sequence[Expression]) -> Dict[str, Any]:
        env = self._current_env.get()
        if env is None:
            raise RuntimeError("m() called outside of a kedi procedure context")
        return self._kedi_run(expressions, env)

    def run_main(self) -> Any:
        if self._main is None:
            raise RuntimeError("No main procedure defined")
        return self._invoke_procedure(self._main, [])

    # Internal helpers
    def _get_proc(self, name: str) -> Callable[..., Any]:
        try:
            return self._procedures[name]
        except KeyError as e:
            raise KeyError(f"Unknown procedure: {name}") from e

    def _evaluate_expression(
        self, expr: Expression, env: ValueEnv
    ) -> Tuple[Optional[str], Optional[OutputExpr]]:
        if isinstance(expr, str):
            return expr, None
        if isinstance(expr, InputExpr):
            if expr.key not in env:
                raise KeyError(f"Value not found in environment: {expr.key}")
            return str(env[expr.key]), None
        if isinstance(expr, CallExpr):
            evaluated_args = [self._evaluate_value(a, env) for a in expr.args]
            callee = self._get_proc(expr.func_name)
            return str(self._invoke_procedure(callee, evaluated_args)), None
        if isinstance(expr, OutputExpr):
            return f"[{expr.key}]", expr
        return str(expr), None

    def _evaluate_value(self, value: Any, env: ValueEnv) -> Any:
        if isinstance(value, InputExpr):
            if value.key not in env:
                raise KeyError(f"Value not found in environment: {value.key}")
            return env[value.key]
        if isinstance(value, CallExpr):
            evaluated_args = [self._evaluate_value(a, env) for a in value.args]
            callee = self._get_proc(value.func_name)
            return self._invoke_procedure(callee, evaluated_args)
        if isinstance(value, OutputExpr):
            raise ValueError("Output expressions are not allowed as arguments")
        return value

    def _invoke_procedure(self, fn: Callable[..., Any], args: Sequence[Any]) -> Any:
        import inspect

        sig = inspect.signature(fn)
        bound = sig.bind_partial(*args)
        bound.apply_defaults()
        local_env: ValueEnv = {name: bound.arguments[name] for name in bound.arguments}

        token = self._current_env.set(local_env)
        try:
            result = fn(*args)
        finally:
            self._current_env.reset(token)

        return (
            self._evaluate_value(result, local_env)
            if isinstance(result, (InputExpr, CallExpr))
            else result
        )

    def _kedi_run(self, expressions: Sequence[Expression], env: ValueEnv) -> Dict[str, Any]:
        NotFound = object()
        prompt_parts: List[str] = []
        outputs: List[OutputExpr] = []
        for expr in expressions:
            text, out = self._evaluate_expression(expr, env)
            if out is not None:
                outputs.append(out)
            if text is not None:
                prompt_parts.append(text)

        description = "".join(prompt_parts)
        required = [out.key for out in outputs]

        # If there are no required outputs, there's nothing to ask the LLM.
        # Treat as a no-op and return empty mapping.
        if not required:
            return {}

        # Build output schema mapping (default to str when unspecified)
        typed_fields: Dict[str, Any] = {
            out.key: (out.pytype if out.pytype is not None else str) for out in outputs
        }
        # Keep question format stable for tests; append description after to preserve key parsing
        base_q = (
            f"Fill in the fields: [{required[0]}]."
            if len(required) == 1
            else f"Fill in the fields: [{', '.join(required)}]."
        )
        # Provide the full description as guidance for better tool performance
        question = base_q + "\n" + description

        if self.adapter is None:
            raise RuntimeError("No adapter provided to KediRuntime")

        result = self.adapter.produce_sync(template=question, output_schema=typed_fields)

        for key in required:
            value = getattr(result, key, NotFound)  # Ensure result has the attribute
            if value is NotFound:
                raise ValueError(f"LLM did not return required key: {key}")
            env[key] = value
        return {k: env[k] for k in required}
