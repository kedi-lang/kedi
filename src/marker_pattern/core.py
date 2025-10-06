from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from contextvars import ContextVar

from pydantic_ai import Agent
from pydantic_ai.output import StructuredDict

# -----------------------------
# Marker runtime implementation
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


@dataclass
class CallExpr:
    func_name: str
    args: Tuple[Any, ...]


Expression = Union[str, InputExpr, OutputExpr, CallExpr]


def i(key: str) -> InputExpr:
    return InputExpr(key)


def o(key: str) -> OutputExpr:
    return OutputExpr(key)


def c(func_name: str, *args: Any) -> CallExpr:
    return CallExpr(func_name=func_name, args=tuple(args))


class MarkerRuntime:
    """Isolated marker runtime with its own registry and environment."""

    def __init__(self, model: str = "groq:qwen/qwen3-32b") -> None:
        self.model = model
        self._procedures: Dict[str, Callable[..., Any]] = {}
        self._main: Optional[Callable[..., Any]] = None
        self._current_env: ContextVar[Optional[ValueEnv]] = ContextVar(
            "marker_current_env", default=None
        )

    # Expressions
    def i(self, key: str) -> InputExpr:
        return i(key)

    def o(self, key: str) -> OutputExpr:
        return o(key)

    def c(self, func_name: str, *args: Any) -> CallExpr:
        return c(func_name, *args)

    # Decorators
    def procedure(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        if fn.__name__ in self._procedures:
            raise ValueError(f"Procedure already registered: {fn.__name__}")
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
            raise RuntimeError("m() called outside of a marker procedure context")
        return self._marker_run(expressions, env)

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

    def _evaluate_expression(self, expr: Expression, env: ValueEnv) -> Tuple[Optional[str], Optional[OutputExpr]]:
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

        return self._evaluate_value(result, local_env) if isinstance(result, (InputExpr, CallExpr)) else result

    def _marker_run(self, expressions: Sequence[Expression], env: ValueEnv) -> Dict[str, Any]:
        prompt_parts: List[str] = []
        outputs: List[OutputExpr] = []
        for expr in expressions:
            text, out = self._evaluate_expression(expr, env)
            if out is not None:
                outputs.append(out)
            if text is not None:
                prompt_parts.append(text)

        description = "".join(prompt_parts)
        properties = {out.key: {"type": "string"} for out in outputs}
        required = [out.key for out in outputs]

        # If there are no required outputs, there's nothing to ask the LLM.
        # Treat as a no-op and return empty mapping.
        if not required:
            return {}

        out_type = StructuredDict(
            {
                "type": "object",
                "properties": properties,
                "required": required,
            },
            name="result",
            description=description,
        )
        try:
            setattr(out_type, "description", description)
        except Exception:
            pass

        agent = Agent(self.model, output_type=out_type)
        question = f"Fill in the fields: [{required[0]}]." if len(required) == 1 else f"Solve [{', '.join(required)}]."
        result = agent.run_sync(question)
        
        #print(f"description: {description}")
        #print(f"question: {question}")
        #print(f"result: {result}")
        data = result.output
        if hasattr(data, "model_dump"):
            data_dict: Dict[str, Any] = data.model_dump()
        elif isinstance(data, dict):
            data_dict = data
        else:
            data_dict = dict(data)

        for key in required:
            if key not in data_dict:
                raise ValueError(f"LLM did not return required key: {key}")
            env[key] = data_dict[key]

        return {k: env[k] for k in required}


# No default runtime instance; users should create their own `MarkerRuntime()`

