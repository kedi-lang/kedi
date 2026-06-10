from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, replace
from types import TracebackType
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from .errors import KediExecutionError, KediPythonOrigin, KediTraceFrame, build_execution_error
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
        adapter: AgentAdapter[Any] | None = None,
        executor: Executor | None = None,
    ) -> None:
        self.adapter = adapter
        self.executor = executor or DefaultExecutor()
        self._procedures: Dict[str, Callable[..., Any]] = {}
        self._main: Optional[Callable[..., Any]] = None
        self._import_env: ValueEnv = {}
        self._current_env: ContextVar[Optional[ValueEnv]] = ContextVar(
            "kedi_current_env", default=None
        )
        self._trace_stack: ContextVar[Tuple[KediTraceFrame, ...]] = ContextVar(
            "kedi_trace_stack", default=()
        )
        self._initial_globals_env: ValueEnv = {}
        self._globals_env: ValueEnv = {}
        self._python_origin_counter = 0
        self._python_origins: Dict[str, KediPythonOrigin] = {}
        self._program: Any | None = None
        self._source_map: Any | None = None
        self._source_text: str | None = None

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
        name = getattr(fn, "__name__", fn.__class__.__name__)
        if name in self._procedures:
            pass  # Silently replace
        self._procedures[name] = fn
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
            return None
        return self._invoke_procedure(self._main, [])

    def set_initial_globals(self, values: ValueEnv) -> None:
        self._initial_globals_env = dict(values)
        self._globals_env = dict(values)

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

        from .lang.procedure_introspection import get_procedure_metadata

        sig = inspect.signature(fn)
        bound = sig.bind_partial(*args)
        bound.apply_defaults()
        local_env: ValueEnv = {name: bound.arguments[name] for name in bound.arguments}
        metadata = get_procedure_metadata(fn)
        proc_span = getattr(getattr(fn, "__kedi_procedure_def__", None), "loc", None)
        proc_name = metadata.name if metadata is not None else getattr(fn, "__name__", None)
        should_trace_procedure = not (fn is self._main and metadata is None)

        token = self._current_env.set(local_env)
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
        finally:
            self._current_env.reset(token)

        return (
            self._evaluate_value(result, local_env)
            if isinstance(result, (InputExpr, CallExpr))
            else result
        )

    @contextmanager
    def trace_frame(self, frame: KediTraceFrame):
        __tracebackhide__ = True
        frame = self._hydrate_frame(frame)
        stack = self._trace_stack.get()
        token = self._trace_stack.set(stack + (frame,))
        try:
            yield
        except KediExecutionError:
            raise
        except Exception as exc:
            kedi_exc = self.build_execution_error(exc)
            traceback = self._last_python_origin_traceback(exc) or exc.__traceback__
            raise kedi_exc.with_traceback(traceback) from None
        finally:
            self._trace_stack.reset(token)

    def current_trace_frames(self) -> tuple[KediTraceFrame, ...]:
        return self._trace_stack.get()

    def build_execution_error(self, exc: BaseException) -> KediExecutionError:
        return build_execution_error(
            exc,
            frames=self.current_trace_frames(),
            python_origins=self._python_origins,
            source_map=self._source_map,
        )

    def _last_python_origin_traceback(self, exc: BaseException) -> TracebackType | None:
        tb = exc.__traceback__
        last = None
        while tb is not None:
            if tb.tb_frame.f_code.co_filename in self._python_origins:
                last = tb
            tb = tb.tb_next
        if last is None:
            return None
        return TracebackType(
            None,
            last.tb_frame,
            last.tb_lasti,
            last.tb_lineno,
        )

    def register_python_origin(
        self,
        *,
        kind: str,
        code: str,
        span: Any,
        compile_line_offset: int = 0,
        wrapper_line_offset: int = 0,
        kedi_line_offset: int = 0,
        procedure_name: str | None = None,
        traceback_code: str | None = None,
    ) -> str:
        import linecache

        self._python_origin_counter += 1
        origin_span = span
        filename = f"<kedi-python:{kind}:{self._python_origin_counter}>"
        if span is not None and getattr(span, "source_path", None):
            filename = span.source_path
        elif (
            span is not None
            and self._source_map is not None
            and hasattr(self._source_map, "resolve_line")
        ):
            resolved = self._source_map.resolve_line(span.start_line)
            if resolved is not None:
                filename = resolved.source_path
                origin_span = replace(
                    span,
                    start_line=resolved.source_line,
                    end_line=resolved.source_line + max(0, span.end_line - span.start_line),
                    source_path=resolved.source_path,
                    source_id=resolved.source_id,
                )
        self._python_origins[filename] = KediPythonOrigin(
            filename=filename,
            kind=kind,
            code=code,
            span=origin_span,
            compile_line_offset=compile_line_offset,
            wrapper_line_offset=wrapper_line_offset,
            kedi_line_offset=kedi_line_offset,
            procedure_name=procedure_name,
        )
        cache_text = "\n" * compile_line_offset + (traceback_code or code)
        lines = [line + "\n" for line in cache_text.splitlines()] or ["\n"]
        linecache.cache[filename] = (len(code), None, lines, filename)
        return filename

    def _hydrate_frame(self, frame: KediTraceFrame) -> KediTraceFrame:
        if frame.source_snippet is not None or frame.span is None or frame.span.source_path is None:
            return frame
        source_map = self._source_map
        line = None
        if source_map is not None and hasattr(source_map, "get_line"):
            line = source_map.get_line(frame.span.source_path, frame.span.start_line)
        return KediTraceFrame(
            kind=frame.kind,
            label=frame.label,
            span=frame.span,
            procedure_name=frame.procedure_name,
            source_snippet=line,
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

        from .agent_adapter.profile_override import SupportsProfileOverride
        from .agent_profile import AgentProfileOverrideUnsupported, current_block_agent_profile

        profile = current_block_agent_profile()
        override_ctx = nullcontext()
        if isinstance(self.adapter, SupportsProfileOverride):
            if profile.needs_model_override(self.adapter.default_model()):
                override_ctx = self.adapter.profile_override(profile)
        elif profile.model is not None:
            raise AgentProfileOverrideUnsupported(
                f"Active adapter does not support model profile override (requested model={profile.model!r})"
            )

        with override_ctx:
            result = self.adapter.produce_sync(template=question, output_schema=typed_fields)

        for key in required:
            value = getattr(result, key, NotFound)  # Ensure result has the attribute
            if value is NotFound:
                raise ValueError(f"LLM did not return required key: {key}")
            env[key] = value
        return {k: env[k] for k in required}
