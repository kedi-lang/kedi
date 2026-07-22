from __future__ import annotations

import atexit
import os
from pathlib import Path
from concurrent.futures import CancelledError
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, replace
from threading import Lock
from types import TracebackType
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
    cast,
)

from pydantic import BaseModel, TypeAdapter

from .errors import (
    KediExecutionError,
    KediPythonOrigin,
    KediTraceFrame,
    build_execution_error,
)
from .executors.base import DefaultExecutor, Executor
from .promise import KediPromise, force

_SHARED_PARALLEL_ENGINES: Dict[int, Any] = {}
_SHARED_PARALLEL_LOCK = Lock()

_PARALLEL_TRUE = frozenset({"1", "true", "yes", "on"})
_PARALLEL_FALSE = frozenset({"", "0", "false", "no", "off"})


def shared_thread_engine(max_workers: int = 8) -> Any:
    """Return a process-wide shared ``ThreadEngine``, one bounded pool per distinct worker count.

    A fresh engine per runtime would spawn a thread pool per program run; sharing a pool per size
    keeps concurrency capped while still honoring a different ``max_workers`` request (the engine's
    own ``__init__`` runs the interpreter routing self-check).
    """
    with _SHARED_PARALLEL_LOCK:
        engine = _SHARED_PARALLEL_ENGINES.get(max_workers)
        if engine is None:
            from .engine import ThreadEngine

            engine = ThreadEngine(max_workers=max_workers)
            _SHARED_PARALLEL_ENGINES[max_workers] = engine
        return engine


@atexit.register
def _shutdown_shared_engines() -> None:
    """Shut down every cached parallel engine's thread pool at interpreter exit.

    The cache holds one bounded pool per distinct worker count for the process lifetime; this
    releases their worker threads on exit instead of leaning solely on interpreter teardown.
    """
    with _SHARED_PARALLEL_LOCK:
        engines = list(_SHARED_PARALLEL_ENGINES.values())
        _SHARED_PARALLEL_ENGINES.clear()
    for engine in engines:
        engine.shutdown()


def _default_engine() -> Any:
    """Default engine: sequential unless parallel is explicitly opted into via ``KEDI_PARALLEL``.

    ``0/false/no/off`` (or unset) -> sequential; ``1/true/yes/on`` -> parallel (default workers);
    a positive integer -> parallel with that many workers; anything else raises (no silent flip).
    """
    from .engine import SequentialEngine

    raw = (os.getenv("KEDI_PARALLEL") or "").strip()
    low = raw.lower()
    if low in _PARALLEL_FALSE:
        return SequentialEngine()
    if low in _PARALLEL_TRUE:
        return shared_thread_engine(max_workers=8)
    try:
        workers = int(raw)
    except ValueError:
        raise RuntimeError(
            f"Invalid KEDI_PARALLEL value {raw!r}: use 0/1/true/false/on/off or a worker count."
        ) from None
    if workers <= 0:
        return SequentialEngine()
    return shared_thread_engine(max_workers=workers)


if TYPE_CHECKING:
    from kedi.agent_adapter import AgentAdapter
    from kedi.lang.ast import EvalSuite, TestSuite

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
        engine: Any | None = None,
    ) -> None:
        self.adapter = adapter
        self.executor = executor or DefaultExecutor()
        self._engine = engine if engine is not None else _default_engine()
        self._selected_adapters: dict[tuple[str, str | None], AgentAdapter[Any]] = {}
        # Outstanding produce Futures (parallel mode) so the end-of-run drain awaits every
        # scheduled template — even unconsumed ones — and surfaces their errors. Lock-guarded
        # because workers add/remove concurrently.
        self._outstanding: list[Any] = []
        self._outstanding_lock = Lock()
        self._origin_counter_lock = Lock()
        self._procedures: Dict[str, Callable[..., Any]] = {}
        self._main: Optional[Callable[..., Any]] = None
        self._import_env: ValueEnv = {}
        self._current_env: ContextVar[Optional[ValueEnv]] = ContextVar(
            "kedi_current_env", default=None
        )
        self._trace_stack: ContextVar[Tuple[KediTraceFrame, ...]] = ContextVar(
            "kedi_trace_stack", default=()
        )
        # Tracks procedure-call nesting so the end-of-run drain fires exactly once, at the
        # outermost invocation, regardless of entry point (run_main / query / direct invoke).
        self._invoke_depth: ContextVar[int] = ContextVar("kedi_invoke_depth", default=0)
        self._initial_globals_env: ValueEnv = {}
        self._globals_env: ValueEnv = {}
        self._python_origin_counter = 0
        self._python_origins: Dict[str, KediPythonOrigin] = {}
        self._program: Any | None = None
        self._source_map: Any | None = None
        self._source_text: str | None = None
        self._type_env_stack: list[dict[str, Any]] = []
        self._profile_registry: dict[str, Any] = {}
        self._prelude_env: ValueEnv | None = None
        self._validation_tests: list[TestSuite] = []
        self._validation_evals: list[EvalSuite] = []

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
        from kedi.lang.procedure_introspection import callable_display_name

        name = callable_display_name(fn)
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

    def invoke(self, expressions: Sequence[Expression], *, capture: str | None = None) -> Any:
        env = self._current_env.get()
        if env is None:
            raise RuntimeError("invoke() called outside of a kedi procedure context")

        parts: List[Any] = []
        deps: List[Any] = []
        for expr in expressions:
            text, out = self._evaluate_expression(expr, env)
            if out is not None:
                raise ValueError("Raw invoke prompts cannot contain output fields.")
            if text is not None:
                parts.append(text)
                if isinstance(text, KediPromise):
                    deps.append(text._future)

        result = self._invoke_text_parts(parts, deps, env, capture_key=capture)
        if capture is None:
            return None
        return result[capture]

    def run_main(self) -> Any:
        if self._main is None:
            return None
        result = self._invoke_procedure(self._main, [])
        # Drain awaits every scheduled template (parallel mode) so side effects fire and errors
        # surface; force resolves a promise result before handing it to the caller.
        self.drain()
        return force(result)

    def set_initial_globals(self, values: ValueEnv) -> None:
        self._initial_globals_env = dict(values)
        self._globals_env = dict(values)

    # Internal helpers
    def _get_proc(self, name: str) -> Callable[..., Any]:
        try:
            return self._procedures[name]
        except KeyError as e:
            raise KeyError(f"Unknown procedure: {name}") from e

    def _lookup_value(self, key: str, env: ValueEnv) -> Any:
        if key in env:
            return env[key]

        current_env = self._current_env.get()
        if isinstance(current_env, dict) and key in current_env:
            return current_env[key]

        for layer in (
            self._globals_env,
            self._initial_globals_env,
            self._prelude_env,
            self._import_env,
        ):
            if isinstance(layer, dict) and key in layer:
                return layer[key]

        raise NameError(f"Value not found in environment: {key}")

    @staticmethod
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
        self, expr: Expression, env: ValueEnv
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
                return value.map(str), None
            return str(value), None
        if isinstance(expr, CallExpr):
            evaluated_args = [self._evaluate_value(a, env) for a in expr.args]
            callee = self._get_proc(expr.func_name)
            # _invoke_procedure force-joins its result, so a cross-procedure call never yields a
            # promise here. If one ever does, str() raising KediPromiseLeak is the correct loud
            # failure — better than silently deferring an unsupported cross-proc dataflow.
            result = self._invoke_procedure(callee, evaluated_args)
            return str(result), None
        if isinstance(expr, OutputExpr):
            return f"[{expr.key}]", expr
        return str(expr), None

    def _evaluate_value(self, value: Any, env: ValueEnv) -> Any:
        # Procedure arguments are forced to concrete values at the call boundary.
        if isinstance(value, InputExpr):
            return force(self._lookup_value(value.key, env))
        if isinstance(value, CallExpr):
            evaluated_args = [self._evaluate_value(a, env) for a in value.args]
            callee = self._get_proc(value.func_name)
            return force(self._invoke_procedure(callee, evaluated_args))
        if isinstance(value, OutputExpr):
            raise ValueError("Output expressions are not allowed as arguments")
        return force(value)

    def _invoke_procedure(self, fn: Callable[..., Any], args: Sequence[Any]) -> Any:
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
        depth_token = self._invoke_depth.set(depth + 1)
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
            final = (
                self._evaluate_value(result, local_env)
                if isinstance(result, (InputExpr, CallExpr))
                else result
            )
        except BaseException as exc:
            primary_exc = exc
            raise
        finally:
            self._current_env.reset(token)
            self._invoke_depth.reset(depth_token)
            # Outermost invocation: drain scheduled template jobs — even on the EXCEPTIONAL exit —
            # so unconsumed templates run, background jobs don't leak, and their errors surface.
            # Nested calls (depth > 0, incl. worker threads) never drain. A drain failure becomes
            # the error only if the body didn't already raise; otherwise it's logged, not masked.
            if depth == 0:
                self._drain_outermost(primary_exc)

        return final

    def _drain_outermost(self, primary_exc: BaseException | None) -> None:
        try:
            self.drain()
        except BaseException:
            if primary_exc is None:
                raise
            import logging

            logging.getLogger("kedi").error(
                "Template failure during drain after a primary error", exc_info=True
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

        with self._origin_counter_lock:
            self._python_origin_counter += 1
            counter = self._python_origin_counter
        origin_span = span
        # Unique virtual filename per snippet so snippets sharing a .kedi file don't
        # collide; the real source path lives on origin_span for rendering.
        filename = f"<kedi-python:{kind}:{counter}>"
        if span is not None and span.source_path is not None:
            origin_span = span
        elif (
            span is not None
            and self._source_map is not None
            and hasattr(self._source_map, "resolve_line")
        ):
            resolved = self._source_map.resolve_line(span.start_line)
            if resolved is not None:
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

    def _invoke_text_parts(
        self,
        parts: Sequence[Any],
        deps: Sequence[Any],
        env: ValueEnv,
        *,
        capture_key: str | None,
    ) -> Dict[str, Any]:
        if capture_key is not None:
            from .lang.compiler.environment import assert_not_cli_args_identifier

            assert_not_cli_args_identifier(capture_key)
        parallel = bool(getattr(self._engine, "parallel", False))

        from .agent_profile import current_block_agent_profile

        block_profile = current_block_agent_profile()
        adapter = self._resolve_adapter(block_profile)
        captured_tool_specs = self._resolve_agent_tool_specs(block_profile, adapter)
        self._assert_profile_override_supported(block_profile, adapter)

        def _call_invoke() -> str:
            resolved_parts = [p.resolve() if isinstance(p, KediPromise) else p for p in parts]
            prompt = "".join(resolved_parts)

            from .agent_adapter.profile_override import SupportsProfileOverride
            from .agent_profile import current_block_agent_profile

            profile = current_block_agent_profile()
            override_ctx = nullcontext()
            if isinstance(adapter, SupportsProfileOverride) and profile.needs_adapter_override(
                adapter.default_model()
            ):
                override_ctx = adapter.profile_override(profile)

            tool_adapter: Any = None
            tool_ctx: Any = nullcontext()
            if captured_tool_specs:
                tool_adapter = cast(Any, adapter)
                tool_ctx = tool_adapter.tool_scope()

            with tool_ctx:
                if tool_adapter is not None:
                    for spec in captured_tool_specs:
                        tool_adapter.register_tool(spec)
                run_kwargs: dict[str, Any] = {}
                instructions = self._agent_instructions(profile)
                if instructions is not None:
                    run_kwargs["instructions"] = instructions
                with override_ctx:
                    return str(adapter.invoke_sync(prompt=prompt, **run_kwargs))

        if not parallel:
            response = _call_invoke()
            if capture_key is None:
                return {}
            env[capture_key] = response
            return {capture_key: response}

        env_snapshot = dict(env)

        def _job() -> Dict[str, Any]:
            token = self._current_env.set(env_snapshot)
            try:
                response = _call_invoke()
                return {capture_key: response} if capture_key is not None else {}
            except KediExecutionError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise self.build_execution_error(exc) from None
            finally:
                self._current_env.reset(token)

        future = self._engine.schedule(deps, _job)
        self._register_outstanding(future)
        if capture_key is None:
            return {}
        view = KediPromise(future, capture_key)
        env[capture_key] = view
        return {capture_key: view}

    def _kedi_run(self, expressions: Sequence[Expression], env: ValueEnv) -> Dict[str, Any]:
        NotFound = object()
        # Prompt parts may include unresolved promises (dependency edges); collect their Futures
        # so the produce job is scheduled only after they resolve.
        parts: List[Any] = []
        outputs: List[OutputExpr] = []
        deps: List[Any] = []
        for expr in expressions:
            text, out = self._evaluate_expression(expr, env)
            if out is not None:
                outputs.append(out)
            if text is not None:
                parts.append(text)
                if isinstance(text, KediPromise):
                    deps.append(text._future)

        required = [out.key for out in outputs]
        if required:
            from .lang.compiler.environment import assert_not_cli_args_identifier

            for key in required:
                assert_not_cli_args_identifier(key)

        if not required:
            if not parts:
                return {}
            return self._invoke_text_parts(parts, deps, env, capture_key=None)

        # Build output schema mapping (default to str when unspecified)
        typed_fields: Dict[str, Any] = {
            out.key: (out.pytype if out.pytype is not None else str) for out in outputs
        }
        explicit_typed_fields = {out.key for out in outputs if out.pytype is not None}
        # Keep question format stable for tests; append description after to preserve key parsing
        prompt_template = (
            "Fields to be substituted into the template string for final construction: ({})."
        )
        base_q = (
            prompt_template.format(required[0])
            if len(required) == 1
            else prompt_template.format(", ".join(required))
        )

        parallel = bool(getattr(self._engine, "parallel", False))

        # Resolve tool specs NOW, on the main thread, while the lexical scope is correct. A
        # deferred job runs after the defining scope (e.g. a nested-procedure tool) has exited, so
        # re-resolving by name from rt._procedures then would bind the wrong procedure. Building
        # the spec here captures the compiled procedure fn for both registration AND invocation.
        from .agent_profile import current_block_agent_profile

        block_profile = current_block_agent_profile()
        adapter = self._resolve_adapter(block_profile)
        captured_tool_specs = self._resolve_agent_tool_specs(block_profile, adapter)
        # Validate the profile override on the main thread too, so it raises synchronously at the
        # call site in BOTH modes — symmetric with the tool guard above, instead of surfacing late
        # at drain (wrapped, differently typed) when _produce runs on a parallel worker.
        self._assert_profile_override_supported(block_profile, adapter)

        def _produce() -> Dict[str, Any]:
            # Resolve any promise prompt parts (already complete by the time this job runs in the
            # parallel engine; concrete already in sequential), then build the prompt + call the
            # model + validate — all inside the captured context so profile/tools are correct.
            resolved_parts = [p.resolve() if isinstance(p, KediPromise) else p for p in parts]
            instruction = (
                "The template string must maintain grammatical and semantic integrity once the output fields are substituted into"
                " their corresponding locations in the template after the final construction.\n"
            )
            question = instruction + base_q + "\n\nThe template string:\n" + "".join(resolved_parts)

            from .agent_adapter.profile_override import SupportsProfileOverride
            from .agent_profile import current_block_agent_profile

            # Support was already validated synchronously on the main thread
            # (_assert_profile_override_supported); here we only enter the override scope.
            profile = current_block_agent_profile()
            override_ctx = nullcontext()
            if isinstance(adapter, SupportsProfileOverride) and profile.needs_adapter_override(
                adapter.default_model()
            ):
                override_ctx = adapter.profile_override(profile)

            # Register the pre-resolved (scope-correct) tool specs inside the adapter's per-call
            # tool scope (ToolScope's ContextVar — isolated per worker by copy_context).
            # tool_adapter is typed Any: _resolve_tool_specs already verified SupportsToolRegistration.
            tool_adapter: Any = None
            tool_ctx: Any = nullcontext()
            if captured_tool_specs:
                tool_adapter = cast(Any, adapter)  # _resolve_tool_specs verified tool support
                tool_ctx = tool_adapter.tool_scope()
            with tool_ctx:
                if tool_adapter is not None:
                    for spec in captured_tool_specs:
                        tool_adapter.register_tool(spec)
                run_kwargs: dict[str, Any] = {}
                instructions = self._agent_instructions(profile)
                if instructions is not None:
                    run_kwargs["instructions"] = instructions

                with override_ctx:
                    result = adapter.produce_sync(
                        template=question,
                        output_schema=typed_fields,
                        **run_kwargs,
                    )

            out_map: Dict[str, Any] = {}
            for key in required:
                value = getattr(result, key, NotFound)  # Ensure result has the attribute
                if value is NotFound:
                    raise ValueError(f"LLM did not return required key: {key}")
                out_map[key] = self._validate_adapter_output(
                    value,
                    typed_fields[key],
                    explicit_type=key in explicit_typed_fields,
                )
            return out_map

        # Sequential default: run inline on the main thread and return concrete values. This is the
        # hot path, so the parallel-only env snapshot and job closure below are NOT built here.
        if not parallel:
            result_map = _produce()  # raises on the main thread, wrapped by the trace frame
            for key in required:
                env[key] = result_map[key]
            return {k: env[k] for k in required}

        # Parallel path. Snapshot the value-env by VALUE so the deferred worker sees the env as of
        # schedule time; copy_context() snapshots only the contextvar binding, leaving the dict
        # shared by reference, which would let later main-thread writes leak into a scheduled job.
        # Every per-call adapter scope (tool/profile ContextVars) is isolated per worker by the
        # copy_context() snapshot, so even tool-using templates pipeline safely.
        env_snapshot = dict(env)

        def _job() -> Dict[str, Any]:
            # The job runs on a worker AFTER the originating template's trace frame has exited on the
            # main thread; attribute failures to the captured frames (carried via copy_context) so
            # errors still point at the template line.
            token = self._current_env.set(env_snapshot)
            try:
                return _produce()
            except KediExecutionError:
                raise
            except Exception as exc:  # noqa: BLE001
                # build_execution_error keeps the provider error on `.original`, so the job-manager
                # retry policy can still classify it (a 429/timeout stays retryable) even though the
                # job surfaces the wrapped KediExecutionError. `from None` avoids re-chaining it.
                raise self.build_execution_error(exc) from None
            finally:
                self._current_env.reset(token)

        future = self._engine.schedule(deps, _job)
        self._register_outstanding(future)
        views: Dict[str, Any] = {key: KediPromise(future, key) for key in required}
        for key in required:
            env[key] = views[key]
        return dict(views)

    @staticmethod
    def _assert_profile_override_supported(profile: Any, adapter: Any) -> None:
        """Raise if the adapter can't honor a requested model/effort/mcp/settings override.

        Called on the main thread at the template call site so the error surfaces synchronously in
        both modes — symmetric with the tool guard in :meth:`_resolve_tool_specs`. ``_produce`` then
        only has to enter the override scope, never re-validate (which would defer the error to a
        worker in parallel mode).
        """
        from .agent_adapter.profile_override import SupportsProfileOverride
        from .agent_profile import AgentProfileOverrideUnsupported

        if isinstance(adapter, SupportsProfileOverride):
            return
        if (
            profile.model is not None
            or profile.effort is not None
            or profile.mcp_servers
            or profile.settings
        ):
            raise AgentProfileOverrideUnsupported(
                "Active adapter does not support agent profile override "
                f"(requested model={profile.model!r}, effort={profile.effort!r}, "
                f"mcp_servers={profile.mcp_servers!r}, settings={dict(profile.settings)!r})"
            )

    def _resolve_adapter(self, profile: Any) -> AgentAdapter[Any]:
        shortname = getattr(profile, "adapter_shortname", None)
        if shortname is None:
            if self.adapter is None:
                raise RuntimeError("No adapter provided to KediRuntime")
            return self.adapter

        adapter_shortname = (
            getattr(self.adapter, "shortname", None) if self.adapter is not None else None
        )
        if self.adapter is not None and shortname == adapter_shortname:
            return self.adapter

        model = getattr(profile, "model", None)
        cache_key = (shortname, model)
        selected = self._selected_adapters.get(cache_key)
        if selected is None:
            from .agent_adapter.registry import build_adapter

            selected = build_adapter(shortname, model)
            self._selected_adapters[cache_key] = selected
        return selected

    def _resolve_tool_specs(self, profile: Any, adapter: Any) -> tuple[Any, ...]:
        """Resolve a template's tools to fully-bound ToolSpecs (on the main thread, in scope).

        Returns () when the template has no tools. Raises if the adapter cannot register tools or
        a named tool is unknown — surfaced synchronously at the call site, not deferred.
        """
        tool_names = tuple(profile.tools)
        if not tool_names:
            return ()

        from .agent_adapter.tool_override import SupportsToolRegistration
        from .agent_adapter.tool_override import with_approval
        from .agent_adapter.utils import (
            kedi_procedure_tool_spec,
            python_callable_tool_spec,
        )
        from .agent_profile import AgentToolOverrideUnsupported
        from .agent_profile.tools import (
            procedure_def_for_tool,
            python_callable_for_tool,
        )

        if not isinstance(adapter, SupportsToolRegistration):
            raise AgentToolOverrideUnsupported(
                "Active adapter does not support tool registration "
                f"(requested tools={tool_names!r})"
            )

        specs: list[Any] = []
        for tool_name in tool_names:
            proc = procedure_def_for_tool(self, tool_name)
            if proc is not None:
                specs.append(kedi_procedure_tool_spec(self, proc))
                continue
            tool_fn = python_callable_for_tool(self, tool_name)
            if tool_fn is None:
                raise KediExecutionError(f"Unknown tool procedure: {tool_name!r}")
            specs.append(python_callable_tool_spec(tool_name, tool_fn))
        if self._adapter_uses_native_approvals(adapter):
            return tuple(specs)
        return tuple(
            with_approval(
                spec,
                policy=profile.approval,
                adapter_shortname=getattr(adapter, "shortname", None),
            )
            for spec in specs
        )

    def _resolve_agent_tool_specs(self, profile: Any, adapter: Any) -> tuple[Any, ...]:
        """Resolve explicit Kedi tools plus the opt-in project-local skill tools."""
        specs = self._resolve_tool_specs(profile, adapter)
        if not getattr(profile, "skills", False):
            return specs

        from .agent_adapter.tool_override import SupportsToolRegistration
        from .agent_profile import AgentToolOverrideUnsupported

        if not isinstance(adapter, SupportsToolRegistration):
            raise AgentToolOverrideUnsupported(
                "Active adapter does not support skill tool registration"
            )
        from .agent_adapter.skills import skill_tool_specs
        from .agent_adapter.tool_override import with_approval

        skill_specs = skill_tool_specs(self._skill_root())
        if not self._adapter_uses_native_approvals(adapter):
            skill_specs = tuple(
                with_approval(
                    spec,
                    policy=profile.approval,
                    adapter_shortname=getattr(adapter, "shortname", None),
                )
                for spec in skill_specs
            )
        return (*specs, *skill_specs)

    @staticmethod
    def _adapter_uses_native_approvals(adapter: Any) -> bool:
        from .agent_adapter.meta import AdapterCapabilities

        capabilities = getattr(adapter, "capabilities", None)
        return isinstance(capabilities, AdapterCapabilities) and capabilities.supports(
            "native_approvals"
        )

    def _skill_root(self) -> Path:
        """Resolve skills beside the executed Kedi source, else from the process CWD."""
        program_loc = getattr(self._program, "loc", None)
        source_path = getattr(program_loc, "source_path", None)
        if isinstance(source_path, str) and not source_path.startswith("<"):
            return Path(source_path).resolve().parent / ".agents" / "skills"
        return Path.cwd() / ".agents" / "skills"

    @staticmethod
    def _agent_instructions(profile: Any) -> str | None:
        """Append the skill contract without replacing user-provided system instructions."""
        instructions = getattr(profile, "system", None)
        if not getattr(profile, "skills", False):
            return instructions
        from .agent_adapter.skills import skill_instructions

        skill_prompt = skill_instructions()
        return skill_prompt if instructions is None else f"{instructions}\n\n{skill_prompt}"

    def _register_outstanding(self, future: Any) -> None:
        with self._outstanding_lock:
            self._outstanding.append(future)

    def drain(self) -> None:
        """Await every scheduled produce Future (parallel mode) at end of run.

        Guarantees that even unconsumed templates execute (their side effects) and that any
        failure surfaces. Fail-fast on the first error (matching today's behavior); additional
        failures are logged loudly, never silently swallowed. Loops until no new work appears
        (a drained template may have scheduled more).
        """
        import logging

        first_exc: BaseException | None = None
        extra: list[BaseException] = []
        while True:
            with self._outstanding_lock:
                pending = list(self._outstanding)
                self._outstanding.clear()
            if not pending:
                break
            for fut in pending:
                try:
                    exc = fut.exception()  # blocks until the Future completes
                except CancelledError as cancelled:
                    exc = cancelled  # a cancelled job surfaces as an error, not a hang
                if exc is not None:
                    if first_exc is None:
                        first_exc = exc
                    else:
                        extra.append(exc)
        for exc in extra:
            logging.getLogger("kedi").error(
                "Suppressed additional template failure during drain: %r", exc
            )
        if first_exc is not None:
            raise first_exc
