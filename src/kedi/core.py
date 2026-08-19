from __future__ import annotations

import atexit
import os
from concurrent.futures import CancelledError
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock, get_ident
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
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
from .telemetry import capture_enabled
from .telemetry.naming import qualified_type_name

if TYPE_CHECKING:
    from .agent_adapter.conversation import ConversationState
    from .agent_adapter.subagents.budget import DelegationBudget, DelegationConcurrency
    from .agent_adapter.subagents.limits import SubagentUsageLimits
    from .agent_adapter.tool_override import ToolSpec
    from .agent_profile import AgentProfile

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
    from kedi.agent_surface import AgentSurface
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


def _tool_request_context(tools: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.json_schema,
            "returns": tool.return_schema,
        }
        for tool in tools
    ]


class KediRuntime:
    """Isolated kedi runtime with its own registry and environment."""

    def __init__(
        self,
        *,
        adapter: AgentAdapter[Any] | None = None,
        executor: Executor | None = None,
        engine: Any | None = None,
        default_agent_profile: Any | None = None,
        subagent_max_depth: int = 5,
        subagent_max_concurrency: int = 4,
        subagent_timeout_seconds: float | None = 120.0,
        subagent_usage_limits: "SubagentUsageLimits | None" = None,
        subagent_state_path: str | Path | None = None,
        conversation_state: "ConversationState | None" = None,
    ) -> None:
        from .agent_profile import AgentProfile
        from .artifacts import ArtifactManager

        self.adapter = adapter
        self.executor = executor or DefaultExecutor()
        self._engine = engine if engine is not None else _default_engine()
        self._selected_adapters: dict[tuple[str, str | None], AgentAdapter[Any]] = {}
        # Outstanding produce Futures are owned by one outer invocation. Concurrent runs on the
        # same runtime must never drain or surface one another's work.
        self._outstanding: dict[object, list[Any]] = {}
        self._outstanding_lock = Lock()
        self._origin_counter_lock = Lock()
        self._conversation_state_lock = Lock()
        self._lifecycle_lock = Lock()
        self._closed = False
        self._procedures: Dict[str, Callable[..., Any]] = {}
        self._main: Optional[Callable[..., Any]] = None
        self._import_env: ValueEnv = {}
        self._current_env: ContextVar[Optional[ValueEnv]] = ContextVar(
            "kedi_current_env", default=None
        )
        self._trace_stack: ContextVar[Tuple[KediTraceFrame, ...]] = ContextVar(
            "kedi_trace_stack", default=()
        )
        self._subagent_stack: ContextVar[tuple[tuple[str, str], ...]] = ContextVar(
            "kedi_subagent_stack", default=()
        )
        # Tracks procedure-call nesting so the end-of-run drain fires exactly once, at the
        # outermost invocation, regardless of entry point (run_main / query / direct invoke).
        self._invoke_depth: ContextVar[int] = ContextVar("kedi_invoke_depth", default=0)
        self._invocation_id: ContextVar[object | None] = ContextVar(
            "kedi_invocation_id",
            default=None,
        )
        from .agent_adapter.request_budget import ModelRequestBudget

        self._request_budget: ContextVar[ModelRequestBudget | None] = ContextVar(
            "kedi_model_request_budget",
            default=None,
        )
        self._initial_globals_env: ValueEnv = {}
        self._globals_env: ValueEnv = {}
        self._python_origin_counter = 0
        self._python_origins: Dict[str, KediPythonOrigin] = {}
        self._python_origin_keys: Dict[tuple[Any, ...], str] = {}
        self._program: Any | None = None
        self._source_map: Any | None = None
        self._source_text: str | None = None
        self._type_env_stack: list[dict[str, Any]] = []
        self._profile_registry: dict[str, Any] = {}
        self.default_agent_profile = (
            default_agent_profile
            if isinstance(default_agent_profile, AgentProfile)
            else AgentProfile.empty()
        )
        self._prelude_env: ValueEnv | None = None
        self._validation_tests: list[TestSuite] = []
        self._validation_evals: list[EvalSuite] = []
        self.conversation_state = conversation_state
        self._implicit_conversation_state: ConversationState | None = None
        self.artifacts = (
            conversation_state.artifacts if conversation_state is not None else ArtifactManager()
        )
        from .agent_adapter.subagents.coordinator import (
            SubagentCoordinator,
            SubagentCoordinatorDependencies,
        )

        self._subagent_coordinator = SubagentCoordinator(
            SubagentCoordinatorDependencies(
                profile_registry=lambda: self._profile_registry,
                evaluate_profile=lambda definition: self._evaluate_profile_definition(definition),
                resolve_adapter=lambda profile: self._resolve_adapter(profile),
                assert_profile_override_supported=lambda profile, adapter: (
                    self._assert_profile_override_supported(profile, adapter)
                ),
                resolve_tools=lambda profile, adapter: self._resolve_agent_tool_specs(
                    profile, adapter
                ),
                instructions_for_profile=lambda profile, adapter: self._agent_instructions(
                    profile, adapter
                ),
                trace_frame=lambda frame: self.trace_frame(frame),
                active_stack=self._subagent_stack,
                default_cwd=Path.cwd,
            ),
            max_depth=subagent_max_depth,
            max_concurrency=subagent_max_concurrency,
            timeout_seconds=subagent_timeout_seconds,
            usage_limits=subagent_usage_limits,
            state_path=subagent_state_path,
        )

    # Expressions
    def i(self, key: str) -> InputExpr:
        return i(key)

    def o(self, key: str, pytype: Optional[Any] = None) -> OutputExpr:
        return o(key, pytype)

    def c(self, func_name: str, *args: Any) -> CallExpr:
        return c(func_name, *args)

    def bind_agent(
        self,
        *,
        adapter: AgentAdapter[Any] | None = None,
        profile: AgentProfile | None = None,
        close_conversation_state: bool = False,
    ) -> AgentSurface:
        """Assemble Kedi runtime services for a native embedded-agent lifecycle."""

        from .agent_surface import AgentSurface

        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("KediRuntime is closed")
        selected_profile = (profile or self.default_agent_profile).frozen_copy()
        selected_adapter = adapter or self._resolve_adapter(selected_profile)
        self._assert_profile_override_supported(selected_profile, selected_adapter)
        tool_specs = cast(
            "tuple[ToolSpec, ...]",
            self._resolve_agent_tool_specs(selected_profile, selected_adapter),
        )
        instructions = self._agent_instructions(selected_profile, selected_adapter)
        effective_profile = replace(selected_profile, system=instructions)
        conversation_state = self._conversation_state_for_profile(selected_profile)
        return AgentSurface(
            agent=selected_adapter,
            profile=effective_profile,
            tool_specs=tool_specs,
            conversation_state=conversation_state,
            close_conversation_state=close_conversation_state,
        )

    async def aclose(self) -> None:
        """Cancel owned child runs and close runtime-owned session resources."""

        with self._lifecycle_lock:
            self._closed = True
        await self._subagent_coordinator.aclose()
        if self._implicit_conversation_state is not None:
            self._implicit_conversation_state.close()
        if self.conversation_state is None:
            self.artifacts.close()

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
        from .lang.compiler.environment import native_value

        return native_value(force(result))

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
        from .lang.compiler.environment import raw_lookup_value

        try:
            return raw_lookup_value(self, key, env)
        except KeyError:
            pass
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
                from .lang.compiler.environment import model_text

                return value.map(model_text), None
            from .lang.compiler.environment import model_text

            return model_text(value), None
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
            from .lang.compiler.environment import native_value

            return native_value(force(self._lookup_value(value.key, env)))
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

        from .telemetry import runtime_span

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
                from .agent_adapter.request_budget import ModelRequestBudget

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

        origin_span = span
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
        span_key = (
            None
            if origin_span is None
            else (
                origin_span.start_byte,
                origin_span.end_byte,
                origin_span.start_line,
                origin_span.start_col,
                origin_span.end_line,
                origin_span.end_col,
                origin_span.source_path,
                origin_span.source_id,
            )
        )
        origin_key = (
            kind,
            code,
            span_key,
            compile_line_offset,
            wrapper_line_offset,
            kedi_line_offset,
            procedure_name,
            traceback_code,
        )
        with self._origin_counter_lock:
            cached = self._python_origin_keys.get(origin_key)
            if cached is not None:
                return cached

            self._python_origin_counter += 1
            filename = f"<kedi-python:{kind}:{self._python_origin_counter}>"
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
            self._python_origin_keys[origin_key] = filename
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
            from .agent_adapter.tool_override import SupportsToolRegistration
            from .agent_profile import current_block_agent_profile

            profile = current_block_agent_profile()
            override_ctx = nullcontext()
            if isinstance(adapter, SupportsProfileOverride):
                override_ctx = adapter.profile_override(profile)

            tool_adapter: Any = None
            tool_ctx: Any = nullcontext()
            if isinstance(adapter, SupportsToolRegistration):
                tool_adapter = adapter
                tool_ctx = tool_adapter.tool_scope()

            with tool_ctx:
                if tool_adapter is not None:
                    for spec in captured_tool_specs:
                        tool_adapter.register_tool(spec)
                run_kwargs: dict[str, Any] = {}
                instructions = self._agent_instructions(profile, adapter)
                if instructions is not None:
                    run_kwargs["instructions"] = instructions
                with override_ctx:
                    return self._run_conversation_call(
                        adapter,
                        prompt,
                        lambda active_prompt: str(
                            adapter.invoke_sync(prompt=active_prompt, **run_kwargs)
                        ),
                        context=(
                            instructions,
                            _tool_request_context(captured_tool_specs),
                        ),
                        instructions=instructions,
                        tools=captured_tool_specs,
                        profile=profile,
                        call_kind="invoke",
                        output_field_count=0,
                    )

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
            from .agent_adapter.tool_override import SupportsToolRegistration
            from .agent_profile import current_block_agent_profile

            # Support was already validated synchronously on the main thread
            # (_assert_profile_override_supported); here we only enter the override scope.
            profile = current_block_agent_profile()
            override_ctx = nullcontext()
            if isinstance(adapter, SupportsProfileOverride):
                override_ctx = adapter.profile_override(profile)

            # Every tool-capable model call gets its own scope, including calls with no Kedi
            # tools. This prevents a nested model call from inheriting its caller's registrations.
            tool_adapter: Any = None
            tool_ctx: Any = nullcontext()
            if isinstance(adapter, SupportsToolRegistration):
                tool_adapter = adapter
                tool_ctx = tool_adapter.tool_scope()

            def produce_and_validate(active_question: str) -> Dict[str, Any]:
                result = adapter.produce_sync(
                    template=active_question,
                    output_schema=typed_fields,
                    **run_kwargs,
                )
                out_map: Dict[str, Any] = {}
                artifact_policy = profile.artifacts
                for key in required:
                    value = getattr(result, key, NotFound)
                    if value is NotFound:
                        raise ValueError(f"LLM did not return required key: {key}")
                    validated = self._validate_adapter_output(
                        value,
                        typed_fields[key],
                        explicit_type=key in explicit_typed_fields,
                    )
                    out_map[key] = (
                        self.artifacts.artifactize(
                            validated,
                            policy=artifact_policy,
                            source=f"output:{key}",
                        )
                        if artifact_policy is not None
                        else validated
                    )
                return out_map

            with tool_ctx:
                if tool_adapter is not None:
                    for spec in captured_tool_specs:
                        tool_adapter.register_tool(spec)
                run_kwargs: dict[str, Any] = {}
                instructions = self._agent_instructions(profile, adapter)
                if instructions is not None:
                    run_kwargs["instructions"] = instructions

                with override_ctx:
                    return self._run_conversation_call(
                        adapter,
                        question,
                        produce_and_validate,
                        context=(
                            instructions,
                            _tool_request_context(captured_tool_specs),
                            {key: repr(value) for key, value in typed_fields.items()},
                        ),
                        instructions=instructions,
                        tools=captured_tool_specs,
                        profile=profile,
                        call_kind="produce",
                        output_field_count=len(required),
                        output_schema={key: repr(value) for key, value in typed_fields.items()},
                    )

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

    def _run_conversation_call(
        self,
        adapter: Any,
        prompt: str,
        call: Callable[[str], Any],
        *,
        context: Sequence[Any] = (),
        instructions: str | None,
        tools: Sequence[ToolSpec] = (),
        profile: AgentProfile,
        call_kind: Literal["produce", "invoke"],
        output_field_count: int,
        output_schema: Mapping[str, str] | None = None,
    ) -> Any:
        """Run one successful adapter turn through explicit continuation state."""
        from .agent_adapter.invocation import observe_invocations
        from .agent_adapter.request_budget import ModelRequestBudget
        from .agent_adapter.telemetry import (
            agent_call_span,
            model_call_span,
            record_conversation_assets,
            resolved_adapter_model,
        )
        from .agent_adapter.tool_override import (
            required_tool_call_scope,
            validate_required_tool_calls,
        )

        budget = self._request_budget.get() or ModelRequestBudget()
        configured_model = profile.model
        frames = self.current_trace_frames()
        frame = frames[-1] if frames else None
        source_span = None if frame is None else frame.span
        source_path = (
            source_span.source_path
            if source_span is not None
            and source_span.source_path is not None
            and capture_enabled("source_path")
            else None
        )
        source_line = None if source_span is None else source_span.start_line
        source_snippet = (
            frame.source_snippet
            if frame is not None and capture_enabled("source_snippet")
            else None
        )
        record_conversation_assets(
            prompt=prompt,
            instructions=instructions,
            profile=profile,
            tools=tools,
            output_schema=output_schema,
            call_kind=call_kind,
            source_path=source_path,
            source_line=source_line,
            source_snippet=source_snippet,
        )

        with agent_call_span(
            adapter,
            profile_name=profile.name,
            call_kind=call_kind,
            prompt_chars=len(prompt),
            output_field_count=output_field_count,
            tool_count=len(tools),
            mcp_server_count=len(profile.mcp_servers),
            artifacts_enabled=bool(profile.artifacts and profile.artifacts.enabled),
            configured_model=configured_model,
        ) as agent_telemetry:

            def run_reserved_call(reservation: Any, active_prompt: str) -> Any:
                with model_call_span(
                    adapter,
                    call_kind=call_kind,
                    configured_model=configured_model,
                ) as model_telemetry:

                    def observe(usage: Any) -> None:
                        reservation.observe(usage)
                        agent_telemetry.observe(usage)
                        model_telemetry.observe(usage)

                    with observe_invocations(observe):
                        result = call(active_prompt)
                    model = resolved_adapter_model(adapter, configured_model)
                    model_telemetry.complete(result, model=model)
                    agent_telemetry.complete(result, model=model)
                    return result

            with required_tool_call_scope(tools):
                state = self._conversation_state_for_profile(profile)
                if state is None:
                    reservation = budget.reserve(prompt, context=context)
                    try:
                        result = run_reserved_call(reservation, prompt)
                        validate_required_tool_calls()
                    except BaseException:
                        reservation.finish(raise_on_exceeded=False)
                        raise
                    reservation.finish()
                    return result

                from .agent_adapter.conversation import SupportsConversationState
                from .lang.compiler.environment import model_value

                capabilities = adapter.capabilities
                adapter_shortname = adapter.shortname
                if profile.history is True and not (
                    capabilities.supports("stateful_history")
                    and isinstance(adapter, SupportsConversationState)
                ):
                    raise RuntimeError(
                        f"Adapter {adapter_shortname!r} does not support stateful history"
                    )
                with state.adapter_turn(adapter_shortname) as conversation:
                    reservation = budget.reserve(
                        prompt,
                        context=context,
                        resume_state=conversation.resume_state,
                    )
                    conversation_scope = (
                        adapter.conversation_scope(conversation)
                        if capabilities.supports("stateful_history")
                        and isinstance(adapter, SupportsConversationState)
                        else nullcontext()
                    )
                    try:
                        with conversation_scope:
                            result = run_reserved_call(reservation, prompt)
                        validate_required_tool_calls()
                    except BaseException:
                        reservation.finish(raise_on_exceeded=False)
                        raise
                    reservation.finish()
                    state.record_exchange(prompt, model_value(result))
                    return result

    def _conversation_state_for_profile(
        self,
        profile: AgentProfile,
    ) -> ConversationState | None:
        if profile.history is False:
            return None
        if self.conversation_state is not None:
            return self.conversation_state
        if profile.history is not True:
            return None
        with self._conversation_state_lock:
            if self._implicit_conversation_state is None:
                from .agent_adapter.conversation import ConversationState

                self._implicit_conversation_state = ConversationState(artifacts=self.artifacts)
            return self._implicit_conversation_state

    @staticmethod
    def _assert_profile_override_supported(profile: Any, adapter: Any) -> None:
        """Raise if the adapter can't honor a requested model/effort/mcp/settings override.

        Called on the main thread at the template call site so the error surfaces synchronously in
        both modes — symmetric with the tool guard in :meth:`_resolve_tool_specs`. ``_produce`` then
        only has to enter the override scope, never re-validate (which would defer the error to a
        worker in parallel mode).
        """
        from .agent_adapter.meta import AdapterCapabilities
        from .agent_adapter.profile_override import SupportsProfileOverride
        from .agent_profile import AgentProfileOverrideUnsupported
        from .agent_profile.compaction import CompactionSettingsError

        compaction = profile.compaction
        if compaction is not None and compaction.mode == "native" and profile.history is not True:
            raise CompactionSettingsError("native compaction requires history to be enabled")
        if profile.history is True and compaction is not None and compaction.mode == "native":
            capabilities = getattr(adapter, "capabilities", None)
            if not isinstance(capabilities, AdapterCapabilities) or not capabilities.supports(
                "native_compaction"
            ):
                shortname = getattr(adapter, "shortname", type(adapter).__name__)
                raise CompactionSettingsError(
                    f"native compaction is not supported by adapter {shortname!r}"
                )

        codemode = profile.codemode
        if codemode is not None and codemode.enabled:
            capabilities = getattr(adapter, "capabilities", None)
            if not isinstance(capabilities, AdapterCapabilities) or not capabilities.supports(
                "codemode"
            ):
                shortname = getattr(adapter, "shortname", type(adapter).__name__)
                raise AgentProfileOverrideUnsupported(
                    f"CodeMode is not supported by adapter {shortname!r}"
                )

        if isinstance(adapter, SupportsProfileOverride):
            return
        if (
            profile.model is not None
            or profile.effort is not None
            or profile.mcp_servers
            or profile.settings
            or profile.compaction is not None
            or profile.codemode is not None
        ):
            raise AgentProfileOverrideUnsupported(
                "Active adapter does not support agent profile override "
                f"(requested model={profile.model!r}, effort={profile.effort!r}, "
                f"mcp_servers={profile.mcp_servers!r}, settings={dict(profile.settings)!r}, "
                f"compaction={profile.compaction!r}, codemode={profile.codemode!r})"
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

        from .agent_adapter.tool_override import SupportsToolRegistration, with_approval
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
        bound_tools = getattr(profile, "tool_bindings", {})
        for tool_name in tool_names:
            bound_spec = bound_tools.get(tool_name)
            if bound_spec is not None:
                specs.append(bound_spec)
                continue
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
        from .agent_adapter.telemetry import adapter_shortname

        adapter_name = adapter_shortname(adapter)
        specs = list(self._resolve_tool_specs(profile, adapter))

        skills = self._skills_settings(profile)
        if skills is not None and skills.enabled:
            from .agent_adapter.tool_override import SupportsToolRegistration
            from .agent_profile import AgentToolOverrideUnsupported

            if not isinstance(adapter, SupportsToolRegistration):
                raise AgentToolOverrideUnsupported(
                    "Active adapter does not support skill tool registration"
                )
            from .agent_adapter.skills import resolve_skill_store, skill_tool_specs
            from .agent_adapter.tool_override import with_approval

            skill_specs = skill_tool_specs(
                store=resolve_skill_store(skills, base_dir=self._skill_base_dir())
            )
            if not self._adapter_uses_native_approvals(adapter):
                skill_specs = tuple(
                    with_approval(
                        spec,
                        policy=profile.approval,
                        adapter_shortname=adapter_name,
                    )
                    for spec in skill_specs
                )
            specs.extend(skill_specs)

        if profile.subagents:
            from .agent_adapter.meta import AdapterCapabilities
            from .agent_adapter.subagents import subagent_tool_specs
            from .agent_adapter.subagents.workflow import dynamic_workflow_tool_spec
            from .agent_profile import AgentSubagentUnsupported

            capabilities = getattr(adapter, "capabilities", None)
            if not isinstance(capabilities, AdapterCapabilities) or not capabilities.supports(
                "subagents"
            ):
                raise AgentSubagentUnsupported(
                    "Active adapter does not support Kedi subagent delegation"
                )
            if profile.workflow_mode == "dynamic":
                specs.append(dynamic_workflow_tool_spec(self, profile))
            else:
                specs.extend(
                    subagent_tool_specs(
                        self,
                        profile,
                        supports_background=capabilities.supports("background_subagents"),
                    )
                )
        artifact_policy = getattr(profile, "artifacts", None)
        artifact_contract_needed = self._artifact_contract_needed(
            profile,
            adapter,
            has_tools=bool(specs),
        )
        if artifact_contract_needed:
            from .agent_adapter.meta import AdapterCapabilities
            from .agent_adapter.tool_override import (
                SupportsToolRegistration,
                with_approval,
                with_artifact_result,
            )
            from .artifacts import ArtifactCapabilityError, artifact_tool_specs

            capabilities = getattr(adapter, "capabilities", None)
            if (
                not isinstance(adapter, SupportsToolRegistration)
                or not isinstance(capabilities, AdapterCapabilities)
                or not capabilities.supports("artifacts")
            ):
                raise ArtifactCapabilityError(
                    "Active adapter does not support Kedi artifact references and tools"
                )
            management_specs = artifact_tool_specs(self.artifacts, artifact_policy)
            if not self._adapter_uses_native_approvals(adapter):
                management_specs = tuple(
                    with_approval(
                        spec,
                        policy=profile.approval,
                        adapter_shortname=adapter_name,
                    )
                    for spec in management_specs
                )
            specs.extend(management_specs)
        if specs and artifact_policy is not None:
            from .agent_adapter.tool_override import with_artifact_result

            specs = [
                with_artifact_result(
                    spec,
                    manager=self.artifacts,
                    policy=artifact_policy,
                )
                for spec in specs
            ]
        from .agent_adapter.tool_override import (
            with_required_before_output,
            with_tool_telemetry,
        )

        specs = [with_required_before_output(spec) for spec in specs]
        specs = [with_tool_telemetry(spec, adapter_shortname=adapter_name) for spec in specs]
        return tuple(specs)

    async def _delegate_task(
        self,
        *,
        allowed_subagents: tuple[str, ...],
        subagent: str,
        task: str,
        final_schema: dict[str, Any] | None,
        parent_profile: Any,
        parent_budget: "DelegationBudget",
        parent_concurrency: "DelegationConcurrency",
        subagent_definitions: Mapping[str, Any] | None = None,
    ) -> Any:
        return await self._subagent_coordinator.delegate(
            allowed_subagents=allowed_subagents,
            subagent=subagent,
            task=task,
            final_schema=final_schema,
            parent_profile=parent_profile,
            parent_budget=parent_budget,
            parent_concurrency=parent_concurrency,
            subagent_definitions=subagent_definitions,
        )

    def _start_subagent(
        self,
        *,
        allowed_subagents: tuple[str, ...],
        subagent: str,
        task: str,
        final_schema: dict[str, Any] | None,
        parent_profile: Any,
        parent_budget: "DelegationBudget",
        parent_concurrency: "DelegationConcurrency",
        subagent_definitions: Mapping[str, Any] | None = None,
    ) -> Any:
        return self._subagent_coordinator.start(
            allowed_subagents=allowed_subagents,
            subagent=subagent,
            task=task,
            final_schema=final_schema,
            parent_profile=parent_profile,
            parent_budget=parent_budget,
            parent_concurrency=parent_concurrency,
            subagent_definitions=subagent_definitions,
        )

    async def _continue_subagent(
        self,
        *,
        run_id: str,
        allowed_subagents: tuple[str, ...],
        task: str,
        final_schema: dict[str, Any] | None,
        parent_profile: Any,
        parent_budget: "DelegationBudget",
        parent_concurrency: "DelegationConcurrency",
        subagent_definitions: Mapping[str, Any] | None = None,
    ) -> Any:
        return await self._subagent_coordinator.continue_run(
            run_id=run_id,
            allowed_subagents=allowed_subagents,
            task=task,
            final_schema=final_schema,
            parent_profile=parent_profile,
            parent_budget=parent_budget,
            parent_concurrency=parent_concurrency,
            subagent_definitions=subagent_definitions,
        )

    def _start_subagent_continuation(
        self,
        *,
        run_id: str,
        allowed_subagents: tuple[str, ...],
        task: str,
        final_schema: dict[str, Any] | None,
        parent_profile: Any,
        parent_budget: "DelegationBudget",
        parent_concurrency: "DelegationConcurrency",
        subagent_definitions: Mapping[str, Any] | None = None,
    ) -> Any:
        return self._subagent_coordinator.start_continuation(
            run_id=run_id,
            allowed_subagents=allowed_subagents,
            task=task,
            final_schema=final_schema,
            parent_profile=parent_profile,
            parent_budget=parent_budget,
            parent_concurrency=parent_concurrency,
            subagent_definitions=subagent_definitions,
        )

    async def _wait_subagent(
        self,
        run_id: str,
        *,
        timeout: float | None = None,
    ) -> Any:
        return await self._subagent_coordinator.wait(run_id, timeout=timeout)

    def _cancel_subagent(self, run_id: str) -> bool:
        return self._subagent_coordinator.cancel(run_id)

    def _subagent_status(self, run_id: str) -> Any:
        return self._subagent_coordinator.status(run_id)

    def _new_subagent_concurrency(self) -> "DelegationConcurrency":
        return self._subagent_coordinator.new_concurrency()

    def _evaluate_profile_definition(self, definition: Any) -> Any:
        from .agent_profile.compile import build_capture_env
        from .agent_profile.model import eval_profile_definition

        return eval_profile_definition(self, definition, build_capture_env(self))

    @staticmethod
    def _adapter_uses_native_approvals(adapter: Any) -> bool:
        from .agent_adapter.meta import AdapterCapabilities

        capabilities = getattr(adapter, "capabilities", None)
        return isinstance(capabilities, AdapterCapabilities) and capabilities.supports(
            "native_approvals"
        )

    def _skill_base_dir(self) -> Path:
        """Resolve the project base beside the executed Kedi source."""
        program_loc = getattr(self._program, "loc", None)
        source_path = getattr(program_loc, "source_path", None)
        if isinstance(source_path, str) and not source_path.startswith("<"):
            return Path(source_path).resolve().parent
        return Path.cwd()

    @staticmethod
    def _skills_settings(profile: Any) -> Any:
        from .agent_profile.skills import skills_settings

        return skills_settings(getattr(profile, "skills", None))

    def _agent_instructions(self, profile: Any, adapter: Any) -> str | None:
        """Append enabled runtime contracts without replacing user instructions."""

        parts = [getattr(profile, "system", None)]
        skills = self._skills_settings(profile)
        if skills is not None and skills.enabled:
            from .agent_adapter.skills import skill_instructions

            parts.append(skill_instructions())
        if self._artifact_contract_needed(
            profile,
            adapter,
            has_tools=bool(
                getattr(profile, "tools", ())
                or (skills is not None and skills.enabled)
                or getattr(profile, "subagents", ())
            ),
        ):
            from .artifacts import artifact_instructions

            parts.append(artifact_instructions())
        return "\n\n".join(part for part in parts if part) or None

    def _artifact_contract_needed(
        self,
        profile: Any,
        adapter: Any,
        *,
        has_tools: bool,
    ) -> bool:
        """Choose a cache-stable artifact contract from static call capabilities."""

        policy = getattr(profile, "artifacts", None)
        if policy is None or not policy.resolved().enabled:
            return False
        from .agent_adapter.meta import AdapterCapabilities

        capabilities = getattr(adapter, "capabilities", None)
        if isinstance(capabilities, AdapterCapabilities) and capabilities.supports("artifacts"):
            return True
        return policy.is_explicit or has_tools

    def _register_outstanding(self, future: Any) -> None:
        owner = self._outstanding_owner()
        with self._outstanding_lock:
            self._outstanding.setdefault(owner, []).append(future)

    def _outstanding_owner(self) -> object:
        invocation_id = self._invocation_id.get()
        if invocation_id is not None:
            return invocation_id
        return ("unscoped", id(self), get_ident())

    def drain(self) -> None:
        """Await every scheduled produce Future (parallel mode) at end of run.

        Guarantees that even unconsumed templates execute (their side effects) and that any
        failure surfaces. Fail-fast on the first error (matching today's behavior); additional
        failures are logged loudly, never silently swallowed. Loops until no new work appears
        (a drained template may have scheduled more).
        """
        import logging

        owner = self._outstanding_owner()
        first_exc: BaseException | None = None
        extra: list[BaseException] = []
        while True:
            with self._outstanding_lock:
                pending = self._outstanding.pop(owner, [])
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
