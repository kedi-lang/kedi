from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

from ..telemetry import TelemetrySpan, capture_enabled, runtime_span
from ..telemetry.naming import qualified_type_name
from .debug import ExecutorDebugEvent, ExecutorDebugExporter

if TYPE_CHECKING:
    from ..core import KediRuntime


@runtime_checkable
class Executor(Protocol):
    """Execution backend for runtime-owned Python evaluation."""

    def set_debug_exporter(self, exporter: ExecutorDebugExporter | None) -> None: ...

    def evaluate_inline(self, rt: KediRuntime, code: str, local_env: dict[str, Any]) -> Any: ...

    def execute_block(self, rt: KediRuntime, code: str, local_env: dict[str, Any]) -> Any: ...

    def execute_side_effects(
        self,
        rt: KediRuntime,
        code: str,
        env_map: dict[str, Any],
        *,
        kedi_line_offset: int = 0,
    ) -> None: ...

    def execute_prelude(self, rt: KediRuntime, code: str) -> dict[str, Any]: ...

    def create_dynamic_function(
        self,
        name: str,
        params: list[str],
        body: Callable[..., Any],
        defaults: dict[str, Any] | None = None,
    ) -> Callable[..., Any]: ...

    def evaluate_type_expression(self, rt: KediRuntime, code: str, env: dict[str, Any]) -> Any: ...


class DefaultExecutor:
    """Default unsandboxed executor backed by Python ``exec`` / ``eval``."""

    def __init__(
        self,
        *,
        debug_exporter: ExecutorDebugExporter | None = None,
    ) -> None:
        self.debug_exporter = debug_exporter

    def set_debug_exporter(self, exporter: ExecutorDebugExporter | None) -> None:
        self.debug_exporter = exporter

    def evaluate_inline(self, rt: KediRuntime, code: str, local_env: dict[str, Any]) -> Any:
        with self._python_execution_span(rt, "inline", code) as active:
            result = self._evaluate_inline(rt, code, local_env)
            active.set_attribute("kedi.result.type", qualified_type_name(result))
            return result

    def _evaluate_inline(self, rt: KediRuntime, code: str, local_env: dict[str, Any]) -> Any:
        self._emit_debug(
            "execution_started",
            "evaluate_inline",
            code=code,
            local_env=local_env,
        )
        env = self._build_python_env(rt, local_env)
        eval_filename = self._python_filename(rt, "inline", code)
        eval_source = self._python_source(rt, code)
        try:
            result = eval(compile(eval_source, eval_filename, "eval", dont_inherit=True), env, env)
            self._emit_debug(
                "execution_completed",
                "evaluate_inline",
                code=code,
                local_env=local_env,
                output=result,
                mode="eval",
            )
            return result
        except SyntaxError:
            try:
                self._run_wrapped_block(rt, "inline", code, local_env)
                self._emit_debug(
                    "execution_completed",
                    "evaluate_inline",
                    code=code,
                    local_env=local_env,
                    output="",
                    mode="exec_after_syntax_error",
                )
                return ""
            except Exception as exc:
                self._emit_debug(
                    "execution_failed",
                    "evaluate_inline",
                    code=code,
                    local_env=local_env,
                    error_type=type(exc).__name__,
                    error=str(exc),
                    mode="exec_after_syntax_error",
                )
                raise
        except Exception as exc:
            self._emit_debug(
                "execution_failed",
                "evaluate_inline",
                code=code,
                local_env=local_env,
                error_type=type(exc).__name__,
                error=str(exc),
                mode="eval",
            )
            raise

    def execute_block(self, rt: KediRuntime, code: str, local_env: dict[str, Any]) -> Any:
        with self._python_execution_span(rt, "block", code) as active:
            result = self._execute_block(rt, code, local_env)
            active.set_attribute("kedi.result.type", qualified_type_name(result))
            return result

    def _execute_block(self, rt: KediRuntime, code: str, local_env: dict[str, Any]) -> Any:
        self._emit_debug(
            "execution_started",
            "execute_block",
            code=code,
            local_env=local_env,
        )
        try:
            result = self._run_wrapped_block(
                rt,
                "block",
                code,
                local_env,
                kedi_line_offset=1,
            )
            self._emit_debug(
                "execution_completed",
                "execute_block",
                code=code,
                local_env=local_env,
                output=result,
            )
            return result
        except Exception as exc:
            self._emit_debug(
                "execution_failed",
                "execute_block",
                code=code,
                local_env=local_env,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise

    def execute_side_effects(
        self,
        rt: KediRuntime,
        code: str,
        env_map: dict[str, Any],
        *,
        kedi_line_offset: int = 0,
    ) -> None:
        with self._python_execution_span(rt, "side_effects", code):
            self._execute_side_effects(
                rt,
                code,
                env_map,
                kedi_line_offset=kedi_line_offset,
            )

    def _execute_side_effects(
        self,
        rt: KediRuntime,
        code: str,
        env_map: dict[str, Any],
        *,
        kedi_line_offset: int = 0,
    ) -> None:
        self._emit_debug(
            "execution_started",
            "execute_side_effects",
            code=code,
            env_map=env_map,
        )
        env = self._build_python_env(rt, env_map)
        filename = self._python_filename(
            rt,
            "side_effects",
            code,
            kedi_line_offset=kedi_line_offset,
        )
        compile_source = self._python_source(rt, code, kedi_line_offset=kedi_line_offset)
        try:
            exec(compile(compile_source, filename, "exec", dont_inherit=True), env, env)
        except Exception as exc:
            self._emit_debug(
                "execution_failed",
                "execute_side_effects",
                code=code,
                env_map=env_map,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        self._sync_existing_variables(rt, env_map, env)
        self._emit_debug(
            "execution_completed",
            "execute_side_effects",
            code=code,
            env_map=env_map,
            output={name: env_map[name] for name in env_map},
        )

    def execute_prelude(self, rt: KediRuntime, code: str) -> dict[str, Any]:
        with self._python_execution_span(rt, "prelude", code) as active:
            result = self._execute_prelude(rt, code)
            active.set_attribute("kedi.result.type", qualified_type_name(result))
            return result

    def _execute_prelude(self, rt: KediRuntime, code: str) -> dict[str, Any]:
        self._emit_debug("execution_started", "execute_prelude", code=code)
        env = self._build_python_env(rt, {})
        filename = self._python_filename(rt, "prelude", code, kedi_line_offset=1)
        compile_source = self._python_source(rt, code, kedi_line_offset=1)
        try:
            exec(compile(compile_source, filename, "exec", dont_inherit=True), env, env)
        except Exception as exc:
            self._emit_debug(
                "execution_failed",
                "execute_prelude",
                code=code,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        self._emit_debug(
            "execution_completed",
            "execute_prelude",
            code=code,
            output=sorted(name for name in env if not name.startswith("__")),
        )
        return env

    def create_dynamic_function(
        self,
        name: str,
        params: list[str],
        body: Callable[..., Any],
        defaults: dict[str, Any] | None = None,
    ) -> Callable[..., Any]:
        self._emit_debug(
            "execution_started",
            "create_dynamic_function",
            function_name=name,
            args=params,
        )
        defaults = defaults or {}
        namespace: dict[str, Any] = {"_body": body}
        param_specs: list[str] = []
        for index, param in enumerate(params):
            if param in defaults:
                default_name = f"__kedi_default_{index}"
                namespace[default_name] = defaults[param]
                param_specs.append(f"{param}={default_name}")
            else:
                param_specs.append(param)
        param_list = ", ".join(param_specs)
        args_map = ", ".join(f"{param}={param}" for param in params)
        src = f"def _impl({param_list}):\n    return _body({args_map})\n"
        exec(src, namespace, namespace)
        fn = namespace["_impl"]
        fn.__name__ = name
        self._emit_debug(
            "execution_completed",
            "create_dynamic_function",
            function_name=name,
            args=params,
            output=name,
        )
        return fn

    def evaluate_type_expression(self, rt: KediRuntime, code: str, env: dict[str, Any]) -> Any:
        with self._python_execution_span(rt, "type_expression", code) as active:
            result = self._evaluate_type_expression(rt, code, env)
            active.set_attribute("kedi.result.type", qualified_type_name(result))
            return result

    def _evaluate_type_expression(
        self,
        rt: KediRuntime,
        code: str,
        env: dict[str, Any],
    ) -> Any:
        self._emit_debug(
            "execution_started",
            "evaluate_type_expression",
            code=code,
        )
        try:
            filename = self._python_filename(rt, "type_expression", code)
            compile_source = self._python_source(rt, code)
            result = eval(compile(compile_source, filename, "eval", dont_inherit=True), env, env)
        except Exception as exc:
            self._emit_debug(
                "execution_failed",
                "evaluate_type_expression",
                code=code,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        self._emit_debug(
            "execution_completed",
            "evaluate_type_expression",
            code=code,
            output=result,
        )
        return result

    @contextmanager
    def _python_execution_span(
        self,
        rt: KediRuntime,
        kind: str,
        code: str,
    ) -> Iterator[TelemetrySpan]:
        block_kind = {"block": "return", "side_effects": "statement"}.get(kind, kind)
        executor_name = type(self).__name__
        attributes: dict[str, bool | str | int] = {
            "kedi.python.block.kind": block_kind,
            "kedi.executor.name": executor_name,
            "kedi.python.code.lines": code.count("\n") + (1 if code else 0),
        }
        frames = rt.current_trace_frames()
        if frames:
            frame = frames[-1]
            if frame.procedure_name is not None:
                attributes["kedi.procedure.name"] = frame.procedure_name
            if frame.span is not None:
                attributes["code.line.number"] = frame.span.start_line
                if frame.span.source_path is not None and capture_enabled("source_path"):
                    attributes["code.file.path"] = frame.span.source_path
        with runtime_span(
            "run python",
            "run_python",
            level="detailed",
            attributes=attributes,
            metric_attributes={
                "executor": executor_name,
                "block_kind": block_kind,
            },
        ) as active:
            try:
                yield active
            except BaseException as exc:
                active.set_attribute("error.type", type(exc).__name__)
                raise

    def _python_filename(
        self,
        rt: KediRuntime,
        kind: str,
        code: str,
        *,
        wrapper_line_offset: int = 0,
        kedi_line_offset: int = 0,
    ) -> str:
        frames = rt.current_trace_frames()
        span = frames[-1].span if frames else None
        procedure_name = None
        for frame in reversed(frames):
            if frame.procedure_name:
                procedure_name = frame.procedure_name
                break
        return rt.register_python_origin(
            kind=kind,
            code=code,
            span=span,
            compile_line_offset=self._python_line_offset(rt, kedi_line_offset),
            wrapper_line_offset=wrapper_line_offset,
            kedi_line_offset=kedi_line_offset,
            procedure_name=procedure_name,
        )

    def _python_source(
        self,
        rt: KediRuntime,
        code: str,
        *,
        kedi_line_offset: int = 0,
    ) -> str:
        return "\n" * self._python_line_offset(rt, kedi_line_offset) + code

    def _python_line_offset(self, rt: KediRuntime, kedi_line_offset: int) -> int:
        frames = rt.current_trace_frames()
        span = frames[-1].span if frames else None
        if span is None:
            return 0
        if span.source_path is not None:
            return max(0, span.start_line + kedi_line_offset - 1)
        source_map = rt._source_map
        if source_map is None or not hasattr(source_map, "resolve_line"):
            return 0
        resolved = source_map.resolve_line(span.start_line)
        if resolved is None:
            return 0
        return max(0, resolved.source_line + kedi_line_offset - 1)

    def _build_python_env(self, rt: KediRuntime, local_env: dict[str, Any]) -> dict[str, Any]:
        from ..lang.compiler.environment import build_python_env

        return build_python_env(rt, local_env)

    def _run_wrapped_block(
        self,
        rt: KediRuntime,
        kind: str,
        code: str,
        local_env: dict[str, Any],
        *,
        kedi_line_offset: int = 0,
    ) -> Any:
        """Run ``code`` as a block. Existing Kedi vars are exposed as ``global`` so reads route
        through ``KediEnv.__getitem__`` (lazy promise resolution) and mutations sync back; new
        Python names stay local and don't leak.

        Note: exposing Kedi vars as module globals (rather than the prior ``nonlocal`` + params
        wrapper) is a deliberate, cross-cutting scoping choice that applies in BOTH sequential and
        parallel mode — a nested ``def``/``lambda`` rebinding a Kedi var must use ``global x`` (not
        ``nonlocal x``), and reflection must read the bare name / ``globals()`` (not ``locals()``).
        Documented for users in docs/KEDI_DSL.md ("Kedi variables are Python globals"). Recommended
        CI hardening: a Python 3.11/3.12 matrix to guard the interpreter routing this depends on.
        """
        env = self._build_python_env(rt, local_env)
        names = list(local_env.keys())
        # A function wrapper is needed so ``return`` is legal; declaring the Kedi vars ``global``
        # makes their reads LOAD_GLOBAL (-> KediEnv.__getitem__) and writes STORE_GLOBAL (-> env).
        header = ["def __kedi_wrapper__():"]
        if names:  # empty ``global`` is a SyntaxError
            header.append(f"    global {', '.join(names)}")
        body = ["    " + line for line in code.splitlines()]
        wrapper_src = "\n".join(
            header + body + ["    pass"]  # guarantees a non-empty body
        )
        filename = self._python_filename(
            rt,
            kind,
            code,
            wrapper_line_offset=len(header),
            kedi_line_offset=kedi_line_offset,
        )
        # Pad so traceback line numbers stay aligned with the Kedi source line.
        compile_source = self._python_source(rt, wrapper_src, kedi_line_offset=kedi_line_offset)
        exec(compile(compile_source, filename, "exec", dont_inherit=True), env, env)
        result = env["__kedi_wrapper__"]()
        # Sync mutated Kedi vars back, reading raw (no KediEnv resolution) so untouched promises
        # stay lazy; touched ones were cached concrete by KediEnv on access.
        updated = {name: dict.__getitem__(env, name) for name in names if name in env}
        self._sync_existing_variables(rt, local_env, updated)
        return result

    def _sync_existing_variables(
        self,
        rt: KediRuntime,
        target: dict[str, Any],
        source: dict[str, Any],
    ) -> None:
        runtime_env = rt._current_env.get()
        for var_name in list(target.keys()):
            if var_name in source:
                # Raw read: never trigger KediEnv promise resolution during sync-back, so an
                # untouched promise stays lazy.
                value = dict.__getitem__(source, var_name)
                target[var_name] = value
                if runtime_env is not None:
                    runtime_env[var_name] = value

    def _emit_debug(self, event: str, step: str, **payload: Any) -> None:
        if self.debug_exporter is None:
            return
        self.debug_exporter.export(
            ExecutorDebugEvent.create(
                event,
                executor=type(self).__name__,
                step=step,
                **payload,
            )
        )
