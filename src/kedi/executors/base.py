from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

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
        self._emit_debug(
            "execution_started",
            "evaluate_inline",
            code=code,
            local_env=local_env,
        )
        env = self._build_python_env(rt, local_env)
        eval_filename = self._python_filename(rt, "inline", code)
        try:
            result = eval(compile(code, eval_filename, "eval", dont_inherit=True), env, env)
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
            exec_filename = self._python_filename(rt, "inline", code)
            try:
                exec(compile(code, exec_filename, "exec", dont_inherit=True), env, env)
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
        self._emit_debug(
            "execution_started",
            "execute_block",
            code=code,
            local_env=local_env,
        )
        env = self._build_python_env(rt, local_env)
        params = ", ".join(local_env.keys())
        wrapper_src = f"def __kedi_block__({params}):\n" + "\n".join(
            "    " + line for line in code.splitlines()
        )
        filename = self._python_filename(
            rt,
            "block",
            code,
            wrapper_line_offset=1,
            kedi_line_offset=1,
        )
        try:
            exec(compile(wrapper_src, filename, "exec", dont_inherit=True), env, env)
            result = env["__kedi_block__"](**local_env)
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
        try:
            exec(compile(code, filename, "exec", dont_inherit=True), env, env)
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
        self._emit_debug("execution_started", "execute_prelude", code=code)
        env = self._build_python_env(rt, {})
        filename = self._python_filename(rt, "prelude", code, kedi_line_offset=1)
        try:
            exec(compile(code, filename, "exec", dont_inherit=True), env, env)
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
        self._emit_debug(
            "execution_started",
            "evaluate_type_expression",
            code=code,
        )
        try:
            filename = self._python_filename(rt, "type_expression", code)
            result = eval(compile(code, filename, "eval", dont_inherit=True), env, env)
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
            wrapper_line_offset=wrapper_line_offset,
            kedi_line_offset=kedi_line_offset,
            procedure_name=procedure_name,
        )

    def _build_python_env(self, rt: KediRuntime, local_env: dict[str, Any]) -> dict[str, Any]:
        from ..lang.compiler.environment import build_python_env

        return build_python_env(rt, local_env)

    def _sync_existing_variables(
        self,
        rt: KediRuntime,
        target: dict[str, Any],
        source: dict[str, Any],
    ) -> None:
        runtime_env = rt._current_env.get()
        for var_name in list(target.keys()):
            if var_name in source:
                target[var_name] = source[var_name]
                if runtime_env is not None:
                    runtime_env[var_name] = source[var_name]

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
