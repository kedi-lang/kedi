from __future__ import annotations

import ast
import builtins
import dataclasses
import inspect
import operator
import symtable
import typing
from functools import reduce
from typing import TYPE_CHECKING, Any, Callable, Mapping, Protocol, cast

import typing_extensions
from pydantic import BaseModel, Field, create_model

from ..promise import KediPromise, force
from .debug import ExecutorDebugEvent, ExecutorDebugExporter

if TYPE_CHECKING:
    from ..core import KediRuntime

_REFERENCE_KEY = "__kedi_pyodide_ref__"


class PyodideBridge(Protocol):
    """Synchronous transport to one browser-owned Pyodide worker."""

    def request(self, payload: Mapping[str, Any], *, timeout: float) -> Mapping[str, Any]: ...


@dataclasses.dataclass(frozen=True)
class PyodideReference:
    """Opaque reference to a Python object owned by the browser worker."""

    handle: str


class PyodideExecutor:
    """Execute user-authored Python in an isolated browser Pyodide worker."""

    def __init__(
        self,
        bridge: PyodideBridge,
        *,
        timeout: float = 60,
        debug_exporter: ExecutorDebugExporter | None = None,
    ) -> None:
        self._bridge = bridge
        self._timeout = timeout
        self._stdout: list[str] = []
        self.debug_exporter = debug_exporter

    def set_debug_exporter(self, exporter: ExecutorDebugExporter | None) -> None:
        self.debug_exporter = exporter

    def drain_stdout(self) -> str:
        output = "".join(self._stdout)
        self._stdout.clear()
        return output

    def evaluate_inline(
        self,
        rt: KediRuntime,
        code: str,
        local_env: dict[str, Any],
    ) -> Any:
        return self._execute(rt, "evaluate_inline", code, local_env)

    def execute_block(
        self,
        rt: KediRuntime,
        code: str,
        local_env: dict[str, Any],
    ) -> Any:
        return self._execute(rt, "execute_block", code, local_env)

    def execute_side_effects(
        self,
        rt: KediRuntime,
        code: str,
        env_map: dict[str, Any],
        *,
        kedi_line_offset: int = 0,
    ) -> None:
        self._execute(
            rt,
            "execute_side_effects",
            code,
            env_map,
            kedi_line_offset=kedi_line_offset,
        )

    def execute_prelude(self, rt: KediRuntime, code: str) -> dict[str, Any]:
        response = self._request(
            "execute_prelude",
            code,
            {},
            sync_names=_prelude_names(code),
        )
        return _response_env(response)

    def create_dynamic_function(
        self,
        name: str,
        params: list[str],
        body: Callable[..., Any],
        defaults: dict[str, Any] | None = None,
    ) -> Callable[..., Any]:
        defaults = defaults or {}
        signature = inspect.Signature(
            [
                inspect.Parameter(
                    param,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=defaults.get(param, inspect.Parameter.empty),
                )
                for param in params
            ]
        )

        def implementation(*args: Any, **kwargs: Any) -> Any:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            return body(**bound.arguments)

        implementation.__name__ = name
        setattr(implementation, "__signature__", signature)
        return implementation

    def evaluate_type_expression(
        self,
        rt: KediRuntime,
        code: str,
        env: dict[str, Any],
    ) -> Any:
        response = self._request("evaluate_type_expression", code, env)
        if "result" in response:
            return _decode_bridge_value(response["result"])
        descriptor = response.get("type")
        if not isinstance(descriptor, dict):
            raise TypeError("Pyodide expression did not return a value or type descriptor")
        return _decode_type_descriptor(descriptor)

    def _execute(
        self,
        rt: KediRuntime,
        action: str,
        code: str,
        target: dict[str, Any],
        *,
        kedi_line_offset: int = 0,
    ) -> Any:
        self._emit_debug("execution_started", action, code=code, local_env=target)
        try:
            from ..lang.compiler.environment import build_python_env

            execution_env = build_python_env(rt, target)
            response = self._request(
                action,
                code,
                execution_env,
                kedi_line_offset=kedi_line_offset,
                sync_names=list(target),
            )
            updates = _response_env(response)
            for name in tuple(target):
                if name in updates:
                    target[name] = updates[name]
            result = response.get("result")
        except Exception as exc:
            self._emit_debug(
                "execution_failed",
                action,
                code=code,
                local_env=target,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        self._emit_debug(
            "execution_completed",
            action,
            code=code,
            local_env=target,
            output=result,
        )
        return result

    def _request(
        self,
        action: str,
        code: str,
        env: dict[str, Any],
        *,
        kedi_line_offset: int = 0,
        sync_names: list[str] | None = None,
    ) -> Mapping[str, Any]:
        referenced = _referenced_names(code, action)
        payload = {
            "operation": "python",
            "action": action,
            "code": code,
            "env": _serialize_env(env, referenced),
            "syncNames": list(env) if sync_names is None else sync_names,
            "kediLineOffset": kedi_line_offset,
        }
        response = self._bridge.request(payload, timeout=self._timeout)
        if not isinstance(response, Mapping):
            raise TypeError("Pyodide bridge response must be an object")
        stdout = response.get("stdout", "")
        if not isinstance(stdout, str):
            raise TypeError("Pyodide bridge stdout must be a string")
        if stdout:
            self._stdout.append(stdout)
        if response.get("ok") is not True:
            error_type = response.get("errorType") or "PythonError"
            message = response.get("error") or "Pyodide execution failed"
            raise RuntimeError(f"{error_type}: {message}")
        return response

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


def _referenced_names(code: str, action: str) -> set[str]:
    tree = ast.parse(code, mode="eval" if action == "evaluate_type_expression" else "exec")
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }


def _prelude_names(code: str) -> list[str]:
    table = symtable.symtable(code, "<kedi-pyodide:prelude>", "exec")
    return [
        symbol.get_name()
        for symbol in table.get_symbols()
        if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()
    ]


def _serialize_env(env: Mapping[str, Any], referenced: set[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in env.items():
        if name == "__builtins__":
            continue
        if _is_worker_builtin(name, value):
            continue
        if isinstance(value, KediPromise):
            if name not in referenced:
                continue
            value = force(value)
        try:
            result[name] = _json_value(value)
        except TypeError:
            if name in referenced:
                raise TypeError(f"Python value {name!r} cannot cross the Pyodide bridge") from None
    return result


def _is_worker_builtin(name: str, value: Any) -> bool:
    if any(value is module for module in (builtins, typing, typing_extensions)):
        return name in {"builtins", "typing", "typing_extensions"}
    for module in (builtins, typing, typing_extensions):
        if getattr(module, name, object()) is value:
            return True
    return False


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, PyodideReference):
        return {_REFERENCE_KEY: value.handle}
    if isinstance(value, BaseModel):
        return _json_value(value.model_dump(mode="json"))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_value(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Pyodide bridge mappings require string keys")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(item) for item in value]
    raise TypeError(f"Unsupported Pyodide bridge value: {type(value).__name__}")


def _response_env(response: Mapping[str, Any]) -> dict[str, Any]:
    value = response.get("env", {})
    if not isinstance(value, dict) or not all(isinstance(name, str) for name in value):
        raise TypeError("Pyodide response env must be an object")
    return {name: _decode_bridge_value(item) for name, item in value.items()}


def _decode_bridge_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode_bridge_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {_REFERENCE_KEY}:
        handle = value[_REFERENCE_KEY]
        if not isinstance(handle, str):
            raise TypeError("Pyodide reference handle must be a string")
        return PyodideReference(handle)
    return {name: _decode_bridge_value(item) for name, item in value.items()}


def _decode_type_descriptor(descriptor: Mapping[str, Any]) -> Any:
    kind = descriptor.get("kind")
    if kind == "none":
        return type(None)
    if kind == "builtin":
        name = descriptor.get("name")
        value = getattr(builtins, name, None) if isinstance(name, str) else None
        if not isinstance(value, type):
            raise TypeError(f"Unknown Pyodide builtin type: {name!r}")
        return value
    if kind == "typing":
        name = descriptor.get("name")
        value = getattr(typing, name, None) if isinstance(name, str) else None
        if value is None:
            raise TypeError(f"Unknown Pyodide typing type: {name!r}")
        return value
    if kind == "generic":
        origin = _decode_type_descriptor(_descriptor_mapping(descriptor, "origin"))
        args = tuple(
            _decode_type_descriptor(_require_mapping(item, "generic argument"))
            for item in _descriptor_list(descriptor, "args")
        )
        return origin[args[0]] if len(args) == 1 else origin[args]
    if kind == "union":
        args = [
            _decode_type_descriptor(_require_mapping(item, "union argument"))
            for item in _descriptor_list(descriptor, "args")
        ]
        if not args:
            raise TypeError("Pyodide union type requires at least one argument")
        return reduce(operator.or_, args)
    if kind == "literal":
        values = tuple(_descriptor_list(descriptor, "values"))
        return cast(Any, typing.Literal)[values]
    if kind == "pydantic_model":
        return _decode_pydantic_model(descriptor)
    raise TypeError(f"Unsupported Pyodide type descriptor: {kind!r}")


def _decode_pydantic_model(descriptor: Mapping[str, Any]) -> type[BaseModel]:
    name = descriptor.get("name")
    if not isinstance(name, str) or not name:
        raise TypeError("Pyodide Pydantic model name must be a non-empty string")
    fields: dict[str, tuple[Any, Any]] = {}
    for item in _descriptor_list(descriptor, "fields"):
        field = _require_mapping(item, "Pydantic model field")
        field_name = field.get("name")
        if not isinstance(field_name, str) or not field_name:
            raise TypeError("Pyodide Pydantic field name must be a non-empty string")
        annotation = _decode_type_descriptor(_descriptor_mapping(field, "annotation"))
        required = field.get("required")
        if not isinstance(required, bool):
            raise TypeError("Pyodide Pydantic field required flag must be a boolean")
        field_kwargs: dict[str, Any] = {}
        description = field.get("description")
        if isinstance(description, str) and description:
            field_kwargs["description"] = description
        examples = field.get("examples")
        if isinstance(examples, list):
            field_kwargs["examples"] = examples
        default = ... if required else field.get("default")
        fields[field_name] = (annotation, Field(default, **field_kwargs))
    factory = cast(Callable[..., type[BaseModel]], create_model)
    return factory(name, **fields)


def _descriptor_mapping(descriptor: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return _require_mapping(descriptor.get(name), name)


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Pyodide {name} must be an object")
    return value


def _descriptor_list(descriptor: Mapping[str, Any], name: str) -> list[Any]:
    value = descriptor.get(name)
    if not isinstance(value, list):
        raise TypeError(f"Pyodide type descriptor {name} must be a list")
    return value


__all__ = ["PyodideBridge", "PyodideExecutor", "PyodideReference"]
