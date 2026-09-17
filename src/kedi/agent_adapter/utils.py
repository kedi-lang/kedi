from __future__ import annotations

import asyncio
import contextvars
import copy
import inspect
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Coroutine,
    Iterator,
    Mapping,
    TypeVar,
    cast,
    get_type_hints,
)

from pydantic import BaseModel, TypeAdapter
from pydantic_ai.tools import Tool

from ..core import CallExpr, InputExpr, KediRuntime
from ..lang.ast import ProcedureDef, Program, ProgramWithValidation
from ..lang.compiler.type_system import KediType, coerce_model_value
from ..lang.docstrings import parse_procedure_docstring
from ..lang.procedure_schema import (
    build_procedure_description,
    procedure_json_schema,
    procedure_return_json_schema,
    program_procedure_json_schemas,
)
from ..module_registry import internal_module_path
from ..tool_context import kedi_tool_call_context
from .approval import ToolRisk
from .invocation import InvocationUsage, report_invocation
from .tool_override import ToolSpec

_DSPY_SYNC_TOOL_MARKER = "__kedi_dspy_sync_tool__"
_T = TypeVar("_T")
_RUN_DEADLINE: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "kedi_run_deadline", default=None
)


class RunDeadlineExceeded(TimeoutError):
    """The enclosing execution deadline expired after cancelling async work."""


@contextmanager
def use_run_deadline(deadline_monotonic: float | None) -> Iterator[None]:
    """Apply an absolute deadline to adapter calls, including copied worker contexts."""
    parent = _RUN_DEADLINE.get()
    deadline = (
        min(parent, deadline_monotonic)
        if parent is not None and deadline_monotonic is not None
        else parent
        if deadline_monotonic is None
        else deadline_monotonic
    )
    token = _RUN_DEADLINE.set(deadline)
    try:
        yield
    finally:
        _RUN_DEADLINE.reset(token)


_FILESYSTEM_READ_ONLY_PROCEDURES = frozenset(
    {"read_text_file", "path_exists", "list_directory", "get_file_info"}
)


def run_sync(awaitable: Awaitable[_T]) -> _T:
    """Run an awaitable to completion from sync code, even inside a running loop.

    When no event loop is running this is just :func:`asyncio.run`. When called
    from within a running loop (``asyncio.run`` would raise there) the coroutine
    is executed on its own loop in a dedicated worker thread.
    """

    async def resolve() -> _T:
        deadline = _RUN_DEADLINE.get()
        if deadline is None:
            return await awaitable
        task = asyncio.ensure_future(awaitable)
        try:
            return await asyncio.wait_for(
                task,
                timeout=max(0.0, deadline - time.monotonic()),
            )
        except asyncio.TimeoutError as exc:
            if task.cancelled():
                raise RunDeadlineExceeded("adapter execution deadline elapsed") from exc
            raise

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _run_awaitable(awaitable, resolve)
    # Copy the caller's context so a dspy.context(lm=...) override reaches the worker.
    ctx = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=1) as executor:
        # cast: Context.run's TypeVar doesn't unify with run_sync's via submit().
        return cast(
            "_T", executor.submit(lambda: ctx.run(_run_awaitable, awaitable, resolve)).result()
        )


def _run_awaitable(
    awaitable: Awaitable[_T],
    resolve: Callable[[], Coroutine[Any, Any, _T]],
) -> _T:
    try:
        return asyncio.run(resolve())
    except RuntimeError as exc:
        if "different loop" not in str(exc) and "attached to a different loop" not in str(exc):
            raise
        raise RuntimeError(
            "run_sync cannot execute an awaitable bound to another event loop; "
            "await it from its owner loop instead"
        ) from exc


def kedi_procedure_tool_spec(
    rt: KediRuntime,
    proc: ProcedureDef,
    *,
    local_env: dict[str, Any] | None = None,
) -> ToolSpec:
    """Build an adapter-neutral tool spec from a Kedi procedure."""
    parsed_docstring = parse_procedure_docstring(proc.docstring)
    json_schema = procedure_json_schema(rt, proc, local_env=local_env)
    return_schema = procedure_return_json_schema(rt, proc, local_env=local_env)
    tool_description = _build_tool_description(
        rt,
        proc,
        parsed_docstring.description,
        local_env=local_env,
    )
    tool_fn = _make_tool_function(rt, proc)
    return ToolSpec(
        fn=tool_fn,
        name=proc.name,
        description=tool_description,
        json_schema=json_schema,
        return_schema=return_schema,
        metadata=_tool_metadata(
            proc,
            json_schema,
            return_schema,
            description=tool_description,
        ),
        proc=proc,
        risk=_procedure_tool_risk(proc),
        risk_resolver=_procedure_tool_risk_resolver(proc),
        argument_validator=_procedure_tool_argument_validator(rt, proc, tool_fn),
    )


def python_callable_tool_spec(name: str, fn: Callable[..., Any]) -> ToolSpec:
    """Build an adapter-neutral tool spec from a Python callable."""
    from kedi.artifacts.stream import callable_with_logical_stream_return

    tool_fn, logical_return = callable_with_logical_stream_return(fn)
    description = getattr(fn, "__kedi_tool_description__", None)
    if not isinstance(description, str):
        description = _callable_description(fn)
    risk = getattr(fn, "__kedi_tool_risk__", "mutating")
    if risk not in ("read_only", "mutating", "sensitive"):
        raise ValueError(f"invalid Kedi tool risk {risk!r} for {name!r}")
    metadata: dict[str, Any] = {}
    kind = getattr(fn, "__kedi_tool_kind__", None)
    if isinstance(kind, str):
        metadata["kedi_tool_kind"] = kind
    return ToolSpec(
        fn=tool_fn,
        name=name,
        description=description,
        return_schema=(
            TypeAdapter(logical_return).json_schema() if logical_return is not None else None
        ),
        metadata=metadata,
        risk=risk,
        argument_validator=_python_callable_argument_validator(tool_fn),
    )


def _procedure_tool_risk(proc: ProcedureDef) -> ToolRisk:
    """Classify bundled filesystem reads; all other Kedi procedures are mutating by default."""

    if _is_builtin_filesystem_procedure(proc) and proc.name in _FILESYSTEM_READ_ONLY_PROCEDURES:
        return "read_only"
    return "mutating"


def _procedure_tool_risk_resolver(
    proc: ProcedureDef,
) -> Callable[[Mapping[str, Any]], ToolRisk] | None:
    """Return an argument-aware risk classifier for bundled filesystem reads."""

    if not _is_builtin_filesystem_procedure(proc):
        return None
    if proc.name not in {"read_text_file", "write_text_file", "apply_patch", "remove_file"}:
        return None

    if proc.name == "read_text_file":

        def filesystem_read_risk(arguments: Mapping[str, Any]) -> ToolRisk:
            if arguments.get("secret_files") is True and _targets_secret_file(arguments):
                return "sensitive"
            return "read_only"

        return filesystem_read_risk

    def filesystem_write_risk(arguments: Mapping[str, Any]) -> ToolRisk:
        return "sensitive" if _targets_secret_file(arguments) else "mutating"

    return filesystem_write_risk


def _is_builtin_filesystem_procedure(proc: ProcedureDef) -> bool:
    source_path = proc.loc.source_path if proc.loc is not None else None
    bundled_path = internal_module_path("filesystem")
    if source_path is None or bundled_path is None:
        return False
    try:
        return Path(source_path).resolve() == bundled_path.resolve()
    except OSError:
        return False


def _targets_secret_file(arguments: Mapping[str, Any]) -> bool:
    raw_path = arguments.get("file_path")
    return isinstance(raw_path, str) and _is_secret_file_name(Path(raw_path).name)


def _is_secret_file_name(name: str) -> bool:
    return name == ".env" or name.startswith(".env.")


def kedi_procedure_to_pydantic_tool(
    rt: KediRuntime,
    proc: ProcedureDef,
    *,
    local_env: dict[str, Any] | None = None,
    sequential: bool = False,
    args_validator: Any = None,
) -> Tool:
    """Build a ``pydantic_ai.Tool`` from a Kedi procedure."""
    from dataclasses import replace

    from .adapters.pydantic import pydantic_tool_from_spec

    spec = kedi_procedure_tool_spec(rt, proc, local_env=local_env)
    return pydantic_tool_from_spec(
        replace(spec, sequential=sequential),
        args_validator=args_validator,
    )


def kedi_program_to_pydantic_tools(
    rt: KediRuntime,
    prog: Program | ProgramWithValidation,
    *,
    include_nested: bool = False,
    local_env: dict[str, Any] | None = None,
    sequential: bool = False,
    args_validator: Any = None,
) -> list[Tool]:
    """Build ``pydantic_ai.Tool`` objects for procedures in a parsed program."""
    from .adapters.pydantic import pydantic_tool_from_spec

    schemas = program_procedure_json_schemas(
        rt,
        prog,
        include_nested=include_nested,
        local_env=local_env,
    )
    procedure_map = {
        proc.name: proc for proc in _all_procedures(prog.procedures, include_nested=include_nested)
    }
    tools: list[Tool] = []
    for name, json_schema in schemas.items():
        proc = procedure_map[name]
        parsed_docstring = parse_procedure_docstring(proc.docstring)
        return_schema = procedure_return_json_schema(rt, proc, local_env=local_env)
        tool_description = _build_tool_description(
            rt,
            proc,
            parsed_docstring.description,
            local_env=local_env,
        )
        tool_fn = _make_tool_function(rt, proc)
        spec = ToolSpec(
            fn=tool_fn,
            name=proc.name,
            description=tool_description,
            json_schema=json_schema,
            return_schema=return_schema,
            metadata=_tool_metadata(
                proc,
                json_schema,
                return_schema,
                description=tool_description,
            ),
            proc=proc,
            risk=_procedure_tool_risk(proc),
            risk_resolver=_procedure_tool_risk_resolver(proc),
            argument_validator=_procedure_tool_argument_validator(
                rt,
                proc,
                tool_fn,
            ),
            sequential=sequential,
        )
        tools.append(
            pydantic_tool_from_spec(
                spec,
                args_validator=args_validator,
            )
        )
    return tools


def dspy_sync_tool_function(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Adapt a sync or async callable to DSPy's sync tool-call path."""
    if getattr(fn, _DSPY_SYNC_TOOL_MARKER, False):
        return fn

    target = fn if inspect.isfunction(fn) or inspect.ismethod(fn) else fn.__call__

    @wraps(target)
    def _sync_tool(*args: Any, **kwargs: Any) -> Any:
        report_invocation(InvocationUsage(tool_calls=1))
        result = target(*args, **kwargs)
        if inspect.isawaitable(result):
            return run_sync(cast(Awaitable[Any], result))
        return result

    setattr(_sync_tool, _DSPY_SYNC_TOOL_MARKER, True)
    return _sync_tool


def dspy_tool_args(json_schema: dict[str, Any]) -> dict[str, Any]:
    defs = json_schema.get("$defs", {})
    return {
        name: _resolve_dspy_arg_schema(schema, defs)
        for name, schema in json_schema.get("properties", {}).items()
    }


def dspy_tool_arg_desc(json_schema: dict[str, Any]) -> dict[str, str]:
    properties = json_schema.get("properties", {})
    return {
        name: description
        for name, schema in properties.items()
        if isinstance(schema, dict) and isinstance((description := schema.get("description")), str)
    }


def _make_tool_function(rt: KediRuntime, proc: ProcedureDef) -> Callable[..., Any]:
    proc_fn = _get_or_compile_proc_function(rt, proc)

    @wraps(proc_fn)
    def _tool_function(**kwargs: Any) -> Any:
        with kedi_tool_call_context():
            return _invoke_procedure_with_kwargs(rt, proc, proc_fn, kwargs)

    _tool_function.__name__ = proc.name
    _tool_function.__doc__ = getattr(proc_fn, "__doc__", None)
    return _tool_function


def _resolve_dspy_arg_schema(
    schema: dict[str, Any],
    defs: dict[str, Any],
) -> dict[str, Any]:
    resolved = copy.deepcopy(schema)
    ref = resolved.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        resolved = copy.deepcopy(defs[ref.removeprefix("#/$defs/")])
    return _strip_json_schema_titles(resolved)


def _strip_json_schema_titles(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_json_schema_titles(item) for key, item in value.items() if key != "title"
        }
    if isinstance(value, list):
        return [_strip_json_schema_titles(item) for item in value]
    return value


def _callable_description(fn: Callable[..., Any]) -> str | None:
    if not (inspect.isfunction(fn) or inspect.ismethod(fn)) and hasattr(fn, "__call__"):
        return inspect.getdoc(fn.__call__) or inspect.getdoc(fn)
    return inspect.getdoc(fn)


def _get_or_compile_proc_function(rt: KediRuntime, proc: ProcedureDef) -> Callable[..., Any]:
    try:
        return rt._get_proc(proc.name)
    except KeyError:
        from ..lang.compiler.procedure_compiler import ProcedureCompiler

        return ProcedureCompiler(rt)._make_procedure_impl(proc, proc.lexical_env)


def _invoke_procedure_with_kwargs(
    rt: KediRuntime,
    proc: ProcedureDef,
    fn: Any,
    kwargs: dict[str, Any],
) -> Any:
    call_kwargs = _validated_procedure_kwargs(rt, proc, fn, kwargs)
    token = rt._current_env.set(dict(call_kwargs))
    try:
        result = fn(**call_kwargs)
    finally:
        rt._current_env.reset(token)

    return (
        rt._evaluate_value(result, dict(call_kwargs))
        if isinstance(result, (InputExpr, CallExpr))
        else result
    )


def _procedure_tool_argument_validator(
    rt: KediRuntime,
    proc: ProcedureDef,
    tool_fn: Callable[..., Any],
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    proc_fn = getattr(tool_fn, "__wrapped__", tool_fn)

    def validate(arguments: Mapping[str, Any]) -> dict[str, Any]:
        return _validated_procedure_kwargs(rt, proc, proc_fn, dict(arguments))

    return validate


def _validated_procedure_kwargs(
    rt: KediRuntime,
    proc: ProcedureDef,
    fn: Callable[..., Any],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    sig = inspect.signature(fn)
    bound = sig.bind(**kwargs)
    bound.apply_defaults()
    local_env = dict(bound.arguments)
    _coerce_tool_arguments(rt, proc, local_env)
    return {name: local_env[name] for name in bound.arguments}


def _python_callable_argument_validator(
    fn: Callable[..., Any],
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    signature = inspect.signature(fn)
    try:
        type_hints = get_type_hints(fn)
    except (NameError, TypeError):
        type_hints = {}

    def validate(arguments: Mapping[str, Any]) -> dict[str, Any]:
        bound = signature.bind(**arguments)
        bound.apply_defaults()
        validated: dict[str, Any] = {}
        for name, parameter in signature.parameters.items():
            if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
                continue
            value = bound.arguments.get(name)
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                validated.update(value or {})
                continue
            annotation = type_hints.get(name, parameter.annotation)
            if isinstance(annotation, str):
                annotation = inspect.Parameter.empty
            if annotation is not inspect.Parameter.empty:
                value = TypeAdapter(annotation).validate_python(value)
            validated[name] = value
        return validated

    return validate


def _coerce_tool_arguments(
    rt: KediRuntime,
    proc: ProcedureDef,
    kwargs: dict[str, Any],
) -> None:
    if not (proc.param_type_annotations or proc.param_type_exprs):
        return
    param_exprs = proc.param_type_exprs or {}
    param_strs = proc.param_type_annotations or {}

    for param_name in proc.params:
        if param_name not in kwargs:
            raise TypeError(f"Missing parameter: {param_name}")

        kedi_type = KediType.from_node(
            param_exprs.get(param_name),
            param_strs.get(param_name),
        )
        if kedi_type.is_none():
            continue

        provided = kwargs[param_name]
        resolved = kedi_type.resolve(rt, None)
        try:
            if isinstance(resolved, type) and issubclass(resolved, BaseModel):
                kwargs[param_name] = coerce_model_value(resolved, provided)
            else:
                kwargs[param_name] = kedi_type.validate(rt, None, provided)
        except Exception as exc:
            raise TypeError(f"Parameter '{param_name}' does not match annotated type") from exc


def _tool_metadata(
    proc: ProcedureDef,
    json_schema: dict[str, Any],
    return_schema: dict[str, Any],
    *,
    description: str | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "kedi_schema": {
            "name": proc.name,
            "parameters": json_schema,
            "returns": return_schema,
            **({"description": description} if description else {}),
        },
        "kedi_input_schema": json_schema,
        "kedi_return_schema": return_schema,
    }
    filesystem_path = internal_module_path("filesystem")
    source_path = proc.loc.source_path if proc.loc is not None else None
    if (
        filesystem_path is not None
        and source_path is not None
        and Path(source_path).resolve() == filesystem_path
    ):
        metadata["kedi_tool_kind"] = "filesystem"
    return metadata


def _build_tool_description(
    rt: KediRuntime,
    proc: ProcedureDef,
    base_description: str | None,
    *,
    local_env: dict[str, Any] | None,
) -> str:
    return build_procedure_description(rt, proc, base_description, local_env=local_env)


def _all_procedures(procedures: list[ProcedureDef], *, include_nested: bool) -> list[ProcedureDef]:
    out: list[ProcedureDef] = []

    def visit(proc: ProcedureDef) -> None:
        out.append(proc)
        if not include_nested:
            return
        for stmt in proc.body:
            nested = getattr(stmt, "proc", None)
            if nested is not None:
                visit(nested)

    for proc in procedures:
        visit(proc)
    return out
