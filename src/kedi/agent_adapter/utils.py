from __future__ import annotations

import asyncio
import contextvars
import copy
import inspect
import importlib
import keyword
import linecache
import re
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, wraps
from importlib import metadata as importlib_metadata
from typing import Any, Callable, Coroutine, Sequence, TypeVar, cast

import dspy
from pydantic import BaseModel
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
from .monty_interpreter import MontyInterpreter
from .tool_override import ToolSpec

_DSPY_SYNC_TOOL_MARKER = "__kedi_dspy_sync_tool__"
_INVALID_IDENT_CHARS = re.compile(r"\W+")

_T = TypeVar("_T")


def run_sync(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run ``coro`` to completion from sync code, even inside a running loop.

    When no event loop is running this is just :func:`asyncio.run`. When called
    from within a running loop (``asyncio.run`` would raise there) the coroutine
    is executed on its own loop in a dedicated worker thread.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Copy the caller's context so a dspy.context(lm=...) override reaches the worker.
    ctx = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=1) as executor:
        # cast: Context.run's TypeVar doesn't unify with run_sync's via submit().
        return cast("_T", executor.submit(lambda: ctx.run(asyncio.run, coro)).result())


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
    return ToolSpec(
        fn=_make_tool_function(rt, proc),
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
    )


def python_callable_tool_spec(name: str, fn: Callable[..., Any]) -> ToolSpec:
    """Build an adapter-neutral tool spec from a Python callable."""
    description = getattr(fn, "__kedi_tool_description__", None)
    if not isinstance(description, str):
        description = _callable_description(fn)
    return ToolSpec(fn=fn, name=name, description=description)


def kedi_procedure_to_pydantic_tool(
    rt: KediRuntime,
    proc: ProcedureDef,
    *,
    local_env: dict[str, Any] | None = None,
    sequential: bool = False,
    args_validator: Any = None,
) -> Tool:
    """Build a ``pydantic_ai.Tool`` from a Kedi procedure."""
    spec = kedi_procedure_tool_spec(rt, proc, local_env=local_env)
    assert spec.json_schema is not None
    tool = Tool.from_schema(
        function=spec.fn,
        name=spec.name,
        description=spec.description or "",
        json_schema=spec.json_schema,
        takes_ctx=False,
        sequential=sequential,
        args_validator=args_validator,
    )
    tool.metadata = spec.metadata
    return tool


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
        tool = Tool.from_schema(
            function=_make_tool_function(rt, proc),
            name=proc.name,
            description=tool_description,
            json_schema=json_schema,
            takes_ctx=False,
            sequential=sequential,
            args_validator=args_validator,
        )
        tool.metadata = _tool_metadata(
            proc,
            json_schema,
            return_schema,
            description=tool_description,
        )
        tools.append(tool)
    return tools


def dspy_sync_tool_function(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Adapt a sync or async callable to DSPy's sync tool-call path."""
    if getattr(fn, _DSPY_SYNC_TOOL_MARKER, False):
        return fn

    target = fn if inspect.isfunction(fn) or inspect.ismethod(fn) else fn.__call__

    @wraps(target)
    def _sync_tool(*args: Any, **kwargs: Any) -> Any:
        result = target(*args, **kwargs)
        if inspect.isawaitable(result):
            return run_sync(cast(Coroutine[Any, Any, Any], result))
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


def dspy_code_act(sig: type[dspy.Signature], tools: Sequence[Any]) -> Any:
    bridge_tools, host_tools, skip_sources = dspy_code_act_tools(tools)
    sig = _with_dspy_code_act_instructions(sig)
    return dspy.CodeAct(
        sig,
        tools=list(bridge_tools),
        interpreter=cast(Any, MontyInterpreter(tools=host_tools, skip_sources=skip_sources)),
    )


def dspy_code_act_tools(
    tools: Sequence[Any],
) -> tuple[list[Any], dict[str, Callable[..., Any]], set[str]]:
    bridge_tools: list[Any] = []
    host_tools: dict[str, Callable[..., Any]] = {}
    skip_sources: set[str] = set()
    used_names: set[str] = set()
    for tool in tools:
        bridge_name = unique_python_identifier(str(tool.name), used_names)
        bridge_tool, source = dspy_code_act_bridge_tool(
            bridge_name,
            _dspy_code_act_parameter_names(tool),
        )
        bridge_tools.append(bridge_tool)
        host_tools[bridge_tool.name] = dspy_sync_tool_function(cast(Callable[..., Any], tool.func))
        skip_sources.add(source.strip())
    return bridge_tools, host_tools, skip_sources


def dspy_code_act_bridge_tool(
    name: str, parameter_names: Sequence[str] | None = None
) -> tuple[Any, str]:
    source = dspy_code_act_bridge_source(name, parameter_names=parameter_names)
    func = compile_dspy_code_act_bridge(name, source)
    return (dspy.Tool(func=func, name=name), source)


def dspy_code_act_bridge_source(
    name: str,
    *,
    parameter_names: Sequence[str] | None = None,
) -> str:
    signature = _dspy_code_act_bridge_signature(parameter_names)
    return f'def {name}({signature}):\n    """Runtime tool bridge."""\n    pass\n'


def _dspy_code_act_parameter_names(tool: Any) -> tuple[str, ...] | None:
    args = getattr(tool, "args", None)
    if not isinstance(args, Mapping):
        return None
    return tuple(str(name) for name in args)


def _dspy_code_act_bridge_signature(parameter_names: Sequence[str] | None) -> str:
    if parameter_names is None:
        return "**kwargs"
    if not parameter_names:
        return ""
    if all(sanitize_python_identifier(name) == name for name in parameter_names):
        return "*, " + ", ".join(parameter_names)
    return "**kwargs"


def compile_dspy_code_act_bridge(name: str, source: str) -> Callable[..., Any]:
    filename = f"<kedi-dspy-codeact-{name}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    namespace: dict[str, Any] = {}
    exec(compile(source, filename, "exec"), namespace)
    return cast(Callable[..., Any], namespace[name])


def _with_dspy_code_act_instructions(sig: type[dspy.Signature]) -> type[dspy.Signature]:
    context = monty_compat_prompt_context()
    if not context:
        return sig
    current = getattr(sig, "instructions", None)
    prefix = (
        "DSPy CodeAct is active. Generate Python code for the Monty sandbox. "
        "Use the provided tool functions directly by name and print any information "
        "that should be available to the final extractor. Follow the Monty capability "
        "context below before choosing imports, builtins, exceptions, or language constructs."
    )
    instructions = (
        f"{prefix}\n\n{context}" if not current else f"{current}\n\n{prefix}\n\n{context}"
    )
    return sig.with_instructions(instructions)


def installed_package_version(package_name: str) -> str:
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def monty_compat_capabilities(monty_version: str) -> Any:
    cache_module = importlib.import_module("monty_compat.cache")
    get_capabilities = cast("Callable[..., Any]", getattr(cache_module, "get_capabilities"))
    return get_capabilities(cache="auto", version=monty_version, only_released=True)


@lru_cache(maxsize=4)
def monty_compat_prompt_context() -> str:
    monty_version = installed_package_version("pydantic-monty")
    try:
        capabilities = monty_compat_capabilities(monty_version)
    except Exception as exc:
        return monty_compat_fallback_prompt(monty_version, exc)

    prompt_context = cast(str, capabilities.to_prompt_context()).strip()
    if not prompt_context:
        return monty_compat_fallback_prompt(monty_version, None)
    return (
        f"Installed pydantic-monty version: {monty_version}.\n"
        "The compatibility context below was generated by monty-compat with "
        "`cache='auto'` for this version.\n\n"
        f"{prompt_context}"
    )


def monty_compat_fallback_prompt(monty_version: str, exc: Exception | None) -> str:
    reason = f" Reason: {type(exc).__name__}: {exc}" if exc is not None else ""
    return (
        "## Monty Sandbox — Compatibility Context Unavailable\n"
        f"Installed pydantic-monty version: {monty_version}."
        f"{reason}\n"
        "Generate conservative Monty code: avoid class definitions, avoid unsupported "
        "imports, prefer simple functions, assignments, loops, conditionals, containers, "
        "and available run_code functions, and inspect tool schemas with `search_schema` "
        "before calling application tools."
    )


def _make_tool_function(rt: KediRuntime, proc: ProcedureDef) -> Callable[..., Any]:
    proc_fn = _get_or_compile_proc_function(rt, proc)

    @wraps(proc_fn)
    def _tool_function(**kwargs: Any) -> Any:
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


def sanitize_python_identifier(name: str) -> str:
    sanitized = _INVALID_IDENT_CHARS.sub("_", name)
    if sanitized and sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    if keyword.iskeyword(sanitized):
        sanitized = f"{sanitized}_"
    return sanitized or "_"


def unique_python_identifier(name: str, used: set[str]) -> str:
    base = sanitize_python_identifier(name)
    candidate = base
    index = 2
    while candidate in used:
        candidate = f"{base}_{index}"
        index += 1
    used.add(candidate)
    return candidate


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
    sig = inspect.signature(fn)
    bound = sig.bind_partial(**kwargs)
    bound.apply_defaults()
    local_env = dict(bound.arguments)
    _coerce_tool_arguments(rt, proc, local_env)
    call_kwargs = {name: local_env[name] for name in bound.arguments}

    token = rt._current_env.set(local_env)
    try:
        result = fn(**call_kwargs)
    finally:
        rt._current_env.reset(token)

    return (
        rt._evaluate_value(result, local_env)
        if isinstance(result, (InputExpr, CallExpr))
        else result
    )


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
    return {
        "kedi_schema": {
            "name": proc.name,
            "parameters": json_schema,
            "returns": return_schema,
            **({"description": description} if description else {}),
        },
        "kedi_input_schema": json_schema,
        "kedi_return_schema": return_schema,
    }


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
