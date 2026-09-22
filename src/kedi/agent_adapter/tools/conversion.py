from __future__ import annotations

import inspect
from functools import wraps
from pathlib import Path
from typing import (
    Any,
    Callable,
    Mapping,
)

from pydantic import TypeAdapter

from kedi.agent_adapter.approvals.policy import ToolRisk
from kedi.agent_adapter.tools.retry import (
    ToolRetryPolicy,
    tool_body,
    tool_retry_policy,
    with_body_retry,
)
from kedi.agent_adapter.tools.specs import ToolSpec
from kedi.agent_adapter.tools.validation import (
    _procedure_tool_argument_validator,
    _python_callable_argument_validator,
    _validated_procedure_kwargs,
)
from kedi.core import CallExpr, InputExpr, KediRuntime
from kedi.lang.ast import ProcedureDef
from kedi.lang.docstrings import parse_procedure_docstring
from kedi.lang.procedure_schema import (
    build_procedure_description,
    procedure_json_schema,
    procedure_return_json_schema,
)
from kedi.module_registry import internal_module_path
from kedi.tool_context import kedi_tool_call_context

_FILESYSTEM_READ_ONLY_PROCEDURES = frozenset(
    {"read_text_file", "path_exists", "list_directory", "get_file_info"}
)


def kedi_procedure_tool_spec(
    rt: KediRuntime,
    proc: ProcedureDef,
    *,
    local_env: dict[str, Any] | None = None,
) -> ToolSpec:
    """Build an adapter-neutral tool spec from a Kedi procedure."""
    parsed_docstring = parse_procedure_docstring(proc.docstring)
    declaration = proc.tool
    json_schema = procedure_json_schema(rt, proc, local_env=local_env)
    return_schema = procedure_return_json_schema(rt, proc, local_env=local_env)
    tool_description = _build_tool_description(
        rt,
        proc,
        declaration.description
        if declaration is not None and declaration.description is not None
        else parsed_docstring.description,
        local_env=local_env,
    )
    retry_policy = _procedure_retry_policy(rt, proc)
    tool_fn = _make_tool_function(rt, proc, retry_policy=retry_policy)
    exposed_name = declaration.name if declaration is not None and declaration.name else proc.name
    return ToolSpec(
        fn=tool_fn,
        name=exposed_name,
        description=tool_description,
        json_schema=json_schema,
        return_schema=return_schema,
        metadata=_tool_metadata(
            proc,
            json_schema,
            return_schema,
            exposed_name=exposed_name,
            description=tool_description,
            retry_policy=retry_policy,
        ),
        proc=proc,
        risk=declaration.risk if declaration is not None else _procedure_tool_risk(proc),
        risk_resolver=_procedure_tool_risk_resolver(proc),
        argument_validator=_procedure_tool_argument_validator(rt, proc, tool_fn),
        retry_policy=retry_policy,
    )


def python_callable_tool_spec(name: str, fn: Callable[..., Any]) -> ToolSpec:
    """Build an adapter-neutral tool spec from a Python callable."""
    from kedi.artifacts.stream import callable_with_logical_stream_return

    retry_policy = tool_retry_policy(fn)
    callable_body, logical_return = callable_with_logical_stream_return(tool_body(fn))
    tool_fn = (
        with_body_retry(callable_body, retry_policy, isolate_arguments=True)
        if retry_policy.retries
        else callable_body
    )
    description = getattr(fn, "__kedi_tool_description__", None)
    if not isinstance(description, str):
        description = _callable_description(fn)
    risk = getattr(fn, "__kedi_tool_risk__", "mutating")
    if risk not in ("read_only", "mutating", "sensitive"):
        raise ValueError(f"invalid Kedi tool risk {risk!r} for {name!r}")
    metadata: dict[str, Any] = {
        "kedi_retry": _retry_metadata(retry_policy),
    }
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
        retry_policy=retry_policy,
    )


def _procedure_retry_policy(rt: KediRuntime, proc: ProcedureDef) -> ToolRetryPolicy:
    return tool_retry_policy(_get_or_compile_proc_function(rt, proc))


def _retry_metadata(policy: ToolRetryPolicy) -> dict[str, Any]:
    return {
        "retries": policy.retries,
        "retry_on": (
            None
            if policy.retry_on is None
            else [exception_type.__name__ for exception_type in policy.retry_on]
        ),
    }


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


def _make_tool_function(
    rt: KediRuntime,
    proc: ProcedureDef,
    *,
    retry_policy: ToolRetryPolicy | None = None,
) -> Callable[..., Any]:
    proc_fn = _get_or_compile_proc_function(rt, proc)
    validated_body = getattr(proc_fn, "__kedi_validated_procedure_body__", proc_fn)
    policy = retry_policy or ToolRetryPolicy()

    @wraps(proc_fn)
    def _procedure_body(**kwargs: Any) -> Any:
        with kedi_tool_call_context():
            return _invoke_procedure_body(rt, validated_body, kwargs)

    retried_body = (
        with_body_retry(_procedure_body, policy, isolate_arguments=True)
        if policy.retries
        else _procedure_body
    )

    @wraps(proc_fn)
    def _tool_function(**kwargs: Any) -> Any:
        call_kwargs = _validated_procedure_kwargs(rt, proc, proc_fn, kwargs)
        return retried_body(**call_kwargs)

    _tool_function.__name__ = proc.name
    _tool_function.__doc__ = getattr(proc_fn, "__doc__", None)
    return _tool_function


def _callable_description(fn: Callable[..., Any]) -> str | None:
    if not (inspect.isfunction(fn) or inspect.ismethod(fn)) and hasattr(fn, "__call__"):
        return inspect.getdoc(fn.__call__) or inspect.getdoc(fn)
    return inspect.getdoc(fn)


def _get_or_compile_proc_function(rt: KediRuntime, proc: ProcedureDef) -> Callable[..., Any]:
    try:
        return rt._get_proc(proc.name)
    except KeyError:
        from kedi.lang.compiler.procedure_compiler import ProcedureCompiler

        return ProcedureCompiler(rt)._make_procedure_impl(proc, proc.lexical_env)


def _invoke_procedure_body(
    rt: KediRuntime,
    fn: Any,
    call_kwargs: dict[str, Any],
) -> Any:
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


def _tool_metadata(
    proc: ProcedureDef,
    json_schema: dict[str, Any],
    return_schema: dict[str, Any],
    *,
    exposed_name: str | None = None,
    description: str | None,
    retry_policy: ToolRetryPolicy | None = None,
) -> dict[str, Any]:
    tool_name = exposed_name or proc.name
    effective_retry = retry_policy or ToolRetryPolicy()
    metadata: dict[str, Any] = {
        "kedi_schema": {
            "name": tool_name,
            "parameters": json_schema,
            "returns": return_schema,
            **({"description": description} if description else {}),
        },
        "kedi_input_schema": json_schema,
        "kedi_return_schema": return_schema,
        "kedi_source_name": proc.name,
        "kedi_retry": _retry_metadata(effective_retry),
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
