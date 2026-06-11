from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..errors import KediExecutionError
from .model import eval_profile_definition
from .profile import AgentProfile

if TYPE_CHECKING:
    from kedi.core import KediRuntime
    from kedi.lang.ast import ProcedureDef, UseDirectiveStmt

    from .registry import ProfileDefinition


def eval_use_directive(
    rt: KediRuntime,
    stmt: UseDirectiveStmt,
    registry: dict[str, ProfileDefinition],
    env: dict[str, object],
    *,
    available_procedures: set[str],
) -> AgentProfile:
    if stmt.tool_names:
        return AgentProfile(tools=tuple(stmt.tool_names))
    name = stmt.name
    if not name:
        raise KediExecutionError("Empty '> use:' directive")
    if name in available_procedures:
        return AgentProfile(tools=(name,))
    definition = registry.get(name)
    if definition is not None:
        return eval_profile_definition(rt, definition, env)
    raise KediExecutionError(f"Unknown use target: {name!r}")


def procedure_def_for_tool(rt: KediRuntime, name: str) -> ProcedureDef | None:
    try:
        fn = rt._get_proc(name)
    except KeyError:
        return None
    proc = getattr(fn, "__kedi_procedure_def__", None)
    if proc is None:
        return None
    return proc


def python_callable_for_tool(rt: KediRuntime, name: str) -> Callable[..., Any] | None:
    for env_name in ("_import_env", "_prelude_env", "_globals_env", "_initial_globals_env"):
        env = getattr(rt, env_name, None)
        if isinstance(env, dict):
            value = env.get(name)
            if callable(value):
                return value
    return None
