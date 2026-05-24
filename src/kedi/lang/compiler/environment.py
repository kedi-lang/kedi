"""Environment management for Kedi compiler.

Handles variable lookup, environment building, and context management.
"""

from __future__ import annotations

import builtins
import linecache
import typing
from typing import Any

import typing_extensions
from pydantic import BaseModel

from ...core import KediRuntime
from ..procedure_introspection import get_procedure_metadata


def lookup_value(rt: KediRuntime, name: str, local_env: dict[str, Any]) -> Any:
    """Look up a variable in the environment hierarchy."""
    if name in local_env:
        return local_env[name]

    try:
        current = rt._current_env.get()
        if isinstance(current, dict) and name in current:
            return current[name]
    except Exception:
        pass

    for env_name in ("_globals_env", "_prelude_env"):
        env = getattr(rt, env_name, None)
        if isinstance(env, dict) and name in env:
            return env[name]

    raise KeyError(f"Unknown variable: {name}")


def build_python_env(rt: KediRuntime, local_env: dict[str, Any]) -> dict[str, Any]:
    """Build execution environment for Python code with full context."""
    env: dict[str, Any] = {"__builtins__": builtins.__dict__}

    # Layer environments: prelude -> globals -> current -> local
    for env_name in ("_prelude_env", "_globals_env"):
        source = getattr(rt, env_name, None)
        if isinstance(source, dict):
            env.update(source)

    try:
        current = rt._current_env.get()
        if isinstance(current, dict):
            env.update(current)
    except Exception:
        pass

    env.update(local_env)

    # Add type environment
    _add_type_env(rt, env, local_env)

    # Wrap procedures as callables
    for name, fn in rt._procedures.items():
        env[name] = _make_proc_wrapper(rt, name, fn, local_env)

    return env


def build_type_env(rt: KediRuntime, local_env: dict[str, Any] | None) -> dict[str, Any]:
    """Build environment for type annotation evaluation."""
    env: dict[str, Any] = {
        "__builtins__": builtins.__dict__,
        "builtins": builtins,
        "typing": typing,
        "typing_extensions": typing_extensions,
    }

    # Import common typing names
    for module in (typing, typing_extensions):
        for name in getattr(module, "__all__", ()):
            if not name.startswith("_"):
                try:
                    env.setdefault(name, getattr(module, name))
                except AttributeError:
                    continue

    # Layer type environments
    stack = getattr(rt, "_type_env_stack", None)
    if isinstance(stack, list):
        for layer in stack:
            env.update(layer)

    # Add runtime environments
    for env_name in ("_prelude_env", "_globals_env"):
        source = getattr(rt, env_name, None)
        if isinstance(source, dict):
            env.update(source)

    try:
        current = rt._current_env.get()
        if isinstance(current, dict):
            env.update(current)
    except Exception:
        pass

    if local_env:
        env.update(local_env)

    return env


def _add_type_env(
    rt: KediRuntime, env: dict[str, Any], local_env: dict[str, Any]
) -> None:
    """Add type environment and handle Pydantic model coercion."""
    stack = getattr(rt, "_type_env_stack", None)
    if not isinstance(stack, list) or not stack:
        return

    type_env: dict[str, Any] = {}
    for layer in stack:
        type_env.update(layer)

    env.update(type_env)

    # Coerce string values to models where applicable
    from .type_system import coerce_model_value

    for key, val in list(local_env.items()):
        base = key.split(".")[0]
        if base in type_env and isinstance(val, str):
            model = type_env[base]
            if isinstance(model, type) and issubclass(model, BaseModel):
                wrapped = coerce_model_value(model, val)
                local_env[key] = wrapped
                env[key] = wrapped


def _make_proc_wrapper(
    rt: KediRuntime,
    proc_name: str,
    proc_fn: Any,
    local_env: dict[str, Any],
):
    """Create a Python-facing callable wrapper for a procedure."""
    metadata = get_procedure_metadata(proc_fn)
    if metadata is None:
        return _make_generic_proc_wrapper(rt, proc_name)

    display_source = metadata.display_source()
    executable_source = metadata.executable_source()
    filename = f"<kedi-proc:{proc_name}:{id(proc_fn)}>"
    linecache.cache[filename] = (
        len(display_source),
        None,
        display_source.splitlines(True),
        filename,
    )
    namespace = build_type_env(rt, local_env)
    namespace.update(metadata.default_namespace())
    namespace["__kedi_call__"] = lambda *args: rt._invoke_procedure(
        rt._get_proc(proc_name), args
    )
    try:
        exec(
            compile(executable_source, filename, "exec", dont_inherit=True),
            namespace,
            namespace,
        )
    except Exception:
        return _make_generic_proc_wrapper(rt, proc_name)
    return namespace[metadata.name]


def _make_generic_proc_wrapper(rt: KediRuntime, proc_name: str):
    """Create the fallback callable wrapper for a non-Kedi procedure."""

    def wrapper(*args: Any):
        callee = rt._get_proc(proc_name)
        return rt._invoke_procedure(callee, list(args))

    return wrapper


class ScopedDict(dict[str, Any]):
    """Dictionary that propagates changes to parent closure."""

    def __init__(
        self,
        initial: dict[str, Any] | None = None,
        *,
        closure: dict[str, Any] | None = None,
        readonly: set[str] | None = None,
    ) -> None:
        super().__init__()
        self._closure = closure
        self._readonly = readonly or set()
        if initial:
            for k, v in initial.items():
                super().__setitem__(k, v)

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, value)
        if self._closure is not None and key not in self._readonly:
            self._closure[key] = value
