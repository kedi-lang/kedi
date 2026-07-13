"""Environment management for Kedi compiler.

Handles variable lookup, environment building, and context management.
"""

from __future__ import annotations

import builtins
import linecache
import typing
from datetime import datetime, timedelta
from typing import Any

import typing_extensions
from pydantic import BaseModel

from ...core import KediRuntime
from ..procedure_introspection import get_procedure_def, get_procedure_metadata

KEDI_TYPING_TYPE_NAMES = frozenset(
    {
        "Annotated",
        "Any",
        "Callable",
        "Dict",
        "FrozenSet",
        "Iterable",
        "List",
        "Literal",
        "Mapping",
        "Optional",
        "Sequence",
        "Set",
        "Tuple",
        "Type",
        "Union",
    }
)

KEDI_DATETIME_TYPES = {
    "datetime": datetime,
    "timedelta": timedelta,
}


class KediEnv(dict):
    """Exec namespace that resolves ``KediPromise`` values lazily, on lookup.

    CPython routes ``LOAD_NAME`` (module/exec top level) and ``LOAD_GLOBAL`` (function and
    comprehension free variables — the slow path taken when ``globals`` is a non-exact ``dict``)
    through ``__getitem__``. So a promise output resolves only when the executing Python actually
    references that name — never all of them up front. Resolved values are cached back so repeated
    lookups are cheap and identity is stable.
    """

    def __getitem__(self, key: str) -> Any:
        value = dict.__getitem__(self, key)
        from ...promise import KediPromise

        if isinstance(value, KediPromise):
            resolved = value.resolve()
            dict.__setitem__(self, key, resolved)
            return resolved
        return value


def assert_kedienv_routing() -> None:
    """Fail loudly if this interpreter doesn't route name lookups through ``KediEnv.__getitem__``.

    Parallel mode's lazy promise resolution relies on CPython consulting a non-exact-``dict``
    globals via ``__getitem__`` for ``LOAD_GLOBAL``/``LOAD_NAME`` (a slow-path behavior verified on
    CPython 3.10). If a future interpreter drops it, promises would silently leak into user Python
    as raw objects — so we refuse parallel mode here rather than degrade. Cheap; run once.
    """
    from concurrent.futures import Future

    from ...promise import KediPromise

    fut: Future = Future()
    fut.set_result({"x": "ok"})
    ns = KediEnv()
    builtins_ns = __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__
    ns["__builtins__"] = builtins_ns
    ns["x"] = KediPromise(fut, "x")
    exec(
        compile(
            "def __probe__():\n global x\n return x\n__r__ = __probe__()",
            "<kedienv-selfcheck>",
            "exec",
        ),
        ns,
        ns,
    )
    if isinstance(dict.__getitem__(ns, "__r__"), KediPromise):  # pragma: no cover - guard
        raise RuntimeError(
            "KediEnv routing self-check failed: this Python build does not route LOAD_GLOBAL "
            "through KediEnv.__getitem__, so parallel mode cannot resolve promises lazily. "
            "Run Kedi sequentially (unset KEDI_PARALLEL / do not use kedi.parallel())."
        )


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

    for env in (rt._globals_env, rt._prelude_env, rt._import_env):
        if env is not None and name in env:
            return env[name]

    raise KeyError(f"Unknown variable: {name}")


def build_python_env(rt: KediRuntime, local_env: dict[str, Any]) -> dict[str, Any]:
    """Build execution environment for Python code with full context.

    Returns a :class:`KediEnv` so promise outputs resolve lazily, only when the executing Python
    references them (Concern 1: unused output fields are never awaited). Critically this also
    avoids a self-deadlock: a tool running inside a produce job must not eagerly resolve the
    in-flight template's own (still-running) output promise.
    """
    env: dict[str, Any] = KediEnv()
    env["__builtins__"] = builtins.__dict__

    # Layer environments: imports -> prelude -> globals -> current -> local
    for source in (rt._import_env, rt._prelude_env, rt._globals_env):
        if source is not None:
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

    # Import only names that are meaningful as Kedi type annotations.
    for module in (typing, typing_extensions):
        for name in KEDI_TYPING_TYPE_NAMES:
            value = getattr(module, name, None)
            if value is not None:
                env.setdefault(name, value)

    env.update(KEDI_DATETIME_TYPES)

    # Layer type environments
    for layer in rt._type_env_stack or []:
        env.update(layer)

    # Add runtime environments
    for source in (rt._import_env, rt._prelude_env, rt._globals_env):
        if source is not None:
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


def _add_type_env(rt: KediRuntime, env: dict[str, Any], local_env: dict[str, Any]) -> None:
    """Add type environment and handle Pydantic model coercion."""
    if not rt._type_env_stack:
        return

    type_env: dict[str, Any] = {}
    for layer in rt._type_env_stack:
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
    namespace["__kedi_call__"] = lambda *args: rt._invoke_procedure(rt._get_proc(proc_name), args)
    try:
        exec(
            compile(executable_source, filename, "exec", dont_inherit=True),
            namespace,
            namespace,
        )
    except Exception:
        return _make_generic_proc_wrapper(rt, proc_name)
    wrapper = namespace[metadata.name]
    wrapper.__doc__ = metadata.docstring
    setattr(wrapper, "__kedi_procedure_metadata__", metadata)
    procedure_def = get_procedure_def(proc_fn)
    if procedure_def is not None:
        setattr(wrapper, "__kedi_procedure_def__", procedure_def)
    return wrapper


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
