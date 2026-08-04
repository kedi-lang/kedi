"""Environment management for Kedi compiler.

Handles variable lookup, environment building, and context management.
"""

from __future__ import annotations

import builtins
import dataclasses
import json
import keyword
import linecache
import re
import typing
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from typing import Any

import typing_extensions
from pydantic import BaseModel, EmailStr, FileUrl, HttpUrl

from ...agent_adapter.approval import ApprovalDecision, ApprovalRequest
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

KEDI_RUNTIME_TYPES = {
    "datetime": datetime,
    "date": date,
    "time": time,
    "timedelta": timedelta,
    "Regex": re.Pattern[str],
    "Email": EmailStr,
    "EmailStr": EmailStr,
    "HttpUrl": HttpUrl,
    "FileUrl": FileUrl,
    "ApprovalDecision": ApprovalDecision,
    "ApprovalRequest": ApprovalRequest,
}

KEDI_RUNTIME_TYPE_ORIGINS = {
    "datetime": "datetime",
    "date": "datetime",
    "time": "datetime",
    "timedelta": "datetime",
    "Regex": "re",
    "Email": "pydantic",
    "HttpUrl": "pydantic",
    "FileUrl": "pydantic",
    "ApprovalDecision": "kedi",
    "ApprovalRequest": "kedi",
}

KEDI_RUNTIME_TYPE_JSON_SCHEMA_FORMATS = {
    "datetime": "date-time",
    "date": "date",
    "time": "time",
    "timedelta": "duration",
    "Regex": "regex",
    "Email": "email",
    "EmailStr": "email",
    "HttpUrl": "uri",
    "FileUrl": "uri",
}

KEDI_DATETIME_TYPES = {
    "datetime": datetime,
    "date": date,
    "time": time,
    "timedelta": timedelta,
}


# The runtime injects parsed CLI arguments under this name. Applications embedding Kedi may
# configure a different identifier before compiling a program; the chosen name stays reserved.
CLI_ARGS_IDENTIFIER: str = "args"


def cli_args_identifier() -> str:
    """Return the configured CLI argument binding name after validating it."""
    identifier = CLI_ARGS_IDENTIFIER
    if (
        not isinstance(identifier, str)
        or not identifier.isidentifier()
        or keyword.iskeyword(identifier)
    ):
        raise ValueError("CLI_ARGS_IDENTIFIER must be a valid, non-keyword Python identifier")
    return identifier


def assert_not_cli_args_identifier(name: str) -> None:
    """Reject writes that would replace the runtime-owned CLI argument binding."""
    identifier = cli_args_identifier()
    if name == identifier:
        raise ValueError(
            f"Reserved identifier {identifier!r} belongs to CLI arguments and cannot be overwritten"
        )


class KediEnv(dict[str, Any]):
    """Exec namespace that resolves ``KediPromise`` values lazily, on lookup.

    CPython routes ``LOAD_NAME`` (module/exec top level) and ``LOAD_GLOBAL`` (function and
    comprehension free variables — the slow path taken when ``globals`` is a non-exact ``dict``)
    through ``__getitem__``. So a promise output resolves only when the executing Python actually
    references that name — never all of them up front. Resolved values are cached back so repeated
    lookups are cheap and identity is stable.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        dict.__init__(self)
        if args or kwargs:
            self.update(*args, **kwargs)

    def _set_cli_args(self, value: Any) -> None:
        """Install the runtime-owned CLI binding without exposing an override path."""
        dict.__setitem__(self, cli_args_identifier(), value)

    def __setitem__(self, key: str, value: Any) -> None:
        assert_not_cli_args_identifier(key)
        dict.__setitem__(self, key, value)

    def update(self, *args: Any, **kwargs: Any) -> None:
        pending = dict(*args, **kwargs)
        for key in pending:
            assert_not_cli_args_identifier(key)
        dict.update(self, pending)

    def setdefault(self, key: str, default: Any = None) -> Any:
        assert_not_cli_args_identifier(key)
        if key in self:
            return dict.__getitem__(self, key)
        dict.__setitem__(self, key, default)
        return default

    def __ior__(self, other: Any) -> "KediEnv":
        self.update(other)
        return self

    def __or__(self, other: Any) -> "KediEnv":
        merged = self.copy()
        merged.update(other)
        return merged

    def __ror__(self, other: Any) -> "KediEnv":
        merged = KediEnv(other)
        dict.update(merged, self)
        return merged

    @classmethod
    def fromkeys(cls, iterable: Any, value: Any = None) -> "KediEnv":
        env = cls()
        for key in iterable:
            env[key] = value
        return env

    def copy(self) -> "KediEnv":
        copied = KediEnv()
        dict.update(copied, self)
        return copied

    def __delitem__(self, key: str) -> None:
        assert_not_cli_args_identifier(key)
        dict.__delitem__(self, key)

    def pop(self, key: str, *default: Any) -> Any:
        assert_not_cli_args_identifier(key)
        return dict.pop(self, key, *default)

    def popitem(self) -> tuple[str, Any]:
        if cli_args_identifier() in self:
            raise ValueError("The reserved CLI argument binding cannot be removed")
        try:
            key = next(reversed(self))
        except StopIteration as exc:
            raise KeyError("popitem(): dictionary is empty") from exc
        value = dict.__getitem__(self, key)
        dict.__delitem__(self, key)
        return key, value

    def clear(self) -> None:
        if cli_args_identifier() in self:
            raise ValueError("The reserved CLI argument binding cannot be removed")
        dict.clear(self)

    def __getitem__(self, key: str) -> Any:
        return native_value(dict.__getitem__(self, key))

    def raw_value(self, key: str) -> Any:
        """Return the stored value without resolving promises or artifacts."""
        return dict.__getitem__(self, key)

    def native_value(self, key: str) -> Any:
        """Return the Python-native value while retaining lazy storage."""
        return native_value(self.raw_value(key))


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


def raw_lookup_value(rt: KediRuntime, name: str, local_env: dict[str, Any]) -> Any:
    """Look up a variable without materializing promises or artifacts."""
    if name in local_env:
        return _raw_mapping_value(local_env, name)

    try:
        current = rt._current_env.get()
        if isinstance(current, dict) and name in current:
            return _raw_mapping_value(current, name)
    except Exception:
        pass

    for env in (rt._globals_env, rt._prelude_env, rt._import_env):
        if env is not None and name in env:
            return _raw_mapping_value(env, name)

    raise KeyError(f"Unknown variable: {name}")


def lookup_value(rt: KediRuntime, name: str, local_env: dict[str, Any]) -> Any:
    """Look up and materialize a value for native Kedi/Python execution."""
    return native_value(raw_lookup_value(rt, name, local_env))


def model_lookup_value(rt: KediRuntime, name: str, local_env: dict[str, Any]) -> Any:
    """Look up a value for a model prompt without opening artifact payloads."""
    return model_value(raw_lookup_value(rt, name, local_env))


def native_value(value: Any, *, _memo: dict[int, Any] | None = None) -> Any:
    """Resolve promises and artifacts recursively into their logical Python values."""
    from ...artifacts import ArtifactHandle
    from ...promise import KediPromise

    if isinstance(value, KediPromise):
        return native_value(value.resolve(), _memo=_memo)
    if isinstance(value, ArtifactHandle):
        return native_value(value.get_value(), _memo=_memo)
    if not _contains_lazy_value(value):
        return value
    return _map_nested(value, model=False, memo={} if _memo is None else _memo)


def model_value(value: Any, *, _memo: dict[int, Any] | None = None) -> Any:
    """Resolve promises while replacing artifacts with compact JSON-safe refs."""
    from ...artifacts import ArtifactHandle
    from ...promise import KediPromise

    if isinstance(value, KediPromise):
        return model_value(value.resolve(), _memo=_memo)
    if isinstance(value, ArtifactHandle):
        return value.as_ref().model_dump(mode="json")
    if not _contains_lazy_value(value):
        return value
    return _map_nested(value, model=True, memo={} if _memo is None else _memo)


def model_text(value: Any) -> str:
    """Render a model-facing value, using stable JSON for artifact references."""
    resolved = model_value(value)
    if _contains_artifact_ref(value):
        try:
            return json.dumps(resolved, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError, RecursionError):
            return repr(resolved)
    return str(resolved)


def _raw_mapping_value(mapping: dict[str, Any], key: str) -> Any:
    if isinstance(mapping, KediEnv):
        return mapping.raw_value(key)
    return dict.__getitem__(mapping, key)


def _contains_artifact_ref(value: Any, *, seen: set[int] | None = None) -> bool:
    from ...artifacts import ArtifactHandle
    from ...promise import KediPromise

    if isinstance(value, ArtifactHandle):
        return True
    if isinstance(value, KediPromise):
        return _contains_artifact_ref(value.resolve(), seen=seen)
    seen = set() if seen is None else seen
    identity = id(value)
    if identity in seen:
        return False
    seen.add(identity)
    if isinstance(value, Mapping):
        return any(
            _contains_artifact_ref(key, seen=seen) or _contains_artifact_ref(item, seen=seen)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_contains_artifact_ref(item, seen=seen) for item in value)
    if isinstance(value, BaseModel):
        return _contains_artifact_ref(value.__dict__, seen=seen)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return any(
            _contains_artifact_ref(getattr(value, item.name), seen=seen)
            for item in dataclasses.fields(value)
        )
    return False


def _contains_lazy_value(value: Any, *, seen: set[int] | None = None) -> bool:
    from ...artifacts import ArtifactHandle
    from ...promise import KediPromise

    if isinstance(value, (ArtifactHandle, KediPromise)):
        return True
    seen = set() if seen is None else seen
    identity = id(value)
    if identity in seen:
        return False
    seen.add(identity)
    if isinstance(value, Mapping):
        return any(
            _contains_lazy_value(key, seen=seen) or _contains_lazy_value(item, seen=seen)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_contains_lazy_value(item, seen=seen) for item in value)
    if isinstance(value, BaseModel):
        return _contains_lazy_value(value.__dict__, seen=seen)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return any(
            _contains_lazy_value(getattr(value, item.name), seen=seen)
            for item in dataclasses.fields(value)
        )
    return False


def _map_nested(value: Any, *, model: bool, memo: dict[int, Any]) -> Any:
    from ...artifacts import ArtifactHandle
    from ...promise import KediPromise

    if isinstance(value, KediPromise):
        return model_value(value, _memo=memo) if model else native_value(value, _memo=memo)
    if isinstance(value, ArtifactHandle):
        return (
            value.as_ref().model_dump(mode="json")
            if model
            else native_value(value.get_value(), _memo=memo)
        )
    if isinstance(value, (str, bytes, bytearray, int, float, bool, type(None))):
        return value
    identity = id(value)
    if identity in memo:
        return memo[identity]
    if isinstance(value, list):
        result: list[Any] = []
        memo[identity] = result
        result.extend(_map_nested(item, model=model, memo=memo) for item in value)
        return result
    if isinstance(value, dict):
        result_dict: dict[Any, Any] = {}
        memo[identity] = result_dict
        for key, item in value.items():
            result_dict[_map_nested(key, model=model, memo=memo)] = _map_nested(
                item, model=model, memo=memo
            )
        return result_dict
    if isinstance(value, tuple):
        result_tuple = tuple(_map_nested(item, model=model, memo=memo) for item in value)
        memo[identity] = result_tuple
        return result_tuple
    if isinstance(value, set):
        if model:
            result_list = [_map_nested(item, model=True, memo=memo) for item in value]
            memo[identity] = result_list
            return result_list
        result_set = {_map_nested(item, model=False, memo=memo) for item in value}
        memo[identity] = result_set
        return result_set
    if isinstance(value, frozenset):
        if model:
            result_list = [_map_nested(item, model=True, memo=memo) for item in value]
            memo[identity] = result_list
            return result_list
        result_frozen = frozenset(_map_nested(item, model=False, memo=memo) for item in value)
        memo[identity] = result_frozen
        return result_frozen
    if isinstance(value, BaseModel):
        if model:
            model_fields: dict[str, Any] = {}
            memo[identity] = model_fields
            model_fields.update(
                {
                    name: _map_nested(item, model=True, memo=memo)
                    for name, item in value.__dict__.items()
                }
            )
            return model_fields
        updates = {
            name: _map_nested(item, model=False, memo=memo) for name, item in value.__dict__.items()
        }
        result_model = value.model_copy(update=updates)
        memo[identity] = result_model
        return result_model
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        if model:
            dataclass_fields: dict[str, Any] = {}
            memo[identity] = dataclass_fields
            dataclass_fields.update(
                {
                    item.name: _map_nested(getattr(value, item.name), model=True, memo=memo)
                    for item in dataclasses.fields(value)
                }
            )
            return dataclass_fields
        updates = {
            item.name: _map_nested(getattr(value, item.name), model=False, memo=memo)
            for item in dataclasses.fields(value)
        }
        result_dataclass = dataclasses.replace(value, **updates)
        memo[identity] = result_dataclass
        return result_dataclass
    return value


def build_python_env(rt: KediRuntime, local_env: dict[str, Any]) -> KediEnv:
    """Build execution environment for Python code with full context.

    Returns a :class:`KediEnv` so promise outputs resolve lazily, only when the executing Python
    references them (Concern 1: unused output fields are never awaited). Critically this also
    avoids a self-deadlock: a tool running inside a produce job must not eagerly resolve the
    in-flight template's own (still-running) output promise.
    """
    env = KediEnv()
    env["__builtins__"] = builtins.__dict__

    identifier = cli_args_identifier()
    missing = object()
    runtime_args = rt._initial_globals_env.get(identifier, missing)
    if runtime_args is not missing:
        env._set_cli_args(runtime_args)

    def merge(source: dict[str, Any]) -> None:
        for name, value in source.items():
            if name == identifier:
                if value is runtime_args:
                    continue
                assert_not_cli_args_identifier(name)
            env[name] = value

    # Layer environments: imports -> prelude -> globals -> current -> local
    for source in (rt._import_env, rt._prelude_env, rt._globals_env):
        if source is not None:
            merge(source)

    try:
        current = rt._current_env.get()
        if isinstance(current, dict):
            merge(current)
    except Exception:
        pass

    merge(local_env)

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

    env.update(KEDI_RUNTIME_TYPES)

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
    setattr(wrapper, "__signature__", metadata.introspection_signature())
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
