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
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any

import typing_extensions
from pydantic import BaseModel, EmailStr, FileUrl, HttpUrl

from ...agent_adapter.approvals.policy import ApprovalDecision, ApprovalRequest
from ...core import KediRuntime
from ..procedure_introspection import get_procedure_def, get_procedure_metadata

KEDI_TYPING_TYPE_NAMES = frozenset(
    {
        "Annotated",
        "Any",
        "Callable",
        "Dict",
        "FrozenSet",
        "List",
        "Literal",
        "Optional",
        "Set",
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


@dataclass(frozen=True, slots=True)
class BindingMetadata:
    """Semantic metadata owned by one lexical binding."""

    kind: str = "value"
    type_contract: Any | None = None
    source: Any | None = None
    assignable: bool = True
    synthetic: bool = False
    decision: Any | None = None


class ScopeFrame(dict[str, Any]):
    """One lexical value scope linked to its containing scope."""

    def __init__(
        self,
        initial: Mapping[str, Any] | None = None,
        *,
        parent: dict[str, Any] | None = None,
    ) -> None:
        dict.__init__(self)
        self.parent = parent
        self._binding_metadata: dict[str, BindingMetadata] = {}
        if initial:
            for name, value in initial.items():
                if name == cli_args_identifier():
                    dict.__setitem__(self, name, value)
                    self._binding_metadata[name] = BindingMetadata(
                        kind="runtime",
                        assignable=False,
                        synthetic=True,
                    )
                else:
                    self.declare(name, value)

    def declare(
        self,
        name: str,
        value: Any,
        metadata: BindingMetadata | None = None,
    ) -> None:
        """Create or replace a binding owned by this frame."""
        assert_not_cli_args_identifier(name)
        dict.__setitem__(self, name, value)
        self._binding_metadata[name] = metadata or BindingMetadata()

    def declare_synthetic(self, name: str, value: Any) -> None:
        """Install a runtime-owned local such as a loop binder."""
        self.declare(name, value, BindingMetadata(kind="synthetic", synthetic=True))

    def owns(self, name: str) -> bool:
        return dict.__contains__(self, name)

    def resolve_owner(self, name: str) -> dict[str, Any] | None:
        if self.owns(name):
            return self
        parent = self.parent
        if isinstance(parent, ScopeFrame):
            return parent.resolve_owner(name)
        if isinstance(parent, dict) and name in parent:
            return parent
        return None

    def binding_metadata(self, name: str) -> BindingMetadata | None:
        """Return metadata from the nearest frame that owns ``name``."""
        owner = self.resolve_owner(name)
        if isinstance(owner, ScopeFrame):
            return owner._binding_metadata.get(name)
        return None

    def local_binding_metadata(self) -> dict[str, BindingMetadata]:
        """Return a copy of metadata owned directly by this frame."""
        return dict(self._binding_metadata)

    def restore_locals(
        self,
        values: Mapping[str, Any],
        metadata: Mapping[str, BindingMetadata],
    ) -> None:
        """Replace local state from a validated persistence snapshot."""
        if set(values) != set(metadata):
            raise ValueError("Restored values and binding metadata must contain the same names")

        reserved = cli_args_identifier()
        dict.clear(self)
        self._binding_metadata.clear()
        for name, value in values.items():
            binding = metadata[name]
            if name == reserved:
                if binding.kind != "runtime" or binding.assignable or not binding.synthetic:
                    raise ValueError(
                        f"Restored reserved binding {reserved!r} must remain runtime-owned"
                    )
                dict.__setitem__(self, name, value)
                self._binding_metadata[name] = binding
                continue
            self.declare(name, value, binding)

    def raw_value(self, name: str) -> Any:
        owner = self.resolve_owner(name)
        if owner is None:
            raise KeyError(name)
        if isinstance(owner, ScopeFrame):
            return dict.__getitem__(owner, name)
        return _raw_mapping_value(owner, name)

    def assign(self, name: str, value: Any) -> None:
        """Write the nearest existing binding without creating a new one."""
        assert_not_cli_args_identifier(name)
        owner = self.resolve_owner(name)
        if owner is None:
            raise NameError(f"Cannot assign unknown Kedi binding: {name}")
        if isinstance(owner, ScopeFrame):
            metadata = owner._binding_metadata.get(name)
            if metadata is not None and not metadata.assignable:
                raise ValueError(f"Kedi binding {name!r} is read-only")
            dict.__setitem__(owner, name, value)
            if metadata is not None and metadata.decision is not None:
                owner._binding_metadata[name] = dataclasses.replace(metadata, decision=None)
            return
        owner[name] = value

    def visible_bindings(self) -> dict[str, tuple[dict[str, Any], Any]]:
        """Return visible names paired with their nearest owning mapping."""
        visible: dict[str, tuple[dict[str, Any], Any]] = {}
        parent = self.parent
        if isinstance(parent, ScopeFrame):
            visible.update(parent.visible_bindings())
        elif isinstance(parent, dict):
            visible.update((name, (parent, _raw_mapping_value(parent, name))) for name in parent)
        visible.update((name, (self, dict.__getitem__(self, name))) for name in dict.keys(self))
        return visible

    def __contains__(self, name: object) -> bool:
        if dict.__contains__(self, name):
            return True
        parent = self.parent
        return isinstance(name, str) and isinstance(parent, dict) and name in parent

    def __getitem__(self, name: str) -> Any:
        return self.raw_value(name)

    def get(self, name: object, default: Any = None) -> Any:
        if not isinstance(name, str):
            return default
        try:
            return self.raw_value(name)
        except KeyError:
            return default

    def __setitem__(self, name: str, value: Any) -> None:
        self.declare(name, value)

    def __delitem__(self, name: str) -> None:
        assert_not_cli_args_identifier(name)
        dict.__delitem__(self, name)
        self._binding_metadata.pop(name, None)

    def update(self, *args: Any, **kwargs: Any) -> None:
        pending = dict(*args, **kwargs)
        for name, value in pending.items():
            self.declare(name, value)

    def setdefault(self, name: str, default: Any = None) -> Any:
        owner = self.resolve_owner(name)
        if owner is not None:
            return self.raw_value(name)
        self.declare(name, default)
        return default

    def pop(self, name: object, *default: Any) -> Any:
        if isinstance(name, str):
            assert_not_cli_args_identifier(name)
        if len(default) > 1:
            raise TypeError(f"pop expected at most 2 arguments, got {len(default) + 1}")
        if not isinstance(name, str) or not self.owns(name):
            if default:
                return default[0]
            raise KeyError(name)
        value = dict.pop(self, name)
        self._binding_metadata.pop(name, None)
        return value

    def popitem(self) -> tuple[str, Any]:
        if not dict.__len__(self):
            raise KeyError("popitem(): dictionary is empty")
        name = next(reversed(dict.keys(self)))
        return name, self.pop(name)

    def clear(self) -> None:
        reserved = cli_args_identifier()
        if self.owns(reserved):
            raise ValueError(f"Cannot clear scope containing reserved CLI binding {reserved!r}")
        dict.clear(self)
        self._binding_metadata.clear()

    def __ior__(self, other: Any, /) -> ScopeFrame:
        self.update(other)
        return self


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
        self._binding_owners: dict[str, dict[str, Any]] = {}
        self._dirty_bindings: set[str] = set()
        # STORE_GLOBAL (``global x`` in a nested function) bypasses ``__setitem__``, so a
        # write-back is also detected by comparing against the installed value.
        self._installed_values: dict[str, Any] = {}
        if args or kwargs:
            self.update(*args, **kwargs)

    def _set_binding(self, name: str, value: Any, owner: dict[str, Any]) -> None:
        """Install a Python-visible value with its Kedi write-back owner."""
        assert_not_cli_args_identifier(name)
        dict.__setitem__(self, name, value)
        self._binding_owners[name] = owner
        self._installed_values[name] = value
        self._dirty_bindings.discard(name)

    def _set_unowned(self, name: str, value: Any) -> None:
        """Install an internal Python name that must never sync into Kedi."""
        assert_not_cli_args_identifier(name)
        dict.__setitem__(self, name, value)
        self._binding_owners.pop(name, None)
        self._dirty_bindings.discard(name)

    def binding_owners(self) -> dict[str, dict[str, Any]]:
        return dict(self._binding_owners)

    def changed_binding_owners(self) -> dict[str, dict[str, Any]]:
        return {
            name: owner
            for name, owner in self._binding_owners.items()
            if name in self._dirty_bindings
            or (name in self and dict.__getitem__(self, name) is not self._installed_values[name])
        }

    def _set_cli_args(self, value: Any) -> None:
        """Install the runtime-owned CLI binding without exposing an override path."""
        dict.__setitem__(self, cli_args_identifier(), value)

    def __setitem__(self, key: str, value: Any) -> None:
        assert_not_cli_args_identifier(key)
        dict.__setitem__(self, key, value)
        if key in self._binding_owners:
            self._dirty_bindings.add(key)

    def update(self, *args: Any, **kwargs: Any) -> None:
        pending = dict(*args, **kwargs)
        for key in pending:
            assert_not_cli_args_identifier(key)
        dict.update(self, pending)
        self._dirty_bindings.update(self._binding_owners.keys() & pending.keys())

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
        copied._binding_owners.update(self._binding_owners)
        copied._installed_values.update(self._installed_values)
        copied._dirty_bindings.update(self._dirty_bindings)
        return copied

    def __delitem__(self, key: str) -> None:
        assert_not_cli_args_identifier(key)
        dict.__delitem__(self, key)
        self._binding_owners.pop(key, None)
        self._dirty_bindings.discard(key)

    def pop(self, key: object, *default: Any) -> Any:
        if not isinstance(key, str):
            return dict.pop(self, key, *default)
        assert_not_cli_args_identifier(key)
        value = dict.pop(self, key, *default)
        self._binding_owners.pop(key, None)
        self._dirty_bindings.discard(key)
        return value

    def popitem(self) -> tuple[str, Any]:
        if cli_args_identifier() in self:
            raise ValueError("The reserved CLI argument binding cannot be removed")
        try:
            key = next(reversed(self))
        except StopIteration as exc:
            raise KeyError("popitem(): dictionary is empty") from exc
        value = dict.__getitem__(self, key)
        dict.__delitem__(self, key)
        self._binding_owners.pop(key, None)
        self._dirty_bindings.discard(key)
        return key, value

    def clear(self) -> None:
        if cli_args_identifier() in self:
            raise ValueError("The reserved CLI argument binding cannot be removed")
        dict.clear(self)
        self._binding_owners.clear()
        self._dirty_bindings.clear()

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
            "Run Kedi sequentially (KEDI_PARALLEL=0 or kedi.configure(parallel=False))."
        )


def raw_lookup_value(rt: KediRuntime, name: str, local_env: dict[str, Any]) -> Any:
    """Look up a variable without materializing promises or artifacts."""
    if isinstance(local_env, ScopeFrame):
        try:
            return local_env.raw_value(name)
        except KeyError:
            pass
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
    _, text = model_value_and_text(value)
    return text


def prompt_part(value: Any, binding: str | None = None) -> Any:
    """Render one prompt substitution: media become a mention, other values model text.

    ``binding`` names the substituted value (``<cat_image>``); media without an explicit
    name are shown to the model under it.
    """
    from ...media import MediaMention, prompt_media

    items = prompt_media(value)
    if items is None:
        return model_text(value)
    return MediaMention(items=items, binding=binding, is_list=isinstance(value, (list, tuple)))


def model_value_and_text(value: Any) -> tuple[Any, str]:
    """Resolve and render once so request text and decision identity share a snapshot."""
    contains_artifact = _contains_artifact_ref(value)
    resolved = model_value(value)
    if contains_artifact:
        try:
            return resolved, json.dumps(resolved, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError, RecursionError):
            return resolved, repr(resolved)
    return resolved, str(resolved)


def _raw_mapping_value(mapping: dict[str, Any], key: str) -> Any:
    if isinstance(mapping, ScopeFrame):
        return mapping.raw_value(key)
    if isinstance(mapping, KediEnv):
        return mapping.raw_value(key)
    return dict.__getitem__(mapping, key)


def visible_bindings(
    mapping: dict[str, Any],
) -> dict[str, tuple[dict[str, Any], Any]]:
    """Return a mapping's Python-visible values and their semantic owners."""
    if isinstance(mapping, ScopeFrame):
        return mapping.visible_bindings()
    return {name: (mapping, _raw_mapping_value(mapping, name)) for name in mapping}


def snapshot_visible_values(mapping: dict[str, Any]) -> dict[str, Any]:
    """Copy every currently visible raw value for deferred read-only execution."""
    return {name: value for name, (_, value) in visible_bindings(mapping).items()}


def write_owned_value(owner: dict[str, Any], name: str, value: Any) -> None:
    """Update a binding through its recorded owner without redeclaring it."""
    assert_not_cli_args_identifier(name)
    if isinstance(owner, ScopeFrame):
        if not owner.owns(name):
            raise RuntimeError(f"Scope frame does not own binding {name!r}")
        metadata = owner._binding_metadata.get(name)
        if metadata is not None and not metadata.assignable:
            raise ValueError(f"Kedi binding {name!r} is read-only")
        dict.__setitem__(owner, name, value)
        if metadata is not None and metadata.decision is not None:
            owner._binding_metadata[name] = dataclasses.replace(metadata, decision=None)
        return
    owner[name] = value


def validate_owned_value(
    rt: KediRuntime,
    owner: dict[str, Any],
    name: str,
    value: Any,
    local_env: dict[str, Any],
) -> Any:
    """Validate one Python write-back against its owner's Kedi type contract."""
    from .type_system import KediType

    if not isinstance(owner, ScopeFrame):
        return value
    metadata = owner._binding_metadata.get(name)
    contract = metadata.type_contract if metadata is not None else None
    if not isinstance(contract, KediType) or contract.is_none():
        return value
    try:
        return contract.validate(rt, local_env, value)
    except Exception as exc:
        raise TypeError(
            f"Python write-back for {name!r} does not match its annotated type"
        ) from exc


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
    env._set_unowned("__builtins__", builtins.__dict__)

    identifier = cli_args_identifier()
    missing = object()
    runtime_args = rt._initial_globals_env.get(identifier, missing)
    if runtime_args is not missing:
        env._set_cli_args(runtime_args)

    def merge(source: dict[str, Any], *, writable: bool) -> None:
        for name, (owner, value) in visible_bindings(source).items():
            if name == identifier:
                if value is runtime_args:
                    continue
                assert_not_cli_args_identifier(name)
            if writable:
                env._set_binding(name, value, owner)
            else:
                env._set_unowned(name, value)

    # Layer environments: imports -> prelude -> globals -> current -> local
    for source in (rt._import_env, rt._prelude_env):
        if source is not None:
            merge(source, writable=False)
    if rt._globals_env is not None:
        merge(rt._globals_env, writable=True)

    try:
        current = rt._current_env.get()
        if isinstance(current, dict):
            merge(current, writable=True)
    except Exception:
        pass

    merge(local_env, writable=True)

    # Add type environment
    _add_type_env(rt, env, local_env)

    # Wrap procedures as callables. A Kedi binding with the same name shadows the
    # procedure, as it does for `<name>` substitutions.
    bound = env.binding_owners()
    for name, fn in rt._procedures.items():
        if name not in bound:
            env._set_unowned(name, _make_proc_wrapper(rt, name, fn, local_env))

    return env


def build_type_env(rt: KediRuntime, local_env: dict[str, Any] | None) -> dict[str, Any]:
    """Build environment for type annotation evaluation."""
    env: dict[str, Any] = {"__builtins__": builtins.__dict__}

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
            env.update((name, value) for name, (_, value) in visible_bindings(source).items())

    try:
        current = rt._current_env.get()
        if isinstance(current, dict):
            env.update((name, value) for name, (_, value) in visible_bindings(current).items())
    except Exception:
        pass

    if local_env:
        env.update((name, value) for name, (_, value) in visible_bindings(local_env).items())

    return env


def _add_type_env(rt: KediRuntime, env: dict[str, Any], local_env: dict[str, Any]) -> None:
    """Add type environment and handle Pydantic model coercion."""
    if not rt._type_env_stack:
        return

    type_env: dict[str, Any] = {}
    for layer in rt._type_env_stack:
        type_env.update(layer)

    if isinstance(env, KediEnv):
        for name, value in type_env.items():
            env._set_unowned(name, value)
    else:
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
    definition_rt = getattr(proc_fn, "__kedi_definition_runtime__", rt)
    namespace = build_type_env(definition_rt, local_env if definition_rt is rt else None)
    namespace.update(metadata.default_namespace())
    namespace[metadata.call_name] = lambda *args: rt._invoke_procedure(
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
    wrapper = namespace[metadata.name]
    wrapper.__doc__ = metadata.docstring
    setattr(wrapper, "__kedi_procedure_target__", proc_fn)
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
