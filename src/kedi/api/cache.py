from __future__ import annotations

import asyncio
import builtins
import copy
import hashlib
import re
import weakref
from collections.abc import Mapping
from concurrent.futures import Future
from dataclasses import dataclass
from itertools import count
from threading import Lock, RLock, get_ident
from typing import (
    Any,
    Callable,
)

from kedi.agent_adapter import AgentAdapterKind
from kedi.agent_adapter.adapters.lazy import LazyAdapter
from kedi.agent_adapter.approvals.policy import ApprovalPolicy
from kedi.agent_adapter.hooks.models import HOOK_EVENT_NAMES
from kedi.agent_profile import AgentProfile, HookSettings
from kedi.api.registration import _callable_name
from kedi.api.state import AdapterLike
from kedi.lang import parse_program

_MEMORY_ADDRESS_RE = re.compile(r" at 0x[0-9a-fA-F]+")


_PARSE_CACHE: dict[tuple[str, str | None, str], Any] = {}


_RESPONSE_CACHE: dict[str, Any] = {}


_RESPONSE_CACHE_LOCK = Lock()


_RESPONSE_CACHE_GENERATION = 0


_RESPONSE_IN_FLIGHT: dict[str, _ResponseFlight] = {}


_CACHE_MISS = object()
_IDENTITY_LOCK = RLock()
_IDENTITY_SERIAL = count()
_IDENTITIES: dict[int, tuple[Callable[[], object | None], int]] = {}


def _object_identity(value: object) -> int:
    """Assign process-local tokens without confusing a recycled Python object address."""
    address = id(value)
    with _IDENTITY_LOCK:
        previous = _IDENTITIES.get(address)
        if previous is not None and previous[0]() is value:
            return previous[1]
        serial = next(_IDENTITY_SERIAL)

        def forget(reference: weakref.ReferenceType[object]) -> None:
            with _IDENTITY_LOCK:
                current = _IDENTITIES.get(address)
                if current is not None and current[0] is reference:
                    _IDENTITIES.pop(address)

        try:
            reference = weakref.ref(value, forget)
        except TypeError:
            # Slot-only callables cannot be weak-referenced. Retain them until clear_cache.
            _IDENTITIES[address] = (lambda: value, serial)
        else:
            _IDENTITIES[address] = (reference, serial)
        return serial


@dataclass(frozen=True)
class _ResponseFlight:
    future: Future[Any]
    owner_thread_id: int
    generation: int


@dataclass(frozen=True)
class CacheInfo:
    parse_entries: int
    response_entries: int


def _source_hash(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _parse_cached(
    *,
    kind: str,
    source: str,
    source_path: str | None = None,
) -> tuple[Any, str]:
    digest = _source_hash(source)
    key = (kind, source_path, digest)
    program = _PARSE_CACHE.get(key)
    if program is None:
        program = parse_program(source, source_path=source_path)
        _PARSE_CACHE[key] = program
    return program, digest


def cache_info() -> CacheInfo:
    with _RESPONSE_CACHE_LOCK:
        response_entries = len(_RESPONSE_CACHE)
    return CacheInfo(parse_entries=len(_PARSE_CACHE), response_entries=response_entries)


def clear_cache() -> None:
    global _RESPONSE_CACHE_GENERATION
    _PARSE_CACHE.clear()
    with _RESPONSE_CACHE_LOCK:
        _RESPONSE_CACHE.clear()
        _RESPONSE_CACHE_GENERATION += 1
    with _IDENTITY_LOCK:
        _IDENTITIES.clear()


def _stable_repr(value: Any) -> str:
    if callable(value):
        return f"callable:{_object_identity(value)}"
    return _MEMORY_ADDRESS_RE.sub(" at 0x<addr>", repr(value))


def _isolate_cached(value: Any) -> Any:
    """Deep-copy a cached value so callers can't mutate the shared entry.

    Falls back to the value itself when it isn't deep-copyable (e.g. it wraps a file handle, lock,
    or DB connection) — that restores stable's by-reference behavior for those exotic results
    instead of turning a previously-working ``cache=True`` call into a crash. JSON / pydantic /
    dict / list results copy cleanly and stay fully isolated.
    """
    try:
        return copy.deepcopy(value)
    except Exception:  # noqa: BLE001 - non-copyable result: degrade to by-reference, don't crash
        return value


def _profile_identity(profile: AgentProfile) -> tuple[Any, ...]:
    """Everything about a resolved profile that changes the model's answer."""
    return (
        profile.model,
        profile.adapter_kind,
        profile.adapter_shortname,
        _stable_repr(profile.agent_command),
        profile.a2a_connection.identity() if profile.a2a_connection is not None else None,
        profile.effort,
        profile.system,
        profile.capability_instructions,
        tuple(
            (
                cap.identity,
                cap.deferred,
                cap.instructions_enabled,
                cap.instructions_restored,
                _profile_identity(cap.profile),
            )
            for cap in profile.capabilities
        ),
        tuple((key, _stable_repr(value)) for key, value in sorted(profile.settings.items())),
        tuple(profile.tools),
        tuple(
            (name, _object_identity(spec)) for name, spec in sorted(profile.tool_bindings.items())
        ),
        tuple(_stable_repr(server) for server in profile.mcp_servers),
        _approval_identity(profile.approval),
        _hooks_identity(profile.hooks),
        _hooks_identity(profile._hook_enforcement),
        profile.skills,
        profile.codemode,
        profile.subagents,
        tuple(
            (name, _object_identity(binding))
            for name, binding in sorted(profile.subagent_bindings.items())
        ),
        profile.max_agents,
        profile.workflow_mode,
        profile.workflow_tools,
        profile.history,
        (
            _object_identity(profile.history_processing.processor),
            _object_identity(profile.history_processing.condition),
            profile.history_processing.inherit_processor,
        )
        if profile.history_processing is not None
        else None,
        _stable_repr(profile.compaction),
        profile.requirements,
        _stable_repr(profile.output_type),
        profile.run_budget,
        profile.artifacts.as_cache_key() if profile.artifacts is not None else None,
    )


def _hooks_identity(settings: HookSettings | None) -> tuple[Any, ...] | None:
    if settings is None:
        return None
    return (
        settings.enabled,
        tuple(
            (
                event,
                tuple(
                    (
                        getattr(handler, "__module__", None),
                        getattr(handler, "__qualname__", None),
                        _object_identity(handler),
                    )
                    for handler in (getattr(settings, event) or ())
                ),
            )
            for event in HOOK_EVENT_NAMES
        ),
    )


def _approval_identity(approval_policy: ApprovalPolicy | None) -> tuple[Any, ...] | None:
    if approval_policy is None:
        return None
    handler = approval_policy.handler
    if handler is None:
        return approval_policy.mode, None
    return (
        approval_policy.mode,
        getattr(handler, "__module__", None),
        getattr(handler, "__qualname__", None),
        _object_identity(handler),
    )


def _response_cache_key(
    *,
    kind: str,
    inner: Callable[..., Any],
    source_id: str,
    source_hash: str,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    config_env: Mapping[str, Any],
    env: Mapping[str, Any],
    adapter_identity: tuple[Any, ...],
    adapter_kwargs: Mapping[str, Any],
    profile: AgentProfile,
    tools: Mapping[str, Callable[..., Any]],
) -> str:
    payload = (
        kind,
        getattr(inner, "__module__", ""),
        _callable_name(inner),
        source_id,
        source_hash,
        tuple(_stable_repr(arg) for arg in args),
        tuple((key, _stable_repr(value)) for key, value in sorted(kwargs.items())),
        tuple((key, _stable_repr(value)) for key, value in sorted(config_env.items())),
        tuple((key, _stable_repr(value)) for key, value in sorted(env.items())),
        # The resolved backend + profile + tools decide the answer; omitting them returns a stale
        # cached response across kedi.context(model=...)/configure(...)/different system or tools.
        adapter_identity,
        tuple((key, _stable_repr(value)) for key, value in sorted(adapter_kwargs.items())),
        _profile_identity(profile),
        tuple((name, _object_identity(tool)) for name, tool in sorted(tools.items())),
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def _adapter_identity(
    adapter: AdapterLike,
    *,
    kind: AgentAdapterKind,
    shortname: str,
) -> tuple[Any, ...]:
    if isinstance(adapter, LazyAdapter):
        return kind, shortname, "registry"
    adapter_type = builtins.type(adapter)
    return (
        kind,
        shortname,
        adapter_type.__module__,
        adapter_type.__qualname__,
        _object_identity(adapter),
    )


def _claim_response_cache_key(
    cache_key: str,
) -> tuple[Any, _ResponseFlight | None, bool]:
    """Return a cache hit or atomically claim the one producer slot for ``cache_key``."""

    with _RESPONSE_CACHE_LOCK:
        cached = _RESPONSE_CACHE.get(cache_key, _CACHE_MISS)
        if cached is not _CACHE_MISS:
            return cached, None, False
        generation = _RESPONSE_CACHE_GENERATION
        flight = _RESPONSE_IN_FLIGHT.get(cache_key)
        if flight is not None and flight.generation == generation:
            if flight.owner_thread_id == get_ident():
                raise RuntimeError(f"Recursive Kedi response cache request for key {cache_key!r}")
            return _CACHE_MISS, flight, False
        flight = _ResponseFlight(
            future=Future(),
            owner_thread_id=get_ident(),
            generation=generation,
        )
        _RESPONSE_IN_FLIGHT[cache_key] = flight
        return _CACHE_MISS, flight, True


def _finish_response_flight(
    cache_key: str,
    flight: _ResponseFlight,
    *,
    result: Any = _CACHE_MISS,
    error: BaseException | None = None,
) -> None:
    if error is None:
        stored_result = _isolate_cached(result)
    else:
        stored_result = _CACHE_MISS
    with _RESPONSE_CACHE_LOCK:
        if error is None and flight.generation == _RESPONSE_CACHE_GENERATION:
            _RESPONSE_CACHE[cache_key] = stored_result
        if _RESPONSE_IN_FLIGHT.get(cache_key) is flight:
            _RESPONSE_IN_FLIGHT.pop(cache_key)
    if error is None:
        flight.future.set_result(stored_result)
    else:
        flight.future.set_exception(error)


def _wait_response_flight(flight: _ResponseFlight) -> Any:
    from kedi.agent_adapter.execution import run_sync

    async def wait() -> Any:
        # A cancelled consumer must not cancel the shared producer's result.
        return await asyncio.shield(asyncio.wrap_future(flight.future))

    return run_sync(wait())
