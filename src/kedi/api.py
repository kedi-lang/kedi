from __future__ import annotations

import asyncio
import builtins
import copy
import dataclasses
import functools
import hashlib
import inspect
import os
import re
import textwrap
from collections.abc import Mapping, Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, ParamSpec, TypeAlias, TypeVar, cast, overload

import dotenv
from typing_extensions import Sentinel

from .agent_adapter.approval import (
    ApprovalDecision,
    ApprovalHandler,
    ApprovalMode,
    ApprovalPolicy,
    ApprovalRequest,
    ToolRisk,
)
from .agent_adapter import AgentAdapter, AgentAdapterKind
from .agent_adapter.adapters.lazy import LazyAdapter
from .agent_adapter.registry import adapter_factory, build_adapter, known_adapter_shortnames
from .agent_profile import AgentProfile, McpServerSpec, block_agent_profile_scope
from .effort import EffortValue
from .errors import KediExecutionError
from .lang import compile_program, parse_program

P = ParamSpec("P")
R = TypeVar("R")
ClassT = TypeVar("ClassT", bound=builtins.type[Any])

FrameworkAdapterName = Literal["pydantic", "dspy", "langchain"]
AgentName = Literal["claude", "codex", "acp"]
AdapterLike: TypeAlias = AgentAdapter[Any] | LazyAdapter
_UNSET = Sentinel("_UNSET")
_TOOL_NAME_ATTR = "__kedi_tool_name__"
_TOOL_DESCRIPTION_ATTR = "__kedi_tool_description__"
_TOOL_RISK_ATTR = "__kedi_tool_risk__"
_TYPE_INJECT_ATTR = "__kedi_type_inject__"
_MEMORY_ADDRESS_RE = re.compile(r" at 0x[0-9a-fA-F]+")
_TYPE_REGISTRY: dict[str, dict[str, builtins.type[Any]]] = {}
_PARSE_CACHE: dict[tuple[str, str | None, str], Any] = {}
_RESPONSE_CACHE: dict[str, Any] = {}


@dataclass(frozen=True)
class _APIConfig:
    adapter: AdapterLike | None = None
    adapter_kind: AgentAdapterKind = "agent-framework"
    adapter_shortname: str = "pydantic"
    model: str | None = None
    adapter_kwargs: Mapping[str, Any] = field(default_factory=dict)
    profile: AgentProfile = field(default_factory=AgentProfile.empty)
    tools: Mapping[str, Callable[..., Any]] = field(default_factory=dict)
    env: Mapping[str, Any] = field(default_factory=dict)
    parallel: bool = False
    max_workers: int = 8


@dataclass(frozen=True)
class CacheInfo:
    parse_entries: int
    response_entries: int


_DEFAULT_CONFIG = _APIConfig()
_config_var: ContextVar[_APIConfig] = ContextVar("kedi_api_config", default=_DEFAULT_CONFIG)


def _callable_name(fn: Callable[..., Any]) -> str:
    tool_name = getattr(fn, _TOOL_NAME_ATTR, None)
    if isinstance(tool_name, str) and tool_name:
        return tool_name
    name = getattr(fn, "__name__", None)
    return name if isinstance(name, str) else builtins.type(fn).__name__


def _tool_mapping(tools: Sequence[Callable[..., Any]]) -> dict[str, Callable[..., Any]]:
    return {_callable_name(tool): tool for tool in tools}


def _is_existing_type_surface(cls: builtins.type[Any]) -> bool:
    from pydantic import BaseModel
    from pydantic.dataclasses import is_pydantic_dataclass

    return issubclass(cls, BaseModel) or is_pydantic_dataclass(cls) or dataclasses.is_dataclass(cls)


def _register_type(cls: ClassT, *, inject: bool) -> ClassT:
    if not isinstance(cls, builtins.type):
        raise TypeError("@kedi.type can only decorate classes.")

    registered = cls if _is_existing_type_surface(cls) else dataclasses.dataclass(cls)
    setattr(registered, _TYPE_INJECT_ATTR, inject)

    module_registry = _TYPE_REGISTRY.setdefault(registered.__module__, {})
    if inject:
        module_registry[registered.__name__] = registered
    else:
        module_registry.pop(registered.__name__, None)
    return cast(ClassT, registered)


def _auto_type_env(fn: Callable[..., Any]) -> dict[str, Any]:
    module = getattr(fn, "__module__", None)
    if not isinstance(module, str):
        return {}
    return dict(_TYPE_REGISTRY.get(module, {}))


def _extract_kedi_source(fn: Callable[..., Any]) -> str:
    name = _callable_name(fn)
    doc = inspect.getdoc(fn)
    if not doc:
        raise RuntimeError(f"{name} must contain a Kedi docstring.")

    lines = doc.splitlines()
    if not lines or lines[0].strip() != "kedi":
        raise RuntimeError(f"{name} docstring must start with a standalone `kedi` header.")

    source = "\n".join(lines[1:]).strip()
    if not source:
        raise RuntimeError(f"{name} has an empty Kedi program.")
    return source


def _bound_runtime_globals(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    signature = inspect.signature(fn)
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


def _wrap_query_source(source: str) -> str:
    body = textwrap.indent(source, "  ")
    return f"@__kedi_query_entry__() -> Any:\n{body}\n"


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
    return CacheInfo(
        parse_entries=len(_PARSE_CACHE),
        response_entries=len(_RESPONSE_CACHE),
    )


def clear_cache() -> None:
    _PARSE_CACHE.clear()
    _RESPONSE_CACHE.clear()


def _stable_repr(value: Any) -> str:
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
        profile.effort,
        profile.system,
        tuple((key, _stable_repr(value)) for key, value in sorted(profile.settings.items())),
        tuple(profile.tools),
        tuple(_stable_repr(server) for server in profile.mcp_servers),
        _approval_identity(profile.approval),
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
        id(handler),
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
    tool_names: Sequence[str],
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
        tuple(tool_names),
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def _agent_profile_overrides(
    *,
    model: str | None | object,
    adapter_kind: AgentAdapterKind | None,
    adapter_shortname: str | None,
    system: str | None,
    effort: EffortValue | None,
    settings: Mapping[str, Any] | None,
    tools: Sequence[Callable[..., Any]],
    mcp_servers: Sequence[McpServerSpec],
    approval_policy: ApprovalPolicy | ApprovalMode | ApprovalHandler | None,
) -> tuple[dict[str, Callable[..., Any]], AgentProfile]:
    tool_map = _tool_mapping(tools)
    return tool_map, AgentProfile(
        model=None if model is _UNSET else cast(str | None, model),
        adapter_kind=adapter_kind,
        adapter_shortname=adapter_shortname,
        effort=effort,
        system=system,
        settings=dict(settings or {}),
        tools=tuple(tool_map),
        mcp_servers=tuple(mcp_servers),
        approval=_coerce_approval(approval_policy),
    )


def _coerce_approval(
    value: ApprovalPolicy | ApprovalMode | ApprovalHandler | None,
) -> ApprovalPolicy | None:
    if value is None:
        return None
    if isinstance(value, ApprovalPolicy):
        return value
    if isinstance(value, str):
        if value == "allow":
            return ApprovalPolicy.allow()
        if value == "deny":
            return ApprovalPolicy.deny()
        raise TypeError("approval must be 'allow', 'deny', ApprovalPolicy, or an approval handler")
    if callable(value):
        return ApprovalPolicy.dynamic(value)
    raise TypeError("approval must be 'allow', 'deny', ApprovalPolicy, or an approval handler")


def _known_adapter_names(kind: AgentAdapterKind) -> tuple[str, ...]:
    return tuple(name for name in known_adapter_shortnames() if adapter_factory(name).kind == kind)


def _selection_metadata(
    value: str | AdapterLike,
    *,
    expected_kind: AgentAdapterKind,
    parameter: str,
) -> tuple[str, AgentAdapterKind, AdapterLike | None]:
    if isinstance(value, str):
        try:
            factory = adapter_factory(value)
        except KediExecutionError as exc:
            known = ", ".join(_known_adapter_names(expected_kind))
            raise ValueError(f"unknown {parameter} {value!r}; expected one of: {known}") from exc
        if factory.kind != expected_kind:
            other = "agent" if expected_kind == "agent-framework" else "adapter"
            raise ValueError(
                f"{value!r} is an {factory.kind}; pass it with {other}= instead of {parameter}="
            )
        return value, factory.kind, None

    kind = getattr(value, "kind", None)
    shortname = getattr(value, "shortname", None)
    if kind not in ("agent-framework", "agent-harness") or not isinstance(shortname, str):
        raise TypeError(
            f"{parameter} instances must implement AgentAdapter and define kind and shortname"
        )
    if kind != expected_kind:
        other = "agent" if expected_kind == "agent-framework" else "adapter"
        raise ValueError(
            f"{shortname!r} is an {kind}; pass it with {other}= instead of {parameter}="
        )
    return shortname, kind, value


def _resolve_selection(
    *,
    adapter: FrameworkAdapterName | AdapterLike | object,
    agent: AgentName | AdapterLike | object,
) -> tuple[str, AgentAdapterKind, AdapterLike | None] | None:
    if adapter is not _UNSET and agent is not _UNSET:
        raise ValueError("adapter and agent are mutually exclusive")
    if agent is not _UNSET:
        return _selection_metadata(
            cast(str | AdapterLike, agent),
            expected_kind="agent-harness",
            parameter="agent",
        )
    if adapter is not _UNSET:
        return _selection_metadata(
            cast(str | AdapterLike, adapter),
            expected_kind="agent-framework",
            parameter="adapter",
        )
    return None


def _adapter_identity(
    adapter: AdapterLike,
    *,
    kind: AgentAdapterKind,
    shortname: str,
) -> tuple[Any, ...]:
    if isinstance(adapter, LazyAdapter):
        return kind, shortname, "registry"
    adapter_type = builtins.type(adapter)
    return kind, shortname, adapter_type.__module__, adapter_type.__qualname__, id(adapter)


def _resolve_bind_file(file: str | os.PathLike[str], fn: Callable[..., Any]) -> Path:
    path = Path(file)
    if path.is_absolute():
        return path

    source_file = inspect.getsourcefile(fn)
    if source_file is None:  # pragma: no cover - interactive/session fallback.
        return path.resolve()
    return (Path(source_file).resolve().parent / path).resolve()


def _read_bind_file(file: str | os.PathLike[str], fn: Callable[..., Any]) -> tuple[str, Path]:
    path = _resolve_bind_file(file, fn)
    if not path.exists():
        raise FileNotFoundError(f"Kedi bind file not found: {path}")
    return path.read_text(encoding="utf-8"), path


def _decorate_kedi_program(
    *,
    kind: str,
    inner: Callable[P, R],
    program: Any,
    source_id: str,
    source_hash: str,
    reload_program: Callable[[], tuple[Any, str]] | None,
    tool_map: Mapping[str, Callable[..., Any]],
    env: Mapping[str, Any],
    profile_override: AgentProfile,
    adapter_override: AdapterLike | None,
    use_main: bool,
    cache: bool,
) -> Callable[P, R]:
    def run(*args: P.args, **kwargs: P.kwargs) -> R:
        config = _config_var.get()
        active_program = program
        active_source_hash = source_hash
        if reload_program is not None:
            active_program, active_source_hash = reload_program()
        profile = config.profile.merge(profile_override)
        active_adapter = adapter_override or _build_adapter(config)
        tool_namespace = {
            *config.tools,
            *tool_map,
        }  # single source for the cache key + shadow guard
        cache_key = _response_cache_key(
            kind=kind,
            inner=inner,
            source_id=source_id,
            source_hash=active_source_hash,
            args=args,
            kwargs=kwargs,
            config_env=config.env,
            env=env,
            adapter_identity=_adapter_identity(
                active_adapter,
                kind=profile.adapter_kind or config.adapter_kind,
                shortname=profile.adapter_shortname or config.adapter_shortname,
            ),
            adapter_kwargs=config.adapter_kwargs,
            profile=profile,
            tool_names=sorted(tool_namespace),
        )
        if cache and cache_key in _RESPONSE_CACHE:
            # Hand back a copy: the cache holds mutable results (dict/list/pydantic model) by value,
            # so a caller mutating a returned object must not corrupt the shared cached entry.
            return cast(R, _isolate_cached(_RESPONSE_CACHE[cache_key]))

        bound_globals = _bound_runtime_globals(inner, args, kwargs)
        tool_collisions = tool_namespace & set(bound_globals)
        if tool_collisions:
            from .errors import KediExecutionError

            raise KediExecutionError(
                f"Kedi argument name(s) shadow registered tool(s): {sorted(tool_collisions)}. "
                "Rename the parameter or the tool so the tool stays callable."
            )
        runtime_globals = {
            **config.tools,
            **tool_map,
            **bound_globals,
            **_auto_type_env(inner),
            **config.env,
            **env,
        }
        engine = None
        if config.parallel:
            from .core import shared_thread_engine

            engine = shared_thread_engine(max_workers=config.max_workers)
        runtime = compile_program(
            active_program,
            adapter=active_adapter,
            runtime_globals=runtime_globals,
            engine=engine,
        )
        with block_agent_profile_scope(profile):
            if use_main:
                result = cast(R, runtime.run_main())
            else:
                result = cast(
                    R,
                    runtime._invoke_procedure(runtime._procedures["__kedi_query_entry__"], []),
                )
        if cache:
            # Store a copy so the caller's later mutation of `result` can't reach back into the
            # cache (and so every future hit is isolated via the copy on the hit path above).
            _RESPONSE_CACHE[cache_key] = _isolate_cached(result)
        return result

    if _is_async_callable(inner):

        @functools.wraps(inner)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # run() does blocking model I/O; offload it to a worker thread so an async caller's
            # event loop keeps serving other coroutines. to_thread copies the context, so the
            # active kedi config/profile contextvars still reach run().
            return await asyncio.to_thread(run, *args, **kwargs)

        return cast(Callable[P, R], async_wrapper)

    @functools.wraps(inner)
    def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return run(*args, **kwargs)

    return sync_wrapper


def _lazy_adapter(
    shortname: str,
    model: str | None,
    adapter_kwargs: Mapping[str, Any],
) -> LazyAdapter:
    metadata = adapter_factory(shortname)

    def factory() -> AgentAdapter[Any]:
        return build_adapter(shortname, model, **dict(adapter_kwargs))

    return LazyAdapter(factory, kind=metadata.kind, shortname=metadata.shortname)


def _build_config(
    *,
    base: _APIConfig | None,
    load_env: bool,
    model: str | None | object,
    adapter: FrameworkAdapterName | AdapterLike | object,
    agent: AgentName | AdapterLike | object,
    system: str | None,
    effort: EffortValue | None,
    settings: Mapping[str, Any] | None,
    tools: Sequence[Callable[..., Any]],
    env: Mapping[str, Any] | None,
    mcp_servers: Sequence[McpServerSpec],
    approval: ApprovalPolicy | ApprovalMode | ApprovalHandler | None,
    adapter_kwargs: Mapping[str, Any],
    parallel: bool | None = None,
    max_workers: int | None = None,
) -> _APIConfig:
    if load_env:
        dotenv.load_dotenv()

    base_parallel = False if base is None else base.parallel
    base_max_workers = 8 if base is None else base.max_workers
    resolved_parallel = base_parallel if parallel is None else parallel
    resolved_max_workers = base_max_workers if max_workers is None else max_workers

    selection = _resolve_selection(adapter=adapter, agent=agent)
    selection_is_explicit = selection is not None

    if base is None:
        if selection is None:
            env_adapter = os.getenv("KEDI_ADAPTER")
            env_agent = os.getenv("KEDI_AGENT")
            if env_adapter and env_agent:
                raise ValueError("KEDI_ADAPTER and KEDI_AGENT are mutually exclusive")
            selection = _resolve_selection(
                adapter=cast(FrameworkAdapterName, env_adapter or "pydantic")
                if not env_agent
                else _UNSET,
                agent=cast(AgentName, env_agent) if env_agent else _UNSET,
            )
        assert selection is not None
        model_default = (os.getenv("KEDI_ADAPTER_MODEL") or None) if model is _UNSET else model
        base_profile = AgentProfile.empty()
        base_tools: Mapping[str, Callable[..., Any]] = {}
        base_env: Mapping[str, Any] = {}
        base_adapter_kwargs: Mapping[str, Any] = {}
    else:
        if selection is None:
            selection = (base.adapter_shortname, base.adapter_kind, base.adapter)
        model_default = base.model if model is _UNSET else model
        base_profile = base.profile
        base_tools = base.tools
        base_env = base.env
        base_adapter_kwargs = base.adapter_kwargs

    shortname, kind, selected_instance = selection
    same_selection = (
        base is not None and shortname == base.adapter_shortname and kind == base.adapter_kind
    )
    inherited_kwargs = base_adapter_kwargs if not selection_is_explicit or same_selection else {}
    merged_kwargs = {**dict(inherited_kwargs), **dict(adapter_kwargs)}
    if selected_instance is None:
        adapter_obj: AdapterLike | None = _lazy_adapter(
            shortname,
            cast(str | None, model_default),
            merged_kwargs,
        )
    else:
        adapter_obj = selected_instance

    tool_map = {**dict(base_tools), **_tool_mapping(tools)}
    merged_env = {**dict(base_env), **dict(env or {})}
    profile_patch = AgentProfile(
        model=cast(str | None, model_default),
        adapter_kind=kind,
        adapter_shortname=shortname,
        effort=effort,
        system=system,
        settings=dict(settings or {}),
        tools=tuple(_tool_mapping(tools)),
        mcp_servers=tuple(mcp_servers),
        approval=_coerce_approval(approval),
    )
    profile = base_profile.merge(profile_patch)
    return _APIConfig(
        adapter=adapter_obj,
        adapter_kind=kind,
        adapter_shortname=shortname,
        model=cast(str | None, model_default),
        adapter_kwargs=merged_kwargs,
        profile=profile,
        tools=tool_map,
        env=merged_env,
        parallel=resolved_parallel,
        max_workers=resolved_max_workers,
    )


def _build_adapter(config: _APIConfig) -> AdapterLike:
    if config.adapter is None:
        return _lazy_adapter(config.adapter_shortname, config.model, config.adapter_kwargs)
    return config.adapter


def configure(
    *,
    model: str | None | object = _UNSET,
    adapter: FrameworkAdapterName | AdapterLike | object = _UNSET,
    agent: AgentName | AdapterLike | object = _UNSET,
    system: str | None = None,
    effort: EffortValue | None = None,
    settings: Mapping[str, Any] | None = None,
    tools: Sequence[Callable[..., Any]] = (),
    env: Mapping[str, Any] | None = None,
    mcp_servers: Sequence[McpServerSpec] = (),
    approval: ApprovalPolicy | ApprovalMode | ApprovalHandler | None = None,
    parallel: bool | None = None,
    max_workers: int | None = None,
    **adapter_kwargs: Any,
) -> None:
    """Set the default Python API configuration for the current context.

    ``parallel=True`` runs independent template calls concurrently (promise pipelining); it is
    opt-in. ``max_workers`` bounds the shared worker pool.
    """
    _config_var.set(
        _build_config(
            base=None,
            load_env=True,
            model=model,
            adapter=adapter,
            agent=agent,
            system=system,
            effort=effort,
            settings=settings,
            tools=tools,
            env=env,
            mcp_servers=mcp_servers,
            approval=approval,
            adapter_kwargs=adapter_kwargs,
            parallel=parallel,
            max_workers=max_workers,
        )
    )


def reset_config() -> None:
    _config_var.set(_DEFAULT_CONFIG)


def approval(handler: ApprovalHandler) -> ApprovalHandler:
    """Register ``handler`` as the current Python API's default approval policy.

    Explicit ``approval=`` values on :func:`query`, :func:`bind`, or
    :func:`context` take precedence for their own scope.
    """

    policy = ApprovalPolicy.dynamic(handler)
    config = _config_var.get()
    _config_var.set(
        dataclasses.replace(
            config,
            profile=config.profile.merge(AgentProfile(approval=policy)),
        )
    )
    return handler


class _KediContext:
    def __init__(self, config: _APIConfig) -> None:
        self._config = config
        self._token: Token[_APIConfig] | None = None

    def __enter__(self) -> _KediContext:
        self._token = _config_var.set(self._config)
        return self

    def __exit__(self, *args: object) -> None:
        if self._token is None:
            return
        _config_var.reset(self._token)
        self._token = None

    async def __aenter__(self) -> _KediContext:
        return self.__enter__()

    async def __aexit__(self, *args: object) -> None:
        self.__exit__(*args)


def context(
    *,
    model: str | None | object = _UNSET,
    adapter: FrameworkAdapterName | AdapterLike | object = _UNSET,
    agent: AgentName | AdapterLike | object = _UNSET,
    system: str | None = None,
    effort: EffortValue | None = None,
    settings: Mapping[str, Any] | None = None,
    tools: Sequence[Callable[..., Any]] = (),
    env: Mapping[str, Any] | None = None,
    mcp_servers: Sequence[McpServerSpec] = (),
    approval: ApprovalPolicy | ApprovalMode | ApprovalHandler | None = None,
    parallel: bool | None = None,
    max_workers: int | None = None,
    **adapter_kwargs: Any,
) -> _KediContext:
    return _KediContext(
        _build_config(
            base=_config_var.get(),
            load_env=False,
            model=model,
            adapter=adapter,
            agent=agent,
            system=system,
            effort=effort,
            settings=settings,
            tools=tools,
            env=env,
            mcp_servers=mcp_servers,
            approval=approval,
            adapter_kwargs=adapter_kwargs,
            parallel=parallel,
            max_workers=max_workers,
        )
    )


def parallel(*, max_workers: int = 8) -> _KediContext:
    """Context manager enabling concurrent (pipelined) template execution within its scope.

    Sugar for ``context(parallel=True, max_workers=...)``::

        with kedi.parallel():
            result = my_query(...)
    """
    return context(parallel=True, max_workers=max_workers)


@overload
def tool(fn: Callable[P, R]) -> Callable[P, R]: ...


@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    retries: int = 0,
    risk: ToolRisk = "mutating",
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def tool(
    fn: Callable[P, R] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    retries: int = 0,
    risk: ToolRisk = "mutating",
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    if retries < 0:
        raise ValueError("tool retries must be >= 0")
    if risk not in ("read_only", "mutating", "sensitive"):
        raise ValueError("tool risk must be 'read_only', 'mutating', or 'sensitive'")

    def decorate(inner: Callable[P, R]) -> Callable[P, R]:
        tool_name = name or _callable_name(inner)
        tool_description = description or inspect.getdoc(inner)

        if _is_async_callable(inner):

            @functools.wraps(inner)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                attempts = retries + 1
                last_error: Exception | None = None
                for _ in range(attempts):
                    try:
                        return await cast(Any, inner)(*args, **kwargs)
                    except Exception as exc:
                        last_error = exc
                assert last_error is not None
                raise last_error

            wrapped = cast(Callable[P, R], async_wrapper)
        else:

            @functools.wraps(inner)
            def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                attempts = retries + 1
                last_error: Exception | None = None
                for _ in range(attempts):
                    try:
                        return inner(*args, **kwargs)
                    except Exception as exc:
                        last_error = exc
                assert last_error is not None
                raise last_error

            wrapped = sync_wrapper

        setattr(wrapped, _TOOL_NAME_ATTR, tool_name)
        setattr(wrapped, _TOOL_RISK_ATTR, risk)
        if tool_description is not None:
            setattr(wrapped, _TOOL_DESCRIPTION_ATTR, tool_description)
            wrapped.__doc__ = tool_description
        return wrapped

    if fn is not None:
        return decorate(fn)
    return decorate


def _is_async_callable(fn: Callable[..., Any]) -> bool:
    return inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(
        getattr(fn, "__call__", None)
    )


@overload
def type(cls: ClassT) -> ClassT: ...


@overload
def type(*, inject: bool = True) -> Callable[[ClassT], ClassT]: ...


def type(
    cls: ClassT | None = None,
    *,
    inject: bool = True,
) -> ClassT | Callable[[ClassT], ClassT]:
    def decorate(inner: ClassT) -> ClassT:
        return _register_type(inner, inject=inject)

    if cls is not None:
        return decorate(cls)
    return decorate


@overload
def query(fn: Callable[P, R]) -> Callable[P, R]: ...


@overload
def query(
    *,
    model: str | None | object = _UNSET,
    adapter: FrameworkAdapterName | AdapterLike | object = _UNSET,
    agent: AgentName | AdapterLike | object = _UNSET,
    system: str | None = None,
    effort: EffortValue | None = None,
    settings: Mapping[str, Any] | None = None,
    tools: Sequence[Callable[..., Any]] = (),
    env: Mapping[str, Any] | None = None,
    mcp_servers: Sequence[McpServerSpec] = (),
    approval: ApprovalPolicy | ApprovalMode | ApprovalHandler | None = None,
    cache: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def query(
    fn: Callable[P, R] | None = None,
    *,
    model: str | None | object = _UNSET,
    adapter: FrameworkAdapterName | AdapterLike | object = _UNSET,
    agent: AgentName | AdapterLike | object = _UNSET,
    system: str | None = None,
    effort: EffortValue | None = None,
    settings: Mapping[str, Any] | None = None,
    tools: Sequence[Callable[..., Any]] = (),
    env: Mapping[str, Any] | None = None,
    mcp_servers: Sequence[McpServerSpec] = (),
    approval: ApprovalPolicy | ApprovalMode | ApprovalHandler | None = None,
    cache: bool = False,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    selection = _resolve_selection(adapter=adapter, agent=agent)
    adapter_shortname, adapter_kind, adapter_override = (
        selection if selection is not None else (None, None, None)
    )
    tool_map, profile_override = _agent_profile_overrides(
        model=model,
        adapter_kind=adapter_kind,
        adapter_shortname=adapter_shortname,
        system=system,
        effort=effort,
        settings=settings,
        tools=tools,
        mcp_servers=mcp_servers,
        approval_policy=approval,
    )

    def decorate(inner: Callable[P, R]) -> Callable[P, R]:
        source = _extract_kedi_source(inner)
        wrapped_source = _wrap_query_source(source)
        program, digest = _parse_cached(kind="query", source=wrapped_source)
        return _decorate_kedi_program(
            kind="query",
            inner=inner,
            program=program,
            source_id=_callable_name(inner),
            source_hash=digest,
            reload_program=None,
            tool_map=tool_map,
            env=dict(env or {}),
            profile_override=profile_override,
            adapter_override=adapter_override,
            use_main=False,
            cache=cache,
        )

    if fn is not None:
        return decorate(fn)
    return decorate


def bind(
    *,
    file: str | os.PathLike[str],
    model: str | None | object = _UNSET,
    adapter: FrameworkAdapterName | AdapterLike | object = _UNSET,
    agent: AgentName | AdapterLike | object = _UNSET,
    system: str | None = None,
    effort: EffortValue | None = None,
    settings: Mapping[str, Any] | None = None,
    tools: Sequence[Callable[..., Any]] = (),
    env: Mapping[str, Any] | None = None,
    mcp_servers: Sequence[McpServerSpec] = (),
    approval: ApprovalPolicy | ApprovalMode | ApprovalHandler | None = None,
    cache: bool = False,
    reload: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    selection = _resolve_selection(adapter=adapter, agent=agent)
    adapter_shortname, adapter_kind, adapter_override = (
        selection if selection is not None else (None, None, None)
    )
    tool_map, profile_override = _agent_profile_overrides(
        model=model,
        adapter_kind=adapter_kind,
        adapter_shortname=adapter_shortname,
        system=system,
        effort=effort,
        settings=settings,
        tools=tools,
        mcp_servers=mcp_servers,
        approval_policy=approval,
    )

    def decorate(inner: Callable[P, R]) -> Callable[P, R]:
        source, path = _read_bind_file(file, inner)
        source_path = str(path)
        program, digest = _parse_cached(kind="bind", source=source, source_path=source_path)

        def reload_bound_program() -> tuple[Any, str]:
            reloaded_source, _ = _read_bind_file(path, inner)
            return _parse_cached(kind="bind", source=reloaded_source, source_path=source_path)

        return _decorate_kedi_program(
            kind="bind",
            inner=inner,
            program=program,
            source_id=source_path,
            source_hash=digest,
            reload_program=reload_bound_program if reload else None,
            tool_map=tool_map,
            env=dict(env or {}),
            profile_override=profile_override,
            adapter_override=adapter_override,
            use_main=True,
            cache=cache,
        )

    return decorate


__all__ = [
    "AdapterLike",
    "ApprovalDecision",
    "ApprovalHandler",
    "ApprovalMode",
    "ApprovalPolicy",
    "ApprovalRequest",
    "AgentName",
    "CacheInfo",
    "FrameworkAdapterName",
    "bind",
    "approval",
    "cache_info",
    "clear_cache",
    "configure",
    "context",
    "parallel",
    "query",
    "reset_config",
    "tool",
    "type",
]
