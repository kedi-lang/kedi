from __future__ import annotations

import functools
import inspect
import os
import textwrap
from collections.abc import Mapping, Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, ParamSpec, TypeVar, cast, overload

import dotenv
from typing_extensions import Sentinel

from .agent_adapter import AgentAdapter
from .agent_adapter.lazy import LazyAdapter
from .agent_profile import AgentProfile, McpServerSpec, block_agent_profile_scope
from .effort import EffortValue
from .lang import compile_program, parse_program

P = ParamSpec("P")
R = TypeVar("R")

AdapterName = Literal["pydantic", "dspy"]
_UNSET = Sentinel("_UNSET")
_TOOL_NAME_ATTR = "__kedi_tool_name__"
_TOOL_DESCRIPTION_ATTR = "__kedi_tool_description__"


@dataclass(frozen=True)
class _APIConfig:
    adapter: AgentAdapter[Any] | LazyAdapter | None = None
    adapter_name: AdapterName = "pydantic"
    model: str | None = None
    adapter_kwargs: Mapping[str, Any] = field(default_factory=dict)
    profile: AgentProfile = field(default_factory=AgentProfile.empty)
    tools: Mapping[str, Callable[..., Any]] = field(default_factory=dict)


_DEFAULT_CONFIG = _APIConfig()
_config_var: ContextVar[_APIConfig] = ContextVar("kedi_api_config", default=_DEFAULT_CONFIG)


def _callable_name(fn: Callable[..., Any]) -> str:
    tool_name = getattr(fn, _TOOL_NAME_ATTR, None)
    if isinstance(tool_name, str) and tool_name:
        return tool_name
    name = getattr(fn, "__name__", None)
    return name if isinstance(name, str) else type(fn).__name__


def _tool_mapping(tools: Sequence[Callable[..., Any]]) -> dict[str, Callable[..., Any]]:
    return {_callable_name(tool): tool for tool in tools}


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


def _lazy_adapter(adapter_name: AdapterName, model: str | None, adapter_kwargs: Mapping[str, Any]):
    def factory() -> AgentAdapter[Any]:
        if adapter_name == "pydantic":
            from .agent_adapter.adapters import PydanticAdapter

            return PydanticAdapter(model, **dict(adapter_kwargs))

        if model is None:
            raise ValueError("DSPy adapter requires an explicit model.")

        from .agent_adapter.adapters import DSPyAdapter

        return DSPyAdapter(model, **dict(adapter_kwargs))

    return LazyAdapter(factory)


def _build_config(
    *,
    base: _APIConfig | None,
    load_env: bool,
    model: str | None | object,
    adapter: AdapterName | AgentAdapter[Any] | LazyAdapter | object,
    system: str | None,
    effort: EffortValue | None,
    settings: Mapping[str, Any] | None,
    tools: Sequence[Callable[..., Any]],
    mcp_servers: Sequence[McpServerSpec],
    adapter_kwargs: Mapping[str, Any],
) -> _APIConfig:
    if load_env:
        dotenv.load_dotenv()

    if base is None:
        adapter_default: AdapterName | AgentAdapter[Any] | LazyAdapter = (
            cast(AdapterName, os.getenv("KEDI_ADAPTER") or "pydantic")
            if adapter is _UNSET
            else cast(AdapterName | AgentAdapter[Any] | LazyAdapter, adapter)
        )
        model_default = os.getenv("KEDI_ADAPTER_MODEL") or None if model is _UNSET else model
        base_profile = AgentProfile.empty()
        base_tools: Mapping[str, Callable[..., Any]] = {}
        base_adapter_kwargs: Mapping[str, Any] = {}
    else:
        adapter_default = (
            (base.adapter if base.adapter is not None else base.adapter_name)
            if adapter is _UNSET
            else cast(AdapterName | AgentAdapter[Any] | LazyAdapter, adapter)
        )
        model_default = base.model if model is _UNSET else model
        base_profile = base.profile
        base_tools = base.tools
        base_adapter_kwargs = base.adapter_kwargs

    adapter_obj: AgentAdapter[Any] | LazyAdapter | None
    adapter_name: AdapterName
    if isinstance(adapter_default, str):
        if adapter_default not in ("pydantic", "dspy"):
            raise ValueError("adapter must be 'pydantic', 'dspy', or an AgentAdapter instance")
        adapter_name = cast(AdapterName, adapter_default)
        merged_kwargs = {**dict(base_adapter_kwargs), **dict(adapter_kwargs)}
        adapter_obj = _lazy_adapter(adapter_name, cast(str | None, model_default), merged_kwargs)
    else:
        adapter_obj = adapter_default
        adapter_name = "pydantic"
        merged_kwargs = dict(adapter_kwargs)

    tool_map = {**dict(base_tools), **_tool_mapping(tools)}
    profile_patch = AgentProfile(
        model=cast(str | None, model_default),
        effort=effort,
        system=system,
        settings=dict(settings or {}),
        tools=tuple(_tool_mapping(tools)),
        mcp_servers=tuple(mcp_servers),
    )
    profile = base_profile.merge(profile_patch)
    return _APIConfig(
        adapter=adapter_obj,
        adapter_name=adapter_name,
        model=cast(str | None, model_default),
        adapter_kwargs=merged_kwargs,
        profile=profile,
        tools=tool_map,
    )


def _build_adapter(config: _APIConfig) -> AgentAdapter[Any] | LazyAdapter:
    if config.adapter is None:
        return _lazy_adapter(config.adapter_name, config.model, config.adapter_kwargs)
    return config.adapter


def configure(
    *,
    model: str | None | object = _UNSET,
    adapter: AdapterName | AgentAdapter[Any] | LazyAdapter | object = _UNSET,
    system: str | None = None,
    effort: EffortValue | None = None,
    settings: Mapping[str, Any] | None = None,
    tools: Sequence[Callable[..., Any]] = (),
    mcp_servers: Sequence[McpServerSpec] = (),
    **adapter_kwargs: Any,
) -> None:
    """Set the default Python API configuration for the current context."""
    _config_var.set(
        _build_config(
            base=None,
            load_env=True,
            model=model,
            adapter=adapter,
            system=system,
            effort=effort,
            settings=settings,
            tools=tools,
            mcp_servers=mcp_servers,
            adapter_kwargs=adapter_kwargs,
        )
    )


def reset_config() -> None:
    _config_var.set(_DEFAULT_CONFIG)


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
    adapter: AdapterName | AgentAdapter[Any] | LazyAdapter | object = _UNSET,
    system: str | None = None,
    effort: EffortValue | None = None,
    settings: Mapping[str, Any] | None = None,
    tools: Sequence[Callable[..., Any]] = (),
    mcp_servers: Sequence[McpServerSpec] = (),
    **adapter_kwargs: Any,
) -> _KediContext:
    return _KediContext(
        _build_config(
            base=_config_var.get(),
            load_env=False,
            model=model,
            adapter=adapter,
            system=system,
            effort=effort,
            settings=settings,
            tools=tools,
            mcp_servers=mcp_servers,
            adapter_kwargs=adapter_kwargs,
        )
    )


@overload
def tool(fn: Callable[P, R]) -> Callable[P, R]: ...


@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    retries: int = 0,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def tool(
    fn: Callable[P, R] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    retries: int = 0,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    if retries < 0:
        raise ValueError("tool retries must be >= 0")

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
def query(fn: Callable[P, R]) -> Callable[P, R]: ...


@overload
def query(
    *,
    system: str | None = None,
    effort: EffortValue | None = None,
    settings: Mapping[str, Any] | None = None,
    tools: Sequence[Callable[..., Any]] = (),
    use: Sequence[Callable[..., Any]] = (),
    mcp_servers: Sequence[McpServerSpec] = (),
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def query(
    fn: Callable[P, R] | None = None,
    *,
    system: str | None = None,
    effort: EffortValue | None = None,
    settings: Mapping[str, Any] | None = None,
    tools: Sequence[Callable[..., Any]] = (),
    use: Sequence[Callable[..., Any]] = (),
    mcp_servers: Sequence[McpServerSpec] = (),
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    query_tools = tuple(tools) + tuple(use)
    query_tool_map = _tool_mapping(query_tools)
    query_profile = AgentProfile(
        effort=effort,
        system=system,
        settings=dict(settings or {}),
        tools=tuple(query_tool_map),
        mcp_servers=tuple(mcp_servers),
    )

    def decorate(inner: Callable[P, R]) -> Callable[P, R]:
        source = _extract_kedi_source(inner)
        program = parse_program(_wrap_query_source(source))

        def run(*args: P.args, **kwargs: P.kwargs) -> R:
            config = _config_var.get()
            adapter = _build_adapter(config)
            runtime_globals = {
                **config.tools,
                **query_tool_map,
                **_bound_runtime_globals(inner, args, kwargs),
            }
            profile = config.profile.merge(query_profile)
            runtime = compile_program(
                program,
                adapter=adapter,
                runtime_globals=runtime_globals,
            )
            with block_agent_profile_scope(profile):
                return cast(
                    R,
                    runtime._invoke_procedure(runtime._procedures["__kedi_query_entry__"], []),
                )

        if _is_async_callable(inner):

            @functools.wraps(inner)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                return run(*args, **kwargs)

            return cast(Callable[P, R], async_wrapper)

        @functools.wraps(inner)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return run(*args, **kwargs)

        return sync_wrapper

    if fn is not None:
        return decorate(fn)
    return decorate


__all__ = ["configure", "context", "query", "reset_config", "tool"]
