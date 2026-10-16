from __future__ import annotations

import builtins
import functools
import inspect
import os
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import (
    Any,
    Callable,
    cast,
    overload,
)

from kedi.agent_adapter import AdapterCapability, ConversationState
from kedi.agent_adapter.approvals.policy import ApprovalHandler, ApprovalMode, ApprovalPolicy
from kedi.agent_adapter.execution import run_in_worker
from kedi.agent_profile import (
    AgentProfile,
    HookInput,
    HookSettings,
    McpServerSpec,
    SkillsSettings,
    block_agent_profile_scope,
)
from kedi.api.cache import (
    _CACHE_MISS,
    _adapter_identity,
    _claim_response_cache_key,
    _finish_response_flight,
    _isolate_cached,
    _parse_cached,
    _response_cache_key,
    _ResponseFlight,
)
from kedi.api.configuration import (
    _agent_profile_overrides,
    _build_adapter,
    _execution_engine,
    _resolve_selection,
)
from kedi.api.registration import _auto_type_env, _callable_name, _is_async_callable
from kedi.api.state import _UNSET, AdapterLike, AgentName, FrameworkAdapterName, P, R, _config_var
from kedi.artifacts import ArtifactPolicyInput
from kedi.effort import EffortValue
from kedi.lang import compile_program
from kedi.telemetry import capture_enabled, current_span
from kedi.telemetry.naming import program_name, qualified_type_name


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
    conversation_override: ConversationState | None | object,
) -> Callable[P, R]:
    from kedi.explain import _InspectionSource

    inspection_source = _InspectionSource(
        program=program,
        kind=kind,
        reload_program=reload_program,
        profile=profile_override,
        adapter_shortname=profile_override.adapter_shortname,
        tools=dict(tool_map),
    )

    def _run(*args: P.args, **kwargs: P.kwargs) -> R:
        config = _config_var.get()
        active_program = program
        active_source_hash = source_hash
        if reload_program is not None:
            active_program, active_source_hash = reload_program()
        profile = config.profile.merge(profile_override)
        active_adapter = adapter_override or _build_adapter(config)
        current_span().set_attribute("kedi.adapter.name", active_adapter.shortname)
        conversation = (
            config.conversation
            if conversation_override is _UNSET
            else cast(ConversationState | None, conversation_override)
        )
        if cache and conversation is not None:
            raise ValueError("Kedi response caching cannot be combined with a conversation session")
        tool_namespace = {
            *config.tools,
            *tool_map,
        }  # single source for the cache key + shadow guard
        cache_key = (
            _response_cache_key(
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
                tools={**config.tools, **tool_map},
            )
            if cache
            else ""
        )
        bound_globals = _bound_runtime_globals(inner, args, kwargs)
        tool_collisions = tool_namespace & set(bound_globals)
        if tool_collisions:
            from kedi.errors import KediExecutionError

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
        cache_flight: _ResponseFlight | None = None
        if cache:
            cached, cache_flight, owns_flight = _claim_response_cache_key(cache_key)
            if cached is not _CACHE_MISS:
                current_span().add_event(
                    "cache hit",
                    {"kedi.cache.kind": "response"},
                )
                return cast(R, _isolate_cached(cached))
            current_span().add_event(
                "cache miss",
                {"kedi.cache.kind": "response"},
            )
            if not owns_flight:
                assert cache_flight is not None
                from kedi.api.cache import _wait_response_flight

                return cast(R, _isolate_cached(_wait_response_flight(cache_flight)))
        try:
            runtime = compile_program(
                active_program,
                adapter=active_adapter,
                runtime_globals=runtime_globals,
                engine=_execution_engine(config),
                default_agent_profile=profile,
                conversation_state=conversation,
                loop_iteration_limit=config.loop_iteration_limit,
            )
            with block_agent_profile_scope(profile):
                if use_main:
                    result = cast(R, runtime.run_main())
                else:
                    result = cast(
                        R,
                        runtime._invoke_procedure(runtime._procedures["__kedi_query_entry__"], []),
                    )
        except BaseException as exc:
            if cache_flight is not None:
                _finish_response_flight(cache_key, cache_flight, error=exc)
            raise
        if cache_flight is not None:
            _finish_response_flight(cache_key, cache_flight, result=result)
        return result

    def run(*args: P.args, **kwargs: P.kwargs) -> R:
        display_name = program_name(source_id, fallback=_callable_name(inner))
        attributes: dict[str, bool | str] = {
            "kedi.program.name": display_name,
            "kedi.program.source_type": "file" if kind == "bind" else "python_api",
            "kedi.runtime.entrypoint": kind,
            "kedi.runtime.cached_parse": True,
            "kedi.runtime.cached_compile": False,
        }
        if kind == "bind" and capture_enabled("source_path"):
            attributes["code.file.path"] = source_id
        from kedi.telemetry import runtime_span

        source_type = "file" if kind == "bind" else "python_api"
        with runtime_span(
            f"kedi run {display_name}",
            "run_program",
            attributes=attributes,
            metric_attributes={"entrypoint": kind, "source_type": source_type},
        ) as active:
            try:
                result = _run(*args, **kwargs)
            except BaseException as exc:
                active.set_attribute("error.type", builtins.type(exc).__name__)
                raise
            active.set_attribute("kedi.result.type", qualified_type_name(result))
            return result

    if _is_async_callable(inner):

        @functools.wraps(inner)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return await run_in_worker(run, *args, **kwargs)

        setattr(async_wrapper, "__kedi_inspection_source__", inspection_source)
        return cast(Callable[P, R], async_wrapper)

    @functools.wraps(inner)
    def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return run(*args, **kwargs)

    setattr(sync_wrapper, "__kedi_inspection_source__", inspection_source)
    return sync_wrapper


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
    requires: Sequence[AdapterCapability] = (),
    tools: Sequence[Callable[..., Any]] = (),
    env: Mapping[str, Any] | None = None,
    mcp_servers: Sequence[McpServerSpec] = (),
    approval: ApprovalPolicy | ApprovalMode | ApprovalHandler | None = None,
    hooks: HookSettings | HookInput = None,
    skills: SkillsSettings | bool | None = None,
    artifacts: ArtifactPolicyInput = None,
    conversation: ConversationState | None | object = _UNSET,
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
    requires: Sequence[AdapterCapability] = (),
    tools: Sequence[Callable[..., Any]] = (),
    env: Mapping[str, Any] | None = None,
    mcp_servers: Sequence[McpServerSpec] = (),
    approval: ApprovalPolicy | ApprovalMode | ApprovalHandler | None = None,
    hooks: HookSettings | HookInput = None,
    skills: SkillsSettings | bool | None = None,
    artifacts: ArtifactPolicyInput = None,
    conversation: ConversationState | None | object = _UNSET,
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
        requires=requires,
        tools=tools,
        mcp_servers=mcp_servers,
        approval_policy=approval,
        hooks=hooks,
        skills=skills,
        artifacts=artifacts,
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
            conversation_override=conversation,
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
    requires: Sequence[AdapterCapability] = (),
    tools: Sequence[Callable[..., Any]] = (),
    env: Mapping[str, Any] | None = None,
    mcp_servers: Sequence[McpServerSpec] = (),
    approval: ApprovalPolicy | ApprovalMode | ApprovalHandler | None = None,
    hooks: HookSettings | HookInput = None,
    skills: SkillsSettings | bool | None = None,
    artifacts: ArtifactPolicyInput = None,
    conversation: ConversationState | None | object = _UNSET,
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
        requires=requires,
        tools=tools,
        mcp_servers=mcp_servers,
        approval_policy=approval,
        hooks=hooks,
        skills=skills,
        artifacts=artifacts,
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
            conversation_override=conversation,
        )

    return decorate
