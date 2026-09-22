from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import (
    Any,
    Callable,
    cast,
)

import dotenv

from kedi.agent_adapter import (
    AdapterCapability,
    AgentAdapter,
    AgentAdapterKind,
    ConversationState,
)
from kedi.agent_adapter.adapters.lazy import LazyAdapter
from kedi.agent_adapter.approvals.policy import (
    ApprovalHandler,
    ApprovalMode,
    ApprovalPolicy,
)
from kedi.agent_adapter.registry import (
    adapter_capabilities,
    adapter_factory,
    build_adapter,
    known_adapter_shortnames,
)
from kedi.agent_profile import (
    AgentProfile,
    HookInput,
    HookSettings,
    McpServerSpec,
    SkillsSettings,
    normalize_hook_settings,
)
from kedi.api.registration import _coerce_approval, _tool_mapping
from kedi.api.state import (
    _DEFAULT_CONFIG,
    _UNSET,
    AdapterLike,
    AgentName,
    FrameworkAdapterName,
    _APIConfig,
    _config_var,
)
from kedi.artifacts import ArtifactPolicyInput, normalize_artifact_policy
from kedi.effort import EffortValue
from kedi.errors import KediExecutionError


def _execution_engine(config: _APIConfig) -> Any:
    from kedi.core.engines import _default_engine, shared_thread_engine
    from kedi.engine import SequentialEngine

    if config.parallel is False:
        return SequentialEngine()
    if config.parallel is True:
        return shared_thread_engine(max_workers=config.max_workers)
    return _default_engine(max_workers=config.max_workers)


def _agent_profile_overrides(
    *,
    model: str | None | object,
    adapter_kind: AgentAdapterKind | None,
    adapter_shortname: str | None,
    system: str | None,
    effort: EffortValue | None,
    settings: Mapping[str, Any] | None,
    requires: Sequence[AdapterCapability],
    tools: Sequence[Callable[..., Any]],
    mcp_servers: Sequence[McpServerSpec],
    approval_policy: ApprovalPolicy | ApprovalMode | ApprovalHandler | None,
    hooks: HookSettings | HookInput,
    skills: SkillsSettings | bool | None,
    artifacts: ArtifactPolicyInput,
) -> tuple[dict[str, Callable[..., Any]], AgentProfile]:
    tool_map = _tool_mapping(tools)
    hook_settings = normalize_hook_settings(hooks)
    return tool_map, AgentProfile(
        model=None if model is _UNSET else cast(str | None, model),
        adapter_kind=adapter_kind,
        adapter_shortname=adapter_shortname,
        effort=effort,
        system=system,
        settings=dict(settings or {}),
        requirements=tuple(requires),
        tools=tuple(tool_map),
        mcp_servers=tuple(mcp_servers),
        approval=_coerce_approval(approval_policy),
        hooks=hook_settings,
        _hook_enforcement=hook_settings,
        skills=skills,
        artifacts=normalize_artifact_policy(artifacts),
    )


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


def _lazy_adapter(
    shortname: str,
    model: str | None,
    adapter_kwargs: Mapping[str, Any],
) -> LazyAdapter:
    metadata = adapter_factory(shortname)

    def factory() -> AgentAdapter[Any]:
        return build_adapter(shortname, model, **dict(adapter_kwargs))

    return LazyAdapter(
        factory,
        kind=metadata.kind,
        shortname=metadata.shortname,
        capabilities=adapter_capabilities(shortname),
    )


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
    hooks: HookSettings | HookInput,
    skills: SkillsSettings | bool | None,
    artifacts: ArtifactPolicyInput,
    conversation: ConversationState | None | object,
    adapter_kwargs: Mapping[str, Any],
    parallel: bool | None = None,
    max_workers: int | None = None,
    loop_iteration_limit: int | None = None,
) -> _APIConfig:
    if load_env:
        dotenv.load_dotenv()

    base_parallel = None if base is None else base.parallel
    base_max_workers = 8 if base is None else base.max_workers
    resolved_parallel = base_parallel if parallel is None else parallel
    resolved_max_workers = base_max_workers if max_workers is None else max_workers
    base_loop_iteration_limit = 10_000 if base is None else base.loop_iteration_limit
    resolved_loop_iteration_limit = (
        base_loop_iteration_limit if loop_iteration_limit is None else loop_iteration_limit
    )
    if isinstance(resolved_loop_iteration_limit, bool) or resolved_loop_iteration_limit <= 0:
        raise ValueError("loop_iteration_limit must be a positive integer")
    base_conversation = None if base is None else base.conversation
    resolved_conversation = (
        base_conversation
        if conversation is _UNSET
        else cast(ConversationState | None, conversation)
    )

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
    hook_settings = normalize_hook_settings(hooks)
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
        hooks=hook_settings,
        _hook_enforcement=hook_settings,
        skills=skills,
        artifacts=normalize_artifact_policy(artifacts),
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
        conversation=resolved_conversation,
        parallel=resolved_parallel,
        max_workers=resolved_max_workers,
        loop_iteration_limit=resolved_loop_iteration_limit,
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
    hooks: HookSettings | HookInput = None,
    skills: SkillsSettings | bool | None = None,
    artifacts: ArtifactPolicyInput = None,
    conversation: ConversationState | None | object = _UNSET,
    parallel: bool | None = None,
    max_workers: int | None = None,
    loop_iteration_limit: int | None = None,
    **adapter_kwargs: Any,
) -> None:
    """Set the default Python API configuration for the current context.

    Independent template calls run concurrently by default (promise pipelining).
    ``parallel=False`` selects sequential execution; ``max_workers`` bounds the shared pool.
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
            hooks=hooks,
            skills=skills,
            artifacts=artifacts,
            conversation=conversation,
            adapter_kwargs=adapter_kwargs,
            parallel=parallel,
            max_workers=max_workers,
            loop_iteration_limit=loop_iteration_limit,
        )
    )


def reset_config() -> None:
    _config_var.set(_DEFAULT_CONFIG)
