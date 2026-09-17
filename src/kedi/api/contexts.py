from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextvars import Token
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
)

from kedi.agent_adapter import ConversationState
from kedi.agent_adapter.approvals.policy import ApprovalHandler, ApprovalMode, ApprovalPolicy
from kedi.agent_adapter.subagents.limits import SubagentUsageLimits
from kedi.agent_profile import HookInput, HookSettings, McpServerSpec, SkillsSettings
from kedi.api.configuration import _build_adapter, _build_config, _execution_engine
from kedi.api.state import (
    _UNSET,
    AdapterLike,
    AgentName,
    FrameworkAdapterName,
    _APIConfig,
    _config_var,
)
from kedi.artifacts import ArtifactPolicyInput
from kedi.effort import EffortValue
from kedi.executors.base import Executor

if TYPE_CHECKING:
    from kedi.interactive_session import InteractiveSession


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
    hooks: HookSettings | HookInput = None,
    skills: SkillsSettings | bool | None = None,
    artifacts: ArtifactPolicyInput = None,
    conversation: ConversationState | None | object = _UNSET,
    parallel: bool | None = None,
    max_workers: int | None = None,
    loop_iteration_limit: int | None = None,
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


class _KediSessionContext:
    def __init__(self, state: ConversationState | None = None) -> None:
        self.state = state or ConversationState()
        self._context: _KediContext | None = None

    def __enter__(self) -> ConversationState:
        self._context = context(conversation=self.state)
        self._context.__enter__()
        return self.state

    def __exit__(self, *args: object) -> None:
        try:
            if self._context is not None:
                self._context.__exit__(*args)
        finally:
            self._context = None
            self.state.close()

    async def __aenter__(self) -> ConversationState:
        return self.__enter__()

    async def __aexit__(self, *args: object) -> None:
        self.__exit__(*args)


def session(state: ConversationState | None = None) -> _KediSessionContext:
    """Continue model history and artifact ownership explicitly within one scope."""
    return _KediSessionContext(state)


def interactive(
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
    executor: Executor | None = None,
    engine: Any | None = None,
    cwd: str | Path | None = None,
    subagent_max_depth: int = 5,
    subagent_max_concurrency: int = 4,
    subagent_timeout_seconds: float | None = 120.0,
    subagent_usage_limits: SubagentUsageLimits | None = None,
    subagent_state_path: str | Path | None = None,
    loop_iteration_limit: int | None = None,
    **adapter_kwargs: Any,
) -> InteractiveSession:
    """Create a process-local incremental Kedi execution session."""
    from kedi.interactive_session import InteractiveSession

    config = _build_config(
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
        hooks=hooks,
        skills=skills,
        artifacts=artifacts,
        conversation=conversation,
        adapter_kwargs=adapter_kwargs,
        parallel=parallel,
        max_workers=max_workers,
        loop_iteration_limit=loop_iteration_limit,
    )
    active_conversation = config.conversation
    owns_conversation = active_conversation is None
    if active_conversation is None:
        active_conversation = ConversationState()
    active_engine = engine if engine is not None else _execution_engine(config)
    return InteractiveSession(
        adapter=_build_adapter(config),
        executor=executor,
        runtime_globals={**config.tools, **config.env},
        engine=active_engine,
        default_agent_profile=config.profile,
        cwd=cwd,
        subagent_max_depth=subagent_max_depth,
        subagent_max_concurrency=subagent_max_concurrency,
        subagent_timeout_seconds=subagent_timeout_seconds,
        subagent_usage_limits=subagent_usage_limits,
        subagent_state_path=subagent_state_path,
        conversation_state=active_conversation,
        close_conversation_state=owns_conversation,
        loop_iteration_limit=config.loop_iteration_limit,
    )


def parallel(*, max_workers: int = 8) -> _KediContext:
    """Context manager enabling concurrent (pipelined) template execution within its scope.

    Sugar for ``context(parallel=True, max_workers=...)``::

        with kedi.parallel():
            result = my_query(...)
    """
    return context(parallel=True, max_workers=max_workers)
