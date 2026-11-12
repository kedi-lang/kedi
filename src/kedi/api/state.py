from __future__ import annotations

import builtins
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Literal,
    ParamSpec,
    TypeAlias,
    TypeVar,
)

from typing_extensions import Sentinel

from kedi.agent_adapter import AgentAdapter, AgentAdapterKind, ConversationState
from kedi.agent_adapter.adapters.lazy import LazyAdapter
from kedi.agent_profile import AgentProfile

P = ParamSpec("P")


R = TypeVar("R")


ClassT = TypeVar("ClassT", bound=builtins.type[Any])


FrameworkAdapterName = Literal["pydantic", "langchain"]


AgentName = Literal["claude", "codex", "acp"]


AdapterLike: TypeAlias = AgentAdapter[Any] | LazyAdapter


_UNSET = Sentinel("_UNSET")


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
    conversation: ConversationState | None = None
    parallel: bool | None = None
    max_workers: int = 8
    loop_iteration_limit: int = 10_000


_DEFAULT_CONFIG = _APIConfig()


_config_var: ContextVar[_APIConfig] = ContextVar("kedi_api_config", default=_DEFAULT_CONFIG)
