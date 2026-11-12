from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from kedi.errors import KediExecutionError

from .meta import AdapterCapabilities, AgentAdapter, AgentAdapterKind


@dataclass(frozen=True)
class AdapterFactory:
    shortname: str
    kind: AgentAdapterKind
    build: Callable[[str | None, Mapping[str, Any]], AgentAdapter[Any]]


def _pydantic(model: str | None, kwargs: Mapping[str, Any]) -> AgentAdapter[Any]:
    from .adapters.pydantic import PydanticAdapter

    return PydanticAdapter(model=model, **dict(kwargs))


def _langchain(model: str | None, kwargs: Mapping[str, Any]) -> AgentAdapter[Any]:
    from .adapters.langchain import LangChainAdapter

    return LangChainAdapter(model=model, **dict(kwargs))


def _claude(model: str | None, kwargs: Mapping[str, Any]) -> AgentAdapter[Any]:
    from .adapters.claude import ClaudeAdapter

    return ClaudeAdapter(model=model, **dict(kwargs))


def _codex(model: str | None, kwargs: Mapping[str, Any]) -> AgentAdapter[Any]:
    from .adapters.codex import CodexAdapter

    return CodexAdapter(model=model, **dict(kwargs))


def _acp(model: str | None, kwargs: Mapping[str, Any]) -> AgentAdapter[Any]:
    from .adapters.acp import ACPAdapter

    return ACPAdapter(model=model, **dict(kwargs))


def _a2a(model: str | None, kwargs: Mapping[str, Any]) -> AgentAdapter[Any]:
    from .adapters.a2a import A2AAdapter

    return A2AAdapter(model=model, **dict(kwargs))


def _pydantic_capabilities() -> AdapterCapabilities:
    from .adapters.pydantic import PydanticAdapter

    return PydanticAdapter.capabilities


def _langchain_capabilities() -> AdapterCapabilities:
    from .adapters.langchain import LangChainAdapter

    return LangChainAdapter.capabilities


def _claude_capabilities() -> AdapterCapabilities:
    from .adapters.claude import ClaudeAdapter

    return ClaudeAdapter.capabilities


def _codex_capabilities() -> AdapterCapabilities:
    from .adapters.codex import CodexAdapter

    return CodexAdapter.capabilities


def _acp_capabilities() -> AdapterCapabilities:
    from .adapters.acp import ACPAdapter

    return ACPAdapter.capabilities


def _a2a_capabilities() -> AdapterCapabilities:
    from .adapters.a2a import A2AAdapter

    return A2AAdapter.capabilities


_ADAPTERS: dict[str, AdapterFactory] = {
    "pydantic": AdapterFactory("pydantic", "agent-framework", _pydantic),
    "langchain": AdapterFactory("langchain", "agent-framework", _langchain),
    "claude": AdapterFactory("claude", "agent-harness", _claude),
    "codex": AdapterFactory("codex", "agent-harness", _codex),
    "acp": AdapterFactory("acp", "agent-harness", _acp),
    "a2a": AdapterFactory("a2a", "agent-harness", _a2a),
}

_CAPABILITY_LOADERS: dict[str, Callable[[], AdapterCapabilities]] = {
    "pydantic": _pydantic_capabilities,
    "langchain": _langchain_capabilities,
    "claude": _claude_capabilities,
    "codex": _codex_capabilities,
    "acp": _acp_capabilities,
    "a2a": _a2a_capabilities,
}


def known_adapter_shortnames() -> tuple[str, ...]:
    return tuple(_ADAPTERS)


def adapter_factory(shortname: str) -> AdapterFactory:
    if shortname == "dspy":
        raise KediExecutionError(
            "The DSPy runtime adapter moved to https://github.com/kedi-lang/dspy-adapter. "
            "Use 'pydantic' or 'langchain' for built-in runtime execution. "
            "GEPA still uses DSPy through kedi[dspy]."
        )
    try:
        return _ADAPTERS[shortname]
    except KeyError as exc:
        known = ", ".join(sorted(_ADAPTERS))
        raise KediExecutionError(
            f"Unknown agent adapter {shortname!r}. Known adapters: {known}"
        ) from exc


def validate_adapter_kind(shortname: str, expected_kind: AgentAdapterKind) -> None:
    directive = "agent" if expected_kind == "agent-harness" else "adapter"
    kind_label = "agent harness" if expected_kind == "agent-harness" else "agent framework"
    try:
        factory = adapter_factory(shortname)
    except KediExecutionError as exc:
        known = ", ".join(
            name for name, candidate in _ADAPTERS.items() if candidate.kind == expected_kind
        )
        raise KediExecutionError(
            f"> {directive}: unknown {kind_label} adapter {shortname!r}. "
            f"Known {kind_label} adapters: {known}"
        ) from exc
    if factory.kind != expected_kind:
        actual = "agent" if factory.kind == "agent-harness" else "adapter"
        raise KediExecutionError(
            f"> {directive}: cannot select {shortname!r}; it is an {actual} adapter"
        )


def adapter_capabilities(shortname: str) -> AdapterCapabilities:
    adapter_factory(shortname)
    loader = _CAPABILITY_LOADERS.get(shortname)
    if loader is None:
        return AdapterCapabilities()
    return loader()


def build_adapter(
    shortname: str,
    model: str | None = None,
    **kwargs: Any,
) -> AgentAdapter[Any]:
    return adapter_factory(shortname).build(model, kwargs)
