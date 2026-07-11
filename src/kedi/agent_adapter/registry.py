from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from kedi.errors import KediExecutionError

from .meta import AgentAdapter, AgentAdapterKind


@dataclass(frozen=True)
class AdapterFactory:
    shortname: str
    kind: AgentAdapterKind
    build: Callable[[str | None, Mapping[str, Any]], AgentAdapter[Any]]


def _pydantic(model: str | None, kwargs: Mapping[str, Any]) -> AgentAdapter[Any]:
    from .adapters.pydantic import PydanticAdapter

    return PydanticAdapter(model=model, **dict(kwargs))


def _dspy(model: str | None, kwargs: Mapping[str, Any]) -> AgentAdapter[Any]:
    from .adapters.dspy import DSPyAdapter

    if model is None:
        raise ValueError("DSPy adapter requires a model")
    return DSPyAdapter(model=model, **dict(kwargs))


def _langchain(model: str | None, kwargs: Mapping[str, Any]) -> AgentAdapter[Any]:
    from .adapters.langchain import LangChainAdapter

    return LangChainAdapter(model=model, **dict(kwargs))


def _claude(model: str | None, kwargs: Mapping[str, Any]) -> AgentAdapter[Any]:
    from .adapters.claude import ClaudeAdapter

    return ClaudeAdapter(model=model, **dict(kwargs))


def _codex(model: str | None, kwargs: Mapping[str, Any]) -> AgentAdapter[Any]:
    from .adapters.codex import CodexAdapter

    return CodexAdapter(model=model, **dict(kwargs))


_ADAPTERS: dict[str, AdapterFactory] = {
    "pydantic": AdapterFactory("pydantic", "agent-framework", _pydantic),
    "dspy": AdapterFactory("dspy", "agent-framework", _dspy),
    "langchain": AdapterFactory("langchain", "agent-framework", _langchain),
    "claude": AdapterFactory("claude", "agent-harness", _claude),
    "codex": AdapterFactory("codex", "agent-harness", _codex),
}


def known_adapter_shortnames() -> tuple[str, ...]:
    return tuple(_ADAPTERS)


def adapter_factory(shortname: str) -> AdapterFactory:
    try:
        return _ADAPTERS[shortname]
    except KeyError as exc:
        known = ", ".join(sorted(_ADAPTERS))
        raise KediExecutionError(
            f"Unknown agent adapter {shortname!r}. Known adapters: {known}"
        ) from exc


def validate_adapter_kind(shortname: str, expected_kind: AgentAdapterKind) -> None:
    factory = adapter_factory(shortname)
    if factory.kind != expected_kind:
        directive = "agent" if expected_kind == "agent-harness" else "adapter"
        actual = "agent" if factory.kind == "agent-harness" else "adapter"
        raise KediExecutionError(
            f"> {directive}: cannot select {shortname!r}; it is an {actual} adapter"
        )


def build_adapter(
    shortname: str,
    model: str | None = None,
    **kwargs: Any,
) -> AgentAdapter[Any]:
    return adapter_factory(shortname).build(model, kwargs)
