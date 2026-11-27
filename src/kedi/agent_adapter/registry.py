from __future__ import annotations

import importlib
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from kedi.errors import KediExecutionError

from .meta import AdapterCapabilities, AgentAdapter, AgentAdapterKind

if TYPE_CHECKING:
    from kedi.package_adapters import PackageAdapter


@dataclass(frozen=True)
class AdapterFactory:
    shortname: str
    kind: AgentAdapterKind
    build: Callable[[str | None, Mapping[str, Any]], AgentAdapter[Any]]
    owns_connection: bool = False
    """Each profile command selects its own instance, which the runtime closes."""


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


def _rpc(model: str | None, kwargs: Mapping[str, Any]) -> AgentAdapter[Any]:
    from .adapters.jsonrpc import JsonRpcAdapter

    options = dict(kwargs)
    adapter = JsonRpcAdapter(options.pop("command"), model=model, **options)
    if adapter.kind != "agent-harness":
        adapter.close()
        raise KediExecutionError(
            f"> agent: rpc: the bridge {adapter.backend['name']!r} is an {adapter.kind} bridge; "
            "select it from Python with compile_program(adapter=JsonRpcAdapter(...))"
        )
    return adapter


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


def _rpc_capabilities() -> AdapterCapabilities:
    from .adapters.jsonrpc import JsonRpcAdapter

    return JsonRpcAdapter.capabilities


_ADAPTERS: dict[str, AdapterFactory] = {
    "pydantic": AdapterFactory("pydantic", "agent-framework", _pydantic),
    "langchain": AdapterFactory("langchain", "agent-framework", _langchain),
    "claude": AdapterFactory("claude", "agent-harness", _claude),
    "codex": AdapterFactory("codex", "agent-harness", _codex),
    "acp": AdapterFactory("acp", "agent-harness", _acp),
    "a2a": AdapterFactory("a2a", "agent-harness", _a2a),
    "rpc": AdapterFactory("rpc", "agent-harness", _rpc, owns_connection=True),
}

_CAPABILITY_LOADERS: dict[str, Callable[[], AdapterCapabilities]] = {
    "pydantic": _pydantic_capabilities,
    "langchain": _langchain_capabilities,
    "claude": _claude_capabilities,
    "codex": _codex_capabilities,
    "acp": _acp_capabilities,
    "a2a": _a2a_capabilities,
    "rpc": _rpc_capabilities,
}


def owns_connection(shortname: str) -> bool:
    """Whether a profile's command selects its own instance of this adapter."""

    factory = _ADAPTERS.get(shortname)
    return factory is not None and factory.owns_connection


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


def validate_adapter_kind(
    shortname: str, expected_kind: AgentAdapterKind, *, imported: Sequence[str] = ()
) -> None:
    """Rejects a built-in shortname of another kind, or a name neither built in nor imported."""

    directive = "agent" if expected_kind == "agent-harness" else "adapter"
    kind_label = "agent harness" if expected_kind == "agent-harness" else "agent framework"
    try:
        factory = adapter_factory(shortname)
    except KediExecutionError as exc:
        known = ", ".join(
            [name for name, candidate in _ADAPTERS.items() if candidate.kind == expected_kind]
            + [f"{name} (imported)" for name in imported]
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


def package_runtime_environment(package: str) -> dict[str, str]:
    """Where a package's bridge keeps its run-time files: inside the installed package, so they go
    with it. The bridge SDKs keep their preflight cache in ``KEDI_BRIDGE_CACHE``; a bridge's own
    files (a harness's home) go under ``KEDI_PACKAGE_RUNTIME_DIR``."""

    from kedi.package_lifecycle import RUNTIME_DIRECTORY
    from kedi.package_registry import installed_package_root

    runtime = installed_package_root(package) / RUNTIME_DIRECTORY
    runtime.mkdir(exist_ok=True)
    return {
        "KEDI_PACKAGE_RUNTIME_DIR": str(runtime),
        "KEDI_BRIDGE_CACHE": str(runtime / "bridge-checks"),
    }


def build_package_adapter(adapter: PackageAdapter, model: str | None) -> AgentAdapter[Any]:
    """The adapter a package declares: its Python class, or its JSON-RPC bridge launched."""

    label = f"{adapter.name!r} from package {adapter.package!r}"
    if adapter.command is not None:
        from .adapters.jsonrpc import JsonRpcAdapter

        if adapter.environment == "uv":
            from kedi.package_environment import environment_ready
            from kedi.package_registry import installed_package_root

            if not environment_ready(installed_package_root(adapter.package)):
                raise KediExecutionError(
                    f"> {adapter.directive}: {label} runs in the package's environment, whose "
                    "installation did not finish. Install the package again: "
                    f"kedi add {adapter.package}, or kedi install for a local package."
                )
        program = adapter.command[0]
        if shutil.which(program) is None:
            raise KediExecutionError(
                f"> {adapter.directive}: {label} runs {program!r}, which is not installed "
                "or not on PATH"
            )
        built: AgentAdapter[Any] = JsonRpcAdapter(
            list(adapter.command), model=model, env=package_runtime_environment(adapter.package)
        )
    else:
        assert adapter.class_path is not None
        module_name, _, attribute = adapter.class_path.partition(":")
        try:
            cls = getattr(importlib.import_module(module_name), attribute)
        except (ImportError, AttributeError) as exc:
            raise KediExecutionError(
                f"> {adapter.directive}: {label} names class {adapter.class_path!r}, which "
                f"cannot be imported: {exc}"
            ) from exc
        built = cls(model=model)
    if getattr(built, "kind", None) != adapter.adapter_kind:
        close = getattr(built, "close", None)
        if callable(close):
            close()
        actual = "agent" if getattr(built, "kind", None) == "agent-harness" else "adapter"
        raise KediExecutionError(
            f"> {adapter.directive}: {label} is declared an {adapter.kind}, but it is an "
            f"{actual} (kind {getattr(built, 'kind', None)!r})"
        )
    return built


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
