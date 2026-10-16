from __future__ import annotations

# pyright: reportMissingImports=false
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

from .meta import TemplatePromptMode

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from pydantic_ai.models import Model

    CodexModel: TypeAlias = Model | BaseChatModel
else:
    CodexModel: TypeAlias = Any

CodexModelAdapter = Literal["pydantic", "langchain"]
CodexResponsesConnection = Literal["http", "websocket"]
CodexResponsesFallback = Literal["error", "http"]
CODEX_MODEL_PREFIX = "codex/"
TYPESAFE_MODEL_PREFIX = "typesafe/"
CODEX_DEFAULT_INSTRUCTIONS = (
    "You are a precise assistant running through Kedi. Follow the active instructions, "
    "available tools, and output contract."
)


def is_codex_model_id(model_id: str) -> bool:
    """Return whether a model id selects the Codex-authenticated model bridge."""

    return model_id.startswith(CODEX_MODEL_PREFIX)


def is_typesafe_model_id(model_id: str) -> bool:
    """Return whether a model id selects the TypeSafe Jev bridge."""

    return model_id.startswith(TYPESAFE_MODEL_PREFIX)


def is_decision_model_id(model_id: str) -> bool:
    """Select hosted Jev or an optional local Laya decision model."""
    return is_typesafe_model_id(model_id) or model_id.startswith(("laya/", "laya-mlx/"))


def model_template_prompt_mode(model: object) -> TemplatePromptMode:
    """Use the model contract, including older registered Jev integrations."""
    if isinstance(model, str):
        return "decision" if is_decision_model_id(model) else "completion"
    # Published Jev integrations predate the optional model-level contract.
    legacy_decision = (
        getattr(model, "system", None) == "typesafe"
        or getattr(model, "_llm_type", None) == "typesafe-jev"
    )
    mode = getattr(model, "kedi_prompt_mode", "decision" if legacy_decision else "completion")
    if mode not in ("completion", "decision"):
        raise ValueError("Model kedi_prompt_mode must be 'completion' or 'decision'")
    return mode


def laya_model(model_id: str, adapter: Literal["pydantic", "langchain"]) -> Any:
    """Select Laya independently of its optional local inference backend."""
    provider, _, checkpoint = model_id.partition("/")
    if provider not in {"laya", "laya-mlx"} or not checkpoint.strip():
        raise ValueError("Laya model id must be laya/<checkpoint> or laya-mlx/<checkpoint>")
    if adapter not in {"pydantic", "langchain"}:
        raise ValueError("Laya models require the Pydantic or LangChain adapter")
    try:
        import kedi_laya
    except ModuleNotFoundError as exc:
        if exc.name != "kedi_laya":
            raise
        raise ModuleNotFoundError(
            "Local Laya models require kedi-laya. Install 'kedi-decisions[laya]' "
            "for PyTorch or 'kedi-decisions[laya-mlx]' for Apple Silicon. "
            "For LangChain also install 'kedi-decisions[langchain]'."
        ) from exc
    backend = "mlx" if provider == "laya-mlx" else "auto"
    if adapter == "pydantic":
        return kedi_laya.LayaModel(checkpoint, backend=backend)
    return kedi_laya.LayaChatModel(checkpoint, backend=backend)


def typesafe_pydantic_model(model_id: str) -> Model:
    """Build the optional TypeSafe Jev model for Pydantic AI."""

    model_name = _typesafe_model_name(model_id)
    try:
        from kedi_typesafe.integrations.pydantic import TypeSafeModel
    except ModuleNotFoundError as exc:
        if exc.name != "kedi_typesafe":
            raise
        raise ModuleNotFoundError(
            "TypeSafe-backed models require kedi-typesafe==0.2.1. "
            "Install Kedi with: uv add 'kedi[typesafe]'"
        ) from exc
    return TypeSafeModel(model_name)


def typesafe_langchain_model(model_id: str) -> BaseChatModel:
    """Build the optional TypeSafe Jev chat model for LangChain."""

    model_name = _typesafe_model_name(model_id)
    try:
        from kedi_typesafe.integrations.langchain import TypeSafeChatModel
    except ModuleNotFoundError as exc:
        if exc.name not in {"kedi_typesafe", "langchain_core", "langchain_typesafe"}:
            raise
        raise ModuleNotFoundError(
            "TypeSafe-backed LangChain models require kedi-typesafe[langchain]==0.2.1. "
            "Install Kedi with: uv add 'kedi[langchain,typesafe]' 'kedi-typesafe[langchain]'"
        ) from exc
    return TypeSafeChatModel(model_name)


async def close_typesafe_model_current(model: object) -> None:
    """Close a TypeSafe transport before its current event loop exits."""

    if model_template_prompt_mode(model) != "decision":
        return
    close_current = getattr(model, "aclose_current", None)
    if callable(close_current):
        await cast(Callable[[], Awaitable[None]], close_current)()


def codex_responses_model(
    model_id: str,
    adapter: CodexModelAdapter,
    *,
    connection: CodexResponsesConnection = "http",
    fallback: CodexResponsesFallback = "error",
    transport_observer: Callable[[object], None] | None = None,
) -> CodexModel:
    """Build a Codex-authenticated model for a supported agent framework.

    ``model_id`` accepts either ``codex/<model>`` or the bare Codex model name.
    Authentication is read by ``codex-auth-helper`` from the user's Codex login.
    """

    model_name = _codex_model_name(model_id)
    try:
        import codex_auth_helper  # Optional codex-model extra.
    except ModuleNotFoundError as exc:
        if exc.name != "codex_auth_helper":
            raise
        raise ModuleNotFoundError(
            "Codex-backed models require Python 3.11+ and codex-auth-helper[websocket]==1.8.0. "
            "Install Kedi with: uv add 'kedi[codex-model]'"
        ) from exc

    if connection not in {"http", "websocket"}:
        raise ValueError(f"Unsupported Codex Responses connection: {connection!r}")
    if fallback not in {"error", "http"}:
        raise ValueError(f"Unsupported Codex Responses fallback: {fallback!r}")
    if connection == "http" and fallback != "error":
        raise ValueError("`fallback` applies only when `connection='websocket'`.")
    if connection == "websocket" and not hasattr(
        codex_auth_helper,
        "CodexResponsesConnection",
    ):
        raise RuntimeError(
            "Codex Responses WebSocket support requires a newer codex-auth-helper build."
        )
    if transport_observer is not None and not hasattr(
        codex_auth_helper,
        "CodexResponsesTransportEvent",
    ):
        raise RuntimeError(
            "Codex Responses transport observation requires a newer codex-auth-helper build."
        )

    factory_kwargs: dict[str, Any] = {"instructions": CODEX_DEFAULT_INSTRUCTIONS}
    if auth_path := os.environ.get("KEDI_CODEX_AUTH_FILE"):
        factory_kwargs["config"] = codex_auth_helper.CodexAuthConfig(
            auth_path=Path(auth_path).expanduser(),
        )
    if connection != "http":
        factory_kwargs["connection"] = connection
    if fallback != "error":
        factory_kwargs["fallback"] = fallback
    if transport_observer is not None:
        factory_kwargs["transport_observer"] = transport_observer

    try:
        if adapter == "pydantic":
            return codex_auth_helper.create_codex_responses_model(
                model_name,
                **factory_kwargs,
            )
        if adapter == "langchain":
            return codex_auth_helper.create_codex_chat_openai(
                model_name,
                **factory_kwargs,
            )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Codex authentication is unavailable. Run `codex login`, then retry."
        ) from exc

    raise ValueError(f"Unsupported Codex model adapter: {adapter!r}")


def _codex_model_name(model_id: str) -> str:
    model_name = model_id[len(CODEX_MODEL_PREFIX) :] if is_codex_model_id(model_id) else model_id
    model_name = model_name.strip()
    if not model_name:
        raise ValueError("Codex model id must name a model after 'codex/'")
    if "://" in model_name or model_name.startswith("/"):
        raise ValueError(f"Invalid Codex model id: {model_id!r}")
    return model_name


def _typesafe_model_name(model_id: str) -> str:
    model_name = (
        model_id[len(TYPESAFE_MODEL_PREFIX) :] if is_typesafe_model_id(model_id) else model_id
    )
    model_name = model_name.strip()
    if not model_name:
        raise ValueError("TypeSafe model id must name a model after 'typesafe/'")
    if "://" in model_name or model_name.startswith("/"):
        raise ValueError(f"Invalid TypeSafe model id: {model_id!r}")
    return model_name


__all__ = [
    "CODEX_DEFAULT_INSTRUCTIONS",
    "CODEX_MODEL_PREFIX",
    "TYPESAFE_MODEL_PREFIX",
    "CodexModel",
    "CodexModelAdapter",
    "CodexResponsesConnection",
    "CodexResponsesFallback",
    "codex_responses_model",
    "close_typesafe_model_current",
    "is_codex_model_id",
    "is_typesafe_model_id",
    "typesafe_langchain_model",
    "typesafe_pydantic_model",
]
