from __future__ import annotations

# pyright: reportMissingImports=false

import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

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
CODEX_DEFAULT_INSTRUCTIONS = (
    "You are a precise assistant running through Kedi. Follow the active instructions, "
    "available tools, and output contract."
)


def is_codex_model_id(model_id: str) -> bool:
    """Return whether a model id selects the Codex-authenticated model bridge."""

    return model_id.startswith(CODEX_MODEL_PREFIX)


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
        import codex_auth_helper  # ty: ignore[unresolved-import]  # Optional codex-model extra.
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


__all__ = [
    "CODEX_DEFAULT_INSTRUCTIONS",
    "CODEX_MODEL_PREFIX",
    "CodexModel",
    "CodexModelAdapter",
    "CodexResponsesConnection",
    "CodexResponsesFallback",
    "codex_responses_model",
    "is_codex_model_id",
]
