from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from copy import copy
from types import FrameType
from typing import Any, Sequence, cast

from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import AbstractToolset, ToolsetTool
from pydantic_ai.capabilities._deferred_capability_loader import (
    DeferredCapabilityLoader,
    _render_deferred_capability_catalog,
)

NativeDeferredCapabilityLoader = DeferredCapabilityLoader


def _transport_profile_model(model: Model) -> Model:
    if isinstance(model, WrapperModel):
        inner = _transport_profile_model(model.wrapped)
        if inner is model.wrapped:
            return model
        # WrapperModel forwards unknown attributes before copy reconstructs wrapped.
        prepared = object.__new__(type(model))
        vars(prepared).update(vars(model))
        prepared.wrapped = inner
        return prepared
    if getattr(model, "responses_connection", None) != "websocket":
        return model
    try:
        from codex_auth_helper.model import CodexResponsesModel
    except ModuleNotFoundError as exc:
        if exc.name != "codex_auth_helper":
            raise
        return model

    if not isinstance(model, CodexResponsesModel) or not any(
        model.profile.get(key) is not None for key in ("tool_addition_mode", "tool_deferral_mode")
    ):
        return model
    # Codex's WebSocket gateway cannot consume native tool-addition contracts.
    # Keep the transport/client and let Pydantic reveal loaded tools in its list.
    prepared = copy(model)
    profile = model.profile.copy()
    profile["tool_addition_mode"] = None
    profile["tool_deferral_mode"] = None
    prepared.profile = profile
    return prepared


@contextmanager
def transport_model_scope(agent: Agent[Any, Any], model: Model | None) -> Iterator[None]:
    if model is None:
        yield
        return
    prepared = _transport_profile_model(model)
    if prepared is model:
        yield
        return
    from .budget import budget_model

    with agent.override(model=budget_model(prepared)):
        yield


async def native_deferred_catalog(ctx: RunContext[Any]) -> str:
    """Keep native loader wording/identity while Kedi filters reachable entries."""
    return await _render_deferred_capability_catalog(ctx)


class PydanticAICompatibilityError(RuntimeError):
    """Raised when the installed Pydantic AI layout is unsupported by Kedi."""


def build_base_toolset_list(
    agent: Agent[Any, Any],
    capability_toolsets: Sequence[Any] | None = None,
    *,
    ignore_overrides: bool = False,
) -> list[AbstractToolset[Any]]:
    """Build Pydantic AI's configured toolsets without dispatching to Kedi's override."""

    try:
        if not ignore_overrides:
            return cast(
                "list[AbstractToolset[Any]]",
                Agent._build_toolset_list(agent, cap_toolsets=cast(Any, capability_toolsets)),
            )
        return cast(
            "list[AbstractToolset[Any]]",
            Agent._build_toolset_list(
                agent,
                cap_toolsets=cast(Any, capability_toolsets),
                ignore_overrides=ignore_overrides,
            ),
        )
    except AttributeError as exc:
        raise _unsupported_layout(exc) from exc


def application_toolset_count(agent: Agent[Any, Any], *, ignore_overrides: bool = False) -> int:
    """Count constructor, overridden, and dynamic application toolsets."""

    try:
        if ignore_overrides:
            return 1 + len(agent._user_toolsets) + agent._constructor_dynamic_toolset_count
        overridden_toolsets = agent._override_toolsets.get()
        if overridden_toolsets is not None:
            return 1 + len(overridden_toolsets.value)
        return 1 + len(agent._user_toolsets) + len(agent._dynamic_toolsets)
    except AttributeError as exc:
        raise _unsupported_layout(exc) from exc


def has_application_tools(agent: Agent[Any, Any]) -> bool:
    """Return whether an application-provided function or toolset can be called."""

    try:
        overridden_tools = agent._override_tools.get()
        if overridden_tools is not None:
            return bool(overridden_tools.value)
        if agent._function_toolset.tools:
            return True
        overridden_toolsets = agent._override_toolsets.get()
        if overridden_toolsets is not None:
            return bool(overridden_toolsets.value)
        return bool(agent._user_toolsets or agent._dynamic_toolsets)
    except AttributeError as exc:
        raise _unsupported_layout(exc) from exc


def application_toolsets(agent: Agent[Any, Any]) -> list[AbstractToolset[Any]]:
    """Return configured application toolsets, excluding capability toolsets."""

    toolsets = build_base_toolset_list(agent)
    return toolsets[: application_toolset_count(agent)]


def infer_agent_name(agent: Agent[Any, Any], frame: FrameType | None) -> None:
    """Delegate native name inference while keeping the private contract isolated."""

    try:
        agent._infer_name(frame)
    except AttributeError as exc:
        raise _unsupported_layout(exc) from exc


def effective_model(
    agent: Agent[Any, Any],
    model: Model | str | None = None,
) -> Model:
    """Resolve the model selected by run arguments, overrides, capabilities, or defaults."""

    try:
        return agent._get_model_outside_run(model)
    except AttributeError as exc:
        raise _unsupported_layout(exc) from exc


def configured_prompt_model(agent: Agent[Any, Any], model: str | None) -> object:
    """Inspect static bindings without loading a provider or invoking a dynamic selector."""
    try:
        override = agent._override_model.get()
        if override is not None:
            return override.value
        if model is not None:
            return model
        contribution = agent._effective_root_capability().get_model()
        if contribution is not None:
            return contribution
        return agent._model
    except AttributeError as exc:
        raise _unsupported_layout(exc) from exc


def build_nested_tool_manager(
    *,
    ctx: RunContext[Any],
    toolset: AbstractToolset[Any],
    tools: dict[str, ToolsetTool[Any]],
) -> Any:
    """Build the supported Pydantic AI nested-call manager for CodeMode."""

    try:
        from pydantic_ai.tool_manager import ToolManager

        parent = ctx.tool_manager
        if parent is None:
            raise PydanticAICompatibilityError(
                "Pydantic AI did not attach a tool manager to the active run context"
            )
        return ToolManager(
            toolset=toolset,
            root_capability=parent.root_capability,
            ctx=ctx,
            tools=tools,
        )
    except AttributeError as exc:
        raise _unsupported_layout(exc) from exc


def _unsupported_layout(exc: AttributeError) -> PydanticAICompatibilityError:
    return PydanticAICompatibilityError(
        "The installed Pydantic AI version has an unsupported Agent tool layout. "
        "Install the Kedi-supported pydantic-ai-slim==2.45.0 release. "
        f"Missing private contract: {exc}"
    )
