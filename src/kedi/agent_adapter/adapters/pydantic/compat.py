from __future__ import annotations

from types import FrameType
from typing import Any, Sequence, cast

from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import AbstractToolset, ToolsetTool


class PydanticAICompatibilityError(RuntimeError):
    """Raised when the installed Pydantic AI layout is unsupported by Kedi."""


def build_base_toolset_list(
    agent: Agent[Any, Any],
    capability_toolsets: Sequence[Any] | None = None,
) -> list[AbstractToolset[Any]]:
    """Build Pydantic AI's configured toolsets without dispatching to Kedi's override."""

    try:
        return cast(
            "list[AbstractToolset[Any]]",
            Agent._build_toolset_list(agent, cap_toolsets=cast(Any, capability_toolsets)),
        )
    except AttributeError as exc:
        raise _unsupported_layout(exc) from exc


def application_toolset_count(agent: Agent[Any, Any]) -> int:
    """Count constructor, overridden, and dynamic application toolsets."""

    try:
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
        "Install the Kedi-supported pydantic-ai-slim==2.23.0 release. "
        f"Missing private contract: {exc}"
    )
