"""Cancellation-safe LangGraph snapshots for live child tasks."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast
from uuid import uuid4

from langchain.agents.middleware.types import InputAgentState
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.typing import ContextT, InputT, OutputT, StateT

from kedi.agent_adapter.adapters.langchain.runtime import _RUN
from kedi.agent_adapter.subagents.control import LiveInput, controlled_attempt


async def run_with_input(
    agent: CompiledStateGraph[StateT, ContextT, InputT, OutputT],
    payload: InputAgentState,
    config: RunnableConfig,
    control: LiveInput,
    capture: Callable[[list[AnyMessage]], None],
) -> OutputT:
    config = {
        **config,
        "configurable": {**config.get("configurable", {}), "thread_id": uuid4().hex},
    }
    try:
        # LangGraph's values-mode annotation erases its declared output state type.
        return cast(
            OutputT,
            await controlled_attempt(control, agent.ainvoke(cast(InputT, payload), config=config)),
        )
    finally:
        snapshot = await agent.aget_state(config)
        if snapshot.values:
            messages: list[AnyMessage] = list(snapshot.values.get("messages", []))
            answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
            run_state = _RUN.get()
            completed = run_state.completed_tools if run_state is not None else {}
            for message in tuple(messages):
                if isinstance(message, AIMessage):
                    for call in message.tool_calls:
                        if call["id"] is None:
                            continue
                        if call["id"] not in answered:
                            messages.append(
                                completed[call["id"]]
                                if call["id"] in completed
                                else ToolMessage(
                                    content="This tool call was interrupted. Its result is unavailable; do not assume it succeeded.",
                                    tool_call_id=call["id"],
                                    name=call["name"],
                                    status="error",
                                )
                            )
                            answered.add(call["id"])
            capture(messages)
