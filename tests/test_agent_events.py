from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from threading import Event
from typing import Any, cast

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.test import TestModel

from kedi.agent_adapter import events as event_module
from kedi.agent_adapter.adapters import PydanticAdapter
from kedi.agent_adapter.adapters import pydantic as pydantic_adapter_module
from kedi.agent_adapter.events import (
    AgentEventDispatcher,
    AgentMessageEvent,
    AgentRunIdentity,
    AgentRunStateEvent,
    AgentStreamEvent,
    AsyncAgentEventQueue,
    agent_event_run,
    bind_agent_event_dispatcher,
    bind_agent_run,
    current_agent_event_dispatcher,
    current_agent_run_identity,
    observe_agent_events,
)


@dataclass
class _Sink:
    events: list[AgentStreamEvent] = field(default_factory=list)

    def emit(self, event: AgentStreamEvent) -> None:
        self.events.append(event)


def _run_state(run_id: str = "run") -> AgentRunStateEvent:
    return AgentRunStateEvent(
        run_id=run_id,
        adapter="fake",
        sequence=0,
        state="started",
    )


def _commentary(message_id: str, content: str = "note") -> AgentMessageEvent:
    return AgentMessageEvent(
        run_id="run",
        message_id=message_id,
        adapter="fake",
        sequence=0,
        phase="commentary",
        content=content,
    )


def test_semantic_events_resolve_commentary_and_authoritative_final() -> None:
    sink = _Sink()

    with observe_agent_events(sink):
        assert current_agent_event_dispatcher() is not None
        with bind_agent_run("root", parent_run_id="parent", agent_name="worker"):
            with agent_event_run("fake") as resolver:
                assert resolver is not None
                assert current_agent_run_identity() == resolver.identity
                resolver.append_delta("message-1", "Checking ")
                resolver.append_delta("message-1", "the parser.")
                resolver.complete_message("message-1")
                resolver.mark_tool_boundary()
                resolver.complete_message("message-2", "Fixed.")
                resolver.complete_run(final_text="Fixed.")
        assert current_agent_run_identity() is None

    assert [event.sequence for event in sink.events] == [1, 2, 3, 4]
    assert [
        (type(event).__name__, getattr(event, "phase", None), getattr(event, "state", None))
        for event in sink.events
    ] == [
        ("AgentRunStateEvent", None, "started"),
        ("AgentMessageEvent", "commentary", None),
        ("AgentMessageEvent", "final", None),
        ("AgentRunStateEvent", None, "completed"),
    ]
    assert all(event.run_id == "root" for event in sink.events)
    assert all(event.parent_run_id == "parent" for event in sink.events)
    assert all(event.agent_name == "worker" for event in sink.events)
    assert isinstance(sink.events[1], AgentMessageEvent)
    assert sink.events[1].content == "Checking the parser."
    assert isinstance(sink.events[2], AgentMessageEvent)
    assert sink.events[2].content == "Fixed."


def test_structured_run_discards_unresolved_text_and_has_no_final_message() -> None:
    events: list[AgentStreamEvent] = []

    with observe_agent_events(events.append):
        with agent_event_run("fake") as resolver:
            assert resolver is not None
            resolver.complete_message("schema-json", '{"answer": "ok"}')

    assert [event.state for event in events if isinstance(event, AgentRunStateEvent)] == [
        "started",
        "completed",
    ]
    assert not any(isinstance(event, AgentMessageEvent) for event in events)


def test_explicit_phases_and_message_boundaries_do_not_publish_partials() -> None:
    events: list[AgentStreamEvent] = []

    with observe_agent_events(events.append):
        with agent_event_run("fake") as resolver:
            assert resolver is not None
            resolver.append_delta("partial", "not complete")
            resolver.complete_message(
                "commentary",
                "I am checking the next path.",
                explicit_phase="commentary",
            )
            resolver.complete_message("older", "First complete message.")
            resolver.complete_message("newer", "Terminal response.")
            resolver.mark_message_boundary()
            resolver.complete_message(
                "native-final",
                "Terminal response.",
                explicit_phase="final",
            )
            resolver.complete_run(final_text="Terminal response.")

    messages = [event for event in events if isinstance(event, AgentMessageEvent)]
    assert [(event.message_id, event.phase, event.content) for event in messages] == [
        ("commentary", "commentary", "I am checking the next path."),
        ("older", "commentary", "First complete message."),
        ("native-final", "final", "Terminal response."),
    ]


@pytest.mark.parametrize(
    ("error", "terminal_state"),
    [(RuntimeError("failed"), "failed"), (asyncio.CancelledError(), "cancelled")],
)
def test_failed_and_cancelled_runs_discard_messages_and_emit_terminal_state(
    error: BaseException,
    terminal_state: str,
) -> None:
    events: list[AgentStreamEvent] = []

    with pytest.raises(type(error)), observe_agent_events(events.append):
        with agent_event_run("fake") as resolver:
            assert resolver is not None
            resolver.complete_message("pending", "Must not escape.")
            raise error

    assert [event.state for event in events if isinstance(event, AgentRunStateEvent)] == [
        "started",
        terminal_state,
    ]
    assert not any(isinstance(event, AgentMessageEvent) for event in events)


def test_no_observer_is_a_noop() -> None:
    assert current_agent_event_dispatcher() is None
    with agent_event_run("fake") as resolver:
        assert resolver is None


def test_pydantic_event_capability_accepts_none_and_rejects_invalid_container() -> None:
    adapter = PydanticAdapter(TestModel())
    events: list[AgentStreamEvent] = []

    with observe_agent_events(events.append):
        assert adapter.run_sync("emit events").output is not None
    assert events

    async def reject_invalid() -> None:
        with (
            observe_agent_events(events.append),
            pytest.raises(
                TypeError,
                match="capabilities must be a list or tuple",
            ),
        ):
            async with adapter._iter_with_kedi_state(
                "invalid",
                {"capabilities": cast(Any, object())},
            ):
                pass

    asyncio.run(reject_invalid())


def test_dispatcher_lifecycle_and_lane_ordering() -> None:
    delivered: list[AgentStreamEvent] = []
    dispatcher = AgentEventDispatcher(delivered.append)
    assert dispatcher.active

    dispatcher.publish(_run_state())
    dispatcher.close()
    assert delivered == [replace(_run_state(), sequence=1)]
    assert not dispatcher.active

    dispatcher.publish(_run_state("late"))
    dispatcher.close()
    assert delivered == [replace(_run_state(), sequence=1)]
    with pytest.raises(RuntimeError, match="closed Kedi agent event dispatcher"):
        with bind_agent_event_dispatcher(dispatcher):
            pass

    first_commentary = _commentary("first")
    later_control = _run_state("control")
    dispatcher._commentary.append((1, first_commentary))
    dispatcher._control.append((2, later_control))
    assert dispatcher._pop_next() is first_commentary
    assert dispatcher._pop_next() is later_control

    earlier_control = _run_state("earlier-control")
    later_commentary = _commentary("later")
    dispatcher._control.append((3, earlier_control))
    dispatcher._commentary.append((4, later_commentary))
    assert dispatcher._pop_next() is earlier_control
    assert dispatcher._pop_next() is later_commentary


def test_dispatcher_bounds_commentary_and_detaches_slow_sink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(event_module, "_COMMENTARY_CAPACITY", 1)
    entered = Event()
    release = Event()
    delivered: list[AgentStreamEvent] = []

    def slow_sink(event: AgentStreamEvent) -> None:
        delivered.append(event)
        entered.set()
        release.wait(timeout=2)

    dispatcher = AgentEventDispatcher(slow_sink)
    dispatcher.publish(_run_state())
    assert entered.wait(timeout=1)
    dispatcher.publish(_commentary("discarded"))
    dispatcher.publish(_commentary("retained"))
    assert [item[1].message_id for item in dispatcher._commentary] == ["retained"]

    dispatcher.close(timeout=0)
    assert not dispatcher.active
    release.set()
    dispatcher._worker.join(timeout=1)
    assert not dispatcher._worker.is_alive()


def test_dispatcher_isolates_repeated_sink_failures() -> None:
    failed = Event()
    calls = 0

    def broken_sink(_event: AgentStreamEvent) -> None:
        nonlocal calls
        calls += 1
        if calls == event_module._SINK_FAILURE_LIMIT:
            failed.set()
        raise RuntimeError("sink failed")

    dispatcher = AgentEventDispatcher(broken_sink)
    for index in range(event_module._SINK_FAILURE_LIMIT):
        dispatcher.publish(_run_state(f"run-{index}"))

    assert failed.wait(timeout=1)
    dispatcher._worker.join(timeout=1)
    assert not dispatcher.active
    assert calls == event_module._SINK_FAILURE_LIMIT
    dispatcher.close()


def test_resolver_handles_empty_duplicate_explicit_and_terminal_messages() -> None:
    events: list[AgentStreamEvent] = []

    with observe_agent_events(events.append):
        with agent_event_run("fake") as resolver:
            assert resolver is not None
            resolver.deactivate()
            resolver.deactivate()
            resolver.append_delta("empty", "")
            resolver.append_delta("explicit", "A note", explicit_phase="commentary")
            resolver.complete_message("explicit")
            resolver.complete_message(
                "explicit",
                "A duplicate note",
                explicit_phase="commentary",
            )
            resolver.complete_message("blank", "   ")
            resolver.complete_message("older", "Older result")
            resolver.complete_message("answer", "Final result", explicit_phase="final")
            resolver.mark_tool_boundary()
            resolver.mark_message_boundary()
            resolver.complete_run(final_text="Final result")
            resolver.append_delta("late", "ignored")
            resolver.complete_message("late", "ignored")
            resolver.mark_tool_boundary()
            resolver.mark_message_boundary()
            resolver.complete_run(final_text="ignored")
            resolver.fail_run()

    messages = [event for event in events if isinstance(event, AgentMessageEvent)]
    assert [(event.phase, event.content) for event in messages] == [
        ("commentary", "A note"),
        ("commentary", "Older result"),
        ("final", "Final result"),
    ]


def test_resolver_promotes_only_matching_terminal_message() -> None:
    events: list[AgentStreamEvent] = []

    with observe_agent_events(events.append):
        with agent_event_run("fake") as resolver:
            assert resolver is not None
            resolver.complete_message("older", "First complete message")
            resolver.complete_message("final", "Terminal response")
            resolver.complete_run(final_text="Terminal response")

    messages = [event for event in events if isinstance(event, AgentMessageEvent)]
    assert [(event.phase, event.content) for event in messages] == [
        ("commentary", "First complete message"),
        ("final", "Terminal response"),
    ]


def test_resolver_direct_terminal_guards() -> None:
    dispatcher = AgentEventDispatcher(lambda _event: None)
    resolver = event_module.AgentMessageResolver(
        dispatcher,
        adapter="fake",
        identity=AgentRunIdentity("run"),
    )
    resolver.deactivate()
    resolver.complete_run()
    resolver.cancel_run()
    dispatcher.close()


def test_async_agent_event_queue_crosses_thread_boundary() -> None:
    async def scenario() -> None:
        implicit = AsyncAgentEventQueue()
        explicit = AsyncAgentEventQueue(loop=asyncio.get_running_loop())
        first = _run_state("implicit")
        second = _run_state("explicit")

        implicit.emit(first)
        explicit.emit(second)
        assert await asyncio.wait_for(implicit.queue.get(), timeout=1) is first
        assert await asyncio.wait_for(explicit.queue.get(), timeout=1) is second

    asyncio.run(scenario())


def test_pydantic_adapter_emits_final_only_after_completed_invoke() -> None:
    events: list[AgentStreamEvent] = []
    adapter = PydanticAdapter(TestModel(custom_output_text="done"))

    with observe_agent_events(events.append):
        assert asyncio.run(adapter.invoke(prompt="reply")) == "done"

    messages = [event for event in events if isinstance(event, AgentMessageEvent)]
    assert [(event.phase, event.content) for event in messages] == [("final", "done")]


def test_pydantic_event_capability_seals_text_at_tool_boundary() -> None:
    events: list[AgentStreamEvent] = []

    async def scenario() -> None:
        with observe_agent_events(events.append):
            with agent_event_run("pydantic") as resolver:
                assert resolver is not None
                capability = pydantic_adapter_module._KediAgentEventCapability(resolver)
                response = ModelResponse(
                    parts=[TextPart("Inspecting."), ToolCallPart("read_file", {})]
                )
                observed = await capability.after_model_request(
                    None,  # type: ignore[arg-type]
                    request_context=None,  # type: ignore[arg-type]
                    response=response,
                )
                assert observed is response

    asyncio.run(scenario())
    messages = [event for event in events if isinstance(event, AgentMessageEvent)]
    assert [(event.phase, event.content) for event in messages] == [("commentary", "Inspecting.")]


def test_pydantic_event_capability_accepts_tuple_and_rejects_invalid_capabilities() -> None:
    adapter = PydanticAdapter(TestModel(custom_output_text="done"))

    with observe_agent_events(lambda _event: None):
        assert asyncio.run(adapter.invoke(prompt="reply", capabilities=())) == "done"
        with pytest.raises(TypeError, match="list or tuple"):
            asyncio.run(adapter.invoke(prompt="reply", capabilities="invalid"))
