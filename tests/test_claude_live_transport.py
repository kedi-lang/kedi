from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import TracebackType

import pytest
from pydantic import JsonValue, TypeAdapter

pytest.importorskip("claude_agent_sdk")
import claude_agent_sdk
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    Message,
    ResultMessage,
)
from claude_agent_sdk._internal.transport import Transport
from claude_agent_sdk.types import HookInput, NotificationHookInput

from kedi.agent_adapter.adapters.claude.live import run_query
from kedi.agent_adapter.subagents import SubagentConversationContext
from kedi.agent_adapter.subagents.control import LiveInput, RunInterrupted, live_input_scope

WIRE_MESSAGE = TypeAdapter(dict[str, JsonValue])


class RecordingTransport(Transport):
    """Exercise the installed SDK's control protocol without a CLI or paid model."""

    def __init__(self) -> None:
        self.incoming: asyncio.Queue[dict[str, JsonValue] | None] = asyncio.Queue()
        self.started = asyncio.Event()
        self.connected = False
        self.closed = False
        self.initialization: dict[str, JsonValue] = {}
        self.hook_outputs: list[dict[str, JsonValue]] = []
        self.interrupts = 0

    async def connect(self) -> None:
        self.connected = True

    def is_ready(self) -> bool:
        return self.connected and not self.closed

    async def close(self) -> None:
        self.closed = True
        await self.incoming.put(None)

    async def end_input(self) -> None:
        return None

    async def read_messages(self) -> AsyncIterator[dict[str, JsonValue]]:
        while True:
            message = await self.incoming.get()
            if message is None:
                return
            yield message

    async def write(self, data: str) -> None:
        message = WIRE_MESSAGE.validate_json(data)
        if message["type"] == "control_request":
            request = message["request"]
            assert isinstance(request, dict)
            if request["subtype"] == "initialize":
                self.initialization = request
            elif request["subtype"] == "interrupt":
                self.interrupts += 1
            else:
                pytest.fail(f"Unexpected control request: {request['subtype']}")
            await self.incoming.put(
                {
                    "type": "control_response",
                    "response": {
                        "subtype": "success",
                        "request_id": message["request_id"],
                        "response": {},
                    },
                }
            )
            if request["subtype"] == "interrupt":
                await self.finish()
        elif message["type"] == "user":
            self.started.set()
        elif message["type"] == "control_response":
            response = message["response"]
            assert isinstance(response, dict) and response["subtype"] == "success"
            output = response["response"]
            assert isinstance(output, dict)
            self.hook_outputs.append(output)
            await self.finish()
        else:
            pytest.fail(f"Unexpected SDK write: {message['type']}")

    async def finish(self) -> None:
        await self.incoming.put(
            {
                "type": "assistant",
                "session_id": "retained-session",
                "message": {
                    "model": "test-model",
                    "content": [{"type": "text", "text": "Updated review."}],
                },
            }
        )
        await self.incoming.put(
            {
                "type": "result",
                "subtype": "success",
                "duration_ms": 1,
                "duration_api_ms": 1,
                "is_error": False,
                "num_turns": 1,
                "session_id": "retained-session",
                "result": "Updated review.",
            }
        )

    async def request_stop_hook(self) -> None:
        await self.request_hook_callback(
            {
                "hook_event_name": "Stop",
                "session_id": "retained-session",
                "transcript_path": "/tmp/transcript.jsonl",
                "cwd": "/tmp",
                "stop_hook_active": False,
            }
        )

    async def request_hook_callback(self, input: HookInput) -> None:
        hooks = self.initialization["hooks"]
        assert isinstance(hooks, dict)
        matchers = hooks["Stop"]
        assert isinstance(matchers, list) and isinstance(matchers[-1], dict)
        callbacks = matchers[-1]["hookCallbackIds"]
        assert isinstance(callbacks, list) and isinstance(callbacks[0], str)
        await self.incoming.put(
            {
                "type": "control_request",
                "request_id": "stop-hook",
                "request": {
                    "subtype": "hook_callback",
                    "callback_id": callbacks[0],
                    "input": WIRE_MESSAGE.validate_python(input),
                    "tool_use_id": None,
                },
            }
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt", [False, True])
@pytest.mark.parametrize("through_adapter", [False, True])
async def test_real_claude_sdk_delivers_input_and_drains_interrupt_with_shared_task_context(
    monkeypatch: pytest.MonkeyPatch, interrupt: bool, through_adapter: bool
) -> None:
    from kedi.agent_adapter.adapters.claude import ClaudeAdapter

    transport = RecordingTransport()
    real_client = ClaudeSDKClient

    def client(*, options: ClaudeAgentOptions) -> ClaudeSDKClient:
        return real_client(options=options, transport=transport)

    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", client)
    control = LiveInput("worker")
    control.configure(input=True, interrupt=True)
    initial = control.admit("Initial evidence.", sender_id="parent")
    claimed = control.take_pending()
    observed: list[Message] = []
    conversation = SubagentConversationContext(conversation_id="live-review")

    async def execute() -> ResultMessage | str:
        prompt = "Review.\n\n" + claimed[0].render()
        if through_adapter:
            adapter = ClaudeAdapter(model="test-model")
            with live_input_scope(control), adapter.conversation_scope(conversation):
                return await adapter.invoke(prompt=prompt)
        return await run_query(ClaudeAgentOptions(), prompt, control, observed.append)

    task = asyncio.create_task(execute())
    try:
        await asyncio.wait_for(transport.started.wait(), 2)
        assert initial.message_id in control.delivered
        receipt = control.admit("Review tokens only.", sender_id="parent", interrupt=interrupt)
        if interrupt:
            with pytest.raises(RunInterrupted):
                await asyncio.wait_for(task, 2)
            assert transport.interrupts == 1
        else:
            await transport.request_stop_hook()
            result = await asyncio.wait_for(task, 2)
            assert (result if isinstance(result, str) else result.result) == "Updated review."
            assert transport.interrupts == 0
            assert transport.hook_outputs[0]["decision"] == "block"
            assert "Review tokens only." in str(transport.hook_outputs[0]["reason"])
            assert receipt.message_id in control.delivered
        if through_adapter:
            assert conversation.result_state["claude_session_id"] == "retained-session"
            assert "claude_session_fingerprint" in conversation.result_state
        else:
            assert len(observed) == 2
            assert isinstance(observed[0], AssistantMessage)
            assert isinstance(observed[1], ResultMessage)
            assert observed[1].session_id == "retained-session"
        assert transport.closed
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("suppress_context", [False, True])
async def test_real_claude_sdk_query_write_failure_does_not_confirm_input(
    monkeypatch: pytest.MonkeyPatch,
    suppress_context: bool,
) -> None:
    from kedi.agent_adapter.subagents.control import UnconfirmedStop

    class DisconnectedTransport(RecordingTransport):
        async def write(self, data: str) -> None:
            if WIRE_MESSAGE.validate_json(data)["type"] == "user":
                raise ConnectionError("query write failed")
            await super().write(data)

    transport = DisconnectedTransport()
    real_client = ClaudeSDKClient

    class SuppressingClient(ClaudeSDKClient):
        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType | None,
        ) -> bool:
            await super().__aexit__(exc_type, exc_val, exc_tb)
            return True

    def client(*, options: ClaudeAgentOptions) -> ClaudeSDKClient:
        client_type = SuppressingClient if suppress_context else real_client
        return client_type(options=options, transport=transport)

    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", client)
    control = LiveInput("worker")
    control.configure(input=True)
    control.admit("Initial evidence.", sender_id="parent")
    claimed = control.take_pending()
    with pytest.raises(UnconfirmedStop, match="terminal confirmation") as failure:
        await asyncio.wait_for(
            run_query(ClaudeAgentOptions(), claimed[0].render(), control, lambda message: None), 2
        )
    if suppress_context:
        assert failure.value.__cause__ is None
        assert "client closed" in str(failure.value)
    else:
        assert isinstance(failure.value.__cause__, ConnectionError)
    assert not control.delivered and control.waiting_input
    assert transport.closed


@pytest.mark.asyncio
async def test_real_claude_sdk_unexpected_hook_does_not_consume_queued_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = RecordingTransport()
    real_client = ClaudeSDKClient

    def client(*, options: ClaudeAgentOptions) -> ClaudeSDKClient:
        return real_client(options=options, transport=transport)

    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", client)
    control = LiveInput("worker")
    control.configure(input=True)
    task = asyncio.create_task(
        run_query(ClaudeAgentOptions(), "Review.", control, lambda message: None)
    )
    try:
        await asyncio.wait_for(transport.started.wait(), 2)
        receipt = control.admit("Additional evidence.", sender_id="parent")
        event: NotificationHookInput = {
            "hook_event_name": "Notification",
            "session_id": "retained-session",
            "transcript_path": "/tmp/transcript.jsonl",
            "cwd": "/tmp",
            "message": "SDK lifecycle notification",
            "notification_type": "idle_prompt",
        }
        await transport.request_hook_callback(event)
        result = await asyncio.wait_for(task, 2)
        assert result.result == "Updated review."
        assert transport.hook_outputs == [{}]
        assert receipt.message_id not in control.delivered
        assert control.waiting_input
        assert transport.closed
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_real_claude_sdk_redirect_during_connection_never_submits_old_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = LiveInput("worker")
    control.configure(input=True, interrupt=True)

    class RedirectedTransport(RecordingTransport):
        async def connect(self) -> None:
            await super().connect()
            control.admit("replacement", sender_id="parent", interrupt=True)

    transport = RedirectedTransport()
    real_client = ClaudeSDKClient

    def client(*, options: ClaudeAgentOptions) -> ClaudeSDKClient:
        return real_client(options=options, transport=transport)

    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", client)
    with pytest.raises(RunInterrupted):
        await asyncio.wait_for(
            run_query(ClaudeAgentOptions(), "Obsolete task.", control, lambda message: None), 2
        )
    assert not transport.started.is_set()
    assert transport.closed
