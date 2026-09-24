from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import Any

import pytest
from langchain_core.messages import ToolMessage

from kedi.agent_adapter.adapters.langchain.middleware import _langchain_approval_middleware
from kedi.agent_adapter.hooks import (
    AdapterHookSnapshot,
    HookDispatcher,
    HookDispatcherSnapshot,
    PostToolResultFailureEvent,
    PostToolUseEvent,
    bind_hook_dispatcher,
    with_hook_lifecycle,
)
from kedi.agent_adapter.hooks.tool_wrapper import note_result_observer_failure
from kedi.agent_adapter.tools.execution import with_artifact_result
from kedi.agent_adapter.tools.records import (
    ToolCallTracker,
    ToolResultHandoff,
    attach_tool_call,
    track_tool_call,
)
from kedi.agent_adapter.tools.retry import ToolRetryPolicy, with_body_retry
from kedi.agent_adapter.tools.specs import ToolSpec
from kedi.agent_profile import AgentProfile, HookSettings


def dispatcher(handler: Any) -> HookDispatcher:
    return HookDispatcher(
        HookDispatcherSnapshot.compose(
            AdapterHookSnapshot.empty(),
            HookSettings(post_tool_use=(handler,), post_tool_result_failure=(handler,)),
        )
    )


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("invalid_result", [False, True])
def test_result_failure_observers_follow_the_notification_contract(
    asynchronous: bool, invalid_result: bool
) -> None:
    from kedi.agent_adapter.hooks.tool_wrapper import (
        dispatch_post_tool_result_failure,
        dispatch_post_tool_result_failure_sync,
    )

    events = []

    def first(event):
        events.append(("first", event))
        return "invalid" if invalid_result else None

    def second(event):
        events.append(("second", event))

    chain = HookDispatcher(
        HookDispatcherSnapshot.compose(
            AdapterHookSnapshot.empty(),
            HookSettings(post_tool_result_failure=(first, second)),
        )
    )
    error = ValueError("schema rejected result")
    tracker = ToolCallTracker("one")
    tracker.start()
    tracker.returned(None)
    record = tracker.result_failed("result_validation")

    def dispatch():
        args = (ToolSpec(name="body", fn=lambda: None), {}, error, record)
        kwargs = {"adapter_shortname": "fake", "tool_call_id": "one"}
        if asynchronous:
            return asyncio.run(dispatch_post_tool_result_failure(*args, **kwargs))
        return dispatch_post_tool_result_failure_sync(*args, **kwargs)

    with bind_hook_dispatcher(chain):
        if invalid_result:
            with pytest.raises(
                TypeError, match="post_tool_result_failure handlers must return None"
            ):
                dispatch()
        else:
            assert dispatch() is None
    assert [name for name, _ in events] == (["first"] if invalid_result else ["first", "second"])
    assert events[0][1].record is record
    assert events[0][1].sequence == 1


@pytest.mark.parametrize("callable_object", [False, True])
def test_partial_tool_preserves_bound_signature_for_both_framework_schemas(
    callable_object: bool,
) -> None:
    from functools import partial
    from inspect import signature
    from pydantic_ai.tools import Tool
    from langchain_core.tools import StructuredTool
    from kedi.agent_adapter.tools.conversion import python_callable_tool_spec

    def format_number(prefix: str, value: int) -> str:
        return f"{prefix}{value}"

    class Formatter:
        def __call__(self, prefix: str, value: int) -> str:
            return format_number(prefix, value)

    original = partial(Formatter() if callable_object else format_number, "n=")
    spec = python_callable_tool_spec("format_number", original)
    assert signature(spec.fn) == signature(original)
    pydantic_tool = Tool(spec.fn, name=spec.name)
    langchain_tool = StructuredTool.from_function(
        spec.fn, name=spec.name, description="Format a number."
    )
    for schema in (
        pydantic_tool.function_schema.json_schema,
        langchain_tool.get_input_schema().model_json_schema(),
    ):
        assert set(schema["properties"]) == {"value"}
        assert schema["properties"]["value"]["type"] == "integer"
        assert schema["required"] == ["value"]
    assert spec.fn(value=4) == "n=4"


def test_async_body_errors_keep_execution_failure_record() -> None:
    async def body() -> None:
        raise ValueError("body failure")

    wrapped = with_body_retry(body, ToolRetryPolicy(), isolate_arguments=False)
    tool = with_hook_lifecycle(ToolSpec(name="body", fn=wrapped), adapter_shortname="fake")
    with bind_hook_dispatcher(dispatcher(lambda event: None)):
        with pytest.raises(ValueError, match="body failure") as raised:
            asyncio.run(tool.fn())
        assert raised.value.tool_call.execution_state == "raised"

        middleware = _langchain_approval_middleware(
            profile=AgentProfile.empty(),
            tools=(ToolSpec(name="body", fn=wrapped),),
            approval_enabled=False,
        )

        async def handler(request: Request) -> None:
            await wrapped()

        with pytest.raises(ValueError, match="body failure") as raised:
            asyncio.run(
                middleware.awrap_tool_call(
                    Request({"name": "body", "args": {}, "id": "one"}), handler
                )
            )
        assert raised.value.tool_call.execution_state == "raised"


@pytest.mark.parametrize("recorded", [False, True])
@pytest.mark.parametrize("observer_fails", [False, True])
def test_pydantic_post_hook_errors_preserve_available_records(
    recorded: bool, observer_fails: bool
) -> None:
    from pydantic_ai.models.test import TestModel
    from pydantic_ai.messages import ToolCallPart
    from pydantic_ai.tools import ToolDefinition
    from kedi.agent_adapter.adapters.pydantic import PydanticAdapter
    from kedi.agent_adapter.adapters.pydantic.hooks import _KediHookCapability

    capability = _KediHookCapability(PydanticAdapter(TestModel()))
    observations = []
    if recorded:
        tracker = ToolCallTracker("one")
        tracker.start()
        tracker.returned("native")
        capability._records["one"] = tracker

    def observe(event: Any) -> None:
        if isinstance(event, PostToolUseEvent):
            raise RuntimeError("post observer failed")
        if isinstance(event, PostToolResultFailureEvent):
            observations.append(event)
            if observer_fails:
                raise ValueError("failure observer also failed")

    with (
        bind_hook_dispatcher(dispatcher(observe)),
        pytest.raises(RuntimeError, match="post observer failed") as raised,
    ):
        asyncio.run(
            capability.after_tool_execute(
                None,
                call=ToolCallPart("body", {}, tool_call_id="one"),
                tool_def=ToolDefinition(name="body"),
                args={},
                result="native",
            )
        )
    if recorded:
        assert raised.value.tool_call.output == "native"
        assert raised.value.tool_call.failure_stage == "post_hook"
        assert len(observations) == 1
    else:
        assert raised.value.tool_call is None
        assert not observations
    if recorded and observer_fails and callable(getattr(raised.value, "add_note", None)):
        assert "observer raised" in raised.value.__notes__[0]
    else:
        assert not getattr(raised.value, "__notes__", ())
    assert not capability._records


@pytest.mark.parametrize("style", ["async", "awaitable", "sync"])
@pytest.mark.parametrize("recorded", [True, False])
def test_post_hook_failure_never_repeats_body_and_preserves_original_error(
    style: str, recorded: bool
) -> None:
    bodies = 0
    observations = []

    async def body_async() -> None:
        nonlocal bodies
        bodies += 1

    def body_sync() -> Any:
        nonlocal bodies
        if style == "awaitable":
            return body_async()
        bodies += 1
        return None

    def observer(event: Any) -> None:
        if isinstance(event, PostToolUseEvent):
            raise RuntimeError("post observer failed")
        if isinstance(event, PostToolResultFailureEvent):
            observations.append(event)
            raise ValueError("failure observer also failed")

    body = body_async if style == "async" else body_sync
    wrapped = (
        with_body_retry(body, ToolRetryPolicy(retries=2), isolate_arguments=False)
        if recorded
        else body
    )
    tool = with_hook_lifecycle(ToolSpec(name="body", fn=wrapped), adapter_shortname="fake")
    with (
        bind_hook_dispatcher(dispatcher(observer)),
        pytest.raises(RuntimeError, match="post observer failed") as raised,
    ):
        if style == "sync":
            tool.fn()
        else:
            asyncio.run(tool.fn())
    assert bodies == 1
    if recorded:
        assert raised.value.tool_call.output_available
        assert raised.value.tool_call.output is None
        assert raised.value.tool_call.failure_stage == "post_hook"
        assert len(observations) == 1
        if callable(getattr(raised.value, "add_note", None)):
            assert "observer raised" in raised.value.__notes__[0]
        else:
            assert not getattr(raised.value, "__notes__", ())
    else:
        assert not observations


@pytest.mark.parametrize("style", ["async", "awaitable", "admitter"])
def test_artifact_admission_errors_are_classified_after_success(
    style: str, monkeypatch: Any
) -> None:
    import kedi.agent_adapter.tools.execution as execution

    async def body() -> str:
        return "native"

    def awaitable() -> Any:
        return body()

    async def reject(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("admission failed")

    monkeypatch.setattr(execution, "admit_artifact_result_async", reject)
    tool = with_artifact_result(
        ToolSpec(name="body", fn=body if style == "async" else awaitable),
        manager=object(),
        policy=object(),
    )
    with pytest.raises(RuntimeError) as raised:
        asyncio.run(tool.result_admitter("native", "body") if style == "admitter" else tool.fn())
    assert raised.value.kedi_result_failure_stage == "artifact"


@pytest.mark.parametrize("tracked", [False, True])
@pytest.mark.parametrize(
    "outcome", ["sync_success", "sync_failure", "async_failure", "async_success"]
)
def test_sync_awaitable_retry_tracks_every_kind_of_attempt(tracked: bool, outcome: str) -> None:
    calls = 0

    async def fail() -> None:
        raise TimeoutError("asynchronous failure")

    async def succeed() -> str:
        return "ok"

    def body() -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            return fail()
        if outcome == "sync_failure" or (outcome == "sync_success" and calls == 2):
            raise TimeoutError("synchronous failure")
        if outcome == "async_failure":
            return fail()
        return succeed() if outcome == "async_success" else "ok"

    fn = with_body_retry(body, ToolRetryPolicy(retries=2), isolate_arguments=tracked)

    def run() -> None:
        if outcome.endswith("failure"):
            with pytest.raises(TimeoutError):
                asyncio.run(fn())
        else:
            assert asyncio.run(fn()) == "ok"

    if tracked:
        with track_tool_call("logical") as tracker:
            run()
        assert len(tracker.records) == calls
    else:
        run()
    assert calls == (2 if outcome == "async_success" else 3)


def test_record_helpers_handle_absent_and_unattachable_evidence() -> None:
    tracker = ToolCallTracker("empty")
    assert tracker.ready() is None and tracker.handed_off() is None
    assert tracker.result_failed("handoff") is None
    with pytest.raises(RuntimeError, match="not started"):
        tracker.update(result_state="ready")
    tracker.start()
    assert tracker.result_failed("handoff") is None
    handoff = ToolResultHandoff()
    asyncio.run(handoff.fail(RuntimeError(), "handoff"))
    handoff.tracker = tracker
    asyncio.run(handoff.fail(RuntimeError(), "handoff"))
    tracker.returned(None)
    error = RuntimeError()
    asyncio.run(handoff.fail(error, "handoff"))
    assert error.tool_call.output_available

    class LockedError(RuntimeError):
        def __setattr__(self, name: str, value: Any) -> None:
            raise AttributeError(name)

    attach_tool_call(LockedError(), tracker.latest)
    attach_tool_call(error, None)
    error.add_note = None
    note_result_observer_failure(error, RuntimeError())


@pytest.mark.parametrize("style", ["async", "awaitable", "sync"])
def test_validation_error_after_return_and_control_errors_are_not_execution_failures(
    style: str,
) -> None:
    from kedi.agent_adapter.hooks import HookDeniedError
    from kedi.agent_adapter.tools.errors import correctable_tool_error

    native = with_body_retry(lambda: "native", ToolRetryPolicy(retries=2), isolate_arguments=False)

    async def validated_async() -> str:
        native()
        raise ValueError("result rejected")

    def validated_sync() -> Any:
        if style == "awaitable":
            return validated_async()
        native()
        raise ValueError("result rejected")

    tool = with_hook_lifecycle(
        ToolSpec(name="body", fn=validated_async if style == "async" else validated_sync),
        adapter_shortname="fake",
    )
    with pytest.raises(ValueError) as raised:
        if style == "sync":
            tool.fn()
        else:
            asyncio.run(tool.fn())
    assert raised.value.tool_call.output == "native"
    assert correctable_tool_error(tool, raised.value) is None

    async def denied_async() -> None:
        raise HookDeniedError(event="pre_tool_use", reason="denied")

    def denied_sync() -> Any:
        if style == "awaitable":
            return denied_async()
        raise HookDeniedError(event="pre_tool_use", reason="denied")

    denied = with_body_retry(
        denied_async if style == "async" else denied_sync,
        ToolRetryPolicy(retries=2),
        isolate_arguments=False,
    )
    tool = with_hook_lifecycle(ToolSpec(name="body", fn=denied), adapter_shortname="fake")
    with pytest.raises(HookDeniedError):
        if style == "sync":
            tool.fn()
        else:
            asyncio.run(tool.fn())


def test_result_failure_dispatch_can_be_disabled_and_unresolved_annotations_are_retained() -> None:
    from kedi.agent_adapter.hooks.tool_wrapper import dispatch_post_tool_result_failure_sync

    tracker = ToolCallTracker("one")
    record = tracker.start()
    dispatch_post_tool_result_failure_sync(
        ToolSpec(name="body", fn=lambda: None),
        {},
        ValueError(),
        record,
        adapter_shortname="fake",
        tool_call_id="one",
    )

    class Callback:
        def __call__(self, value: "UnresolvedAnnotation") -> str:  # noqa: F821
            return str(value)

    wrapped = with_body_retry(Callback(), ToolRetryPolicy(), isolate_arguments=False)
    assert wrapped.__annotations__["value"]
    assert wrapped("ok") == "ok"


@pytest.mark.parametrize("observer_fails", [False, True])
def test_pydantic_preserves_returned_evidence_when_framework_validation_fails(
    observer_fails: bool,
) -> None:
    from pydantic_ai.models.test import TestModel
    from pydantic_ai.messages import ToolCallPart
    from pydantic_ai.tools import ToolDefinition
    from kedi.agent_adapter.adapters.pydantic import PydanticAdapter
    from kedi.agent_adapter.adapters.pydantic.hooks import _KediHookCapability

    adapter = PydanticAdapter(TestModel())
    capability = _KediHookCapability(adapter)
    tracker = ToolCallTracker("one")
    tracker.start()
    tracker.returned(None)
    capability._records["one"] = tracker
    adapter._remember_hook_tool("one", ToolSpec(name="body", fn=lambda: None), "kedi")
    events = []

    def observe(event):
        if isinstance(event, PostToolResultFailureEvent):
            events.append(event)
            if observer_fails:
                raise RuntimeError("observer failed")

    error = ValueError("schema rejected result")
    with bind_hook_dispatcher(dispatcher(observe)):
        asyncio.run(
            capability._dispatch_failure(
                call=ToolCallPart("body", {}, tool_call_id="one"),
                tool_def=ToolDefinition(name="body"),
                args={},
                error=error,
            )
        )
    assert error.tool_call.output_available and error.tool_call.output is None
    assert error.tool_call.failure_stage == "result_validation"
    assert len(events) == 1
    if observer_fails and callable(getattr(error, "add_note", None)):
        assert error.__notes__
    else:
        assert not getattr(error, "__notes__", ())


@dataclass(frozen=True)
class Request:
    tool_call: dict[str, Any]
    tool: Any = None

    def override(self, **changes: Any) -> Request:
        return replace(self, **changes)


@pytest.mark.parametrize("mode", ["raise", "artifact", "status"])
@pytest.mark.parametrize("tracked", [False, True])
def test_langchain_result_boundary_failures_do_not_become_body_retries(
    mode: str, tracked: bool, monkeypatch: Any
) -> None:
    import kedi.agent_adapter.adapters.langchain.middleware as module

    def body() -> str:
        return "native"

    fn = (
        with_body_retry(body, ToolRetryPolicy(retries=2), isolate_arguments=False)
        if tracked
        else body
    )
    tool = ToolSpec(name="body", fn=fn)
    middleware = _langchain_approval_middleware(
        profile=AgentProfile.empty(), tools=(tool,), approval_enabled=False
    )
    failures = []

    def observe(event: Any) -> None:
        if isinstance(event, PostToolResultFailureEvent):
            failures.append(event)
            raise RuntimeError("observer failure")

    async def reject(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("artifact failure")

    if mode == "artifact":
        monkeypatch.setattr(module, "_langchain_admit_native_tool_result", reject)

    async def handler(request: Request) -> ToolMessage:
        fn()
        if mode == "raise":
            raise RuntimeError("validation failure")
        return ToolMessage(
            content="invalid" if mode == "status" else "native",
            tool_call_id="one",
            status="error" if mode == "status" else "success",
        )

    async def run() -> None:
        with bind_hook_dispatcher(dispatcher(observe)):
            if mode == "status" and not tracked:
                result = await middleware.awrap_tool_call(
                    Request({"name": "body", "args": {}, "id": "one"}), handler
                )
                assert result.status == "error"
            else:
                with pytest.raises(RuntimeError) as raised:
                    await middleware.awrap_tool_call(
                        Request({"name": "body", "args": {}, "id": "one"}), handler
                    )
                if tracked:
                    assert raised.value.tool_call.output == "native"
                    assert raised.value.tool_call.execution_state == "returned"
                    assert len(failures) == 1

    asyncio.run(run())
