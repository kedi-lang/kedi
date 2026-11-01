from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError, model_validator
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from kedi.agent_adapter import PydanticAdapter
from kedi.agent_adapter.execution import use_async_owner_loop
from kedi.agent_adapter.subagents import SubagentOutputState
from kedi.agent_adapter.subagents.runs import SubagentCompleted, SubagentFailed
from kedi.lang import compile_program, parse_program
from tests.retry_feedback import retry_feedback
from tests.test_subagents import RecordingPydanticAdapter


class Appointment(BaseModel):
    day: date
    starts_at: datetime
    identifier: UUID
    seats: tuple[int, int]
    reminders: list[date | None]


WIRE_APPOINTMENT = {
    "day": "2026-10-19",
    "starts_at": "2026-10-19T12:30:00Z",
    "identifier": "b9bf6a87-4d2c-4680-a0f3-7882c0ddaf41",
    "seats": [1, 2],
    "reminders": ["2026-10-18", None],
}


def task_source(capture: bool) -> str:
    output = "" if capture else "  > output: Appointment\n"
    prompt = "The appointment is [appointment: Appointment]." if capture else "Schedule a visit."
    return (
        "> profile: worker:\n"
        "  > instructions: Return the scheduled appointment.\n"
        f"{output}"
        "> profile: coordinator:\n"
        "  > subagent: worker\n"
        "> use: coordinator\n"
        f"> task [job]: worker:\n  >> {prompt}\n"
        "> await [result]: job\n"
        "> await [again]: job\n"
        f"= `result.output{'.appointment' if capture else ''}`\n"
    )


def child_adapter(
    candidates: list[dict[str, Any]], *, native: bool
) -> tuple[PydanticAdapter[Any, Any], list[dict[str, Any]], list[Any]]:
    requests: list[dict[str, Any]] = []
    feedback: list[Any] = []

    def candidate() -> dict[str, Any]:
        value = candidates[min(len(requests), len(candidates) - 1)]
        requests.append(value)
        return value

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        feedback.extend(retry_feedback(messages))
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, candidate())])

    class FallbackAdapter(RecordingPydanticAdapter):
        async def produce_subagent(
            self, *, template: str, output_schema: dict[str, Any], **kwargs: Any
        ) -> dict[str, Any]:
            return candidate()

    adapter = (
        PydanticAdapter(FunctionModel(respond), retries={"output": 1})
        if native
        else FallbackAdapter()
    )
    return adapter, requests, feedback


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [True, False], ids=["native-repair", "shared-fallback"])
@pytest.mark.parametrize("capture", [True, False], ids=["capture", "profile-output"])
async def test_child_json_values_become_native_types_before_completion(
    tmp_path: Path, native: bool, capture: bool
) -> None:
    validations = 0

    class CountedAppointment(Appointment):
        @model_validator(mode="after")
        def record_validation(self) -> CountedAppointment:
            nonlocal validations
            validations += 1
            return self

    final = {"appointment": WIRE_APPOINTMENT} if capture else WIRE_APPOINTMENT
    envelope = {"task_summary": "Scheduled", "final_result": final}
    adapter, requests, feedback = child_adapter([envelope], native=native)
    state_path = tmp_path / "children.json"
    with use_async_owner_loop():
        runtime = compile_program(
            parse_program(task_source(capture)),
            adapter=adapter,
            runtime_globals={"Appointment": CountedAppointment},
            subagent_state_path=state_path,
        )
        try:
            result = await asyncio.to_thread(runtime.run_main)
            assert isinstance(result, Appointment)
            assert result.day == date(2026, 10, 19)
            assert result.starts_at == datetime(2026, 10, 19, 12, 30, tzinfo=timezone.utc)
            assert result.identifier == UUID(WIRE_APPOINTMENT["identifier"])
            assert result.seats == (1, 2)
            assert result.reminders == [date(2026, 10, 18), None]
            assert runtime._globals_env["result"] is runtime._globals_env["again"]
            events = runtime._subagent_coordinator.events.snapshot()
            assert sum(isinstance(event, SubagentCompleted) for event in events) == 1
            assert not any(isinstance(event, SubagentFailed) for event in events)
        finally:
            await runtime.aclose()
    assert len(requests) == 1
    assert validations == 1
    assert not feedback
    saved = json.loads(state_path.read_text())
    assert next(iter(saved["results"].values()))["final_result"] == final


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [True, False], ids=["native-repair", "shared-fallback"])
@pytest.mark.parametrize("corrected", [True, False], ids=["corrected", "always-invalid"])
async def test_invalid_json_native_value_repairs_only_within_native_retry_loop(
    tmp_path: Path, native: bool, corrected: bool
) -> None:
    bad = {**WIRE_APPOINTMENT, "day": "2026-02-30"}
    candidates = [
        {"task_summary": "Scheduled", "final_result": {"appointment": bad}},
        {
            "task_summary": "Scheduled",
            "final_result": {"appointment": WIRE_APPOINTMENT if corrected else bad},
        },
    ]
    adapter, requests, feedback = child_adapter(candidates, native=native)
    state_path = tmp_path / "children.json"
    with use_async_owner_loop():
        runtime = compile_program(
            parse_program(task_source(True)),
            adapter=adapter,
            runtime_globals={"Appointment": Appointment},
            subagent_state_path=state_path,
        )
        try:
            if native and corrected:
                result = await asyncio.to_thread(runtime.run_main)
                assert result.day == date(2026, 10, 19)
            else:
                message = "Exceeded maximum output retries" if native else "valid date"
                with pytest.raises(Exception, match=message):
                    await asyncio.to_thread(runtime.run_main)
                events = runtime._subagent_coordinator.events.snapshot()
                assert not any(isinstance(event, SubagentCompleted) for event in events)
                assert sum(isinstance(event, SubagentFailed) for event in events) == 1
        finally:
            await runtime.aclose()
    assert len(requests) == (2 if native else 1)
    if native:
        assert len(feedback) == 1
        assert "final_result.appointment.day" in str(feedback[0].content)
    else:
        assert not feedback
    if not (native and corrected):
        assert json.loads(state_path.read_text())["results"] == {}


@pytest.mark.parametrize(
    ("output_type", "value"),
    [
        (int, "3"),
        (bool, "true"),
        (bool, 1),
        (bool, 0),
        (tuple[int, int], [1]),
        (tuple[int, int], [1, 2, 3]),
        (tuple[int, int], [1, "2"]),
    ],
)
def test_child_json_conversion_keeps_primitive_types_strict(output_type: Any, value: Any) -> None:
    state = SubagentOutputState(TypeAdapter(output_type))
    with pytest.raises(ValidationError):
        state.validate_candidate(value)


@pytest.mark.parametrize("value", [object(), float("nan"), float("inf")])
def test_child_json_conversion_rejects_non_json_values(value: Any) -> None:
    state = SubagentOutputState(TypeAdapter(Any))
    with pytest.raises((TypeError, ValueError)):
        state.validate_candidate(value)


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [True, False], ids=["native-repair", "shared-fallback"])
async def test_native_task_materializes_enum_in_supported_capture_schema(native: bool) -> None:
    class State(str, Enum):
        READY = "ready"

    adapter, requests, _ = child_adapter(
        [{"task_summary": "Checked", "final_result": {"state": "ready"}}], native=native
    )
    source = (
        task_source(True)
        .replace(
            "The appointment is [appointment: Appointment].", "The review state is [state: State]."
        )
        .replace("result.output.appointment", "result.output.state")
    )
    with use_async_owner_loop():
        runtime = compile_program(
            parse_program(source), adapter=adapter, runtime_globals={"State": State}
        )
        try:
            assert await asyncio.to_thread(runtime.run_main) is State.READY
        finally:
            await runtime.aclose()
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [True, False], ids=["native-repair", "shared-fallback"])
async def test_unexpected_native_validator_error_is_not_retried(
    tmp_path: Path, native: bool
) -> None:
    class BrokenAppointment(Appointment):
        @model_validator(mode="after")
        def reject(self) -> BrokenAppointment:
            raise RuntimeError("Application validator failed")

    adapter, requests, feedback = child_adapter(
        [{"task_summary": "Scheduled", "final_result": {"appointment": WIRE_APPOINTMENT}}],
        native=native,
    )
    state_path = tmp_path / "children.json"
    with use_async_owner_loop():
        runtime = compile_program(
            parse_program(task_source(True)),
            adapter=adapter,
            runtime_globals={"Appointment": BrokenAppointment},
            subagent_state_path=state_path,
        )
        try:
            with pytest.raises(Exception, match="Application validator failed"):
                await asyncio.to_thread(runtime.run_main)
            assert not any(
                isinstance(event, SubagentCompleted)
                for event in runtime._subagent_coordinator.events.snapshot()
            )
        finally:
            await runtime.aclose()
    assert len(requests) == 1
    assert not feedback
    assert json.loads(state_path.read_text())["results"] == {}


@pytest.mark.asyncio
async def test_concurrent_children_keep_distinct_native_json_contracts() -> None:
    started: set[str] = set()
    both_started = asyncio.Event()

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        schema = info.output_tools[0].parameters_json_schema["properties"]["final_result"]
        field = next(iter(schema["properties"]))
        started.add(field)
        if len(started) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=5)
        value = WIRE_APPOINTMENT["day" if field == "day" else "identifier"]
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {"task_summary": "Checked", "final_result": {field: value}},
                )
            ]
        )

    source = """
> profile: worker:
  > instructions: Return the requested value.
> profile: coordinator:
  > subagent: worker
  > max_agents: 2
> use: coordinator
> task [date_job]: worker:
  >> The scheduled day is [day: date].
> task [id_job]: worker:
  >> The identifier is [identifier: UUID].
> await [dated]: date_job
> await [identified]: id_job
= `(dated.output.day, identified.output.identifier)`
"""
    with use_async_owner_loop():
        runtime = compile_program(
            parse_program(source),
            adapter=PydanticAdapter(FunctionModel(respond), retries={"output": 0}),
            runtime_globals={"date": date, "UUID": UUID},
        )
        try:
            assert await asyncio.to_thread(runtime.run_main) == (
                date(2026, 10, 19),
                UUID(WIRE_APPOINTMENT["identifier"]),
            )
        finally:
            await runtime.aclose()
    assert started == {"day", "identifier"}
