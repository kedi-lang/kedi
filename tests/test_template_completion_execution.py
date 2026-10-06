import asyncio
import re
from pathlib import Path
from threading import Lock

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from kedi.agent_adapter import PydanticAdapter
from kedi.agent_adapter.execution import use_async_owner_loop
from kedi.agent_adapter.invocation import observe_invocations
from kedi.engine import ThreadEngine
from kedi.lang import compile_program, parse_program

ROOT = Path(__file__).parent / "files/completion_forms"
DESCRIPTIONS = (
    "Three sealed cartons contain seven units each. One unit is damaged.",
    "One item is unusable; the shipment consists of seven items per box and three boxes.",
    "Five items are unusable; the shipment consists of seven items per box and three boxes.",
)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["no_output", "paraphrases"])
async def test_templates_execute_effects_and_bind_each_iteration(name):
    effects, outputs, usage = [], [], []
    lock = Lock()
    code = "a" * 32

    async def respond(messages, info):
        if name == "no_output":
            if any(
                isinstance(part, ToolReturnPart)
                for message in messages
                if isinstance(message, ModelRequest)
                for part in message.parts
            ):
                return ModelResponse(parts=[TextPart("Recorded.")])
            supplied = re.search(r"delivery code ([a-f0-9]{32})", str(messages)).group(1)
            return ModelResponse(parts=[ToolCallPart("record_delivery", {"code": supplied})])
        value = 16 if "Five items" in str(messages) else 20
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"usable": value})])

    def record_delivery(code: str) -> str:
        effects.append(code)
        return "recorded"

    def record_result(description, usable):
        with lock:
            outputs.append((description, usable))

    engine = ThreadEngine(2)
    runtime = compile_program(
        parse_program((ROOT / f"{name}.kedi").read_text()),
        adapter=PydanticAdapter(FunctionModel(respond)),
        engine=engine,
        runtime_globals={
            "delivery_code": code,
            "record_delivery": record_delivery,
            "descriptions": DESCRIPTIONS,
            "record_result": record_result,
        },
    )
    try:
        with use_async_owner_loop(), observe_invocations(usage.append):
            assert await asyncio.wait_for(asyncio.to_thread(runtime.run_main), 10) == "done"
        if name == "no_output":
            assert effects == [code]
        else:
            assert len(outputs) == 3
            assert dict(outputs) == dict(zip(DESCRIPTIONS, (20, 20, 16), strict=True))
            assert all(type(value) is int for _, value in outputs)
            assert sum(item.requests for item in usage) == 3
    finally:
        await runtime.aclose()
        engine.shutdown()
