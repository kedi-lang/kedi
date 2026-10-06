import asyncio
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel

from kedi.agent_adapter import PydanticAdapter
from kedi.agent_adapter.execution import use_async_owner_loop
from kedi.engine import SequentialEngine
from kedi.lang import compile_program, parse_program

ROOT = Path(__file__).parent / "files/decision_composition"
POLICY = "The refund request is pending. No operator has approved it. Only an operator can approve a refund. There is no decision deadline."


@pytest.mark.parametrize("revise", [False, True])
def test_composition_uses_scoped_judge_and_bounded_revision(revise):
    writers, judges = [], []

    def writer(messages, info):
        text = " ".join(
            str(part.content)
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, UserPromptPart)
        )
        status = "pending" if not revise or "Correct this customer reply" in text else "approved"
        if not writers:
            assert ("The refund is approved." if revise else "The refund is pending.") in text
        writers.append(status)
        tool = info.output_tools[0]
        key = next(iter(tool.parameters_json_schema["properties"]))
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool.name, {key: {"message": f"Refund status: {status}.", "status": status}}
                )
            ]
        )

    def judge(messages, info):
        supported = not revise or bool(judges)
        judges.append(supported)
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, {"supported": supported})]
        )

    adapter = PydanticAdapter(FunctionModel(writer))
    adapter._resolve_model = lambda model: FunctionModel(judge)
    engine, runtime = SequentialEngine(), None
    try:
        runtime = compile_program(
            parse_program((ROOT / "reply.kedi").read_text()),
            adapter=adapter,
            engine=engine,
            runtime_globals={
                "policy": POLICY,
                "note": "The refund is approved." if revise else "The refund is pending.",
                "judge_model": "typesafe/test-judge",
            },
        )
        result = runtime.run_main()
        assert result["revisions"] == int(revise)
        assert result["initial_accepted"] is not revise
        assert result["final_accepted"] is True
        assert result["final"]["status"] == "pending"
        assert writers == (["approved", "pending"] if revise else ["pending"])
        assert judges == ([False, True] if revise else [True])
    finally:
        if runtime is not None:
            asyncio.run(runtime.aclose())
        engine.shutdown()


@pytest.mark.asyncio
async def test_writer_and_scoped_judge_execute_on_the_host_event_loop(monkeypatch):
    owner = asyncio.get_running_loop()
    calls = []

    async def answer(messages, info):
        assert asyncio.get_running_loop() is owner
        tool = info.output_tools[0]
        key = next(iter(tool.parameters_json_schema["properties"]))
        calls.append(key)
        value = True if key == "supported" else {"message": "Refund pending.", "status": "pending"}
        return ModelResponse(parts=[ToolCallPart(tool.name, {key: value})])

    model = FunctionModel(answer)
    adapter = PydanticAdapter(model)
    monkeypatch.setattr(adapter, "_resolve_model", lambda model_id: model)
    engine = SequentialEngine()
    runtime = compile_program(
        parse_program((ROOT / "reply.kedi").read_text()),
        adapter=adapter,
        engine=engine,
        runtime_globals={
            "policy": POLICY,
            "note": "The refund is pending.",
            "judge_model": "typesafe/test-judge",
        },
    )
    try:
        with use_async_owner_loop():
            result = await asyncio.wait_for(asyncio.to_thread(runtime.run_main), 5)
        assert result["revisions"] == 0
        assert result["final"]["status"] == "pending"
        assert result["initial_accepted"] is True
        assert result["final_accepted"] is True
        assert calls == ["reply", "supported"]
    finally:
        await runtime.aclose()
        engine.shutdown()
