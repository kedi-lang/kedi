import asyncio
import re
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from threading import Lock

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel

from kedi.agent_adapter import PydanticAdapter
from kedi.agent_adapter.execution import use_async_owner_loop
from kedi.agent_adapter.invocation import observe_invocations
from kedi.engine import SequentialEngine, ThreadEngine
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["map_failure", "show_return"])
async def test_failed_map_drains_and_show_respects_early_return(name):
    effects, usages, prompts = [], [], []
    items = [f"record-{i}:{i:032x}" for i in range(3)]
    receipt, output, lock = "f" * 32, StringIO(), Lock()

    async def answer(messages, info):
        text = " ".join(
            str(part.content)
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, UserPromptPart)
        )
        nonce = re.search(r"\b[0-9a-f]{32}\b", text).group(0)
        tool = info.output_tools[0]
        field = next(iter(tool.parameters_json_schema["properties"]))
        return ModelResponse(parts=[ToolCallPart(tool.name, {field: nonce})])

    def commit(item, identity):
        with lock:
            effects.append((item, identity))
        if item == items[1]:
            raise ValueError("Injected map item failure: record-1")

    adapter = PydanticAdapter(FunctionModel(answer))
    adapter.on("user_prompt_submit")(lambda event: prompts.append(event.content))
    engine = ThreadEngine(2) if name == "map_failure" else SequentialEngine()
    path = Path(__file__).parent / "files/template_control_flow" / f"{name}.kedi"
    runtime = compile_program(
        parse_program(path.read_text()),
        adapter=adapter,
        engine=engine,
        runtime_globals={
            "items": items,
            "commit": commit,
            "receipt": receipt,
            "record_effect": effects.append,
        },
    )
    try:
        with use_async_owner_loop(), observe_invocations(usages.append), redirect_stdout(output):
            if name == "map_failure":
                with pytest.raises(KediExecutionError, match="Injected map item failure"):
                    await asyncio.wait_for(asyncio.to_thread(runtime.run_main), 10)
                await runtime.aclose()
                assert sorted(effects) == [(item, item.split(":")[1]) for item in items]
                assert sum(u.requests for u in usages) == 3
            else:
                assert await asyncio.wait_for(asyncio.to_thread(runtime.run_main), 10) == receipt
                assert output.getvalue() == receipt + "\n"
                assert effects == []
                assert sum(u.requests for u in usages) == 1
                assert prompts and all("discarded-secret" not in prompt for prompt in prompts)
    finally:
        await runtime.aclose()
        engine.shutdown()
