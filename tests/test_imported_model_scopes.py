import asyncio
import re
from collections import Counter
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.profiles import ModelProfile

from kedi.agent_adapter import PydanticAdapter
from kedi.agent_adapter.execution import use_async_owner_loop
from kedi.engine import SequentialEngine
from kedi.lang import compile_program, parse_program


@pytest.mark.parametrize("name", ["library", "imported", "switch"])
def test_scope_contract_program_parses(name):
    source = Path(__file__).parent / "files/model_scopes" / f"{name}.kedi"
    parse_program(source.read_text(), source_path=str(source))


@pytest.mark.asyncio
async def test_imported_scope_fixture_observes_all_four_model_bindings(monkeypatch):
    requests = []

    def model(label):
        async def respond(messages, info):
            code = re.search(r"The scoped code is (\w+)\.", info.instructions).group(1)
            requests.append((label, info.model_request_parameters.thinking, code))
            schema = info.output_tools[0]
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        schema.name,
                        {field: code for field in schema.parameters_json_schema["properties"]},
                    )
                ]
            )

        return FunctionModel(respond, profile=ModelProfile(supports_thinking=True))

    models = {"root": model("root"), "secondary": model("secondary")}
    adapter = PydanticAdapter(models["root"])
    monkeypatch.setattr(adapter, "_resolve_model", models.__getitem__)
    path = (Path(__file__).parent / "files/model_scopes/imported.kedi").resolve()
    engine = SequentialEngine()
    runtime = compile_program(
        parse_program(path.read_text(), source_path=str(path)),
        adapter=adapter,
        engine=engine,
        runtime_globals={
            "root_model": "root",
            "imported_model": "secondary",
            "root_code": "root_code",
            "imported_code": "imported_code",
            "local_code": "local_code",
        },
    )
    try:
        with use_async_owner_loop():
            actual = await asyncio.to_thread(runtime.run_main)
        assert actual == {
            "imported": "imported_code",
            "before": "root_code",
            "local": "local_code",
            "after": "root_code",
        }
        assert Counter(requests) == Counter(
            {
                ("root", "medium", "root_code"): 2,
                ("secondary", "low", "imported_code"): 1,
                ("secondary", "high", "local_code"): 1,
            }
        )
    finally:
        await runtime.aclose()
        engine.shutdown()
