import json
import re

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from kedi.agent_adapter import PydanticAdapter
from kedi.lang import parse_program
from tests.host_boundary_helpers import ROOT, filesystem_boundary, thread_boundary


def test_programs_parse():
    for path in ROOT.glob("*.kedi"):
        parse_program(path.read_text())


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["outside", "symlink", "secret", "secret-approved", "allowed"])
async def test_filesystem_rejects_unauthorized_reads_before_open(mode):
    calls = 0

    async def model(messages, info):
        nonlocal calls
        calls += 1
        if calls == 1:
            prompt = str(messages)
            target = re.search(r"file_path='([^']+)'", prompt).group(1)
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "read_text_file",
                        {
                            "file_path": target,
                            "secret_files": mode == "secret-approved",
                            "offset": 0,
                            "max_chars": 100,
                        },
                        "fixture_read",
                    )
                ]
            )
        results = [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
        text = str(results[-1].content) if mode == "allowed" and results else "blocked"
        return ModelResponse(parts=[TextPart(text)])

    evidence = {}
    checks = await filesystem_boundary(PydanticAdapter(FunctionModel(model)), mode, evidence)
    assert all(checks.values()), evidence


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["interrupt", "cancel"])
async def test_subagent_replacement_waits_for_synchronous_tool(mode):
    calls = 0

    async def model(messages, info):
        nonlocal calls
        calls += 1
        if calls <= 2:
            name = "checkpoint" if calls == 1 else "read_checkpoint"
            return ModelResponse(parts=[ToolCallPart(name, {}, "fixture_" + str(calls))])
        results = [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
        receipt = results[-1].content
        payload = {
            "task_summary": "Recovered.",
            "final_result": {"scope": "revised", "receipt": receipt},
        }
        if info.output_tools:
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, payload)])
        return ModelResponse(parts=[TextPart(json.dumps(payload))])

    evidence = {}
    checks = await thread_boundary(PydanticAdapter(FunctionModel(model)), mode, evidence)
    assert all(checks.values()), evidence
