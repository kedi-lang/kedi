"""Media captures of a ``> task``: the child names one of its media, the program gets it."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from kedi.agent_adapter import PydanticAdapter
from kedi.agent_adapter.subagents import SubagentOutputState
from kedi.lang import compile_program, parse_program
from kedi.media import Image
from tests.test_media_values import png_bytes

RED = png_bytes(255)
BLUE = png_bytes(20)

PROGRAM = """> import: media
> profile: looker:
    > adapter: pydantic

> profile: coordinator:
    > adapter: pydantic
    > subagent: looker

> use: coordinator

[cats: list[Image]] = `[image("red.png"), image("blue.png")]`
> task [job]: looker:
    >> Pick the blue one of <<cats>> as [photo: Image] and say why as [reason].
> await [result]: job
= `[result.output.photo, result.output.reason]`
"""


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KEDI_WORKSPACE_POLICY", raising=False)
    (tmp_path / "red.png").write_bytes(RED)
    (tmp_path / "blue.png").write_bytes(BLUE)
    return tmp_path


def _run(source: str, adapter: Any) -> Any:
    runtime = compile_program(parse_program(source), adapter=adapter)
    try:
        return runtime.run_main()
    finally:
        asyncio.run(runtime.aclose())


def test_a_task_child_names_one_of_its_attachments(workspace: Path) -> None:
    prompts: list[str] = []

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompts.extend(
            str(part.content)
            for message in messages
            for part in getattr(message, "parts", ())
            if isinstance(part, UserPromptPart)
        )
        (output,) = info.output_tools
        final = {"photo": "cats_2", "reason": "it is blue"}
        return ModelResponse(
            parts=[ToolCallPart(output.name, {"task_summary": "picked", "final_result": final})]
        )

    photo, reason = _run(PROGRAM, PydanticAdapter(model=FunctionModel(respond)))
    assert isinstance(photo, Image) and photo.data == BLUE
    assert reason == "it is blue"
    assert any("write the name it was given" in prompt for prompt in prompts)


def test_delegated_results_stay_json_and_name_no_media() -> None:
    from pydantic import TypeAdapter, ValidationError

    state = SubagentOutputState(TypeAdapter(Image))
    with pytest.raises(ValidationError, match="Expected an image as a data URI"):
        state.validate_candidate("cats_2")
