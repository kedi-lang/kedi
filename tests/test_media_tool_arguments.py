"""Media in tool arguments: a model hands a tool an image of its request by name."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from kedi.agent_adapter import LangChainAdapter, PydanticAdapter
from kedi.agent_adapter.tools.validation import _python_callable_argument_validator
from kedi.lang import compile_program, parse_program
from kedi.media import Image
from tests.test_media_values import png_bytes

SAT = png_bytes(10)
SOURCE = (
    "> import: media\n"
    '@satellite(region: str) -> Image:\n  = `image("sat.png")`\n\n'
    "@size(photo: Image) -> str:\n  = `str(len(photo.data))`\n\n"
    "> approval: allow\n> use: satellite\n> use: size\n"
    "[reply] << Fetch Ankara and measure the image.\n= <reply>\n"
)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KEDI_WORKSPACE_POLICY", raising=False)
    (tmp_path / "sat.png").write_bytes(SAT)
    return tmp_path


def _run(source: str, adapter: Any) -> Any:
    runtime = compile_program(parse_program(source), adapter=adapter)
    try:
        return runtime.run_main()
    finally:
        asyncio.run(runtime.aclose())


def _pydantic(photo: str) -> tuple[FunctionModel, list[str], list[list[ModelMessage]]]:
    results: list[str] = []
    requests: list[list[ModelMessage]] = []

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        requests.append(list(messages))
        returned = [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]
        if not returned:
            return ModelResponse(parts=[ToolCallPart("satellite", {"region": "Ankara"})])
        if returned[0].tool_name == "satellite":
            return ModelResponse(parts=[ToolCallPart("size", {"photo": photo})])
        results.append(str(returned[0].content))
        return ModelResponse(parts=[TextPart("measured")])

    return FunctionModel(respond), results, requests


def test_a_model_hands_a_tool_image_to_another_tool_by_name(workspace: Path) -> None:
    model, results, requests = _pydantic("satellite_result_1")
    assert _run(SOURCE, PydanticAdapter(model=model)) == "measured"
    assert results == [str(len(SAT))]
    # The request that offered the tools told the model to use names.
    assert "<MediaOutputInstructions>" in str(requests[0][0])


def test_a_wrong_name_tells_the_model_the_names_it_can_use(workspace: Path) -> None:
    model, results, _ = _pydantic("ankara.png")
    assert _run(SOURCE, PydanticAdapter(model=model)) == "measured"
    (error,) = results
    assert "Parameter 'photo' does not match annotated type" in error
    assert "the name of an image in this request (satellite_result_1)" in error


class _ChatModel(FakeMessagesListChatModel):
    seen: list[list[Any]] = []

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self

    def _generate(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        type(self).seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


def test_langchain_tools_take_media_names_on_their_worker_threads(workspace: Path) -> None:
    _ChatModel.seen = []
    model = _ChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "satellite", "args": {"region": "Ankara"}, "id": "c1"}],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "size", "args": {"photo": "<image satellite_result_1>"}, "id": "c2"}
                ],
            ),
            AIMessage(content="measured"),
        ]
    )
    model.profile = {"tool_calling": True}
    assert _run(SOURCE, LangChainAdapter(model=model)) == "measured"
    sizes = [
        message.content
        for message in _ChatModel.seen[-1]
        if isinstance(message, ToolMessage) and message.name == "size"
    ]
    assert sizes == [str(len(SAT))]


def test_python_tools_take_media_names_only_from_a_model() -> None:
    def size(photo: Image) -> int:
        return len(photo.data or b"")

    validate = _python_callable_argument_validator(size)
    photo = Image.from_bytes(SAT, "image/png")
    assert validate({"photo": photo}) == {"photo": photo}
    # Outside a request nothing can be named, and a string never becomes media.
    with pytest.raises(ValueError, match="which holds none"):
        validate({"photo": "photo_1"})
