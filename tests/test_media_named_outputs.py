"""Media in outputs: a model names an attachment or a tool result of its request."""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypedDict, cast

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from pydantic import BaseModel, TypeAdapter, ValidationError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from kedi.agent_adapter import LangChainAdapter, PydanticAdapter
from kedi.agent_adapter.request.media import (
    MediaAttachment,
    media_validation_context,
    require_media_output,
)
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program
from kedi.media import MEDIA_NAMES, Audio, Image, MediaError, type_holds_media
from tests.lsp_helpers import diagnostics_for
from tests.test_media_values import png_bytes

SAT = png_bytes(10)
RED = png_bytes(255)
BLUE = png_bytes(20)
DRAWN = png_bytes(200)

TOOL = (
    "> import: media\n"
    "@satellite(region: str) -> Image:\n"
    '  = `image("sat.png")`\n\n'
    "> approval: allow\n"
    "> use: satellite\n"
)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KEDI_WORKSPACE_POLICY", raising=False)
    (tmp_path / "sat.png").write_bytes(SAT)
    (tmp_path / "red.png").write_bytes(RED)
    (tmp_path / "blue.png").write_bytes(BLUE)
    return tmp_path


def _run(source: str, adapter: Any) -> Any:
    runtime = compile_program(parse_program(source), adapter=adapter)
    try:
        return runtime.run_main()
    finally:
        asyncio.run(runtime.aclose())


def _answers(*answers: Any, tool: bool = True) -> tuple[FunctionModel, list[list[ModelMessage]]]:
    """A model that calls ``satellite`` (when ``tool``), then gives each answer in turn."""

    requests: list[list[ModelMessage]] = []
    pending = list(answers)

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        requests.append(list(messages))
        if tool and len(requests) == 1:
            return ModelResponse(parts=[ToolCallPart("satellite", {"region": "Ankara"})])
        (output,) = info.output_tools
        return ModelResponse(parts=[ToolCallPart(output.name, pending.pop(0))])

    return FunctionModel(respond), requests


def _feedback(requests: list[list[ModelMessage]]) -> list[str]:
    """What each request after the first sent back about the model's previous answer."""

    return [
        str(part.content)
        for messages in requests[1:]
        for part in messages[-1].parts
        if isinstance(part, (RetryPromptPart, ToolReturnPart))
    ]


# A model that cannot draw names an image of its request.


def test_an_image_output_names_the_image_a_tool_returned(workspace: Path) -> None:
    model, requests = _answers({"response": "satellite_result_1"})
    source = TOOL + "[img: Image] << Fetch the satellite image of Ankara.\n= `img`\n"
    image = _run(source, PydanticAdapter(model=model))
    assert isinstance(image, Image) and image.data == SAT
    # The model read the name next to the image, and how to give it back.
    returned = [
        part
        for message in requests[-1]
        for part in getattr(message, "parts", ())
        if isinstance(part, ToolReturnPart)
    ]
    assert "<image satellite_result_1>" in str(returned[0].content)
    instructions = [
        message.instructions for message in requests[-1] if isinstance(message, ModelRequest)
    ]
    assert any("<MediaOutputInstructions>" in (text or "") for text in instructions)


def test_fields_and_lists_name_media_too(workspace: Path) -> None:
    model, _ = _answers({"photo": "<image satellite_result_1>", "region": "Ankara"})
    source = (
        TOOL + ">> Fetch the satellite image of Ankara as [photo: Image] for [region].\n"
        "= `[photo, region]`\n"
    )
    photo, region = _run(source, PydanticAdapter(model=model))
    assert (photo.data, region) == (SAT, "Ankara")

    model, _ = _answers({"response": ["satellite_result_1", "satellite_result_1"]})
    source = TOOL + "[shots: list[Image]] << Fetch Ankara twice.\n= `shots`\n"
    assert [shot.data for shot in _run(source, PydanticAdapter(model=model))] == [SAT, SAT]


def test_a_model_names_one_of_the_prompt_attachments(workspace: Path) -> None:
    model, _ = _answers({"response": "cats_2"}, tool=False)
    source = (
        "> import: media\n"
        '[cats: list[Image]] = `[image("red.png"), image("blue.png")]`\n'
        "[blue: Image] << Which of <<cats>> is blue?\n= `blue`\n"
    )
    assert _run(source, PydanticAdapter(model=model)).data == BLUE


def test_a_wrong_name_or_a_made_up_url_is_refused_with_the_names_to_use(
    workspace: Path,
) -> None:
    model, requests = _answers(
        {"response": "https://example.com/ankara.png"},
        {"response": "satellite_result_1"},
    )
    source = TOOL + "[img: Image] << Fetch the satellite image of Ankara.\n= `img`\n"
    assert _run(source, PydanticAdapter(model=model)).data == SAT
    retry = _feedback(requests)[-1]
    assert "the name of an image in this request (satellite_result_1)" in retry
    assert "https://example.com/ankara.png" in retry


def test_a_text_model_with_nothing_to_name_is_refused_before_the_request(
    workspace: Path,
) -> None:
    model, requests = _answers({"response": "image_1"}, tool=False)
    with pytest.raises(KediExecutionError, match="cannot produce images; use an image model"):
        _run(
            "> import: media\n[img: Image] << Draw a cat.\n= `img`\n", PydanticAdapter(model=model)
        )
    assert requests == []


# Audio: a raw invoke speaks, unless the model has audio to name


def test_an_audio_output_names_an_attached_clip(workspace: Path) -> None:
    from tests.test_media_speech_output import WAV

    (workspace / "bark.wav").write_bytes(WAV)
    (workspace / "meow.wav").write_bytes(WAV + b"\x00\x00")
    model, requests = _answers({"response": "clips_2"}, tool=False)
    source = (
        "> import: media\n"
        '[clips: list[Audio]] = `[audio("bark.wav"), audio("meow.wav")]`\n'
        "[clip: Audio] << Which of <<clips>> is the cat?\n= `clip`\n"
    )
    clip = _run(source, PydanticAdapter(model=model))
    assert isinstance(clip, Audio) and clip.data == WAV + b"\x00\x00"
    assert len(requests) == 1


def test_an_audio_output_names_what_a_tool_declared_to_return_audio(workspace: Path) -> None:
    from tests.test_media_speech_output import WAV

    (workspace / "bark.wav").write_bytes(WAV)
    requests: list[list[ModelMessage]] = []

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        requests.append(list(messages))
        if len(requests) == 1:
            return ModelResponse(parts=[ToolCallPart("record", {})])
        (output,) = info.output_tools
        return ModelResponse(parts=[ToolCallPart(output.name, {"response": "record_result_1"})])

    source = (
        "> import: media\n"
        '@record() -> Audio:\n  = `audio("bark.wav")`\n\n'
        "> approval: allow\n> use: record\n"
        "[clip: Audio] << Record the dog.\n= `clip`\n"
    )
    assert _run(source, PydanticAdapter(model=FunctionModel(respond))).data == WAV


def test_tools_that_return_no_audio_leave_speech_as_it_was(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_media_speech_output import _fake_realtime, _live_model

    record = _fake_realtime(monkeypatch)
    source = (
        "> import: media\n"
        "@weather(city: str) -> str:\n  = sunny\n\n"
        "> use: weather\n"
        "[speech: Audio] << Read this aloud: sunny in Ankara.\n= `speech`\n"
    )
    speech = _run(source, PydanticAdapter(model=_live_model()))
    assert isinstance(speech, Audio) and record["prompt"].endswith("sunny in Ankara.")


def test_tool_specs_record_the_media_their_return_type_holds() -> None:
    from kedi.agent_adapter.tools.conversion import python_callable_tool_spec

    def record() -> Audio: ...

    def shots() -> list[Image]: ...

    def plain() -> str: ...

    kinds = {
        fn.__name__: python_callable_tool_spec(fn.__name__, fn).metadata.get("kedi_return_media")
        for fn in (record, shots, plain)
    }
    assert kinds == {"record": ["audio"], "shots": ["image"], "plain": None}


# Models that draw


def _openai_responses(outputs: list[list[dict[str, Any]]], bodies: list[dict[str, Any]]) -> Any:
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

    def handle(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content or b"{}"))
        output = outputs[min(len(bodies), len(outputs)) - 1]
        return httpx.Response(
            200,
            json={
                "id": f"resp_{len(bodies)}",
                "object": "response",
                "created_at": 0,
                "model": "gpt-5",
                "status": "completed",
                "output": output,
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    return OpenAIResponsesModel("gpt-5", provider=OpenAIProvider(api_key="t", http_client=client))


def _function_call(name: str, arguments: dict[str, Any], call: str) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": f"fc_{call}",
        "call_id": call,
        "name": name,
        "arguments": json.dumps(arguments),
        "status": "completed",
    }


def test_a_model_that_draws_and_takes_tools_may_name_a_tool_image(workspace: Path) -> None:
    bodies: list[dict[str, Any]] = []
    model = _openai_responses(
        [
            [_function_call("satellite", {"region": "Ankara"}, "c1")],
            [_function_call("final_result", {"response": "satellite_result_1"}, "c2")],
        ],
        bodies,
    )
    source = TOOL + "[img: Image] << Fetch the satellite image of Ankara.\n= `img`\n"
    assert _run(source, PydanticAdapter(model=model)).data == SAT
    tool_names = [tool.get("name", tool["type"]) for tool in bodies[0]["tools"]]
    assert "image_generation" in tool_names and "final_result" in tool_names
    assert "<MediaOutputInstructions>" in json.dumps(bodies[0])

    bodies.clear()
    drawn = {
        "type": "image_generation_call",
        "id": "ig_1",
        "status": "completed",
        "result": base64.b64encode(DRAWN).decode(),
        "output_format": "png",
    }
    model = _openai_responses([[drawn]], bodies)
    source = "> import: media\n[img: Image] << Draw a cat.\n= `img`\n"
    assert _run(source, PydanticAdapter(model=model)).data == DRAWN
    # With nothing to name, the request is a drawing request only, as it always was.
    (body,) = bodies
    assert "final_result" not in [tool.get("name") for tool in body.get("tools", [])]
    assert "MediaOutputInstructions" not in json.dumps(body)


def test_an_image_model_that_takes_no_tools_is_refused_with_tools(workspace: Path) -> None:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    sent: list[httpx.Request] = []
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: sent.append(r) or httpx.Response(500)),
        timeout=60,
    )
    model = GoogleModel(
        "gemini-2.5-flash-image", provider=GoogleProvider(api_key="t", http_client=client)
    )
    source = TOOL + "[img: Image] << Fetch the satellite image of Ankara.\n= `img`\n"
    with pytest.raises(KediExecutionError, match="takes no tools, so it cannot call satellite"):
        _run(source, PydanticAdapter(model=model))
    assert sent == []


# LangChain


class _ChatModel(FakeMessagesListChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self


def test_langchain_text_models_name_a_tool_image(workspace: Path) -> None:
    model = _ChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "satellite", "args": {"region": "Ankara"}, "id": "c1"}],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "KediTypedOutput",
                        "args": {"response": "satellite_result_1"},
                        "id": "c2",
                    }
                ],
            ),
        ]
    )
    model.profile = {"tool_calling": True}
    source = TOOL + "[img: Image] << Fetch the satellite image of Ankara.\n= `img`\n"
    assert _run(source, LangChainAdapter(model=model)).data == SAT


# The rule itself


def test_names_resolve_only_in_a_model_validation_context() -> None:
    red = Image.from_bytes(RED, "image/png")
    clip = Audio.from_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 24, "audio/wav")
    images = TypeAdapter(list[Image])
    context = {MEDIA_NAMES: {"red": red, "clip": clip}}
    assert images.validate_python(["red", " <image red> "], context=context) == [red, red]
    assert images.validate_json('["red"]', context=context) == [red]
    with pytest.raises(ValidationError, match="clip is <audio"):
        images.validate_python(["clip"], context=context)
    with pytest.raises(ValidationError, match="which holds none"):
        TypeAdapter(Audio).validate_python("song", context={MEDIA_NAMES: {"red": red}})
    # Without the context, strings never become media, and JSON reads back serialized media.
    with pytest.raises(ValidationError, match="strings are never converted"):
        images.validate_python(["red"])
    assert images.validate_json(images.dump_json([red]))[0].data == RED
    with pytest.raises(ValidationError, match="the name of an image in this request, got int"):
        TypeAdapter(Image).validate_python(5, context=context)
    # Outside a request, only this scope's tool media can be named; a caller's own mapping
    # context keeps its keys, and any other context is left as it is.
    names = media_validation_context({"own": 1})
    assert names["own"] == 1 and len(names[MEDIA_NAMES]) == 0
    assert media_validation_context("own") == "own"


def test_a_tool_image_never_takes_an_attachment_name(workspace: Path) -> None:
    model, _ = _answers({"response": "red_2"})
    source = (
        "> import: media\n"
        '@satellite(region: str) -> Image:\n  = `image("blue.png").with_name("red")`\n\n'
        "> approval: allow\n> use: satellite\n"
        '[red: Image] = `image("red.png")`\n'
        "[img: Image] << Compare <<red>> with the satellite image of Ankara.\n= `img`\n"
    )
    assert _run(source, PydanticAdapter(model=model)).data == BLUE


class _Node(BaseModel):
    label: str
    child: _Node | None = None


@dataclasses.dataclass
class _Frame:
    photo: Image


class _Clip(TypedDict):
    sound: Audio


@dataclasses.dataclass
class _Unresolved:
    photo: MissingType  # type: ignore[name-defined]  # noqa: F821


def test_type_holds_media_walks_containers_and_models() -> None:
    class Shot(BaseModel):
        region: str
        photo: Image | None = None

    assert type_holds_media(Image) and type_holds_media(list[Shot])
    assert type_holds_media(dict[str, Audio])
    assert type_holds_media(_Frame) and type_holds_media(_Clip)
    assert not type_holds_media(str) and not type_holds_media(dict[str, int])
    # A recursive model is walked once; hints that do not resolve hold no media to name.
    assert not type_holds_media(_Node)
    assert not type_holds_media(_Unresolved)


def test_harness_adapters_name_attachments_only() -> None:
    from kedi.agent_adapter.adapters.claude import ClaudeAdapter

    adapter = ClaudeAdapter(model="claude-sonnet-4")
    red = MediaAttachment(name="red", media=Image.from_bytes(RED, "image/png"))
    require_media_output(adapter, "image", (red,))
    with pytest.raises(MediaError, match="holds no image to name"):
        require_media_output(adapter, "image", ())
    # An adapter without structured output cannot have the model name one either.
    from kedi.agent_adapter.adapters.acp import ACPAdapter

    acp = SimpleNamespace(shortname="acp", capabilities=ACPAdapter.capabilities)
    with pytest.raises(MediaError, match="Adapter 'acp' cannot produce images; Pydantic AI"):
        require_media_output(acp, "image", (red,))


PICK = (
    "> import: media\n"
    '[cats: list[Image]] = `[image("red.png"), image("blue.png")]`\n'
    "[blue: Image] << Which of <<cats>> is blue?\n= `blue`\n"
)


def test_the_claude_agent_sdk_resolves_names_of_attachments(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kedi.agent_adapter.adapters.claude import ClaudeAdapter
    from tests.test_agent_adapters import _install_fake_claude_sdk

    events = _install_fake_claude_sdk(monkeypatch)
    events["structured_output"] = "cats_2"
    assert _run(PICK, ClaudeAdapter(model="claude-sonnet-4")).data == BLUE


def test_codex_resolves_names_of_attachments(workspace: Path) -> None:
    from kedi.agent_adapter.adapters import CodexAdapter
    from kedi.agent_adapter.adapters.codex.app_server import CodexTurnResult

    class FakeClient:
        def run_stateless_turn_sync(self, **kwargs: Any) -> CodexTurnResult:
            return CodexTurnResult(text=json.dumps({"response": "cats_2"}), turn={})

        def close(self) -> None:
            pass

        async def aclose(self) -> None:
            pass

    assert _run(PICK, CodexAdapter(client=cast(Any, FakeClient()))).data == BLUE


def test_the_editor_accepts_audio_outputs_that_name_an_attachment() -> None:
    source = (
        "> agent: claude\n> import: media\n"
        '[clips: list[Audio]] = `[audio("bark.wav")]`\n'
        "[clip: Audio] << Which of <<clips>> barks?\n= `clip`\n"
    )
    assert [
        diagnostic
        for diagnostic in diagnostics_for(source, "/tmp/pick.kedi")
        if "speech output" in diagnostic.message
    ] == []


def test_the_editor_accepts_image_outputs_that_name_an_attachment() -> None:
    source = (
        "> agent: claude\n> import: media\n"
        '[cats: list[Image]] = `[image("red.png")]`\n'
        "[best: Image] << Pick the best of <<cats>>.\n= `best`\n"
    )
    assert [
        diagnostic
        for diagnostic in diagnostics_for(source, "/tmp/pick.kedi")
        if "image output" in diagnostic.message
    ] == []
