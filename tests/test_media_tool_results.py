"""Media returned by tools (#185)."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from kedi.agent_adapter import LangChainAdapter, PydanticAdapter
from kedi.agent_adapter.adapters.langchain.media import split_tool_media
from kedi.agent_adapter.tools.media import refuse_tool_media, sandbox_value, tool_media
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program
from kedi.media import Document, Image, MediaError
from tests.test_media_values import png_bytes

PNG = png_bytes()
BIG_PNG = PNG + b"\x00" * 4096
WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x10\x00\x00\x00\x01\x00\x01\x00" + b"\x00" * 16
CAT = Image.from_bytes(PNG, "image/png")


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KEDI_WORKSPACE_POLICY", raising=False)
    (tmp_path / "cat.png").write_bytes(PNG)
    (tmp_path / "big.png").write_bytes(BIG_PNG)
    (tmp_path / "voice.wav").write_bytes(WAV)
    return tmp_path


def _program(body: str = '`image("cat.png")`', *, head: str = "", returns: str = "Image") -> str:
    return (
        f"{head}> import: media\n"
        f"@snap() -> {returns}:\n  = {body}\n\n"
        "> approval: allow\n> use: snap\n"
        "[reply] << Take a photo.\n= <reply>\n"
    )


def _run(source: str, adapter: Any) -> Any:
    runtime = compile_program(parse_program(source), adapter=adapter)
    try:
        return runtime.run_main()
    finally:
        asyncio.run(runtime.aclose())


# The adapter-neutral form


def test_results_holding_media_become_placeholders_and_attachments() -> None:
    other = Image.from_url("https://example.com/dog.png")
    notes = Document.from_bytes(b"decision: ship", "text/plain")
    with agent_run_request_scope_for_test():
        first = tool_media(CAT, tool_name="snap")
        second = tool_media(CAT.with_name(None), tool_name="snap")
        nested = tool_media({"photo": other, "notes": notes, "n": 1}, tool_name="look")
        listed = tool_media([CAT, CAT], tool_name="burst")
    assert first is not None and first.text == "<image snap_result_1>"
    assert [attachment.name for attachment in first.attachments] == ["snap_result_1"]
    # The same image again keeps its name within one request.
    assert second is not None and second.text == "<image snap_result_1>"
    assert nested is not None
    assert nested.text.startswith(
        '{"photo":"<image look_result_photo>","notes":"<document look_result_notes>","n":1}'
    )
    assert "decision: ship" in nested.text
    assert [attachment.name for attachment in nested.attachments] == ["look_result_photo"]
    assert (
        listed is not None and listed.text == '["<image snap_result_1>", "<image snap_result_1>"]'
    )
    assert len(listed.attachments) == 1
    assert tool_media("plain", tool_name="snap") is None


def test_names_stay_unique_across_tool_calls_of_one_request() -> None:
    dog = Image.from_url("https://example.com/dog.png")
    with agent_run_request_scope_for_test():
        first = tool_media(CAT, tool_name="snap")
        second = tool_media(dog, tool_name="snap")
    assert first is not None and second is not None
    assert (first.text, second.text) == ("<image snap_result_1>", "<image snap_result_2>")


def test_codemode_code_receives_placeholders() -> None:
    with agent_run_request_scope_for_test():
        assert sandbox_value({"photo": CAT}, tool_name="snap") == {
            "photo": "<image snap_result_photo>"
        }
        # The same image keeps its name; another one is generated.
        assert sandbox_value(CAT, tool_name="snap") == "<image snap_result_photo>"
        dog = Image.from_url("https://example.com/dog.png")
        assert sandbox_value(dog, tool_name="snap") == "<image snap_result_1>"
        assert sandbox_value({"n": 1}, tool_name="snap") == {"n": 1}


def test_text_only_tool_bridges_refuse_media() -> None:
    from kedi.agent_adapter.tools.execution import tool_result_text

    with pytest.raises(MediaError, match="tool bridge sends text only"):
        refuse_tool_media({"photo": CAT})
    with pytest.raises(MediaError, match="tool bridge sends text only"):
        tool_result_text(CAT)
    assert tool_result_text({"n": 1}) == '{"n":1}'


# Pydantic AI


def test_pydantic_sends_tool_media_natively(workspace: Path) -> None:
    seen: list[list[Any]] = []

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]
        if returns:
            seen.append(list(returns[0].content))
            return ModelResponse(parts=[TextPart("a cat")])
        return ModelResponse(parts=[ToolCallPart("snap", {}, tool_call_id="c1")])

    assert _run(_program(), PydanticAdapter(model=FunctionModel(respond))) == "a cat"
    ((text, image),) = seen
    assert text == "<image snap_result_1>"
    assert isinstance(image, BinaryContent) and image.data == PNG
    assert image.identifier == "snap_result_1"


def test_gemini_reads_tool_media_inside_the_function_response(workspace: Path) -> None:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    bodies: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        part = (
            {"functionCall": {"name": "snap", "args": {}, "id": "c1"}}
            if len(bodies) == 1
            else {"text": "a cat"}
        )
        candidate = {"content": {"role": "model", "parts": [part]}, "finishReason": "STOP"}
        return httpx.Response(200, json={"candidates": [candidate]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    model = GoogleModel(
        "gemini-3.5-flash", provider=GoogleProvider(api_key="t", http_client=client)
    )
    # A large image is still media, never an artifact reference.
    source = _program('`image("big.png")`', head="> artifacts:\n    threshold: 1kb\n")
    assert _run(source, PydanticAdapter(model=model)) == "a cat"
    response = bodies[-1]["contents"][-1]["parts"][0]["functionResponse"]
    assert response["response"] == {"return_value": "<image snap_result_1>"}
    (inline,) = response["parts"]
    assert base64.b64decode(inline["inline_data"]["data"]) == BIG_PNG


def test_provider_rules_apply_to_tool_media(workspace: Path) -> None:
    from tests.test_media_provider_bodies import Bodies, pydantic_model

    calls = 0

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        return ModelResponse(parts=[ToolCallPart("snap", {}, tool_call_id="c1")])

    # Audio from a tool fails on Anthropic as audio in a prompt does.
    bodies = Bodies()
    source = _program('`audio("voice.wav")`', returns="Audio")
    with pytest.raises(KediExecutionError, match="Anthropic models do not accept audio"):
        _run(source, PydanticAdapter(model=_tool_calling(pydantic_model("anthropic", bodies))))
    with pytest.raises(KediExecutionError, match="media_limit"):
        _run(
            _program(head="> settings:\n    media_limit: 10\n"),
            PydanticAdapter(model=FunctionModel(respond)),
        )


def _tool_calling(model: Any) -> Any:
    """A model that calls snap first, then defers to ``model`` for the tool result."""

    from pydantic_ai.models.wrapper import WrapperModel

    class FirstCallsSnap(WrapperModel):
        async def request(self, messages: Any, settings: Any, parameters: Any) -> Any:
            if not any(isinstance(part, ToolReturnPart) for part in messages[-1].parts):
                return ModelResponse(parts=[ToolCallPart("snap", {}, tool_call_id="c1")])
            return await super().request(messages, settings, parameters)

    return FirstCallsSnap(model)


def test_codemode_results_attach_the_media_their_code_named(workspace: Path) -> None:
    seen: list[Any] = []
    step = 0

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal step
        step += 1
        returns = [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]
        if step == 1:
            call = ToolCallPart("get_tool_schema", {"tool_names": ["snap"]}, tool_call_id="c0")
        elif step == 2:
            code = "photo = await snap()\n{'photo': photo, 'note': 'ok'}"
            call = ToolCallPart("execute_code", {"code": code}, tool_call_id="c1")
        else:
            seen.append(returns[0].content)
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[call])

    source = _program(head="> codemode: enabled\n")
    assert _run(source, PydanticAdapter(model=FunctionModel(respond))) == "done"
    ((text, image),) = seen
    assert json.loads(text) == {"photo": "<image snap_result_1>", "note": "ok"}
    assert isinstance(image, BinaryContent) and image.data == PNG


def test_history_replays_tool_media(workspace: Path) -> None:
    requests: list[list[ModelMessage]] = []

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        requests.append(list(messages))
        last = messages[-1]
        if any(isinstance(part, ToolReturnPart) for part in last.parts) or len(requests) > 2:
            return ModelResponse(parts=[TextPart("ok")])
        return ModelResponse(parts=[ToolCallPart("snap", {}, tool_call_id="c1")])

    source = _program().replace("= <reply>\n", "[again] << And now?\n= <again>\n")
    source = "> history: enabled\n" + source
    assert _run(source, PydanticAdapter(model=FunctionModel(respond))) == "ok"
    replayed = [
        part
        for message in requests[-1]
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]
    assert isinstance(replayed[0].content[1], BinaryContent)


# LangChain


class _RecordingChatModel(FakeMessagesListChatModel):
    seen: list[list[Any]] = []

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self

    def _generate(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        type(self).seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


def _langchain(profile: dict[str, Any] | None = None) -> Any:
    _RecordingChatModel.seen = []
    model = _RecordingChatModel(
        responses=[
            AIMessage(content="", tool_calls=[{"name": "snap", "args": {}, "id": "c1"}]),
            AIMessage(content="a cat"),
        ]
    )
    model.profile = profile
    return model


def test_langchain_sends_tool_media_in_the_tool_message_where_the_model_takes_it(
    workspace: Path,
) -> None:
    model = _langchain({"image_tool_message": True})
    assert _run(_program(), LangChainAdapter(model=model)) == "a cat"
    tool_message = _RecordingChatModel.seen[-1][-1]
    assert isinstance(tool_message, ToolMessage)
    text, image = tool_message.content
    assert text == {"type": "text", "text": "<image snap_result_1>"}
    assert image["type"] == "image" and base64.b64decode(image["base64"]) == PNG
    assert image["id"] == "snap_result_1"


def test_langchain_moves_tool_media_to_a_user_message_otherwise(workspace: Path) -> None:
    source = _program('`image("big.png")`', head="> artifacts:\n    threshold: 1kb\n")
    assert _run(source, LangChainAdapter(model=_langchain())) == "a cat"
    *_, tool_message, user = _RecordingChatModel.seen[-1]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == [{"type": "text", "text": "<image snap_result_1>"}]
    assert isinstance(user, HumanMessage)
    opening, image, closing = user.content
    assert opening == {"type": "text", "text": '<image id="snap_result_1">'}
    assert base64.b64decode(image["base64"]) == BIG_PNG
    assert closing == {"type": "text", "text": "</image>"}


def test_split_tool_media_places_moved_media_after_the_run_of_tool_messages() -> None:
    image = {"type": "image", "base64": "AAAA", "mime_type": "image/png", "id": "a"}
    messages = [
        AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "1"}]),
        ToolMessage(content=[{"type": "text", "text": "<image a>"}, image], tool_call_id="1"),
        ToolMessage(content="plain", tool_call_id="2"),
        AIMessage(content="done"),
    ]
    moved = split_tool_media(messages, allowed=frozenset())
    assert [type(message).__name__ for message in moved] == [
        "AIMessage",
        "ToolMessage",
        "ToolMessage",
        "HumanMessage",
        "AIMessage",
    ]
    assert split_tool_media(messages, allowed=frozenset({"image"})) == messages


# Text-only tool bridges


def test_claude_tools_report_media_as_the_tools_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from kedi.agent_adapter.adapters.claude.tools import _claude_sdk, _claude_sdk_tool
    from kedi.agent_adapter.tools.specs import ToolSpec
    from tests.test_agent_adapters import _install_fake_claude_sdk

    _install_fake_claude_sdk(monkeypatch)
    tool = _claude_sdk_tool(_claude_sdk(), ToolSpec(fn=lambda: CAT, name="snap", json_schema={}))
    result = asyncio.run(tool({}))
    assert result["is_error"] is True
    assert "tool bridge sends text only" in result["content"][0]["text"]


@contextmanager
def agent_run_request_scope_for_test() -> Iterator[None]:
    """One request's tool media namer, as a request scope sets it."""

    from kedi.agent_adapter.request.context import _TOOL_MEDIA_NAMER
    from kedi.agent_adapter.request.media import MediaNamer

    token = _TOOL_MEDIA_NAMER.set(MediaNamer())
    try:
        yield
    finally:
        _TOOL_MEDIA_NAMER.reset(token)


def test_langchain_codemode_results_attach_the_media_their_code_named(workspace: Path) -> None:
    code = "photo = await snap()\n{'photo': photo}"
    _RecordingChatModel.seen = []
    model = _RecordingChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "get_tool_schema", "args": {"tool_names": ["snap"]}, "id": "c0"}
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[{"name": "execute_code", "args": {"code": code}, "id": "c1"}],
            ),
            AIMessage(content="done"),
        ]
    )
    model.profile = {"image_tool_message": True}
    assert _run(_program(head="> codemode: enabled\n"), LangChainAdapter(model=model)) == "done"
    tool_message = _RecordingChatModel.seen[-1][-1]
    text, image = tool_message.content
    assert json.loads(text["text"]) == {"photo": "<image snap_result_1>"}
    assert base64.b64decode(image["base64"]) == PNG


def test_codemode_results_without_placeholders_stay_text(workspace: Path) -> None:
    seen: list[Any] = []
    step = 0

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal step
        step += 1
        returns = [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]
        if step == 1:
            call = ToolCallPart("get_tool_schema", {"tool_names": ["snap"]}, tool_call_id="c0")
        elif step == 2:
            code = "photo = await snap()\n'taken'"
            call = ToolCallPart("execute_code", {"code": code}, tool_call_id="c1")
        else:
            seen.append(returns[0].content)
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[call])

    _run(_program(head="> codemode: enabled\n"), PydanticAdapter(model=FunctionModel(respond)))
    assert seen == ["taken"]


def test_langchain_lowering_outside_a_run_and_inside_code() -> None:
    from kedi.agent_adapter.adapters.langchain.tool_media import lower_tool_result
    from kedi.agent_adapter.tools.scope import defer_artifact_admission

    with pytest.raises(MediaError, match="returned media outside a LangChain run"):
        asyncio.run(lower_tool_result(CAT, tool_name="snap"))
    with defer_artifact_admission():
        assert asyncio.run(lower_tool_result(CAT, tool_name="snap")) is CAT
    assert asyncio.run(lower_tool_result("plain", tool_name="snap")) == "plain"


def test_post_tool_use_hooks_see_the_media_the_adapter_sends(workspace: Path) -> None:
    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(part, ToolReturnPart) for part in messages[-1].parts):
            return ModelResponse(parts=[TextPart("ok")])
        return ModelResponse(parts=[ToolCallPart("snap", {}, tool_call_id="c1")])

    for adapter in (
        PydanticAdapter(model=FunctionModel(respond)),
        LangChainAdapter(model=_langchain()),
    ):
        results: list[Any] = []
        adapter.on("post_tool_use")(lambda event: results.append(event.result))
        _run(_program(), adapter)
        (result,) = results
        content = result.content if isinstance(result, ToolMessage) else result
        assert content[0] in (
            "<image snap_result_1>",
            {"type": "text", "text": "<image snap_result_1>"},
        )
        assert len(content) == 2


def test_text_only_bridges_refuse_codemode_results_naming_media() -> None:
    from kedi.agent_adapter.tools.execution import tool_result_text

    with agent_run_request_scope_for_test():
        assert sandbox_value(CAT, tool_name="snap") == "<image snap_result_1>"
        with pytest.raises(MediaError, match="tool bridge sends text only"):
            tool_result_text({"p": "<image snap_result_1>"})
        assert tool_result_text({"p": "no media here"}) == '{"p":"no media here"}'


def test_live_journals_record_media_as_the_model_reads_them(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from pydantic_ai.tools import ToolDefinition

    from kedi.agent_adapter.adapters.langchain import runtime
    from kedi.agent_adapter.adapters.pydantic.live import InputDeliveryCapability

    capability = InputDeliveryCapability(control=SimpleNamespace())
    ctx = SimpleNamespace(model=SimpleNamespace(system="google", model_name="gemini-3.5-flash"))
    with agent_run_request_scope_for_test():
        returned = asyncio.run(
            capability.after_tool_execute(
                ctx,  # type: ignore[arg-type]
                call=ToolCallPart("snap", {}, tool_call_id="c1"),
                tool_def=ToolDefinition(name="snap"),
                args={},
                result=CAT,
            )
        )
    assert returned is CAT
    text, image = capability.completed_tools["c1"].return_value
    assert text == "<image snap_result_1>" and isinstance(image, BinaryContent)

    recorded: list[Any] = []
    monkeypatch.setattr(runtime, "record_live_tool_result", recorded.append)
    _run(_program(), LangChainAdapter(model=_langchain({"image_tool_message": True})))
    ((text_block, image_block),) = recorded
    assert text_block["text"] == "<image snap_result_1>" and image_block["type"] == "image"


def test_a_direct_adapter_run_keeps_tool_media_names_apart() -> None:
    dog = Image.from_url("https://example.com/dog.png")
    seen: list[Any] = []

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]
        if returns:
            seen.extend(part.content[0] for part in returns)
            return ModelResponse(parts=[TextPart("ok")])
        return ModelResponse(
            parts=[
                ToolCallPart("cat", {}, tool_call_id="c1"),
                ToolCallPart("dog", {}, tool_call_id="c2"),
            ]
        )

    from kedi.agent_adapter.tools.specs import ToolSpec

    adapter = PydanticAdapter(model=FunctionModel(respond))
    adapter.register_tool(
        ToolSpec(fn=lambda: CAT, name="cat", json_schema={"type": "object"}, risk="read_only")
    )
    adapter.register_tool(
        ToolSpec(fn=lambda: dog, name="dog", json_schema={"type": "object"}, risk="read_only")
    )
    assert adapter.run_sync("Look.").output == "ok"
    assert sorted(seen) == ["<image cat_result_1>", "<image dog_result_1>"]


def test_native_pydantic_media_from_a_tool_takes_the_same_path(workspace: Path) -> None:
    seen: list[Any] = []

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]
        if returns:
            seen.append(returns[0].content)
            return ModelResponse(parts=[TextPart("ok")])
        return ModelResponse(parts=[ToolCallPart("shot", {}, tool_call_id="c1")])

    from kedi.agent_adapter.tools.specs import ToolSpec

    screenshot = ["Screen captured.", BinaryContent(data=PNG, media_type="image/png")]
    adapter = PydanticAdapter(model=FunctionModel(respond))
    adapter.register_tool(
        ToolSpec(
            fn=lambda: screenshot, name="shot", json_schema={"type": "object"}, risk="read_only"
        )
    )
    assert adapter.run_sync("Capture.").output == "ok"
    ((text, image),) = seen
    assert json.loads(text) == ["Screen captured.", "<image shot_result_2>"]
    assert isinstance(image, BinaryContent) and image.identifier == "shot_result_2"


def test_histories_holding_tool_media_survive_session_snapshots() -> None:
    from langchain_core.messages import message_to_dict, messages_from_dict
    from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelRequest

    image = BinaryContent(data=PNG, media_type="image/png", identifier="snap_result_1")
    messages = [
        ModelRequest(parts=[ToolReturnPart("snap", ["<image snap_result_1>", image], "c1")])
    ]
    restored = ModelMessagesTypeAdapter.validate_python(
        ModelMessagesTypeAdapter.dump_python(messages, mode="json")
    )
    assert restored[0].parts[0].content[1].data == PNG
    tool_message = ToolMessage(
        content=[
            {"type": "text", "text": "<image snap_result_1>"},
            {"type": "image", "base64": "AAAA", "mime_type": "image/png", "id": "snap_result_1"},
        ],
        tool_call_id="c1",
    )
    assert messages_from_dict([message_to_dict(tool_message)])[0].content == tool_message.content


def test_codemode_results_without_a_json_form_name_no_media() -> None:
    from kedi.agent_adapter.tools.media import codemode_media

    with agent_run_request_scope_for_test():
        sandbox_value(CAT, tool_name="snap")
        assert codemode_media(object()) is None
        assert codemode_media("<image snap_result_1>") is not None


def test_artifact_admission_never_stores_media() -> None:
    from kedi.agent_adapter.tools.execution import admit_artifact_result

    assert (
        admit_artifact_result(CAT, manager=None, policy=None, tool_name="snap", call_id="c1") is CAT
    )
