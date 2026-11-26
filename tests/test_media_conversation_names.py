"""Media names in a conversation: a name means one media across its requests."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from kedi.agent_adapter import PydanticAdapter
from kedi.agent_adapter.conversation import ConversationState
from kedi.agent_adapter.request import (
    ConversationCoordinates,
    RawPromptBuilder,
    RunRequestAssembler,
    TextOutput,
)
from kedi.agent_adapter.request.context import agent_run_request_scope, conversation_media_scope
from kedi.agent_adapter.request.media import RequestMedia
from kedi.agent_adapter.tools.media import codemode_media, tool_media
from kedi.lang import compile_program, parse_program
from kedi.media import MEDIA_NAMES, Image
from tests.test_media_values import png_bytes

ANKARA = png_bytes(10)
IZMIR = png_bytes(200)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KEDI_WORKSPACE_POLICY", raising=False)
    (tmp_path / "tiles").mkdir()
    (tmp_path / "tiles" / "Ankara.png").write_bytes(ANKARA)
    (tmp_path / "tiles" / "Izmir.png").write_bytes(IZMIR)
    return tmp_path


def _run(source: str, adapter: Any) -> Any:
    runtime = compile_program(parse_program(source), adapter=adapter)
    try:
        return runtime.run_main()
    finally:
        asyncio.run(runtime.aclose())


def _program(head: str) -> str:
    return (
        f"> import: media\n{head}"
        '@satellite(region: str) -> Image:\n  = `image("tiles/" + region + ".png")`\n\n'
        "> approval: allow\n> use: satellite\n"
        "[a] << Fetch Ankara.\n[b] << Fetch Izmir.\n"
    )


def _model(final: Any = None) -> tuple[FunctionModel, list[str]]:
    """Fetch Ankara, then Izmir; then give ``final`` for an output call. Records tool names."""

    placeholders: list[str] = []
    turns = iter(["Ankara", "Izmir"])

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        last = messages[-1]
        returned = [part for part in last.parts if isinstance(part, ToolReturnPart)]
        if returned:
            placeholders.append(str(returned[0].content))
            return ModelResponse(parts=[TextPart("ok")])
        if info.output_tools:
            (output,) = info.output_tools
            return ModelResponse(parts=[ToolCallPart(output.name, final)])
        return ModelResponse(parts=[ToolCallPart("satellite", {"region": next(turns)})])

    return FunctionModel(respond), placeholders


def test_tool_media_names_stay_unique_across_a_conversation(workspace: Path) -> None:
    model, placeholders = _model({"response": "satellite_result_1"})
    source = _program("> history: enabled\n") + "[img: Image] << Give me Ankara again.\n= `img`\n"
    assert _run(source, PydanticAdapter(model=model)).data == ANKARA
    assert ["satellite_result_1" in text for text in placeholders] == [True, False]
    assert "<image satellite_result_2>" in placeholders[1]


def test_requests_without_a_conversation_name_their_own_media(workspace: Path) -> None:
    model, placeholders = _model()
    _run(_program("") + "= done\n", PydanticAdapter(model=model))
    assert all("<image satellite_result_1>" in text for text in placeholders)
    assert len(placeholders) == 2


def _request(prompt: str = "go") -> Any:
    return RunRequestAssembler().assemble(
        prompt=RawPromptBuilder(content=prompt).build(),
        instructions=(),
        conversation=ConversationCoordinates(
            session_id="s", lane_identity="l", cache_epoch=0, cache_key="k", history_count=0
        ),
        tool_specs=(),
        output=TextOutput(),
        run_id="r",
        turn_sequence=1,
        call_kind="invoke",
    )


def test_codemode_attaches_only_media_tools_returned_in_this_request() -> None:
    state = ConversationState()
    earlier = Image.from_bytes(ANKARA, "image/png")
    assert state.media_namer.name_for(earlier, None, generated="snap_result") == "snap_result_1"
    with conversation_media_scope(state.media_namer), agent_run_request_scope(_request()):
        assert codemode_media("<image snap_result_1>") is None
        named = tool_media(Image.from_bytes(IZMIR, "image/png"), tool_name="snap")
        assert named is not None and named.text == "<image snap_result_2>"
        attached = codemode_media("<image snap_result_2> and <image snap_result_1>")
        assert attached is not None
        assert [item.name for item in attached.attachments] == ["snap_result_2"]


def test_checkpoints_keep_names_but_not_media() -> None:
    state = ConversationState()
    state.media_namer.name_for(Image.from_bytes(ANKARA, "image/png"), "tile")
    checkpoint = state.export_checkpoint("pydantic")
    assert checkpoint["media_names"] == ["tile"]

    restored = ConversationState.from_checkpoint(
        checkpoint, adapter_shortname="pydantic", artifacts=state.artifacts
    )
    namer = restored.media_namer
    assert namer.name_for(Image.from_bytes(IZMIR, "image/png"), "tile") == "tile_2"
    names = {MEDIA_NAMES: RequestMedia((), namer)}
    with pytest.raises(ValidationError, match="tile is from before this conversation was restored"):
        TypeAdapter(Image).validate_python("tile", context=names)
    assert TypeAdapter(Image).validate_python("tile_2", context=names).data == IZMIR

    # Checkpoints from before media names restore with none reserved.
    del checkpoint["media_names"]
    older = ConversationState.from_checkpoint(
        checkpoint, adapter_shortname="pydantic", artifacts=state.artifacts
    )
    assert older.media_namer.names == frozenset()
    with pytest.raises(ValueError, match="media names must be a list of strings"):
        ConversationState.from_checkpoint(
            {**checkpoint, "media_names": [1]},
            adapter_shortname="pydantic",
            artifacts=state.artifacts,
        )
