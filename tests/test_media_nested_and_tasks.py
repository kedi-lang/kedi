"""Media inside dicts and models, and media in subagent tasks (#186)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from kedi import current_subagent_execution
from kedi.agent_adapter import PydanticAdapter
from kedi.agent_adapter.request.media import compose_prompt
from kedi.core.model_calls import _snapshot_prompt_input
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program
from kedi.media import Image, MediaError, reference
from tests.lsp_helpers import diagnostics_for
from tests.test_media_provider_bodies import Bodies, pydantic_model
from tests.test_media_values import png_bytes

PNG = png_bytes()
CAT = Image.from_bytes(PNG, "image/png")


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KEDI_WORKSPACE_POLICY", raising=False)
    (tmp_path / "cat.png").write_bytes(PNG)
    return tmp_path


def _run(source: str, adapter: Any) -> Any:
    runtime = compile_program(parse_program(source), adapter=adapter)
    try:
        return runtime.run_main()
    finally:
        asyncio.run(runtime.aclose())


def _layout(parts: list[dict[str, Any]]) -> list[str]:
    return ["<binary>" if "inlineData" in part else part["text"] for part in parts]


# Media inside other values


def test_a_dict_holding_media_attaches_them_and_sends_json_naming_them(workspace: Path) -> None:
    source = (
        "> import: media\n"
        '[cat: Image] = `image("cat.png")`\n'
        '[report: dict] = `{"title": "Q3", "photo": cat, "pets": [cat, "note"]}`\n'
        "[reply] << Summarize <<report>>.\n= <reply>\n"
    )
    bodies = Bodies()
    _run(source, PydanticAdapter(model=pydantic_model("google", bodies, "gemini-3.5-flash")))
    layout = _layout(bodies.last_user()["parts"])
    # One image, attached once under its first path, and the JSON naming it twice.
    assert layout == [
        '<attachments>\n<image id="report_photo" type="image/png">\n',
        "<binary>",
        '\n</image>\n<document id="report" type="application/json">\n'
        '{"title":"Q3","photo":"<image report_photo>","pets":["<image report_photo>","note"]}'
        "\n</document>\n</attachments>\nSummarize report.",
    ]


def test_a_directly_referenced_image_keeps_its_name_inside_a_value() -> None:
    text, attachments = compose_prompt(
        [
            "Compare ",
            reference(CAT, binding="cat"),
            " with ",
            reference({"photo": CAT}, binding="report"),
            ".",
        ]
    )
    assert text == "Compare cat with report."
    assert [attachment.name for attachment in attachments] == ["cat", "report"]
    assert attachments[1].media.text == '{"photo":"<image cat>"}'


def test_models_attach_only_the_media_they_write() -> None:
    class Report(BaseModel):
        title: str
        photo: Image
        gallery: list[Image] = []
        hidden: Image | None = Field(default=None, exclude=True)

    other = Image.from_url("https://example.com/dog.png")
    ref = reference(Report(title="x", photo=CAT, gallery=[other], hidden=other), binding="r")
    assert ref.items == (CAT, other)
    assert ref.layout is not None
    assert ref.layout.preferred == ("r_photo", "r_gallery_1")
    assert json.loads(ref.layout.skeleton) == {
        "title": "x",
        "photo": "<image r_photo>",
        "gallery": ["<image r_gallery_1>"],
    }


def test_mixed_lists_and_call_references_name_media_by_position() -> None:
    ref = reference(["caption", CAT], call="pick")
    assert ref.layout is not None and ref.layout.preferred == ("pick_result_2",)
    text, attachments = compose_prompt([ref])
    # The value itself takes the next generated name, as ``<<pick(...)>>`` always does.
    assert text == "pick_result_1"
    assert [attachment.name for attachment in attachments] == ["pick_result_2", "pick_result_1"]


def test_odd_keys_still_give_identifier_names() -> None:
    ref = reference({"my photo!": CAT}, binding="args.report")
    assert ref.layout is not None and ref.layout.preferred == ("args_report_my_photo_",)


def test_values_holding_media_without_a_text_form_still_fail() -> None:
    with pytest.raises(MediaError, match="has no text form to attach"):
        reference({"photos": {CAT}, "n": float("nan")}, binding="r")


def test_the_prompt_identity_follows_the_value_and_its_media() -> None:
    def fingerprint(value: Any) -> str:
        return _snapshot_prompt_input("report", value, by_reference=True).identity.fingerprint

    other = Image.from_url("https://example.com/dog.png")
    first = fingerprint({"title": "Q3", "photo": CAT})
    assert first == fingerprint({"title": "Q3", "photo": Image.from_bytes(PNG, "image/png")})
    assert first != fingerprint({"title": "Q4", "photo": CAT})
    assert first != fingerprint({"title": "Q3", "photo": other})


def test_substituting_a_value_holding_media_as_text_still_fails(workspace: Path) -> None:
    source = (
        "> import: media\n"
        '[cat: Image] = `image("cat.png")`\n'
        '[report: dict] = `{"photo": cat}`\n'
        "[reply] << Summarize <report>.\n= <reply>\n"
    )
    with pytest.raises(KediExecutionError, match="renders text"):
        _run(source, PydanticAdapter(model=pydantic_model("google", Bodies(), "gemini-3.5-flash")))


# Subagent tasks

TASK_PROGRAM = """> import: media
> profile: looker:
    > adapter: pydantic

> profile: coordinator:
    > adapter: pydantic
    > subagent: looker

> use: coordinator

[cat: Image] = `image("cat.png")`
[report: dict] = `{"title": "Q3", "photo": cat}`
> task [job]: looker:
    >> Describe <<cat>> using <<report>>.
> await [result]: job
= `result.task_summary`
"""


def test_a_task_prompt_passes_its_media_to_the_child(workspace: Path) -> None:
    bodies = Bodies()
    _run(TASK_PROGRAM, PydanticAdapter(model=pydantic_model("google", bodies, "gemini-3.5-flash")))
    (child,) = [body for body in bodies if "contents" in body]
    layout = _layout(child["contents"][-1]["parts"])
    assert layout[1] == "<binary>"
    assert layout[2].startswith(
        '\n</image>\n<document id="report" type="application/json">\n'
        '{"title":"Q3","photo":"<image cat>"}'
    )
    assert layout[2].endswith("Describe cat using report.")


def test_a_task_child_whose_adapter_takes_no_media_fails_at_the_task(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kedi.agent_adapter.meta import AdapterCapabilities

    capabilities = PydanticAdapter.capabilities
    calls: list[Any] = []

    class TextOnly(PydanticAdapter):
        pass

    def resolve(self: Any, profile: Any) -> Any:
        adapter = original(self, profile)
        if profile is not None and profile.name == "looker":
            adapter = TextOnly(model=FunctionModel(lambda *args: calls.append(args)))
            adapter.capabilities = AdapterCapabilities(
                **{**capabilities.__dict__, "supports_media_input": False}
            )
        return adapter

    from kedi.core.runtime import KediRuntime

    original = KediRuntime._resolve_adapter
    monkeypatch.setattr(KediRuntime, "_resolve_adapter", resolve)
    with pytest.raises(KediExecutionError, match="does not accept media input"):
        _run(
            TASK_PROGRAM,
            PydanticAdapter(model=pydantic_model("google", Bodies(), "gemini-3.5-flash")),
        )
    assert calls == []


def test_a_delegated_child_gets_none_of_the_parent_prompts_media(workspace: Path) -> None:
    seen: dict[str, list[Any]] = {"parent": [], "child": []}
    parent_calls = 0

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal parent_calls
        prompts = [
            part.content
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, UserPromptPart)
        ]
        if current_subagent_execution() is not None:
            seen["child"].extend(prompts)
            return ModelResponse(parts=[TextPart("child done")])
        seen["parent"].extend(prompts)
        parent_calls += 1
        if parent_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "delegate_task",
                        {"subagent": "looker", "task": "Describe the image."},
                        tool_call_id="delegate-1",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart("parent done")])

    source = """> import: media
> profile: looker:
    > adapter: pydantic

> profile: coordinator:
    > adapter: pydantic
    > subagent: looker

> use: coordinator

[cat: Image] = `image("cat.png")`
[reply] << Look at <<cat>> and delegate.
= <reply>
"""
    assert _run(source, PydanticAdapter(model=FunctionModel(respond))) == "parent done"

    def has_binary(prompts: list[Any]) -> bool:
        return any(
            isinstance(item, BinaryContent)
            for prompt in prompts
            if isinstance(prompt, list)
            for item in prompt
        )

    assert has_binary(seen["parent"])
    assert seen["child"] and not has_binary(seen["child"])


# Editor


def test_the_editor_accepts_task_references_and_checks_the_childs_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kedi.agent_adapter import registry
    from kedi.agent_adapter.meta import AdapterCapabilities

    source = TASK_PROGRAM.replace(
        "> profile: looker:\n    > adapter: pydantic",
        "> profile: looker:\n    > agent: acp:\n        command: acp-server",
    )

    def media_lines() -> list[tuple[str, int]]:
        return [
            (str(diagnostic.code), diagnostic.range.start.line)
            for diagnostic in diagnostics_for(source, "/tmp/task.kedi")
            if str(diagnostic.code).startswith(("K-ADAPTER-CAPABILITY", "K-REFERENCE"))
        ]

    assert media_lines() == []
    with monkeypatch.context() as patch:
        patch.setitem(registry._CAPABILITY_LOADERS, "acp", AdapterCapabilities)
        assert ("K-ADAPTER-CAPABILITY", 14) in media_lines()


def test_agent_input_still_refuses_references() -> None:
    source = TASK_PROGRAM.replace(
        "> await [result]: job", "> send: job:\n    >> See <<cat>> too.\n> await [result]: job"
    )
    (diagnostic,) = [
        diagnostic
        for diagnostic in diagnostics_for(source, "/tmp/task.kedi")
        if str(diagnostic.code) == "K-REFERENCE-CONTEXT"
    ]
    assert "not allowed in > send and > interrupt input" in diagnostic.message
    assert "> task prompts" in diagnostic.message


def test_dataclasses_cycles_and_unwritable_parts_of_values_holding_media() -> None:
    from dataclasses import dataclass

    @dataclass
    class Card:
        name: str
        photo: Image

    ref = reference(Card(name="cat", photo=CAT), binding="card")
    assert ref.layout is not None and ref.layout.preferred == ("card_photo",)

    loop: dict[str, Any] = {"photo": CAT}
    loop["self"] = loop
    with pytest.raises(MediaError, match="contains itself"):
        reference(loop, binding="loop")

    class Holder(BaseModel):
        photo: Image
        extra: Any

    with pytest.raises(MediaError, match="a Holder value has no JSON form to attach"):
        reference(Holder(photo=CAT, extra=object()), binding="h")


def test_a_value_that_writes_other_media_on_the_second_pass_fails() -> None:
    from pydantic import model_serializer

    class Shifty(BaseModel):
        photo: Image
        passes: int = 0

        @model_serializer
        def write(self) -> Any:
            object.__setattr__(self, "passes", self.passes + 1)
            return {"photos": [self.photo] * self.passes}

    ref = reference(Shifty(photo=CAT), binding="s")
    with pytest.raises(MediaError, match="wrote other media on a second pass"):
        compose_prompt([ref])


def test_a_stored_value_holding_media_is_refused_when_it_must_be_opened(workspace: Path) -> None:
    from kedi.artifacts import ArtifactPolicy

    calls: list[Any] = []
    source = "> artifacts:\n    enabled: false\n[summary] << Summarize <<log>>.\n= done\n"
    runtime = compile_program(
        parse_program(source),
        adapter=PydanticAdapter(FunctionModel(lambda *args: calls.append(args))),
    )
    runtime._initial_globals_env["log"] = runtime.artifacts.artifactize(
        {"photo": CAT, "pad": "x" * 5000},
        policy=ArtifactPolicy.patch({"threshold": "1kb"}),
        source="tool:read_log",
    )
    try:
        with pytest.raises(
            KediExecutionError, match="a stored value holding media cannot be opened"
        ):
            runtime.run_main()
    finally:
        asyncio.run(runtime.aclose())
    assert calls == []
