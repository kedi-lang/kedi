"""References in Kedi prompts: attachments before the text, IDs, reuse and error paths."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel

from kedi.agent_adapter import LangChainAdapter, PydanticAdapter, UserPromptSubmitDecision
from kedi.engine import SequentialEngine, ThreadEngine
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program
from tests.test_media_values import png_bytes

TEMPLATE_PREFIX = "Fields to be substituted into the template string for final construction: "


def image_open(name: str) -> str:
    return f'<image id="{name}" type="image/png">\n'


def attachments_open(name: str) -> str:
    return "<attachments>\n" + image_open(name)


CLOSE = "\n</image>\n</attachments>\n"


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KEDI_WORKSPACE_POLICY", raising=False)
    photos = tmp_path / "photos"
    photos.mkdir()
    for index, name in enumerate(("cat", "dog", "receipt"), start=1):
        (photos / f"{name}.png").write_bytes(png_bytes(index))
    (tmp_path / "notes.md").write_text("# Meeting\n- decision: ship\n", encoding="utf-8")
    return tmp_path


def _arguments(schema: dict[str, Any]) -> dict[str, Any]:
    values = {"boolean": True, "number": 12.5, "integer": 3, "array": ["a"]}
    return {
        key: values.get(spec.get("type"), "x") for key, spec in schema.get("properties", {}).items()
    }


def _pydantic_item(item: Any) -> str:
    if isinstance(item, str):
        return item
    return f"<{getattr(item, 'kind', type(item).__name__)} {getattr(item, 'identifier', '')}>"


def run_pydantic(
    source: str,
    *,
    engine: Any = None,
    hook: Callable[[Any], Any] | None = None,
) -> tuple[Any, list[list[str]]]:
    """Run a program on Pydantic AI and return the result and each request's user content."""

    seen: list[list[str]] = []

    def respond(messages: list[Any], info: Any) -> ModelResponse:
        for part in messages[-1].parts:
            if isinstance(part, UserPromptPart):
                content = [part.content] if isinstance(part.content, str) else part.content
                seen.append([_pydantic_item(item) for item in content])
        if info.output_tools:
            tool = info.output_tools[0]
            return ModelResponse(
                parts=[ToolCallPart(tool.name, _arguments(tool.parameters_json_schema))]
            )
        return ModelResponse(parts=[TextPart("a cat")])

    adapter = PydanticAdapter(FunctionModel(respond), hook_handler=hook)
    return _run(source, adapter, engine), seen


class _FakeChat(BaseChatModel):
    tools: list[dict[str, Any]] = Field(default_factory=list)
    seen: list[list[Any]] = Field(default_factory=list)
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-media"

    def bind_tools(self, tools: Any, **kwargs: Any) -> _FakeChat:
        self.tools = [convert_to_openai_tool(tool) for tool in tools]
        return self

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        self.calls += 1
        last = [message for message in messages if isinstance(message, HumanMessage)][-1]
        content = [last.content] if isinstance(last.content, str) else last.content
        self.seen.append(
            [
                block["text"] if isinstance(block, dict) and block["type"] == "text" else block
                for block in content
            ]
        )
        structured = [tool for tool in self.tools if "artifact" not in tool["function"]["name"]]
        if structured:
            function = structured[-1]["function"]
            call = {
                "name": function["name"],
                "args": _arguments(function["parameters"]),
                "id": f"call_{self.calls}",
            }
            message = AIMessage(content="", tool_calls=[call])
        else:
            message = AIMessage(content="a cat")
        return ChatResult(generations=[ChatGeneration(message=message)])


def run_langchain(source: str) -> tuple[Any, list[list[Any]]]:
    model = _FakeChat()
    return _run(source, LangChainAdapter(model), None), model.seen


def _run(source: str, adapter: Any, engine: Any) -> Any:
    engine = engine or SequentialEngine()
    runtime = compile_program(parse_program(source), adapter=adapter, engine=engine)
    try:
        return runtime.run_main()
    finally:
        asyncio.run(runtime.aclose())
        engine.shutdown()


def _langchain_shape(content: list[Any]) -> list[str]:
    return [
        item if isinstance(item, str) else f"<{item['type']} {item.get('mime_type')}>"
        for item in content
    ]


COMPARE = """> import: media

[cat_image: Image] = `image("photos/cat.png")`
[dog_image: Image] = `image("photos/dog.png")`

>> Compare <<cat_image>> and <<dog_image>>. Is <<cat_image>> happier? [answer: bool]
= <answer>
"""


def test_attachments_come_before_the_text_and_the_text_shows_ids(workspace: Path) -> None:
    result, seen = run_pydantic(COMPARE)
    assert result == "True"
    assert seen == [
        [
            attachments_open("cat_image"),
            "<binary cat_image>",
            "\n</image>\n" + image_open("dog_image"),
            "<binary dog_image>",
            CLOSE
            + TEMPLATE_PREFIX
            + "(answer).\n\nThe template string:\n"
            + "Compare cat_image and dog_image. Is cat_image happier? [answer]",
        ]
    ]


def test_langchain_sends_the_same_structure_as_standard_blocks(workspace: Path) -> None:
    _, seen = run_langchain(COMPARE)
    assert _langchain_shape(seen[0])[:4] == [
        attachments_open("cat_image"),
        "<image image/png>",
        "\n</image>\n" + image_open("dog_image"),
        "<image image/png>",
    ]
    assert seen[0][1]["base64"] == base64.b64encode(png_bytes(1)).decode()
    assert seen[0][-1].endswith("Compare cat_image and dog_image. Is cat_image happier? [answer]")


@pytest.mark.parametrize("runner", [run_pydantic, run_langchain])
def test_ids_lists_dedupe_raw_invokes_and_claims(workspace: Path, runner: Any) -> None:
    source = """> import: media

[cat_image: Image] = `image("photos/cat.png")`
[photos: list[Image]] = `[cat_image, image("photos/dog.png")]`
[same: Image] = `cat_image`
>> Total of <<image(photos/receipt.png)>>: [total: float]
>> Is <<same>> among <<photos>>? [found: bool]
[caption] << Describe <<cat_image>>.
[mode] = no
> if: <<cat_image>> shows a cat
  [mode] := yes
> if: <<image(photos/dog.png)>> shows a dog
  [mode] := <mode>!
= <total>|<found>|<caption>|<mode>
"""
    result, seen = runner(source)
    assert result == "12.5|True|a cat|yes!"
    texts = [
        "".join(
            item for item in request if isinstance(item, str) and not item.startswith("<binary")
        )
        for request in seen
    ]
    assert texts[0].startswith(attachments_open("image_result_1"))
    assert texts[0].endswith("Total of image_result_1: [total]")
    # The same content is attached once and renders the first ID it got in this request.
    assert texts[1].startswith(attachments_open("same") + "\n</image>\n" + image_open("photos_2"))
    assert texts[1].endswith("Is same among same, photos_2? [found]")
    assert texts[2] == attachments_open("cat_image") + CLOSE + "Describe cat_image."
    assert texts[3] == attachments_open("cat_image") + CLOSE + "Claim: cat_image shows a cat"
    assert (
        texts[4] == attachments_open("image_result_1") + CLOSE + "Claim: image_result_1 shows a dog"
    )


@pytest.mark.parametrize("runner", [run_pydantic, run_langchain])
def test_text_documents_and_values_are_document_elements(workspace: Path, runner: Any) -> None:
    source = """> import: media

~Owner(name: str, age: int)

[owner: Owner] = `Owner(name="Ada", age=36)`
[tags: list[str]] = `["a", "b"]`
>> List the decisions in <<document(notes.md)>> for <<owner>> with <<tags>>: [decisions: list[str]]
= done
"""
    _, seen = runner(source)
    assert seen[0][0] == (
        "<attachments>\n"
        '<document id="document_result_1" type="text/markdown">\n'
        "# Meeting\n- decision: ship\n\n"
        "</document>\n"
        '<document id="owner" type="application/json">\n'
        '{"name":"Ada","age":36}\n'
        "</document>\n"
        '<document id="tags" type="application/json">\n'
        '["a","b"]\n'
        "</document>\n"
        "</attachments>\n"
        + TEMPLATE_PREFIX
        + "(decisions).\n\nThe template string:\n"
        + "List the decisions in document_result_1 for owner with tags: [decisions]"
    )


def test_referenced_content_is_never_turned_into_an_artifact_by_its_size(workspace: Path) -> None:
    (workspace / "log.txt").write_text("line\n" * 400, encoding="utf-8")
    source = """> import: media
> artifacts:
    threshold: 1kb

[summary] << Summarize <<document(log.txt)>>.
= <summary>
"""
    _, seen = run_pydantic(source)
    (content,) = seen[0]
    assert content.startswith(
        '<attachments>\n<document id="document_result_1" type="text/plain">\n' + "line\n" * 400
    )
    assert content.endswith("</attachments>\nSummarize document_result_1.")


def _stored_log_program(extra: str) -> tuple[list[list[str]], list[str]]:
    """A template output stored as an artifact, then used again in a prompt."""
    seen: list[list[str]] = []
    texts: list[str] = []

    def respond(messages: list[Any], info: Any) -> ModelResponse:
        part = next(p for p in messages[-1].parts if isinstance(p, UserPromptPart))
        content = [part.content] if isinstance(part.content, str) else list(part.content)
        seen.append([_pydantic_item(item) for item in content])
        if info.output_tools:
            tool = info.output_tools[0]
            return ModelResponse(parts=[ToolCallPart(tool.name, {"log": "line\n" * 400})])
        texts.append(content[-1])
        return ModelResponse(parts=[TextPart("ok")])

    source = f"""> artifacts:
    threshold: 1kb

>> Write the build log as [log].
{extra}
= done
"""
    _run(source, PydanticAdapter(FunctionModel(respond)), None)
    return seen, texts


def test_stored_values_go_by_reference_only_through_double_angles(workspace: Path) -> None:
    seen, _ = _stored_log_program(
        '[note: str] = `"short"`\n[summary] << Summarize <<log>> with <<note>>.'
    )
    (content,) = seen[1]
    opening, reference, closing = content.split("\n", 3)[1:]
    assert opening == '<artifact id="log">'
    assert json.loads(reference)["ref_id"] == "template_output_1"
    assert closing == (
        '</artifact>\n<document id="note" type="text/plain">\nshort\n</document>\n'
        "</attachments>\nSummarize log with note."
    )

    # <log> is the template text itself: the whole log, never an artifact reference.
    seen, _ = _stored_log_program("[summary] << Summarize <log>.")
    assert seen[1] == ["Summarize " + "line\n" * 400 + "."]


def test_stored_values_are_sent_in_full_when_the_model_cannot_read_artifacts(
    workspace: Path,
) -> None:
    seen, _ = _stored_log_program(
        "> artifacts:\n    enabled: false\n[summary] << Summarize <<log>>."
    )
    (content,) = seen[1]
    assert content.startswith('<attachments>\n<document id="log" type="text/plain">\nline\n')


def test_typed_parameters_take_media_values_only(workspace: Path) -> None:
    source = """> import: media

@caption(photo: Image) -> str:
  [text] << Describe <<photo>> in one sentence.
  = <text>

[invoice: Image] = `image("photos/receipt.png")`
= <caption(`invoice`)>
"""
    result, seen = run_pydantic(source)
    assert result == "a cat"
    assert seen == [
        [attachments_open("photo"), "<binary photo>", CLOSE + "Describe photo in one sentence."]
    ]
    # A string is never converted to media, not even at a typed parameter.
    with pytest.raises(KediExecutionError, match="strings are never converted to media"):
        run_pydantic(source.replace("<caption(`invoice`)>", "<caption(photos/cat.png)>"))


def test_parallel_engine_attaches_references_from_promises(workspace: Path) -> None:
    source = """> import: media

@pick() -> Image:
  = `image("photos/dog.png")`

[cat_image: Image] = `image("photos/cat.png")`
[dog: Image] = `pick()`
>> What is <<cat_image>>? [what]
>> Compare <<dog>> with <<cat_image>> given <what>: [comparison]
= <comparison>
"""
    result, seen = run_pydantic(source, engine=ThreadEngine())
    assert result == "x"
    compare = next(request for request in seen if "<binary dog>" in request)
    assert compare[:4] == [
        attachments_open("dog"),
        "<binary dog>",
        "\n</image>\n" + image_open("cat_image"),
        "<binary cat_image>",
    ]


def test_hooks_see_attachments_read_only_and_edit_only_the_text(workspace: Path) -> None:
    events: list[Any] = []

    def hook(event: Any) -> Any:
        if event.event == "user_prompt_submit":
            events.append(event)
            return UserPromptSubmitDecision.edit(event.content + " Be brief.")
        return None

    source = """> import: media
[cat_image: Image] = `image("photos/cat.png")`
[text] << Describe <<cat_image>>.
= <text>
"""
    _, seen = run_pydantic(source, hook=hook)
    (event,) = events
    (summary,) = event.media
    assert (summary.name, summary.kind, summary.media_type, summary.source) == (
        "cat_image",
        "image",
        "image/png",
        "data",
    )
    assert summary.size == len(png_bytes(1)) and len(summary.sha256 or "") == 64
    assert "iVBOR" not in repr(event)
    assert seen == [
        [
            attachments_open("cat_image"),
            "<binary cat_image>",
            CLOSE + "Describe cat_image. Be brief.",
        ]
    ]


HISTORY = """> import: media
> history: enabled

[photo: Image] = `image("photos/cat.png")`
[notes: Document] = `document("notes.md")`
[a] << Describe <<photo>> using <<notes>>.
[b] << How many colors does <<photo>> have? Check <<notes>>.
[photo] := `image("photos/dog.png")`
[c] << Describe <<photo>>.
[photo] := `image("photos/cat.png")`
[d] << Describe <<photo>> again.
= done
"""


@pytest.mark.parametrize("runner", [run_pydantic, run_langchain])
def test_history_reuses_attachments_by_id_and_content(workspace: Path, runner: Any) -> None:
    _, seen = runner(HISTORY)
    sent = [
        [
            name
            for name in ("photo", "notes")
            if any(isinstance(item, str) and f'id="{name}"' in item for item in request)
        ]
        for request in seen
    ]
    # Turn 2 reuses both; a rebound name is attached again, also when it returns to old content.
    assert sent == [["photo", "notes"], [], ["photo"], ["photo"]]
    assert seen[1] == ["How many colors does photo have? Check notes."]


def test_history_processor_attaches_references_on_every_turn(workspace: Path) -> None:
    seen: list[list[str]] = []

    def respond(messages: list[Any], info: Any) -> ModelResponse:
        part = next(p for p in messages[-1].parts if isinstance(p, UserPromptPart))
        content = [part.content] if isinstance(part.content, str) else part.content
        seen.append([_pydantic_item(item) for item in content])
        return ModelResponse(parts=[TextPart("ok")])

    with pytest.warns(UserWarning, match="History processing"):
        adapter = PydanticAdapter(
            FunctionModel(respond), history_processor=lambda context: context.messages
        )
    source = """> import: media
> history: enabled
[photo: Image] = `image("photos/cat.png")`
[a] << Describe <<photo>>.
[b] << Describe <<photo>> again.
= done
"""
    _run(source, adapter, None)
    assert [request[0] for request in seen] == [attachments_open("photo")] * 2


@pytest.mark.parametrize(
    ("statement", "message"),
    [
        (">> See <cat_image> [x]", "<cat_image> renders text, but the value is <image cat.png"),
        ("[m] = See <cat_image>", "<cat_image> renders text"),
        ("[m: Image] = <image(photos/cat.png)>", r"<image\(...\)> renders text"),
        ("[m: Image] = photos/cat.png", "strings are never converted to media"),
        ("= <cat_image>", "<cat_image> renders text"),
        ("[m] = <describe(<cat_image>)>", "<cat_image> renders text"),
        ("> instructions:\n    Look at <cat_image>.\n>> Hi [x]", "<cat_image> renders text"),
        (
            ">> Describe <`{'photo': cat_image}`> [x]",
            "renders text, but the value is of type dict. In a model prompt bind it to a name",
        ),
        (">> Compare <`[cat_image, 'x']`> [x]", r"\(in a list\)"),
        (">> Compare <<pair>> [x]", "media inside a dict, a model or a mixed list"),
        (">> Describe <<nothing>> [x]", "<<nothing>> is None"),
        (">> Describe <nothing> [x]", "<nothing> renders text, but the value is None"),
        ("> show: <cat_image>", "<cat_image> renders text"),
    ],
)
def test_substituting_non_text_fails_loudly(workspace: Path, statement: str, message: str) -> None:
    source = f"""> import: media

@describe(photo: Image) -> str:
  [text] << Describe <<photo>>.
  = <text>

[cat_image: Image] = `image("photos/cat.png")`
[pair: list] = `[cat_image, "x"]`
[nothing: str | None] = `None`
{statement}
"""
    with pytest.raises(KediExecutionError, match=message):
        run_pydantic(source)


@pytest.mark.parametrize(
    "statement",
    [
        "[m] = See <<cat_image>>",
        "= <<cat_image>>",
        "> show: <<cat_image>>",
        "[m] = <describe(<<cat_image>>)>",
    ],
)
def test_references_outside_model_prompts_are_parse_errors(workspace: Path, statement: str) -> None:
    from kedi.lang.parser.diagnostics import KediParseError

    with pytest.raises(KediParseError, match="are not allowed in"):
        parse_program(
            f'> import: media\n[cat_image: Image] = `image("photos/cat.png")`\n{statement}\n'
        )


def test_show_prints_a_byte_free_label_with_backticks(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_pydantic(
        '> import: media\n[cat_image: Image] = `image("photos/cat.png")`\n> show: `cat_image`\n'
    )
    assert capsys.readouterr().out.strip() == "<image cat.png · image/png · 73 B>"


def test_loader_errors_name_the_cause(workspace: Path) -> None:
    with pytest.raises(KediExecutionError, match="path escapes the working directory"):
        run_pydantic('> import: media\n[x: Image] = `image("../outside.png")`\n')
    with pytest.raises(KediExecutionError, match="Media file not found: photos/missing.png"):
        run_pydantic("> import: media\n[t] << See <<image(photos/missing.png)>>.\n")


def test_adapters_without_media_input_fail_before_model_io(workspace: Path) -> None:
    from kedi.agent_adapter.meta import AdapterCapabilities
    from kedi.media import MediaError

    adapter = PydanticAdapter(FunctionModel(lambda messages, info: ModelResponse(parts=[])))
    adapter.capabilities = AdapterCapabilities(
        supports_structured_output=True, supports_media_input=False
    )
    with pytest.raises((KediExecutionError, MediaError), match="does not accept media input"):
        _run(
            '> import: media\n[cat_image: Image] = `image("photos/cat.png")`\n'
            "[t] << See <<cat_image>>.\n",
            adapter,
            None,
        )


def test_debugger_events_carry_metadata_but_no_bytes(workspace: Path) -> None:
    from kedi.debugging import observe_execution

    events: list[Any] = []
    source = (
        '> import: media\n[cat_image: Image] = `image("photos/cat.png")`\n'
        "[t] << See <<cat_image>>.\n"
    )
    with observe_execution(events.append):
        run_pydantic(source)
    payloads = [event.payload for event in events if event.kind in {"model_input", "request"}]
    assert payloads and all(payload["media"][0]["name"] == "cat_image" for payload in payloads)
    encoded = base64.b64encode(png_bytes(1)).decode()
    assert all(encoded not in json.dumps(payload, default=str) for payload in payloads)
