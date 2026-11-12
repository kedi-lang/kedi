"""Media on decision models: Jev reads images and text documents, Laya reads text."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from kedi_typesafe import TypeSafeModel
from kedi_typesafe.integrations.langchain import TypeSafeChatModel
from typesafe_sdk import NoulAnswer, SystemOneResponse, Usage

from kedi.agent_adapter.adapters import LangChainAdapter, PydanticAdapter
from kedi.agent_adapter.request.media import (
    MediaAttachment,
    check_media_support,
    media_provider,
)
from kedi.lang import compile_program, parse_program
from kedi.media import Audio, Image, MediaError
from tests.test_media_values import png_bytes

SOURCE = """\
> import: media
> import: decisions
> adapter: {adapter}
> model: typesafe/jev-latest

~Check(
  legible: Annotated[bool, "Is the receipt legible?", `BooleanCriteria(true="Legible", false="Blurry")`]
)

>> The receipt in <{image}>.
The check is [check: Check].
= `check.legible`
"""


class Client:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    async def system_one(self, state, questions, **kwargs):
        self.calls.append((state, kwargs))
        return SystemOneResponse(
            model="test",
            answers={key: NoulAnswer(noul=0.95) for key in questions},
            usage=Usage(input_tokens=1, output_tokens=1),
        )

    async def aclose(self) -> None:
        pass


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KEDI_WORKSPACE_POLICY", raising=False)
    (tmp_path / "receipt.png").write_bytes(png_bytes())
    return tmp_path


def _run(monkeypatch: pytest.MonkeyPatch, adapter_name: str, image: str) -> tuple[Any, Client]:
    client = Client()
    if adapter_name == "pydantic":
        model = TypeSafeModel(client=client)
        adapter = PydanticAdapter(model)
        monkeypatch.setattr(PydanticAdapter, "_resolve_model", lambda self, name: model)
    else:
        model = TypeSafeChatModel(client=client)
        adapter = LangChainAdapter(model)
        monkeypatch.setattr(LangChainAdapter, "_resolve_model", lambda self, name: model)
    source = SOURCE.format(adapter=adapter_name, image=image)
    runtime = compile_program(parse_program(source), adapter=adapter)
    try:
        return runtime.run_main(), client
    finally:
        runtime.artifacts.close()


@pytest.mark.parametrize("adapter_name", ["pydantic", "langchain"])
def test_jev_reads_prompt_images_as_assets(workspace, monkeypatch, adapter_name):
    result, client = _run(monkeypatch, adapter_name, "image(receipt.png)")

    assert result is True
    [(state, kwargs)] = client.calls
    [(reference, asset)] = kwargs["assets"].items()
    assert reference == str(asset)
    assert reference in str(state)
    assert "receipt.png" not in str(state)


def test_kedi_downloads_image_urls_for_jev(workspace, monkeypatch):
    from kedi.agent_adapter.adapters.pydantic import media as pydantic_media

    async def download(media):
        return Image.from_bytes(png_bytes())

    monkeypatch.setattr(pydantic_media, "download_media", download)
    result, client = _run(monkeypatch, "pydantic", "image(https://example.com/receipt.png)")

    assert result is True
    [(_, kwargs)] = client.calls
    assert len(kwargs["assets"]) == 1


@pytest.mark.parametrize(
    ("system", "media", "message"),
    [
        (
            "typesafe",
            Audio.from_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt "),
            "images and text documents",
        ),
        ("laya", Image.from_bytes(png_bytes()), "read only text"),
    ],
)
def test_decision_models_reject_media_they_cannot_read(system, media, message):
    attachment = MediaAttachment(name="evidence", media=media)
    with pytest.raises(MediaError, match=message):
        check_media_support([attachment], provider=media_provider(system), model="jev-latest")


def test_jev_accepts_images():
    attachment = MediaAttachment(name="photo", media=Image.from_bytes(png_bytes()))
    check_media_support([attachment], provider=media_provider("typesafe"), model="jev-latest")
