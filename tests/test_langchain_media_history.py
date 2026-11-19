from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import message_to_dict, messages_from_dict

from kedi.agent_adapter.adapters.langchain import media as lowering
from kedi.agent_adapter.request.media import MediaAttachment
from kedi.media import Image
from tests.test_media_values import png_bytes


def test_downloaded_url_is_not_attached_again(monkeypatch):
    downloads = []

    async def download(media):
        downloads.append(media.url)
        return Image.from_bytes(png_bytes(), "image/png")

    monkeypatch.setattr(lowering, "download_media", download)
    attachment = MediaAttachment(name="photo", media=Image.from_url("https://example.com/a.png"))
    model = SimpleNamespace(profile={})

    async def run():
        first = await lowering.user_message(
            "Inspect photo.", [attachment], history=None, model=model
        )
        history = messages_from_dict([message_to_dict(first)])
        second = await lowering.user_message(
            "Recall photo.", [attachment], history=history, model=model
        )
        assert second.content == "Recall photo."

    asyncio.run(run())
    assert downloads == [attachment.media.url]


@pytest.mark.parametrize("direct_url", [False, True])
def test_source_identity_requires_retained_matching_media(monkeypatch, direct_url):
    async def download(media):
        return Image.from_bytes(png_bytes(1 if media.url.endswith("a.png") else 2), "image/png")

    monkeypatch.setattr(lowering, "download_media", download)
    model = SimpleNamespace(profile={"image_url_inputs": direct_url})
    first_image = MediaAttachment(name="photo", media=Image.from_url("https://example.com/a.png"))
    changed_image = MediaAttachment(name="photo", media=Image.from_url("https://example.com/b.png"))

    async def run():
        first = await lowering.user_message("Inspect.", [first_image], history=None, model=model)
        assert (
            await lowering.user_message("Recall.", [first_image], history=[first], model=model)
        ).content == "Recall."
        changed = await lowering.user_message(
            "Changed.", [changed_image], history=[first], model=model
        )
        assert isinstance(changed.content, list)
        assert lowering.sent_attachments([first, changed])["photo"] == changed_image.identity
        for history in ([], [first.model_copy(update={"content": "summary"})]):
            resent = await lowering.user_message(
                "Inspect.", [first_image], history=history, model=model
            )
            assert isinstance(resent.content, list)
        altered = first.model_copy(deep=True)
        assert isinstance(altered.content, list)
        altered.content[1] = {"type": "image", "url": "https://example.com/replaced.png"}
        assert lowering.sent_attachments([altered])["photo"] != first_image.identity
        if not direct_url:
            invalid_metadata = first.model_copy(
                update={"additional_kwargs": {"kedi_media_sources": None}}
            )
            resent = await lowering.user_message(
                "Inspect.", [first_image], history=[invalid_metadata], model=model
            )
            assert isinstance(resent.content, list)

    asyncio.run(run())


def test_source_metadata_does_not_reach_openai_wire(monkeypatch):
    from langchain_openai import ChatOpenAI

    async def download(media):
        return Image.from_bytes(png_bytes(), "image/png")

    monkeypatch.setattr(lowering, "download_media", download)
    attachment = MediaAttachment(name="photo", media=Image.from_url("https://example.com/a.png"))
    model = ChatOpenAI(model="test-model", api_key="test", profile={})
    message = asyncio.run(
        lowering.user_message("Inspect.", [attachment], history=None, model=model)
    )
    assert message.additional_kwargs
    payload = model._get_request_payload([message])
    assert "kedi_media_sources" not in str(payload)
    assert "https://example.com/a.png" not in str(payload)


def test_downloaded_url_reuse_through_a_kedi_program(monkeypatch):
    from tests.test_media_prompts import run_langchain

    downloads = []

    async def download(media):
        downloads.append(media.url)
        return Image.from_bytes(png_bytes(), "image/png")

    monkeypatch.setattr(lowering, "download_media", download)
    _, seen = run_langchain("""> import: media
> history: enabled
[photo: Image] = `image("https://example.com/a.png")`
[first] << Inspect <<photo>>.
[second] << Recall <<photo>>.
= <second>
""")
    assert len(seen) == 2
    assert seen[0][1]["type"] == "image"
    assert seen[1] == ["Recall photo."]
    assert downloads == ["https://example.com/a.png"]
