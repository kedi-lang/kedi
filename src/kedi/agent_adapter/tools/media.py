"""Media that tools return: one adapter-neutral form, which each adapter sends natively.

A tool result that holds media becomes text and attachments, as a ``<<x>>`` reference does:
the text names each media value with a placeholder (``<image snap_result_1>``, or JSON holding
placeholders for a dict or a model), and the attachments carry the media. Text documents stay
inline in the text. Results that hold media skip artifact admission: the model reads them as
media, not as a stored reference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from kedi.agent_adapter.request.context import tool_media_namer
from kedi.agent_adapter.request.media import MediaAttachment
from kedi.media import Media, contains_media, reference


@dataclass(frozen=True, slots=True)
class ToolMedia:
    """A tool result holding media: the text the model reads and the media it names.

    ``value`` is the result with each media replaced by its placeholder, the JSON-compatible
    form CodeMode code receives.
    """

    text: str
    attachments: tuple[MediaAttachment, ...]
    value: Any = None


def tool_media(value: Any, *, tool_name: str) -> ToolMedia | None:
    """``value`` as text and attachments when it holds media, else ``None``."""

    if not contains_media(value):
        return None
    namer = tool_media_namer()
    generated = f"{tool_name}_result"
    ref = reference(value, call=tool_name)
    if ref.layout is not None:
        names = [
            namer.name_for(item, preferred)
            for item, preferred in zip(ref.items, ref.layout.preferred, strict=True)
        ]
        text = ref.layout.document(names).text
        placeholder_value = json.loads(text) if ref.layout.structured else text
    else:
        names = [namer.name_for(item, None, generated=generated) for item in ref.items]
        placeholders = [
            _placeholder(item, name) for item, name in zip(ref.items, names, strict=True)
        ]
        placeholder_value = placeholders if ref.is_list else placeholders[0]
        text = json.dumps(placeholders) if ref.is_list else placeholders[0]
    # One image named twice (two fields, or a repeated list item) is attached once.
    attachments = list(
        {
            name: MediaAttachment(name=name, media=item)
            for item, name in zip(ref.items, names, strict=True)
        }.values()
    )
    return _with_inline_text(text, attachments, value=placeholder_value)


def refuse_tool_media(value: Any) -> None:
    """Fail for a tool result holding media on a tool bridge that carries text only.

    The adapter turns the failure into the tool's error, which the model reads; sending a
    label or a data URI instead would hide the media.
    """

    if contains_media(value) or codemode_media(value) is not None:
        from kedi.media import MediaError

        raise MediaError(
            "The tool returned media, and this adapter's tool bridge sends text only; "
            "Pydantic AI and LangChain send tool media to the model"
        )


def sandbox_value(value: Any, *, tool_name: str) -> Any:
    """A nested tool result as CodeMode code receives it: media become placeholders.

    The request's namer keeps the media, so a code result naming a placeholder attaches it.
    """

    media = tool_media(value, tool_name=tool_name) if contains_media(value) else None
    return value if media is None else media.value


def codemode_media(value: Any) -> ToolMedia | None:
    """Media a CodeMode result names by placeholder, from its nested tool calls.

    The code saw only placeholders (``<image snap_result_1>``); the media they name in its
    result are attached to it.
    """

    named = tool_media_namer().attachments
    if not named:
        return None
    text = _result_text(value)
    if text is None:
        return None
    mentioned = [item for item in named if _placeholder(item.media, item.name) in text]
    if not mentioned:
        return None
    return _with_inline_text(text, mentioned, value=value)


def _result_text(value: Any) -> str | None:
    """A result's text as a tool bridge writes it, or ``None`` when it has no JSON form."""

    if isinstance(value, str):
        return value
    from pydantic_core import PydanticSerializationError

    from kedi.agent_adapter.tools.execution import _tool_result_adapter

    try:
        return _tool_result_adapter().dump_json(value).decode("utf-8")
    except PydanticSerializationError:
        return None


def _with_inline_text(text: str, attachments: list[MediaAttachment], *, value: Any) -> ToolMedia:
    inline = [attachment.inline_text() for attachment in attachments if attachment.is_text]
    return ToolMedia(
        text="\n".join([text, *inline]),
        attachments=tuple(attachment for attachment in attachments if not attachment.is_text),
        value=value,
    )


def _placeholder(media: Media, name: str) -> str:
    return f"<{media.kind} {name}>"


__all__ = [
    "ToolMedia",
    "codemode_media",
    "contains_media",
    "refuse_tool_media",
    "sandbox_value",
    "tool_media",
]
