"""Cactus Mirage as a LangChain chat model that reads the writing in an image (#198, P1).

Mirage (21 MB, Apache 2.0, ``pip install cactus-compute``) is a small vision model: it
describes scenes, finds objects and reads text. Here it does one thing, reading text, for
prompts that reference an image, ``<<photo>>``. The model profile declares ``image_inputs``,
so Kedi sends the image itself; Mirage returns each text region with a box, and this module
puts them in reading order, one line per row.

It answers in text, reading Latin script only, as Mirage does. A typed capture or a program
tool fails with a clear error.
"""

from __future__ import annotations

import base64
import os
from functools import cache
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

# Cactus reports usage to its servers unless told not to; this example never sends anything.
os.environ["CACTUS_TELEMETRY"] = "0"
os.environ["DO_NOT_TRACK"] = "1"


class MirageReader(BaseChatModel):
    """Answers with the writing in the image of the last user turn, line by line."""

    # No tool calling: Kedi then offers no tools, and binds one only for a program tool or a
    # typed capture, which this reader refuses.
    profile: Any = {"image_inputs": True, "tool_calling": False, "structured_output": False}

    @property
    def _llm_type(self) -> str:
        return "cactus-mirage"

    def bind_tools(self, tools: Any, **_: Any) -> MirageReader:
        raise NotImplementedError(
            "MirageReader answers with the text it reads: it takes no tools and no typed "
            "output, so capture untyped text, for example [writing] << Read <<photo>>."
        )

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **_: Any
    ) -> ChatResult:
        message = next(m for m in reversed(messages) if isinstance(m, HumanMessage))
        images = [
            part
            for part in message.content
            if isinstance(part, dict) and part.get("type") == "image"
        ]
        if len(images) != 1:
            raise ValueError(f"MirageReader reads one image per prompt, got {len(images)}")
        regions = _mirage().describe(base64.b64decode(images[0]["base64"]), sections=["text"])
        text = reading_order(regions.get("text", []))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


@cache
def _mirage() -> Any:
    from cactus import Mirage

    return Mirage()


def reading_order(regions: list[dict[str, Any]]) -> str:
    """Text regions as lines: regions whose vertical centers are close share a row."""

    rows: list[tuple[float, list[dict[str, Any]]]] = []
    for region in sorted(regions, key=lambda r: (r["bbox"][1] + r["bbox"][3]) / 2):
        top, bottom = region["bbox"][1], region["bbox"][3]
        middle = (top + bottom) / 2
        if rows and abs(rows[-1][0] - middle) < (bottom - top) / 2:
            rows[-1][1].append(region)
        else:
            rows.append((middle, [region]))
    return "\n".join(
        " ".join(r["string"] for r in sorted(row, key=lambda r: r["bbox"][0])) for _, row in rows
    )
