"""Cactus Needle 3 as a Pydantic AI model, for Kedi's typed extraction (#198, P0).

Needle 3 is a small function-calling model (34 MB, Apache 2.0, ``pip install
cactus-compute``). Kedi asks for a typed value through an output tool; this model hands that
tool's schema and Kedi's rendered text to Needle and answers with the tool call. Kedi and
Pydantic AI validate the result, so Cactus's own repair is off (``strict=False``).

Needle answers only with structured output: it does not call Kedi's other tools, and a plain
``<<`` prompt without a type fails clearly.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from typing import Any

from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.settings import ModelSettings

# Cactus reports usage to its servers unless told not to; this example never sends anything.
os.environ["CACTUS_TELEMETRY"] = "0"
os.environ["DO_NOT_TRACK"] = "1"


class NeedleModel(Model):
    """Answers Kedi's output tool with Needle's extraction from the prompt."""

    def __init__(self, *, max_new_tokens: int = 512) -> None:
        super().__init__()
        self.max_new_tokens = max_new_tokens

    @property
    def model_name(self) -> str:
        return "needle3"

    @property
    def system(self) -> str:
        return "cactus"

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        if not model_request_parameters.output_tools:
            raise UserError(
                "Needle 3 answers only with structured output; capture a typed value, "
                "for example [order: Order]"
            )
        tool = model_request_parameters.output_tools[0]
        schema = {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.parameters_json_schema,
        }
        system, prompt = _texts(messages)
        arguments = await asyncio.to_thread(_extract, prompt, schema, system, self.max_new_tokens)
        if arguments is None:
            # Needle called no tool; Pydantic AI then asks again for the output tool.
            return ModelResponse(parts=[TextPart("")], model_name=self.model_name)
        return ModelResponse(parts=[ToolCallPart(tool.name, arguments)], model_name=self.model_name)


def _extract(prompt: str, schema: dict[str, Any], system: str, max_new_tokens: int) -> Any:
    from cactus import extract

    return extract(
        prompt, schema, system=system or None, max_new_tokens=max_new_tokens, strict=False
    )


def _texts(messages: Sequence[ModelMessage]) -> tuple[str, str]:
    """Kedi's system text and the latest user prompt, as Needle takes them."""

    system: list[str] = []
    prompt = ""
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        if message.instructions:
            system.append(message.instructions)
        for part in message.parts:
            if isinstance(part, SystemPromptPart):
                system.append(part.content)
            elif isinstance(part, UserPromptPart) and not prompt:
                # Kedi's prompt; Pydantic AI's later retry text is not something to extract from.
                content = part.content
                prompt = (
                    content
                    if isinstance(content, str)
                    else " ".join(item for item in content if isinstance(item, str))
                )
    return "\n\n".join(dict.fromkeys(system)), prompt
