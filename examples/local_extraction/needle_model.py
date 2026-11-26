"""Cactus Needle 3 as a Pydantic AI model, for Kedi's typed extraction (#198, P0).

Needle 3 is a small function-calling model (34 MB, Apache 2.0, ``pip install
cactus-compute``). Kedi asks for a typed value through an output tool. This model hands
that tool's schema to Needle, with the request Kedi rendered for this call, and answers
with the tool call. Kedi and Pydantic AI validate the result, so Cactus's own repair is
off (``strict=False``).

Needle answers only with structured output. A plain ``<<`` prompt without a type, or a
program tool from ``> use:``, fails with a clear error. Kedi's artifact tools are offered
to every model and are optional, so Needle leaves them unused.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.settings import ModelSettings

from kedi.agent_adapter import current_request, has_current_request

# Cactus reports usage to its servers unless told not to; this example never sends anything.
os.environ["CACTUS_TELEMETRY"] = "0"
os.environ["DO_NOT_TRACK"] = "1"


class NeedleModel(Model):
    """Answers Kedi's output tool with Needle's extraction from Kedi's prompt."""

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
        if not has_current_request():
            raise UserError("NeedleModel answers Kedi requests; call it through a Kedi program")
        if not model_request_parameters.output_tools:
            raise UserError(
                "Needle 3 answers only with structured output; capture a typed value, "
                "for example [order: Order]"
            )
        program_tools = [
            tool.name
            for tool in model_request_parameters.function_tools
            if (tool.metadata or {}).get("kedi_tool_kind") != "artifact_management"
        ]
        if program_tools:
            raise UserError(f"Needle 3 calls no tools; remove {program_tools} from > use:")
        tool = model_request_parameters.output_tools[0]
        schema = {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.parameters_json_schema,
        }
        # Kedi's rendering of this call: a retry or earlier history is not something to
        # extract from.
        rendered = current_request().render()
        arguments = await asyncio.to_thread(
            _extract, rendered.user, schema, rendered.system or "", self.max_new_tokens
        )
        if arguments is None:
            # Needle called no tool; Pydantic AI then asks again for the output tool.
            return ModelResponse(parts=[TextPart("")], model_name=self.model_name)
        return ModelResponse(parts=[ToolCallPart(tool.name, arguments)], model_name=self.model_name)


def _extract(prompt: str, schema: dict[str, Any], system: str, max_new_tokens: int) -> Any:
    from cactus import extract

    return extract(
        prompt, schema, system=system or None, max_new_tokens=max_new_tokens, strict=False
    )
