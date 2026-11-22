"""Recover equivalent final structured calls without another model request."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from jsonschema.validators import validator_for
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain.agents.structured_output import MultipleStructuredOutputsError
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, ValidationError

from kedi.agent_adapter.output_schema import OutputCorrection
from kedi.agent_adapter.subagents import SubagentOutputState, accept_subagent_output


@dataclass(kw_only=True)
class RejectedDuplicateOutput(ModelResponse[Any]):
    """Output-only rejection, with intact messages, for the outer output retry loop."""

    error: MultipleStructuredOutputsError


class LangChainDuplicateOutputMiddleware(AgentMiddleware):
    """Validate duplicate candidates before recovering an output-only final response.

    Place after output correction and before request retry/run middleware. Pass the
    same titled JSON schema as ``response_format``. For ordinary typed output, also
    pass ``output_model``; otherwise candidates use subagent JSON Schema validation
    and, when supplied, a fresh copy of ``output_state``. Only normal outer output
    acceptance may commit the real subagent state.

    Rejected duplicates retain native correction feedback and every call/result
    pair. Output-only rejections carry a typed error for the outer correction loop;
    mixed turns stay on their native tool path. This middleware performs no retries
    and owns no correction counter.
    """

    def __init__(
        self,
        *,
        schema: dict[str, Any],
        output_model: type[BaseModel] | None = None,
        output_state: SubagentOutputState | None = None,
    ) -> None:
        if output_model is not None and output_state is not None:
            raise ValueError("Pass output_model or subagent output_state, not both")
        title = schema.get("title")
        self.output_tool_name = title if isinstance(title, str) and title else None
        self.output_model = output_model
        self.output_state = output_state
        self.validator = validator_for(schema)(schema)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        response = await handler(request)
        name = self.output_tool_name
        if (
            response.structured_response is not None
            or name is None
            or any(_tool_name(tool) == name for tool in request.tools)
        ):
            return response
        ai_messages = [message for message in response.result if isinstance(message, AIMessage)]
        if len(ai_messages) != 1:
            return response
        message = ai_messages[0]
        calls = [call for call in message.tool_calls if call["name"] == name]
        if len(calls) < 2:
            return response

        # Recovery must not hide ambiguous IDs or manufacture missing tool results.
        all_ids = [call.get("id") for call in message.tool_calls]
        if any(not isinstance(call_id, str) or not call_id for call_id in all_ids):
            return response
        if len(set(all_ids)) != len(all_ids):
            return response
        output_ids = {call["id"] for call in calls}
        acknowledgements = {
            index: result
            for index, result in enumerate(response.result)
            if isinstance(result, ToolMessage) and result.tool_call_id in output_ids
        }
        if (
            len(acknowledgements) != len(calls)
            or {result.tool_call_id for result in acknowledgements.values()} != output_ids
            or any(result.name != name for result in acknowledgements.values())
        ):
            return response

        output_only = not message.invalid_tool_calls and len(calls) == len(message.tool_calls)

        def rejected() -> ModelResponse[Any]:
            # LangChain's duplicate-error messages default to success. That status
            # incorrectly ends mixed turns after business tools instead of correcting.
            results = [
                result.model_copy(update={"status": "error"})
                if index in acknowledgements
                else result
                for index, result in enumerate(response.result)
            ]
            if output_only:
                return RejectedDuplicateOutput(
                    result=results,
                    error=MultipleStructuredOutputsError([call["name"] for call in calls], message),
                )
            return replace(response, result=results)

        if not output_only:
            return rejected()
        if any(
            isinstance(result, ToolMessage) and index not in acknowledgements
            for index, result in enumerate(response.result)
        ):
            return response
        try:
            # Compare full wire values, not Python equality (True == 1) or a
            # potentially lossy model serializer. Object key order is immaterial.
            candidates = [
                json.dumps(call["args"], sort_keys=True, separators=(",", ":"), allow_nan=False)
                for call in calls
            ]
        except (TypeError, ValueError):
            return rejected()
        if any(candidate != candidates[0] for candidate in candidates[1:]):
            return rejected()
        try:
            for call in calls:
                if self.output_model is not None:
                    self.output_model.model_validate_json(json.dumps(call["args"], allow_nan=False))
                else:
                    accept_subagent_output(
                        call["args"],
                        self.validator,
                        replace(self.output_state) if self.output_state is not None else None,
                    )
        except (OutputCorrection, ValidationError):
            return rejected()

        return replace(
            response,
            structured_response=calls[0]["args"],
            result=[
                result.model_copy(
                    update={
                        "content": "Equivalent structured responses accepted.",
                        "status": "success",
                    }
                )
                if index in acknowledgements
                else result
                for index, result in enumerate(response.result)
            ],
        )


def _tool_name(tool: Any) -> str | None:
    if isinstance(tool, Mapping):
        function = tool.get("function")
        return function.get("name") if isinstance(function, Mapping) else tool.get("name")
    return getattr(tool, "name", None)
