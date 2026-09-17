from __future__ import annotations

from typing import Any, Protocol

from .context import agent_run_request_scope
from .models import AdapterCallOptions, AgentRunRequest, SchemaOutput, TextOutput, TypedOutput


class RequestLoweringAdapter(Protocol):
    def invoke_sync(self, *, prompt: str, **kwargs: Any) -> Any: ...

    def produce_sync(self, *, template: str, **kwargs: Any) -> Any: ...


def lower_request_sync(
    adapter: RequestLoweringAdapter,
    request: AgentRunRequest,
    *,
    options: AdapterCallOptions | None = None,
) -> Any:
    """Route one semantic request through an adapter's existing native lowering surface."""

    kwargs = dict(options.native_kwargs) if options is not None else {}
    if request.instruction_text is not None:
        kwargs["instructions"] = request.instruction_text
    with agent_run_request_scope(request):
        if isinstance(request.output, TextOutput):
            return adapter.invoke_sync(prompt=request.prompt.content, **kwargs)
        if isinstance(request.output, SchemaOutput):
            return adapter.produce_sync(
                template=request.prompt.content,
                output_schema=dict(request.output.schema),
                **kwargs,
            )
        if isinstance(request.output, TypedOutput):
            return adapter.produce_sync(
                template=request.prompt.content,
                output_type=request.output.output_type,
                **kwargs,
            )
        raise TypeError(f"Unsupported Kedi output contract: {type(request.output).__name__}")


__all__ = ["RequestLoweringAdapter", "lower_request_sync"]
