from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Annotated, Any, Literal, cast

from pydantic import Field, TypeAdapter
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelRetry, UserError
from pydantic_ai.messages import InstructionPart, ToolCallPart, ToolReturn, ToolReturnPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import RunContext, ToolDefinition, ToolDenied
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset, ToolsetTool, WrapperToolset
from pydantic_ai.toolsets.abstract import SchemaValidatorProt
from typing_extensions import NotRequired, Self, TypedDict

from ..adapters.pydantic.compat import build_nested_tool_manager
from ..tools.scope import defer_artifact_admission
from ..tools.specs import ToolResultAdmitter, ToolSpec
from .core import CODEMODE_INSTRUCTIONS, CodeModeLimits, CodeModeSession, CodeModeTool
from .errors import CodeModeCatalogError, CodeModeError


class _SearchToolsArguments(TypedDict):
    query: NotRequired[
        Annotated[
            str | None,
            Field(description="Optional text used to filter tool names."),
        ]
    ]
    limit: NotRequired[
        Annotated[
            int | None,
            Field(description="Maximum names to return. Defaults to 10."),
        ]
    ]
    cursor: NotRequired[
        Annotated[
            str | None,
            Field(description="Opaque cursor returned by an earlier search_tools call."),
        ]
    ]


class _GetToolSchemaArguments(TypedDict):
    tool_names: Annotated[
        list[str],
        Field(description="Exact tool names returned by search_tools."),
    ]


class _ExecuteCodeArguments(TypedDict):
    code: Annotated[str, Field(description="Monty-compatible Python code to execute.")]
    restart: NotRequired[
        Annotated[
            bool,
            Field(description="Reset the per-run sandbox state before executing this code."),
        ]
    ]


_SEARCH_ADAPTER = TypeAdapter(_SearchToolsArguments)
_SCHEMA_ADAPTER = TypeAdapter(_GetToolSchemaArguments)
_EXECUTE_ADAPTER = TypeAdapter(_ExecuteCodeArguments)

_SEARCH_SCHEMA = _SEARCH_ADAPTER.json_schema()
_SCHEMA_SCHEMA = _SCHEMA_ADAPTER.json_schema()
_EXECUTE_SCHEMA = _EXECUTE_ADAPTER.json_schema()

_SEARCH_VALIDATOR = cast("SchemaValidatorProt", _SEARCH_ADAPTER.validator)
_SCHEMA_VALIDATOR = cast("SchemaValidatorProt", _SCHEMA_ADAPTER.validator)
_EXECUTE_VALIDATOR = cast("SchemaValidatorProt", _EXECUTE_ADAPTER.validator)

_CONTROL_KIND = Literal["search", "schema", "execute"]


@dataclass(kw_only=True)
class _CodeModeControlTool(ToolsetTool[Any]):
    control_kind: _CONTROL_KIND
    wrapped_tools: dict[str, ToolsetTool[Any]]
    result_admitter: ToolResultAdmitter | None


@dataclass
class PydanticCodeModeToolset(WrapperToolset[Any]):
    """Project an assembled Pydantic toolset through Kedi's CodeMode controls."""

    limits: CodeModeLimits = field(default_factory=CodeModeLimits)
    _session: CodeModeSession = field(default_factory=CodeModeSession, init=False, repr=False)

    def __post_init__(self) -> None:
        self._session = CodeModeSession(self.limits)

    async def for_run(self, ctx: RunContext[Any]) -> AbstractToolset[Any]:
        wrapped = await self.wrapped.for_run(ctx)
        return type(self)(wrapped=wrapped, limits=self.limits)

    async def for_run_step(self, ctx: RunContext[Any]) -> AbstractToolset[Any]:
        wrapped = await self.wrapped.for_run_step(ctx)
        if wrapped is self.wrapped:
            return self
        updated = replace(self, wrapped=wrapped)
        updated._session = self._session
        return updated

    async def __aenter__(self) -> Self:
        await self.wrapped.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        try:
            return await self.wrapped.__aexit__(*args)
        finally:
            await self._session.close()

    async def get_instructions(
        self,
        ctx: RunContext[Any],
    ) -> str | InstructionPart | Sequence[str | InstructionPart] | None:
        upstream = await self.wrapped.get_instructions(ctx)
        if upstream is None:
            return CODEMODE_INSTRUCTIONS
        if isinstance(upstream, (str, InstructionPart)):
            return [upstream, CODEMODE_INSTRUCTIONS]
        return [*upstream, CODEMODE_INSTRUCTIONS]

    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
        wrapped_tools = await self.wrapped.get_tools(ctx)
        conflicts = sorted(set(wrapped_tools) & {"search_tools", "get_tool_schema", "execute_code"})
        if conflicts:
            rendered = ", ".join(repr(name) for name in conflicts)
            raise UserError(f"CodeMode reserves these tool names: {rendered}")

        native_tools: dict[str, ToolsetTool[Any]] = {}
        catalog_tools: list[CodeModeTool] = []
        for name, tool in wrapped_tools.items():
            definition = tool.tool_def
            if definition.tool_kind is not None:
                native_tools[name] = tool
                continue
            catalog_tools.append(
                CodeModeTool(
                    name=name,
                    description=definition.description,
                    parameters_json_schema=definition.parameters_json_schema,
                    return_schema=definition.return_schema,
                    metadata=definition.metadata,
                    sequential=definition.sequential,
                )
            )

        try:
            self._session.update_tools(catalog_tools)
        except CodeModeCatalogError as exc:
            raise UserError(str(exc)) from exc
        result_admitter = _shared_result_admitter(wrapped_tools)

        result = dict(native_tools)
        result["search_tools"] = self._control_tool(
            kind="search",
            description=(
                "Browse the CodeMode catalog. Returns only tool names and an opaque cursor; "
                "call get_tool_schema before using a selected tool."
            ),
            schema=_SEARCH_SCHEMA,
            validator=_SEARCH_VALIDATOR,
            wrapped_tools=wrapped_tools,
            result_admitter=result_admitter,
        )
        result["get_tool_schema"] = self._control_tool(
            kind="schema",
            description=(
                "Return exact schemas for selected tool names and make those tools callable "
                "inside execute_code."
            ),
            schema=_SCHEMA_SCHEMA,
            validator=_SCHEMA_VALIDATOR,
            wrapped_tools=wrapped_tools,
            result_admitter=result_admitter,
        )
        result["execute_code"] = self._control_tool(
            kind="execute",
            description=(
                "Execute one bounded Monty-compatible notebook cell using only tools previously "
                "hydrated with get_tool_schema. Successful variables persist across calls; "
                "continue from them instead of repeating successful application tool calls. "
                "Failures return captured output and the error in separate fields."
            ),
            schema=_EXECUTE_SCHEMA,
            validator=_EXECUTE_VALIDATOR,
            wrapped_tools=wrapped_tools,
            result_admitter=result_admitter,
        )
        return result

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[Any],
        tool: ToolsetTool[Any],
    ) -> Any:
        if not isinstance(tool, _CodeModeControlTool):
            return await self.wrapped.call_tool(name, tool_args, ctx, tool)

        try:
            if tool.control_kind == "search":
                return self._session.search_tools(
                    query=tool_args.get("query"),
                    limit=tool_args.get("limit"),
                    cursor=tool_args.get("cursor"),
                )
            if tool.control_kind == "schema":
                return self._session.get_tool_schema(tool_names=tool_args["tool_names"])
            return await self._execute_code(tool_args, ctx, tool)
        except CodeModeError as exc:
            raise ModelRetry(str(exc)) from exc

    def _control_tool(
        self,
        *,
        kind: _CONTROL_KIND,
        description: str,
        schema: dict[str, Any],
        validator: SchemaValidatorProt,
        wrapped_tools: dict[str, ToolsetTool[Any]],
        result_admitter: ToolResultAdmitter | None,
    ) -> _CodeModeControlTool:
        name = {
            "search": "search_tools",
            "schema": "get_tool_schema",
            "execute": "execute_code",
        }[kind]
        return _CodeModeControlTool(
            toolset=self,
            tool_def=ToolDefinition(
                name=name,
                description=description,
                parameters_json_schema=schema,
                metadata={
                    "kedi_codemode_control": True,
                    **(
                        {"code_arg_name": "code", "code_arg_language": "python"}
                        if kind == "execute"
                        else {}
                    ),
                },
                sequential=True,
            ),
            max_retries=3,
            args_validator=validator,
            control_kind=kind,
            wrapped_tools=wrapped_tools,
            result_admitter=result_admitter,
        )

    async def _execute_code(
        self,
        args: dict[str, Any],
        ctx: RunContext[Any],
        tool: _CodeModeControlTool,
    ) -> ToolReturn:
        tool_manager = build_nested_tool_manager(
            ctx=ctx,
            toolset=self.wrapped,
            tools=tool.wrapped_tools,
        )
        nested_calls: dict[str, ToolCallPart] = {}
        nested_returns: dict[str, ToolReturnPart] = {}
        call_counter = 0

        async def invoke(tool_name: str, arguments: dict[str, Any]) -> Any:
            nonlocal call_counter
            call_counter += 1
            tool_call_id = f"{ctx.tool_call_id or 'kedi_codemode'}__{call_counter}"
            call = ToolCallPart(
                tool_name=tool_name,
                args=arguments,
                tool_call_id=tool_call_id,
            )
            nested_calls[tool_call_id] = call
            try:
                with defer_artifact_admission():
                    result = await tool_manager.handle_call(call, wrap_validation_errors=False)
            except (ApprovalRequired, CallDeferred) as exc:
                raise RuntimeError(
                    f"Tool {tool_name!r} requires inline approval resolution in CodeMode"
                ) from exc
            if isinstance(result, ToolDenied):
                nested_returns[tool_call_id] = ToolReturnPart(
                    tool_name=tool_name,
                    content=result.message,
                    tool_call_id=tool_call_id,
                    outcome="denied",
                )
                raise RuntimeError(f"Tool {tool_name!r} call denied: {result.message}")

            metadata = None
            if isinstance(result, ToolReturn):
                metadata = result.metadata
                result = result.return_value
            nested_returns[tool_call_id] = ToolReturnPart(
                tool_name=tool_name,
                content=result,
                tool_call_id=tool_call_id,
                metadata=metadata,
            )
            return result

        result = await self._session.execute_code(
            code=args["code"],
            restart=args.get("restart", False),
            invoke=invoke,
        )
        if tool.result_admitter is not None:
            result = await tool.result_admitter(result, "execute_code")
        return ToolReturn(
            return_value=result,
            metadata={
                "code_mode": True,
                "tool_calls": nested_calls,
                "tool_returns": nested_returns,
            },
        )


@dataclass
class PydanticCodeModeCapability(AbstractCapability[Any]):
    """Pydantic projection of Kedi's adapter-neutral CodeMode core."""

    limits: CodeModeLimits = field(default_factory=CodeModeLimits)

    def get_wrapper_toolset(
        self,
        toolset: AbstractToolset[Any],
    ) -> AbstractToolset[Any] | None:
        return PydanticCodeModeToolset(wrapped=toolset, limits=self.limits)

    async def before_model_request(
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        del ctx
        if any(
            native_tool.kind == "mcp_server"
            for native_tool in request_context.model_request_parameters.native_tools
        ):
            raise UserError(
                "Pydantic CodeMode requires MCP servers to use their local toolset path; "
                "provider-native MCP would bypass execute_code"
            )
        return request_context


def _shared_result_admitter(
    tools: dict[str, ToolsetTool[Any]],
) -> ToolResultAdmitter | None:
    admitters: dict[object, ToolResultAdmitter] = {}

    for name, projected_tool in tools.items():

        def collect(toolset: AbstractToolset[Any]) -> None:
            if not isinstance(toolset, FunctionToolset):
                return
            native_tool = toolset.tools.get(name)
            spec = getattr(native_tool, "kedi_spec", None)
            if isinstance(spec, ToolSpec) and spec.result_admitter is not None:
                boundary_key = getattr(
                    spec.result_admitter,
                    "_kedi_artifact_boundary_key",
                    id(spec.result_admitter),
                )
                admitters[boundary_key] = spec.result_admitter

        projected_tool.toolset.apply(collect)

    if len(admitters) > 1:
        raise UserError("CodeMode tools belong to multiple artifact result boundaries")
    return next(iter(admitters.values()), None)


def pydantic_code_mode_capability(
    *,
    limits: CodeModeLimits | None = None,
) -> PydanticCodeModeCapability:
    return PydanticCodeModeCapability(limits=limits or CodeModeLimits())


__all__ = [
    "PydanticCodeModeCapability",
    "PydanticCodeModeToolset",
    "pydantic_code_mode_capability",
]
