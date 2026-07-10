from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Iterator, Sequence, cast

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.models import Model
from pydantic_ai.output import OutputDataT

from ...agent_profile import AgentProfile
from ...agent_profile.settings import PYDANTIC_MODEL_SETTING_KEYS, pydantic_model_settings
from ...dynamic_models import build_dynamic_model
from ...effort import pydantic_effort_value
from ...model_normalization import normalize_for_pydantic_ai
from ...resolved_types import ResolvedKediType
from .._schema import output_field_definition
from ..meta import AdapterCapabilities, AgentAdapterKind
from ..tool_override import ToolScope, ToolSpec

if TYPE_CHECKING:
    from pydantic_ai.tools import Tool as PydanticTool


def _mcp_toolset_from_config(name: str, server_config: dict[str, object]) -> Any:
    from pydantic_ai.mcp import MCPToolset

    config = {"mcpServers": {name: server_config}}
    return MCPToolset(cast(Any, config))


class PydanticAdapter(Agent[AgentDepsT, OutputDataT]):
    """
    Adapter to use Pydantic AI agents.
    """

    kind: AgentAdapterKind = "agent-framework"
    shortname = "pydantic"
    capabilities = AdapterCapabilities(
        supports_structured_output=True,
        supports_tool_registration=True,
        supports_mcp=True,
        supports_profile_override=True,
        supports_model_override=True,
        supports_effort=True,
        supports_settings=True,
        settings=PYDANTIC_MODEL_SETTING_KEYS,
    )

    def __init__(
        self,
        model: str | Model | None = None,
        *,
        model_settings: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """``model`` may be LiteLLM-style (``vendor/a/b``) or Pydantic-style (``vendor:a/b``)."""
        self._tool_scope = ToolScope()
        self._profile_stack_var: ContextVar[tuple[AgentProfile, ...]] = ContextVar(
            f"kedi_pydantic_profile_stack_{id(self)}",
            default=(),
        )
        if isinstance(model, str):
            model = normalize_for_pydantic_ai(model)
        self._agent_model_arg = model
        self._default_model = model if isinstance(model, str) else None
        if model_settings:
            filtered_settings = pydantic_model_settings(model_settings)
            if filtered_settings:
                kwargs["model_settings"] = filtered_settings
        self._agent_init_kwargs = dict(kwargs)
        super().__init__(model, **kwargs)

    def tool_scope(self, tools: Sequence[Any] = ()) -> AbstractContextManager[None]:
        return self._tool_scope.tool_scope(tools)

    def register_tool(self, tool: ToolSpec) -> None:
        from pydantic_ai.tools import Tool

        if tool.json_schema is None:
            native_tool = Tool(tool.fn, name=tool.name)
        else:
            native_tool = Tool.from_schema(
                function=tool.fn,
                name=tool.name,
                description=tool.description or "",
                json_schema=tool.json_schema,
                takes_ctx=False,
            )
        native_tool.metadata = tool.metadata
        self._tool_scope.register_tool(native_tool)

    @property
    def scoped_tools(self) -> tuple[PydanticTool[Any], ...]:
        return cast("tuple[PydanticTool[Any], ...]", self._tool_scope.scoped_tools)

    @property
    def active_profile(self) -> AgentProfile:
        stack_var = getattr(self, "_profile_stack_var", None)
        stack = stack_var.get() if stack_var is not None else ()
        return stack[-1] if stack else AgentProfile.empty()

    @staticmethod
    def type_builder(output_schema: dict[str, ResolvedKediType]) -> type[OutputDataT]:
        """
        Type builder to create a Pydantic model from a given output schema.
        Args:
            output_schema (dict): A dictionary defining the output schema.
        Returns:
            type: A Pydantic model class based on the provided schema.
        """
        if not output_schema:
            raise ValueError("Output schema cannot be empty.")
        field_definitions = {
            field: output_field_definition(type_) for field, type_ in output_schema.items()
        }
        model = build_dynamic_model("_OutputModel", BaseModel, field_definitions)
        return cast(type[OutputDataT], model)

    async def produce(
        self,
        template: str,
        output_schema: dict[str, ResolvedKediType] | None = None,
        output_type: type[OutputDataT] | None = None,
        **kwargs,
    ) -> OutputDataT:
        """
        Invoke the Pydantic AI agent with the provided template and output schema.
        """
        if not output_schema and output_type is None:
            raise ValueError("Output spec must be provided.")
        if output_schema:
            output_type = self.type_builder(output_schema)
        run_kwargs: dict[str, object] = {"output_type": output_type, **kwargs}
        if self.active_profile.system is not None and "instructions" not in run_kwargs:
            run_kwargs["instructions"] = self.active_profile.system
        model_settings = pydantic_model_settings(self.active_profile.settings)
        if model_settings:
            existing_settings = run_kwargs.get("model_settings")
            if isinstance(existing_settings, dict):
                run_kwargs["model_settings"] = {**existing_settings, **model_settings}
            else:
                run_kwargs["model_settings"] = model_settings
        from ..utils import inside_kedi_tool_call

        toolsets = self._toolsets_for_run(include_scoped_tools=not inside_kedi_tool_call())
        if toolsets is not None:
            run_kwargs["toolsets"] = toolsets
        agent = self._agent_for_produce()
        if agent is not self:
            run_kwargs.update(self._profile_override_kwargs(self.active_profile))
        return (await agent.run(template, **cast(Any, run_kwargs))).output

    def _agent_for_produce(self) -> Agent[Any, Any]:
        from ..utils import inside_kedi_tool_call

        if not inside_kedi_tool_call():
            return self
        return Agent(self._agent_model_arg, **self._agent_init_kwargs)

    def _toolsets_for_run(self, *, include_scoped_tools: bool = True) -> list[Any] | None:
        toolsets: list[Any] = []
        scoped_tools = self.scoped_tools if include_scoped_tools else ()
        if scoped_tools:
            from pydantic_ai.toolsets import FunctionToolset

            toolsets.append(FunctionToolset(tools=scoped_tools))
        toolsets.extend(self._mcp_toolsets_for_profile(self.active_profile))
        return toolsets or None

    def _mcp_toolsets_for_profile(self, profile: AgentProfile) -> list[Any]:
        if not profile.mcp_servers:
            return []
        from pydantic_ai.mcp import MCPToolset

        toolsets: list[Any] = []
        for index, spec in enumerate(profile.mcp_servers):
            if spec.transport == "stdio":
                if spec.command is None:
                    raise ValueError("MCP stdio transport requires a command.")
                stdio_config: dict[str, object] = {
                    "command": spec.command,
                    "args": list(spec.args),
                }
                if spec.env is not None:
                    stdio_config["env"] = spec.env
                toolsets.append(_mcp_toolset_from_config(f"kedi_mcp_{index}", stdio_config))
            elif spec.transport == "sse":
                if spec.url is None:
                    raise ValueError("MCP sse transport requires a url.")
                sse_config: dict[str, object] = {"url": spec.url, "transport": "sse"}
                if spec.headers is not None:
                    sse_config["headers"] = spec.headers
                toolsets.append(_mcp_toolset_from_config(f"kedi_mcp_{index}", sse_config))
            elif spec.transport == "http":
                if spec.url is None:
                    raise ValueError("MCP http transport requires a url.")
                toolsets.append(MCPToolset(spec.url, headers=spec.headers))
            else:
                raise ValueError(f"Unsupported MCP transport: {spec.transport}")
        return toolsets

    def produce_sync(
        self,
        template: str,
        output_schema: dict[str, ResolvedKediType] | None = None,
        output_type: type[OutputDataT] | None = None,
        **kwargs,
    ) -> OutputDataT:
        """
        Synchronous version of produce method.
        """
        from ..utils import run_sync

        return run_sync(
            self.produce(template, output_schema=output_schema, output_type=output_type, **kwargs)
        )

    def default_model(self) -> str | None:
        default_model = getattr(self, "_default_model", None)
        if isinstance(default_model, str):
            return default_model
        model = getattr(self, "_model", None)
        return model if isinstance(model, str) else None

    @contextmanager
    def profile_override(self, profile: AgentProfile) -> Iterator[None]:
        override_kwargs = self._profile_override_kwargs(profile)
        stack_var = self._profile_stack_var
        token = stack_var.set((*stack_var.get(), profile))
        try:
            if not override_kwargs:
                yield
                return
            with self.override(**override_kwargs):
                yield
        finally:
            stack_var.reset(token)

    def _profile_override_kwargs(self, profile: AgentProfile) -> dict[str, Any]:
        override_kwargs: dict[str, Any] = {}
        if profile.needs_model_override(self.default_model()):
            if profile.model is None:
                raise RuntimeError("Profile requires a model override but no model is set.")
            override_kwargs["model"] = normalize_for_pydantic_ai(profile.model)
        if profile.effort is not None:
            override_kwargs["spec"] = {
                "capabilities": [
                    {
                        "name": "Thinking",
                        "arguments": {"effort": pydantic_effort_value(profile.effort)},
                    }
                ]
            }
        return override_kwargs
