from __future__ import annotations

import inspect
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable, Coroutine, Iterator, Mapping, Sequence, cast

from pydantic import BaseModel
from pydantic_ai.output import OutputDataT

from ..agent_profile import AgentProfile
from ..agent_profile.settings import LANGCHAIN_MODEL_SETTING_KEYS, langchain_model_settings
from ..dynamic_models import build_dynamic_model
from ..effort import pydantic_effort_value
from ..model_normalization import normalize_for_langchain
from ..resolved_types import ResolvedKediType
from ._schema import output_field_definition
from .meta import AdapterCapabilities, AgentAdapterKind
from .tool_override import ToolScope, ToolSpec


class LangChainAdapter:
    """Adapter to use LangChain agents through Kedi's uniform adapter contract."""

    kind: AgentAdapterKind = "agent-framework"
    shortname = "langchain"
    capabilities = AdapterCapabilities(
        supports_structured_output=True,
        supports_tool_registration=True,
        supports_mcp=True,
        supports_profile_override=True,
        supports_model_override=True,
        supports_effort=True,
        supports_settings=True,
        settings=LANGCHAIN_MODEL_SETTING_KEYS,
    )

    def __init__(
        self,
        model: Any = None,
        *,
        model_settings: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._model = normalize_for_langchain(model) if isinstance(model, str) else model
        self._model_settings = langchain_model_settings(model_settings or {})
        self._model_extra_kwargs = dict(kwargs)
        self._tool_scope = ToolScope()
        self._profile_stack_var: ContextVar[tuple[AgentProfile, ...]] = ContextVar(
            f"kedi_langchain_profile_stack_{id(self)}",
            default=(),
        )

    def tool_scope(self, tools: Sequence[Any] = ()) -> AbstractContextManager[None]:
        return self._tool_scope.tool_scope(tools)

    def register_tool(self, tool: ToolSpec) -> None:
        self._tool_scope.register_tool(_langchain_tool_from_spec(tool))

    @property
    def scoped_tools(self) -> tuple[Any, ...]:
        return self._tool_scope.scoped_tools

    @property
    def active_profile(self) -> AgentProfile:
        stack = self._profile_stack_var.get()
        return stack[-1] if stack else AgentProfile.empty()

    @staticmethod
    def type_builder(output_schema: dict[str, ResolvedKediType]) -> type[BaseModel]:
        if not output_schema:
            raise ValueError("Output schema cannot be empty.")
        field_definitions = {
            field: output_field_definition(type_) for field, type_ in output_schema.items()
        }
        return build_dynamic_model("_OutputModel", BaseModel, field_definitions)

    async def produce(
        self,
        template: str,
        output_schema: dict[str, ResolvedKediType] | None = None,
        output_type: type[OutputDataT] | None = None,
        **kwargs: Any,
    ) -> OutputDataT:
        resolved_output_type = output_type
        if output_schema:
            resolved_output_type = cast(type[OutputDataT], self.type_builder(output_schema))
        if resolved_output_type is None:
            raise ValueError("Output spec must be provided.")

        instructions = cast(str | None, kwargs.pop("instructions", None))
        model = self._chat_model_for_run()
        tools = await self._tools_for_run()
        agent = self._create_agent(
            model=model,
            tools=tools,
            response_format=resolved_output_type,
            system_prompt=instructions,
        )
        payload = {"messages": [{"role": "user", "content": template}], **kwargs}
        result = await agent.ainvoke(payload)
        return cast(OutputDataT, _langchain_structured_response(result))

    def produce_sync(
        self,
        template: str,
        output_schema: dict[str, ResolvedKediType] | None = None,
        output_type: type[OutputDataT] | None = None,
        **kwargs: Any,
    ) -> OutputDataT:
        from .utils import run_sync

        return run_sync(
            self.produce(template, output_schema=output_schema, output_type=output_type, **kwargs)
        )

    def default_model(self) -> str | None:
        model = self._model
        return model if isinstance(model, str) else None

    @contextmanager
    def profile_override(self, profile: AgentProfile) -> Iterator[None]:
        stack_var = self._profile_stack_var
        token = stack_var.set((*stack_var.get(), profile))
        try:
            yield
        finally:
            stack_var.reset(token)

    def _model_for_run(self) -> str | Any:
        profile = self.active_profile
        if profile.model is not None:
            return normalize_for_langchain(profile.model)
        if self._model is None:
            raise RuntimeError("LangChain adapter requires a model.")
        return self._model

    def _settings_for_run(self, model: str | Any) -> dict[str, Any]:
        settings = {
            **dict(self._model_extra_kwargs),
            **dict(self._model_settings),
            **langchain_model_settings(self.active_profile.settings),
        }
        _apply_langchain_effort_settings(settings, model, self.active_profile.effort)
        return settings

    def _chat_model_for_run(self) -> Any:
        model = self._model_for_run()
        settings = self._settings_for_run(model)
        if not isinstance(model, str):
            return model
        try:
            from langchain.chat_models import init_chat_model
        except ImportError as exc:
            raise ModuleNotFoundError(
                "LangChainAdapter requires langchain. Install with: uv add langchain"
            ) from exc
        return init_chat_model(model, **settings)

    def _create_agent(
        self,
        *,
        model: Any,
        tools: Sequence[Any],
        response_format: type[Any],
        system_prompt: str | None,
    ) -> Any:
        try:
            from langchain.agents import create_agent
        except ImportError as exc:
            raise ModuleNotFoundError(
                "LangChainAdapter requires langchain. Install with: uv add langchain"
            ) from exc
        return create_agent(
            model=model,
            tools=list(tools),
            response_format=response_format,
            system_prompt=system_prompt,
        )

    async def _tools_for_run(self) -> list[Any]:
        return [
            *self.scoped_tools,
            *await self._mcp_tools_for_profile(self.active_profile),
        ]

    async def _mcp_tools_for_profile(self, profile: AgentProfile) -> list[Any]:
        if not profile.mcp_servers:
            return []
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except ImportError as exc:
            raise ModuleNotFoundError(
                "LangChainAdapter MCP support requires langchain-mcp-adapters. "
                "Install with: uv add langchain-mcp-adapters"
            ) from exc
        client = MultiServerMCPClient(cast(Any, _langchain_mcp_server_configs(profile)))
        return list(await client.get_tools())


def _langchain_tool_from_spec(tool: ToolSpec) -> Any:
    try:
        from langchain_core.tools import StructuredTool
    except ImportError as exc:
        raise ModuleNotFoundError(
            "LangChainAdapter tool support requires langchain-core. Install with: uv add langchain"
        ) from exc

    description = tool.description or tool.name
    native_tool = StructuredTool.from_function(
        func=_langchain_sync_tool_function(tool.fn),
        name=tool.name,
        description=description,
        args_schema=tool.json_schema,
    )
    native_tool.metadata = tool.metadata
    return native_tool


def _langchain_sync_tool_function(fn: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(fn)
    def _tool(*args: Any, **kwargs: Any) -> Any:
        from .utils import run_sync

        result = fn(*args, **kwargs)
        if inspect.isawaitable(result):
            return run_sync(cast(Coroutine[Any, Any, Any], result))
        return result

    return _tool


def _langchain_mcp_server_configs(profile: AgentProfile) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for index, spec in enumerate(profile.mcp_servers):
        name = f"kedi_mcp_{index}"
        config: dict[str, Any]
        if spec.transport == "stdio":
            if spec.command is None:
                raise ValueError("MCP stdio transport requires a command.")
            config = {
                "transport": "stdio",
                "command": spec.command,
                "args": list(spec.args),
            }
            if spec.env is not None:
                config["env"] = spec.env
        elif spec.transport == "sse":
            if spec.url is None:
                raise ValueError("MCP sse transport requires a url.")
            config = {"transport": "sse", "url": spec.url}
            if spec.headers is not None:
                config["headers"] = spec.headers
        elif spec.transport == "http":
            if spec.url is None:
                raise ValueError("MCP http transport requires a url.")
            config = {"transport": "streamable_http", "url": spec.url}
            if spec.headers is not None:
                config["headers"] = spec.headers
        else:
            raise ValueError(f"Unsupported MCP transport: {spec.transport}")
        configs[name] = config
    return configs


def _apply_langchain_effort_settings(
    settings: dict[str, Any],
    model: str | Any,
    effort: str | None,
) -> None:
    if effort is None:
        return
    mapped = pydantic_effort_value(effort)
    if isinstance(model, str) and model.startswith("openrouter:"):
        existing = settings.get("reasoning")
        reasoning = dict(existing) if isinstance(existing, Mapping) else {}
        reasoning["effort"] = mapped
        settings["reasoning"] = reasoning
        return
    settings["reasoning_effort"] = mapped


def _langchain_structured_response(result: Any) -> Any:
    if isinstance(result, Mapping):
        if "structured_response" in result:
            return result["structured_response"]
        raise ValueError("LangChain agent did not return structured_response.")
    try:
        return result.structured_response
    except AttributeError:
        raise ValueError("LangChain agent did not return structured_response.") from None
