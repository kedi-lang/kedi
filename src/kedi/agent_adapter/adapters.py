from __future__ import annotations

import asyncio
import inspect
import textwrap
from contextlib import AbstractContextManager, AsyncExitStack, contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Iterator, Mapping, Sequence, cast

import dspy
from dspy import LM
from pydantic import BaseModel
from pydantic_ai import Agent, ToolDefinition
from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.models import Model
from pydantic_ai.output import OutputDataT

from ..agent_profile import AgentProfile
from ..agent_profile.settings import (
    code_mode_enabled,
    dspy_lm_settings,
    langchain_model_settings,
    pydantic_model_settings,
)
from ..dynamic_models import build_dynamic_model
from ..effort import pydantic_effort_value
from ..model_normalization import (
    normalize_for_langchain,
    normalize_for_litellm,
    normalize_for_pydantic_ai,
)
from ..resolved_types import ResolvedKediType
from .tool_override import ToolScope, ToolSpec
from .utils import (
    dspy_code_act,
    monty_compat_prompt_context,
    sanitize_python_identifier,
    unique_python_identifier,
)

if TYPE_CHECKING:
    from pydantic_ai.tools import Tool as PydanticTool


def make_repr(cls: type) -> str:
    _type_repr = repr(cls)
    return _type_repr if not _type_repr.startswith("<class") else cls.__name__


def _mcp_toolset_from_config(name: str, server_config: dict[str, object]) -> Any:
    from pydantic_ai.mcp import MCPToolset

    config = {"mcpServers": {name: server_config}}
    return MCPToolset(cast(Any, config))


def _code_mode_capability() -> Any:
    from pydantic_ai_harness.code_mode import CodeMode

    return CodeMode()


def _with_code_mode_capability(capabilities: object = ()) -> list[Any]:
    existing = () if capabilities is None else cast("Sequence[Any]", capabilities)
    return [*existing, _code_mode_capability()]


class PydanticAdapter(Agent[AgentDepsT, OutputDataT]):
    """
    Adapter to use Pydantic AI agents.
    """

    def __init__(
        self,
        model: str | Model | None = None,
        *,
        model_settings: dict[str, Any] | None = None,
        codemode: bool = False,
        **kwargs: Any,
    ) -> None:
        """``model`` may be LiteLLM-style (``vendor/a/b``) or Pydantic-style (``vendor:a/b``)."""
        self._tool_scope = ToolScope()
        self._profile_stack_var: ContextVar[tuple[AgentProfile, ...]] = ContextVar(
            f"kedi_pydantic_profile_stack_{id(self)}",
            default=(),
        )
        if isinstance(model, str):
            model = normalize_for_pydantic_ai(model) if model is not None else None
        if model_settings:
            filtered_settings = pydantic_model_settings(model_settings)
            if filtered_settings:
                kwargs["model_settings"] = filtered_settings
        if codemode:
            kwargs["capabilities"] = _with_code_mode_capability(kwargs.get("capabilities"))
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
        field_definitions = {field: (type_, ...) for field, type_ in output_schema.items()}
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
        model_settings = pydantic_model_settings(self.active_profile.settings)
        if model_settings:
            existing_settings = run_kwargs.get("model_settings")
            if isinstance(existing_settings, dict):
                run_kwargs["model_settings"] = {**existing_settings, **model_settings}
            else:
                run_kwargs["model_settings"] = model_settings
        toolsets = self._toolsets_for_run()
        if toolsets is not None:
            run_kwargs["toolsets"] = toolsets
        if code_mode_enabled(self.active_profile.settings):
            run_kwargs["capabilities"] = _with_code_mode_capability(run_kwargs.get("capabilities"))
        return (await self.run(template, **cast(Any, run_kwargs))).output

    def _toolsets_for_run(self) -> list[Any] | None:
        toolsets: list[Any] = []
        scoped_tools = self.scoped_tools
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
        from .utils import run_sync

        return run_sync(
            self.produce(template, output_schema=output_schema, output_type=output_type, **kwargs)
        )

    def default_model(self) -> str | None:
        model = self.model
        return model if isinstance(model, str) else None

    @contextmanager
    def profile_override(self, profile: AgentProfile) -> Iterator[None]:
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


class LangChainAdapter:
    """Adapter to use LangChain agents through Kedi's uniform adapter contract."""

    def __init__(
        self,
        model: Any = None,
        *,
        model_settings: dict[str, Any] | None = None,
        codemode: bool = False,
        **kwargs: Any,
    ) -> None:
        self._model = normalize_for_langchain(model) if isinstance(model, str) else model
        self._model_settings = langchain_model_settings(model_settings or {})
        self._model_extra_kwargs = dict(kwargs)
        self._codemode = codemode
        self._tool_scope = ToolScope()
        self._code_mode_runner = _LangChainCodeModeRunner()
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
        field_definitions = {field: (type_, ...) for field, type_ in output_schema.items()}
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
        raw_tools = await self._raw_tools_for_run()
        if self._code_mode_active():
            tool_defs = _langchain_code_mode_tool_definitions(raw_tools)
            instructions = _langchain_code_mode_system_prompt(
                base=instructions,
                tool_defs=tool_defs,
            )
            tools = _langchain_code_mode_tools(raw_tools, self._code_mode_runner)
        else:
            tools = raw_tools
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
        tools = await self._raw_tools_for_run()
        if self._code_mode_active():
            return _langchain_code_mode_tools(tools, self._code_mode_runner)
        return tools

    async def _raw_tools_for_run(self) -> list[Any]:
        return [*self.scoped_tools, *await self._mcp_tools_for_profile(self.active_profile)]

    def _code_mode_active(self) -> bool:
        return self._codemode or code_mode_enabled(self.active_profile.settings)

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


_RUN_CODE_ARGS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "code": {
            "type": "string",
            "description": "The Python code to execute in the sandbox.",
        },
        "restart": {
            "type": "boolean",
            "default": False,
            "description": (
                "Set to true to reset sandbox state. When false, definitions from "
                "previous calls are preserved."
            ),
        },
    },
    "required": ["code"],
}

_RUN_CODE_DESCRIPTION = """\
Execute Python code in a sandbox.

The last expression's value is returned automatically. Use print() only for supplementary
output. State is preserved between calls; pass restart=True to clear it.
"""

_SEARCH_SCHEMA_ARGS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Optional substring to match against available code-mode function names.",
        },
        "names": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional exact function names to inspect.",
        },
    },
}

_SEARCH_SCHEMA_DESCRIPTION = "Look up schemas for functions callable from run_code."


class _LangChainCodeModeRunner:
    def __init__(self) -> None:
        self._prelude: list[str] = []

    def reset(self) -> None:
        self._prelude.clear()

    def run(
        self,
        *,
        code: str,
        functions: Mapping[str, Callable[..., Any]],
        restart: bool = False,
    ) -> Any:
        if restart:
            self.reset()
        code = textwrap.dedent(code).strip()
        if not code:
            return ""
        definition_name = _single_python_definition_name(code)
        if definition_name is not None and definition_name not in functions:
            self._prelude.append(code)
            return ""

        Monty = _monty_class()
        stdout: list[str] = []

        def print_callback(stream: str, value: str) -> None:
            if stream == "stdout":
                stdout.append(value)

        sandbox = Monty(_join_python_blocks([*self._prelude, code]), inputs=list(functions))
        result = sandbox.run(
            inputs=dict(functions),
            external_functions=dict(functions),
            print_callback=print_callback,
        )
        captured = "".join(stdout).rstrip("\n")
        if captured and result is not None:
            return {"output": captured, "result": result}
        if captured:
            return captured
        return "" if result is None else result


def _langchain_code_mode_tools(
    tools: Sequence[Any],
    runner: _LangChainCodeModeRunner,
) -> list[Any]:
    try:
        from langchain_core.tools import StructuredTool
    except ImportError as exc:
        raise ModuleNotFoundError(
            "LangChainAdapter code mode requires langchain-core. Install with: uv add langchain"
        ) from exc

    functions = _langchain_code_mode_functions(tools)
    tool_defs = _langchain_code_mode_tool_definitions(tools)

    def search_schema(query: str | None = None, names: list[str] | None = None) -> dict[str, Any]:
        return _langchain_search_schema(tool_defs, query=query, names=names)

    def run_code(code: str, restart: bool = False) -> Any:
        try:
            return runner.run(code=code, functions=functions, restart=restart)
        except Exception as exc:
            return {
                "error": f"{type(exc).__name__}: {exc}",
                "hint": (
                    "Revise the code and call run_code again. Inspect tool schemas with "
                    "search_schema, use keyword arguments from the schema, and avoid assuming "
                    "third-party packages are installed in the sandbox."
                ),
            }

    search_tool = StructuredTool.from_function(
        func=search_schema,
        name="search_schema",
        description=_SEARCH_SCHEMA_DESCRIPTION,
        args_schema=_SEARCH_SCHEMA_ARGS_SCHEMA,
    )
    run_tool = StructuredTool.from_function(
        func=run_code,
        name="run_code",
        description=_RUN_CODE_DESCRIPTION,
        args_schema=_RUN_CODE_ARGS_SCHEMA,
    )
    run_tool.metadata = {"code_arg_name": "code", "code_arg_language": "python"}
    return [search_tool, run_tool]


def _langchain_code_mode_functions(tools: Sequence[Any]) -> dict[str, Callable[..., Any]]:
    functions: dict[str, Callable[..., Any]] = {}
    for function_name, tool in _langchain_code_mode_named_tools(tools):
        parameter_names = _langchain_tool_parameter_names(tool)
        functions[function_name] = _langchain_tool_callable(tool, function_name, parameter_names)
    return functions


def _langchain_code_mode_tool_definitions(tools: Sequence[Any]) -> dict[str, ToolDefinition]:
    tool_defs: dict[str, ToolDefinition] = {}
    for function_name, tool in _langchain_code_mode_named_tools(tools):
        tool_defs[function_name] = ToolDefinition(
            name=function_name,
            description=_langchain_tool_description(tool),
            parameters_json_schema=_langchain_tool_parameters_schema(tool, function_name),
            sequential=True,
        )
    return tool_defs


def _langchain_code_mode_named_tools(tools: Sequence[Any]) -> list[tuple[str, Any]]:
    named_tools: list[tuple[str, Any]] = []
    used_names: set[str] = set()
    for tool in tools:
        name = _langchain_tool_name(tool)
        if name is None:
            continue
        named_tools.append((unique_python_identifier(name, used_names), tool))
    return named_tools


def _langchain_tool_name(tool: Any) -> str | None:
    name = getattr(tool, "name", None)
    if isinstance(name, str) and name:
        return name
    if callable(tool):
        callable_name = getattr(tool, "__name__", None)
        return callable_name if isinstance(callable_name, str) and callable_name else None
    return None


def _langchain_tool_callable(
    tool: Any,
    name: str,
    parameter_names: Sequence[str] = (),
) -> Callable[..., Any]:
    def _call(*args: Any, **kwargs: Any) -> Any:
        kwargs = _merge_positional_tool_args(name, parameter_names, args, kwargs)
        if hasattr(tool, "invoke"):
            result = tool.invoke(kwargs)
        elif callable(tool):
            result = tool(**kwargs)
        else:
            raise TypeError(f"LangChain code mode tool is not callable: {tool!r}")
        if inspect.isawaitable(result):
            from .utils import run_sync

            return run_sync(cast(Coroutine[Any, Any, Any], result))
        return result

    _call.__name__ = name
    _call.__qualname__ = name
    return _call


def _merge_positional_tool_args(
    tool_name: str,
    parameter_names: Sequence[str],
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(kwargs)
    if not args:
        return merged
    if len(args) > len(parameter_names):
        raise TypeError(
            f"{tool_name}() got {len(args)} positional arguments but only "
            f"{len(parameter_names)} schema parameters are available"
        )
    for index, value in enumerate(args):
        parameter_name = parameter_names[index]
        if parameter_name in merged:
            raise TypeError(f"{tool_name}() got multiple values for argument '{parameter_name}'")
        merged[parameter_name] = value
    return merged


def _langchain_tool_description(tool: Any) -> str | None:
    description = getattr(tool, "description", None)
    return description if isinstance(description, str) and description else None


def _langchain_tool_parameters_schema(tool: Any, name: str) -> dict[str, Any]:
    args_schema = getattr(tool, "args_schema", None)
    if isinstance(args_schema, dict):
        schema = dict(args_schema)
    elif hasattr(args_schema, "model_json_schema"):
        schema = cast(Any, args_schema).model_json_schema()
    else:
        args = getattr(tool, "args", None)
        schema = {"type": "object", "properties": dict(args)} if isinstance(args, Mapping) else {}

    if schema.get("type") != "object":
        schema["type"] = "object"
    schema.setdefault("title", name)
    schema.setdefault("properties", {})
    return schema


def _langchain_tool_parameter_names(tool: Any) -> tuple[str, ...]:
    properties = _langchain_tool_parameters_schema(tool, _langchain_tool_name(tool) or "_").get(
        "properties"
    )
    if not isinstance(properties, Mapping):
        return ()
    return tuple(str(name) for name in properties)


def _langchain_search_schema(
    tool_defs: Mapping[str, ToolDefinition],
    *,
    query: str | None = None,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    selected = _langchain_select_tool_defs(tool_defs, query=query, names=names)
    return {
        "available_tools": sorted(tool_defs),
        "matches": [
            {
                "name": name,
                "description": tool_def.description,
                "signature": tool_def.render_signature("...", is_async=False),
                "parameters_json_schema": tool_def.parameters_json_schema,
            }
            for name, tool_def in selected.items()
        ],
    }


def _langchain_select_tool_defs(
    tool_defs: Mapping[str, ToolDefinition],
    *,
    query: str | None = None,
    names: Sequence[str] | None = None,
) -> dict[str, ToolDefinition]:
    if names:
        wanted = set(names)
        return {name: tool_def for name, tool_def in tool_defs.items() if name in wanted}
    if query:
        lowered = query.lower()
        return {
            name: tool_def
            for name, tool_def in tool_defs.items()
            if lowered in name.lower()
            or (tool_def.description is not None and lowered in tool_def.description.lower())
        }
    return dict(tool_defs)


def _langchain_code_mode_system_prompt(
    *,
    base: str | None,
    tool_defs: Mapping[str, ToolDefinition],
    monty_context: str | None = None,
) -> str:
    names = ", ".join(sorted(tool_defs)) if tool_defs else "none"
    monty_context = monty_compat_prompt_context() if monty_context is None else monty_context
    code_mode_prompt = (
        "LangChain code mode is active. The only exposed tools are `search_schema` and "
        "`run_code`. Do not ask for or rely on native tool calls for the underlying "
        "application tools.\n"
        f"Available run_code function names: {names}.\n"
        "Use `search_schema` when you need a function's arguments, JSON schema, or exact "
        "signature. After inspecting the schema, call `run_code` with Python that invokes "
        "the selected function directly by name. Prefer keyword arguments exactly as shown "
        "in the schema, e.g. `tool_name(arg_name=value)`, and do not guess argument names "
        "for tools whose schema you have not inspected. Keep code small and task-focused; "
        "do not assume third-party packages are installed in the sandbox; prefer available "
        "tools and Python standard-library parsing. Code passed to `run_code` must be "
        "Monty-compatible; follow the Monty capability context below before choosing "
        "imports, builtins, exceptions, or language constructs. Use `restart=True` only "
        "when previous sandbox state could interfere."
    )
    if monty_context:
        code_mode_prompt = f"{code_mode_prompt}\n\n{monty_context}"
    if base:
        return f"{base}\n\n{code_mode_prompt}"
    return code_mode_prompt


def _sanitize_python_identifier(name: str) -> str:
    return sanitize_python_identifier(name)


def _monty_class() -> type[Any]:
    try:
        from pydantic_monty import Monty
    except ImportError as exc:
        raise RuntimeError(
            "LangChainAdapter code mode requires pydantic-monty. Install it with "
            '`uv add "pydantic-ai-harness[code-mode]>=0.4.0"`.'
        ) from exc
    return cast(type[Any], Monty)


def _single_python_definition_name(code: str) -> str | None:
    import ast

    try:
        module = ast.parse(code)
    except SyntaxError:
        return None
    if len(module.body) != 1:
        return None
    node = module.body[0]
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    return None


def _join_python_blocks(blocks: Sequence[str]) -> str:
    return "\n\n".join(block for block in blocks if block.strip())


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


class DSPyAdapter:
    # Class-level defaults so the LM config is readable via native attribute access
    # even on adapters built without __init__ (e.g. test doubles).
    _lm_provider: dspy.Provider | None = None
    _lm_max_tokens: int = 4000
    _lm_temperature: float = 1.0

    def __init__(
        self,
        model: str,
        max_tokens: int = 4000,
        temperature: float = 1.0,
        provider: dspy.Provider | None = None,
        model_settings: dict[str, Any] | None = None,
        codemode: bool = False,
        **kwargs,
    ):
        model = normalize_for_litellm(model)
        self._lm_model = model
        self._lm_provider = provider
        self._lm_max_tokens = max_tokens
        self._lm_temperature = temperature
        self._lm_settings = dspy_lm_settings(model_settings or {})
        self._lm_extra_kwargs = dict(kwargs)
        self._codemode = codemode
        self._tool_scope = ToolScope()
        self._profile_stack_var: ContextVar[tuple[AgentProfile, ...]] = ContextVar(
            f"kedi_dspy_profile_stack_{id(self)}",
            default=(),
        )
        # Keep a one-time baseline configure so plain (no-profile) produce has a
        # global LM; per-call correctness comes from the ``dspy.context`` wrap in
        # ``produce_sync`` / ``profile_override`` (innermost context wins).
        dspy.settings.configure(lm=self._default_lm())

    def _default_lm(self) -> LM:
        provider, kwargs = self._lm_kwargs()
        return LM(model=self._lm_model, provider=provider, **kwargs)

    def _lm_kwargs(
        self,
        profile_settings: dict[str, Any] | None = None,
        *,
        effort: str | None = None,
    ) -> tuple[dspy.Provider | None, dict[str, Any]]:
        kwargs = dict(getattr(self, "_lm_extra_kwargs", {}))
        kwargs.update(getattr(self, "_lm_settings", {}))
        kwargs.update(dspy_lm_settings(profile_settings or {}))
        kwargs.setdefault("max_tokens", getattr(self, "_lm_max_tokens", 4000))
        kwargs.setdefault("temperature", getattr(self, "_lm_temperature", 1.0))
        kwargs.setdefault("cache", False)
        if effort is not None:
            kwargs["reasoning_effort"] = effort
        provider = cast(dspy.Provider | None, kwargs.pop("provider", self._lm_provider))
        return provider, kwargs

    def tool_scope(self, tools: Sequence[Any] = ()) -> AbstractContextManager[None]:
        return self._tool_scope.tool_scope(tools)

    def register_tool(self, tool: ToolSpec) -> None:
        from .utils import dspy_sync_tool_function, dspy_tool_arg_desc, dspy_tool_args

        kwargs: dict[str, Any] = {
            "func": dspy_sync_tool_function(tool.fn),
            "name": tool.name,
            "desc": tool.description,
        }
        if tool.json_schema is not None:
            kwargs["args"] = dspy_tool_args(tool.json_schema)
            arg_desc = dspy_tool_arg_desc(tool.json_schema)
            if arg_desc:
                kwargs["arg_desc"] = arg_desc
        self._tool_scope.register_tool(dspy.Tool(**kwargs))

    @property
    def scoped_tools(self) -> tuple[Any, ...]:
        scope = getattr(self, "_tool_scope", None)
        return () if scope is None else scope.scoped_tools

    @property
    def active_profile(self) -> AgentProfile:
        stack_var = getattr(self, "_profile_stack_var", None)
        stack = stack_var.get() if stack_var is not None else ()
        return stack[-1] if stack else AgentProfile.empty()

    @staticmethod
    def type_builder(output_schema: dict[str, ResolvedKediType], **kwargs) -> type[dspy.Signature]:
        """
        Type builder to create a dictionary schema from a given output type.
        Args:
            output_schema (dict): A dictionary defining the output schema.
            string (bool): Whether to use string signatures.
        Returns:
            dict: A dictionary schema based on the provided output type.
        """
        if not output_schema:
            raise ValueError("Output schema must be provided.")
        field_definitions: dict = {
            field: (type_, dspy.OutputField()) for field, type_ in output_schema.items()
        }
        field_definitions.update({"user_prompt": (str, dspy.InputField())})
        return build_dynamic_model("_OutputSignature", dspy.Signature, field_definitions)

    async def produce(
        self,
        template: str,
        output_schema: dict[str, ResolvedKediType],
        **kwargs,
    ) -> dspy.Prediction:
        """
        Adapter to use DSPy agents.
        """
        return await asyncio.to_thread(
            self.produce_sync,
            template,
            output_schema,
            **kwargs,
        )

    def produce_sync(
        self,
        template: str,
        output_schema: dict[str, ResolvedKediType],
        **kwargs,
    ) -> dspy.Prediction:
        """
        Synchronous version of produce method.
        """
        if not output_schema:
            raise ValueError("Output spec must be provided.")
        from .utils import run_sync

        kwargs.pop("user_prompt", None)
        sig = self.type_builder(output_schema)
        instructions = kwargs.pop("instructions", None)
        sig = sig.with_instructions(instructions)
        tools = cast(list[Callable[..., Any]], list(self.scoped_tools))
        with self._produce_lm_context():
            if self.active_profile.mcp_servers:
                return run_sync(self._produce_with_mcp(sig, tools, template, kwargs))
            prediction = self._prediction_for_tools(sig, tools)
            return prediction(**kwargs, user_prompt=template)

    def _code_mode_active(self) -> bool:
        return bool(
            getattr(self, "_codemode", False) or code_mode_enabled(self.active_profile.settings)
        )

    def _prediction_for_tools(
        self,
        sig: type[dspy.Signature],
        tools: list[Callable[..., Any]],
    ) -> Any:
        if self._code_mode_active():
            return dspy_code_act(sig, tools)
        return dspy.ReAct(sig, tools=tools) if tools else dspy.Predict(sig)

    def _produce_lm_context(self) -> AbstractContextManager[Any]:
        """Make the active LM authoritative for one produce call.

        ``profile_override`` already wraps nested calls in ``dspy.context(lm=...)``;
        the innermost context wins, so we only apply the adapter default when no
        profile override is on the stack. The baseline ``dspy.settings.configure``
        from ``__init__`` remains the fallback for first-run safety.
        """
        stack_var = getattr(self, "_profile_stack_var", None)
        if stack_var is not None and stack_var.get():
            return nullcontext()
        if self._lm_model is None:
            return nullcontext()
        return dspy.context(lm=self._default_lm())

    async def _produce_with_mcp(
        self,
        sig: type[dspy.Signature],
        tools: list[Callable[..., Any]],
        template: str,
        kwargs: dict[str, Any],
    ) -> dspy.Prediction:
        mcp_tools: list[Any] = []
        async with AsyncExitStack() as stack:
            for spec in self.active_profile.mcp_servers:
                if spec.transport != "stdio":
                    raise NotImplementedError("DSPy MCP support currently requires stdio transport")
                from mcp import ClientSession, StdioServerParameters
                from mcp.client.stdio import stdio_client

                if spec.command is None:
                    raise ValueError("MCP stdio transport requires a command.")
                server_params = StdioServerParameters(
                    command=spec.command,
                    args=list(spec.args),
                    env=spec.env,
                )
                read, write = await stack.enter_async_context(stdio_client(server_params))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                listed_tools = await session.list_tools()
                mcp_tools.extend(
                    dspy.Tool.from_mcp_tool(session, tool) for tool in listed_tools.tools
                )
            prediction = self._prediction_for_tools(sig, [*tools, *mcp_tools])
            return await prediction.acall(**kwargs, user_prompt=template)

    def default_model(self) -> str | None:
        return self._lm_model

    @contextmanager
    def profile_override(self, profile: AgentProfile) -> Iterator[None]:
        needs_model_override = profile.needs_model_override(self.default_model())
        stack_var = self._profile_stack_var
        token = stack_var.set((*stack_var.get(), profile))
        try:
            if not needs_model_override and profile.effort is None and not profile.settings:
                yield
                return
            model = self._lm_model
            if needs_model_override:
                if profile.model is None:
                    raise RuntimeError("Profile requires a model override but no model is set.")
                model = normalize_for_litellm(profile.model)
            provider, lm_kwargs = self._lm_kwargs(dict(profile.settings), effort=profile.effort)
            lm = LM(
                model=model,
                provider=provider,
                **lm_kwargs,
            )
            with dspy.context(lm=lm):
                yield
        finally:
            stack_var.reset(token)
