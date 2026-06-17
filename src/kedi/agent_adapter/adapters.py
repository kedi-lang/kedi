from __future__ import annotations

import asyncio
from contextlib import (
    AbstractContextManager,
    AsyncExitStack,
    contextmanager,
    nullcontext,
)
from typing import TYPE_CHECKING, Any, Callable, Iterator, Sequence, cast

import dspy
from dspy import LM
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.models import Model
from pydantic_ai.output import OutputDataT

from ..agent_profile import AgentProfile
from ..dynamic_models import build_dynamic_model
from ..effort import pydantic_effort_value
from ..model_normalization import normalize_for_litellm, normalize_for_pydantic_ai
from ..resolved_types import ResolvedKediType
from .tool_override import ToolScope, ToolSpec

if TYPE_CHECKING:
    from pydantic_ai.tools import Tool as PydanticTool


def make_repr(cls: type) -> str:
    _type_repr = repr(cls)
    return _type_repr if not _type_repr.startswith("<class") else cls.__name__


def _mcp_toolset_from_config(name: str, server_config: dict[str, object]) -> Any:
    from pydantic_ai.mcp import MCPToolset

    config = {"mcpServers": {name: server_config}}
    return MCPToolset(cast(Any, config))


class PydanticAdapter(Agent[AgentDepsT, OutputDataT]):
    """
    Adapter to use Pydantic AI agents.
    """

    def __init__(self, model: str | Model | None = None, **kwargs: Any) -> None:
        """``model`` may be LiteLLM-style (``vendor/a/b``) or Pydantic-style (``vendor:a/b``)."""
        self._tool_scope = ToolScope()
        self._profile_stack: list[AgentProfile] = []
        if isinstance(model, str):
            model = normalize_for_pydantic_ai(model) if model is not None else None
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
        stack = getattr(self, "_profile_stack", ())
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
        toolsets = self._toolsets_for_run()
        if toolsets is not None:
            run_kwargs["toolsets"] = toolsets
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
        stack = getattr(self, "_profile_stack", None)
        if stack is None:
            stack = []
            self._profile_stack = stack
        stack.append(profile)
        try:
            if not override_kwargs:
                yield
                return
            with self.override(**override_kwargs):
                yield
        finally:
            stack.pop()


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
        **kwargs,
    ):
        model = normalize_for_litellm(model)
        self._lm_model = model
        self._lm_provider = provider
        self._lm_max_tokens = max_tokens
        self._lm_temperature = temperature
        self._lm_extra_kwargs = dict(kwargs)
        self._tool_scope = ToolScope()
        self._profile_stack: list[AgentProfile] = []
        # Keep a one-time baseline configure so plain (no-profile) produce has a
        # global LM; per-call correctness comes from the ``dspy.context`` wrap in
        # ``produce_sync`` / ``profile_override`` (innermost context wins).
        dspy.settings.configure(lm=self._default_lm())

    def _default_lm(self) -> LM:
        return LM(
            model=self._lm_model,
            provider=self._lm_provider,
            max_tokens=self._lm_max_tokens,
            temperature=self._lm_temperature,
            cache=False,
            **self._lm_extra_kwargs,
        )

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
        stack = getattr(self, "_profile_stack", ())
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
            prediction = dspy.ReAct(sig, tools=tools) if tools else dspy.Predict(sig)
            return prediction(**kwargs, user_prompt=template)

    def _produce_lm_context(self) -> AbstractContextManager[Any]:
        """Make the active LM authoritative for one produce call.

        ``profile_override`` already wraps nested calls in ``dspy.context(lm=...)``;
        the innermost context wins, so we only apply the adapter default when no
        profile override is on the stack. The baseline ``dspy.settings.configure``
        from ``__init__`` remains the fallback for first-run safety.
        """
        if self._profile_stack:
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
            prediction = dspy.ReAct(sig, tools=[*tools, *mcp_tools])
            return await prediction.acall(**kwargs, user_prompt=template)

    def default_model(self) -> str | None:
        return self._lm_model

    @contextmanager
    def profile_override(self, profile: AgentProfile) -> Iterator[None]:
        needs_model_override = profile.needs_model_override(self.default_model())
        stack = getattr(self, "_profile_stack", None)
        if stack is None:
            stack = []
            self._profile_stack = stack
        stack.append(profile)
        try:
            if not needs_model_override and profile.effort is None:
                yield
                return
            model = self._lm_model
            if needs_model_override:
                if profile.model is None:
                    raise RuntimeError("Profile requires a model override but no model is set.")
                model = normalize_for_litellm(profile.model)
            lm_kwargs = dict(self._lm_extra_kwargs)
            if profile.effort is not None:
                lm_kwargs["reasoning_effort"] = profile.effort
            lm = LM(
                model=model,
                provider=self._lm_provider,
                max_tokens=self._lm_max_tokens,
                temperature=self._lm_temperature,
                cache=False,
                **lm_kwargs,
            )
            with dspy.context(lm=lm):
                yield
        finally:
            stack.pop()
