from __future__ import annotations

import asyncio
from contextlib import AbstractContextManager, contextmanager
from typing import TYPE_CHECKING, Any, Callable, Iterator, Sequence, cast

import dspy
from dspy import LM
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.models import Model
from pydantic_ai.output import OutputDataT

from ..effort import pydantic_effort_value
from ..agent_profile import AgentProfile
from ..dynamic_models import build_dynamic_model
from ..model_normalization import normalize_for_litellm, normalize_for_pydantic_ai
from ..resolved_types import ResolvedKediType
from .tool_override import ToolScope

if TYPE_CHECKING:
    from pydantic_ai.tools import Tool as PydanticTool

    from kedi.core import KediRuntime
    from kedi.lang.ast import ProcedureDef

_LOGFIRE_CONFIGURED = False


def configure_logfire() -> None:
    global _LOGFIRE_CONFIGURED
    if _LOGFIRE_CONFIGURED:
        return

    import logfire

    logfire.configure(
        send_to_logfire="if-token-present",
        service_name="kedi",
        console=False,
    )
    logfire.instrument_pydantic_ai()
    _LOGFIRE_CONFIGURED = True


def make_repr(cls: type) -> str:
    _type_repr = repr(cls)
    return _type_repr if not _type_repr.startswith("<class") else cls.__name__


class PydanticAdapter(Agent[AgentDepsT, OutputDataT]):
    """
    Adapter to use Pydantic AI agents.
    """

    def __init__(self, model: str | Model | None = None, **kwargs: Any) -> None:
        """``model`` may be LiteLLM-style (``vendor/a/b``) or Pydantic-style (``vendor:a/b``)."""
        configure_logfire()
        self._tool_scope = ToolScope()
        if isinstance(model, str):
            model = normalize_for_pydantic_ai(model) if model is not None else None
        super().__init__(model, **kwargs)

    def tool_scope(self, tools: Sequence[Any] = ()) -> AbstractContextManager[None]:
        return self._tool_scope.tool_scope(tools)

    def register_tool(self, tool: Any) -> None:
        self._tool_scope.register_tool(tool)

    def register_kedi_tool(self, rt: KediRuntime, proc: ProcedureDef) -> None:
        from .utils import kedi_procedure_to_pydantic_tool

        self.register_tool(kedi_procedure_to_pydantic_tool(rt, proc))

    def register_python_tool(self, name: str, fn: Callable[..., Any]) -> None:
        from pydantic_ai.tools import Tool

        self.register_tool(Tool(fn, name=name))

    @property
    def scoped_tools(self) -> tuple[PydanticTool[Any], ...]:
        return cast("tuple[PydanticTool[Any], ...]", self._tool_scope.scoped_tools)

    @staticmethod
    def type_builder(output_schema: dict[str, ResolvedKediType]) -> type[OutputDataT]:
        """
        Type builder to create a Pydantic model from a given output schema.
        Args:
            output_schema (dict): A dictionary defining the output schema.
        Returns:
            type: A Pydantic model class based on the provided schema.
        """
        assert output_schema, "Output schema cannot be empty."
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
        assert output_schema or output_type, "Output spec must be provided."
        if output_schema:
            output_type = self.type_builder(output_schema)
        run_kwargs: dict[str, object] = {"output_type": output_type, **kwargs}
        toolsets = self._toolsets_for_run()
        if toolsets is not None:
            run_kwargs["toolsets"] = toolsets
        return (await self.run(template, **cast(Any, run_kwargs))).output

    def _toolsets_for_run(self) -> list[Any] | None:
        scoped_tools = self.scoped_tools
        if not scoped_tools:
            return None
        from pydantic_ai.toolsets import FunctionToolset

        return [FunctionToolset(tools=scoped_tools)]

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
        return asyncio.run(
            self.produce(template, output_schema=output_schema, output_type=output_type, **kwargs)
        )

    def default_model(self) -> str | None:
        model = self.model
        return model if isinstance(model, str) else None

    @contextmanager
    def profile_override(self, profile: AgentProfile) -> Iterator[None]:
        override_kwargs: dict[str, Any] = {}
        if profile.needs_model_override(self.default_model()):
            assert profile.model is not None
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
        if not override_kwargs:
            yield
            return
        with self.override(**override_kwargs):
            yield


class DSPyAdapter:
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
        dspy.settings.configure(
            lm=LM(
                model=model,
                provider=provider,
                max_tokens=max_tokens,
                temperature=temperature,
                cache=False,
                **kwargs,
            )
        )

    def tool_scope(self, tools: Sequence[Any] = ()) -> AbstractContextManager[None]:
        return self._tool_scope.tool_scope(tools)

    def register_tool(self, tool: Any) -> None:
        self._tool_scope.register_tool(tool)

    def register_kedi_tool(self, rt: KediRuntime, proc: ProcedureDef) -> None:
        from .utils import kedi_procedure_to_dspy_tool

        self.register_tool(kedi_procedure_to_dspy_tool(rt, proc))

    def register_python_tool(self, name: str, fn: Callable[..., Any]) -> None:
        from .utils import dspy_sync_tool_function

        self.register_tool(dspy.Tool(dspy_sync_tool_function(fn), name=name))

    @property
    def scoped_tools(self) -> tuple[Any, ...]:
        scope = getattr(self, "_tool_scope", None)
        return () if scope is None else scope.scoped_tools

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
        assert output_schema, "Output schema must be provided."
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
        assert output_schema, "Output spec must be provided."
        kwargs.pop("user_prompt", None)
        sig = self.type_builder(output_schema)
        instructions = kwargs.pop("instructions", None)
        sig = sig.with_instructions(instructions)
        tools = cast(list[Callable[..., Any]], list(self.scoped_tools))
        prediction = dspy.ReAct(sig, tools=tools) if tools else dspy.Predict(sig)
        return prediction(**kwargs, user_prompt=template)

    def default_model(self) -> str | None:
        return self._lm_model

    @contextmanager
    def profile_override(self, profile: AgentProfile) -> Iterator[None]:
        needs_model_override = profile.needs_model_override(self.default_model())
        if not needs_model_override and profile.effort is None:
            yield
            return
        model = self._lm_model
        if needs_model_override:
            assert profile.model is not None
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
