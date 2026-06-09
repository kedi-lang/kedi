import asyncio
from contextlib import contextmanager
from typing import Iterator, cast

import dspy
from dspy import LM
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.models import Model
from pydantic_ai.output import OutputDataT

from ..agent_profile import AgentProfile
from ..dynamic_models import build_dynamic_model
from ..model_normalization import normalize_for_litellm, normalize_for_pydantic_ai
from ..resolved_types import ResolvedKediType


def make_repr(cls: type) -> str:
    _type_repr = repr(cls)
    return _type_repr if not _type_repr.startswith("<class") else cls.__name__


class PydanticAdapter(Agent[AgentDepsT, OutputDataT]):
    """
    Adapter to use Pydantic AI agents.
    """

    def __init__(self, model: str | Model | None = None, **kwargs):
        """``model`` may be LiteLLM-style (``vendor/a/b``) or Pydantic-style (``vendor:a/b``)."""
        if isinstance(model, str):
            model = normalize_for_pydantic_ai(model) if model is not None else None
        super().__init__(model, **kwargs)

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
        return (
            await self.run(
                template,
                output_type=output_type,
                **kwargs,
            )
        ).output

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
        if not profile.needs_model_override(self.default_model()):
            yield
            return
        assert profile.model is not None
        with self.override(model=normalize_for_pydantic_ai(profile.model)):
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

    @staticmethod
    async def type_builder(
        output_schema: dict[str, ResolvedKediType], **kwargs
    ) -> type[dspy.Signature]:
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
        model = build_dynamic_model("_OutputSignature", dspy.Signature, field_definitions)
        return cast(type[dspy.Signature], model)

    async def produce(
        self,
        template: str,
        output_schema: dict[str, ResolvedKediType],
        **kwargs,
    ) -> dspy.Prediction:
        """
        Adapter to use DSPy agents.
        """
        assert output_schema, "Output spec must be provided."
        kwargs.pop("user_prompt", None)
        sig = await self.type_builder(output_schema)
        instructions = kwargs.pop("instructions", None)
        prediction = dspy.Predict(sig.with_instructions(instructions))
        return prediction(**kwargs, user_prompt=template)

    def produce_sync(
        self,
        template: str,
        output_schema: dict[str, ResolvedKediType],
        **kwargs,
    ) -> dspy.Prediction:
        """
        Synchronous version of produce method.
        """
        return asyncio.run(self.produce(template, output_schema=output_schema, **kwargs))

    def default_model(self) -> str | None:
        return self._lm_model

    @contextmanager
    def profile_override(self, profile: AgentProfile) -> Iterator[None]:
        if not profile.needs_model_override(self.default_model()):
            yield
            return
        assert profile.model is not None
        lm = LM(
            model=normalize_for_litellm(profile.model),
            provider=self._lm_provider,
            max_tokens=self._lm_max_tokens,
            temperature=self._lm_temperature,
            cache=False,
            **self._lm_extra_kwargs,
        )
        with dspy.context(lm=lm):
            yield
