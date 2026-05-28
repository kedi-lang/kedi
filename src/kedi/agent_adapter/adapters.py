import asyncio

import dspy
from dspy import LM
from pydantic import BaseModel, create_model
from pydantic_ai import Agent
from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.models import Model
from pydantic_ai.output import OutputDataT

from ..model_normalization import normalize_for_litellm, normalize_for_pydantic_ai


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
    def type_builder(output_schema: dict[str, type]) -> type[OutputDataT]:
        """
        Type builder to create a Pydantic model from a given output schema.
        Args:
            output_schema (dict): A dictionary defining the output schema.
        Returns:
            type: A Pydantic model class based on the provided schema.
        """
        assert output_schema, "Output schema cannot be empty."
        field_definitions = {field: (type_, ...) for field, type_ in output_schema.items()}
        return create_model("_OutputModel", __base__=BaseModel, **field_definitions)  # type: ignore[return-value]

    async def produce(
        self,
        template: str,
        output_schema: dict[str, type] | None = None,
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
        output_schema: dict[str, type] | None = None,
        output_type: type[OutputDataT] | None = None,
        **kwargs,
    ) -> OutputDataT:
        """
        Synchronous version of produce method.
        """
        return asyncio.run(
            self.produce(template, output_schema=output_schema, output_type=output_type, **kwargs)
        )


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
    async def type_builder(output_schema: dict[str, type], **kwargs) -> type[dspy.Signature]:
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
        return create_model(
            "_OutputSignature",
            __base__=dspy.Signature,
            **field_definitions,
        )

    async def produce(
        self,
        template: str,
        output_schema: dict[str, type],
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
        output_schema: dict[str, type],
        **kwargs,
    ) -> dspy.Prediction:
        """
        Synchronous version of produce method.
        """
        return asyncio.run(self.produce(template, output_schema=output_schema, **kwargs))
