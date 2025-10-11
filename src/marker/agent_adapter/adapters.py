import asyncio

import dspy
from dspy import LM
from pydantic import BaseModel, create_model
from pydantic_ai import Agent
from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.output import OutputDataT

from .meta import T


def make_repr(cls: type) -> str:
    _type_repr = repr(cls)
    return _type_repr if not _type_repr.startswith("<class") else cls.__name__


class PydanticAdapter(Agent[AgentDepsT, OutputDataT]):
    """
    Adapter to use Pydantic AI agents.
    """

    @staticmethod
    def type_builder(output_schema: dict[str, type]) -> type[BaseModel]:
        """
        Type builder to create a Pydantic model from a given output schema.
        Args:
            output_schema (dict): A dictionary defining the output schema.
        Returns:
            type: A Pydantic model class based on the provided schema.
        """
        assert output_schema, "Output schema cannot be empty."
        field_definitions = {
            field: (type_, ...) for field, type_ in output_schema.items()
        }
        return create_model("_OutputModel", __base__=BaseModel, **field_definitions)

    async def produce(
        self,
        template: str,
        output_schema: dict[str, type] | None = None,
        output_type: type[T] | None = None,
        **kwargs,
    ):
        """
        Invoke the Pydantic AI agent with the provided template and output schema.
        """
        assert output_schema or output_type, "Output spec must be provided."
        return (
            await self.run(
                template,
                output_type=output_type or self.type_builder(output_schema),
                **kwargs,
            )
        ).output

    def produce_sync(
        self,
        template: str,
        output_schema: dict[str, type] | None = None,
        output_type: type[T] | None = None,
        **kwargs,
    ) -> T:
        """
        Synchronous version of produce method.
        """
        return asyncio.run(
            self.produce(
                template, output_schema=output_schema, output_type=output_type, **kwargs
            )
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
        dspy.settings.configure(
            lm=LM(
                model=model,
                provider=provider,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
        )

    @staticmethod
    async def type_builder(
        output_schema: dict[str, type], **kwargs
    ) -> str | dspy.Signature:
        """
        Type builder to create a dictionary schema from a given output type.
        Args:
            output_schema (dict): A dictionary defining the output schema.
            string (bool): Whether to use string signatures.
        Returns:
            dict: A dictionary schema based on the provided output type.
        """
        assert output_schema, "Output schema must be provided."
        field_definitions = {
            field: (type_, dspy.OutputField()) for field, type_ in output_schema.items()
        }
        field_definitions.update({"user_prompt": (str, dspy.InputField())})
        if not kwargs.pop("string", False):
            return create_model(
                "_OutputSignature",
                __base__=dspy.Signature,
                **field_definitions,
            )
        annotations = {
            f"{field}: {make_repr(type_)}" for field, type_ in output_schema.items()
        }
        return f"user_prompt: str -> {', '.join(annotations)}"

    async def produce(
        self,
        template: str,
        output_schema: dict[str, type],
        **kwargs,
    ) -> T:
        """
        Adapter to use DSPy agents.
        """
        assert output_schema, "Output spec must be provided."
        kwargs.pop("user_prompt", None)
        string = kwargs.pop("string", False)
        sig = await self.type_builder(output_schema, string=string)
        instructions = kwargs.pop("instructions", None)
        kw = {} if not instructions else {"instructions": instructions}
        if not string:
            prediction = dspy.Predict(sig.with_instructions(instructions))
            return prediction(**kwargs, user_prompt=template)
        else:
            prediction = dspy.Predict(dspy.Signature(sig, **kw))
            return prediction(user_prompt=template)

    def produce_sync(
        self,
        template: str,
        output_schema: dict[str, type],
        **kwargs,
    ) -> T:
        """
        Synchronous version of produce method.
        """
        return asyncio.run(
            self.produce(template, output_schema=output_schema, **kwargs)
        )
