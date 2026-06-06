from typing import Any, Protocol, TypeVar

from typing_extensions import runtime_checkable

T = TypeVar("T", covariant=True)


@runtime_checkable
class AgentAdapter(Protocol[T]):
    async def produce(
        self,
        *,
        template: str,
        output_schema: dict[str, type],
        **kwargs: Any,
    ) -> T:
        """
        Adapter protocol for agents to produce outputs based on a given template and output type.
        Args:
            template (str): The input template or prompt for the agent.
            output_schema (dict): A dictionary defining the output schema with field names as keys and their types as values.
            **kwargs: Additional keyword arguments for the agent.
        Returns:
            T: The output produced by the agent, matching the specified output type.
        """
        ...

    def produce_sync(
        self,
        *,
        template: str,
        output_schema: dict[str, type],
        **kwargs: Any,
    ) -> T:
        """
        Synchronous version of the produce method.
        Args:
            template (str): The input template or prompt for the agent.
            output_schema (dict): A dictionary defining the output schema with field names as keys and their types as values.
            **kwargs: Additional keyword arguments for the agent.
        Returns:
            T: The output produced by the agent, matching the specified output type.
        """
        ...
