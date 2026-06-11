from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List

from kedi.agent_adapter import AgentAdapter
from kedi.agent_adapter.tool_override import ToolScope
from kedi.agent_profile import AgentProfile
from kedi.resolved_types import ResolvedKediType
from kedi.utils import AttrDict

_OUTPUT_KEY_RE = re.compile(r"\[([A-Za-z_]\w*)(?::[^\]]*)?\]")
_PREV_WORD_RE = re.compile(r"last letter of (\S+)", re.IGNORECASE)


class MockAdapter(AgentAdapter[AttrDict[str, Any]]):
    def __init__(self, *, default_model: str | None = None) -> None:
        self._tool_scope = ToolScope()
        self.default_model_value = default_model
        self.last_effective_model: str | None = default_model
        self.last_effective_effort: str | None = None
        self.model_calls: list[str | None] = []
        self.effort_calls: list[str | None] = []
        self.instruction_calls: list[str | None] = []
        self.tool_calls: list[tuple[str, ...]] = []

    def default_model(self) -> str | None:
        return self.default_model_value

    def tool_scope(self, tools=()):
        return self._tool_scope.tool_scope(tools)

    def register_tool(self, tool):
        self._tool_scope.register_tool(tool)

    @property
    def scoped_tools(self):
        return self._tool_scope.scoped_tools

    @contextmanager
    def profile_override(self, profile: AgentProfile) -> Iterator[None]:
        previous = self.last_effective_model
        previous_effort = self.last_effective_effort
        if profile.model is not None:
            self.last_effective_model = profile.model
        if profile.effort is not None:
            self.last_effective_effort = profile.effort
        try:
            yield
        finally:
            self.last_effective_model = previous
            self.last_effective_effort = previous_effort

    async def produce(
        self,
        *,
        template: str,
        output_schema: dict[str, ResolvedKediType],
        **kwargs: Any,
    ) -> AttrDict[str, Any]:
        # Simple async facade around the synchronous implementation
        return self.produce_sync(template=template, output_schema=output_schema, **kwargs)

    def produce_sync(
        self,
        *,
        template: str,
        output_schema: dict[str, ResolvedKediType] | None = None,
        **kwargs: Any,
    ) -> AttrDict[str, Any]:
        self.model_calls.append(self.last_effective_model)
        self.effort_calls.append(self.last_effective_effort)
        self.instruction_calls.append(kwargs.get("instructions"))
        self.tool_calls.append(tuple(tool.name for tool in self.scoped_tools))
        keys: List[str] = []
        seen: set[str] = set()
        for match in _OUTPUT_KEY_RE.finditer(template):
            key = match.group(1)
            if key not in seen:
                seen.add(key)
                keys.append(key)
        annotations: Dict[str, Any] = dict(output_schema or {})
        data: Dict[str, Any] = {}
        for k in keys:
            ann = annotations.get(k)
            try:
                origin = getattr(ann, "__origin__", None)
            except Exception:
                origin = None
            if ann is int:
                data[k] = 42
            elif ann is float:
                data[k] = 3.14
            elif ann is bool:
                data[k] = True
            elif origin in (list, List):
                arg = getattr(ann, "__args__", (str,))[0]
                if arg is str:
                    data[k] = [f"{k}-a", f"{k}-b"]
                elif arg is int:
                    data[k] = [1, 2]
                else:
                    data[k] = []
            else:
                data[k] = f"{k}-value"
        return AttrDict(data)


class WordChainMockAdapter(MockAdapter):
    """Deterministic adapter for ``word_chain.kedi`` without live LLM calls."""

    def produce_sync(
        self,
        *,
        template: str,
        output_schema: dict[str, ResolvedKediType] | None = None,
        **kwargs: Any,
    ) -> AttrDict[str, Any]:
        if output_schema and "word" in output_schema:
            match = _PREV_WORD_RE.search(template)
            if match is not None:
                previous = match.group(1).rstrip(".,;:")
                chain_word = previous[-1] * 2
                self.model_calls.append(self.last_effective_model)
                self.instruction_calls.append(kwargs.get("instructions"))
                return AttrDict({"word": chain_word})
        return super().produce_sync(
            template=template,
            output_schema=output_schema,
            **kwargs,
        )
