from __future__ import annotations

import re
import threading
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, List, Sequence

from kedi.agent_adapter import AgentAdapter
from kedi.agent_adapter.meta import AdapterCapabilities, AgentAdapterKind
from kedi.agent_adapter.tools.scope import ToolScope
from kedi.agent_adapter.tools.specs import ToolSpec
from kedi.agent_profile import AgentProfile
from kedi.resolved_types import ResolvedKediType
from kedi.utils import AttrDict

_OUTPUT_KEY_RE = re.compile(r"\[([A-Za-z_]\w*)(?::[^\]]*)?\]")
_PREV_WORD_RE = re.compile(r"last letter of (\S+)", re.IGNORECASE)


class MockAdapter(AgentAdapter[AttrDict[str, Any]]):
    kind: AgentAdapterKind = "agent-framework"
    shortname = "mock"
    capabilities = AdapterCapabilities(
        supports_structured_output=True,
        supports_tool_registration=True,
        supports_mcp=True,
        supports_profile_override=True,
        supports_model_override=True,
        supports_effort=True,
        supports_settings=True,
        supports_artifacts=True,
    )

    def __init__(self, *, default_model: str | None = None) -> None:
        self._tool_scope = ToolScope()
        # Recording lists are shared instance state appended from worker threads under the
        # parallel engine; guard the grouped appends so concurrent calls don't interleave.
        self._record_lock = threading.Lock()
        self.default_model_value = default_model
        self._effective_profile: ContextVar[
            tuple[str | None, str | None, dict[str, Any], tuple[str | None, ...]]
        ] = ContextVar(
            f"mock_adapter_profile_{id(self)}",
            default=(default_model, None, {}, ()),
        )
        self.model_calls: list[str | None] = []
        self.effort_calls: list[str | None] = []
        self.settings_calls: list[dict[str, Any]] = []
        self.instruction_calls: list[str | None] = []
        self.tool_calls: list[tuple[str, ...]] = []
        self.mcp_calls: list[tuple[str | None, ...]] = []
        self.request_calls: list[str] = []

    @property
    def last_effective_model(self) -> str | None:
        return self._effective_profile.get()[0]

    @property
    def last_effective_effort(self) -> str | None:
        return self._effective_profile.get()[1]

    @property
    def last_effective_settings(self) -> dict[str, Any]:
        return self._effective_profile.get()[2]

    @property
    def last_effective_mcp_commands(self) -> tuple[str | None, ...]:
        return self._effective_profile.get()[3]

    def default_model(self) -> str | None:
        return self.default_model_value

    def tool_scope(self, tools: Sequence[Any] = ()) -> AbstractContextManager[None]:
        return self._tool_scope.tool_scope(tools)

    def register_tool(self, tool: ToolSpec) -> None:
        self._tool_scope.register_tool(tool)

    @property
    def scoped_tools(self) -> tuple[Any, ...]:
        return self._tool_scope.scoped_tools

    @contextmanager
    def profile_override(self, profile: AgentProfile) -> Iterator[None]:
        model, effort, settings, mcp_commands = self._effective_profile.get()
        effective = (
            profile.model if profile.model is not None else model,
            profile.effort if profile.effort is not None else effort,
            {**settings, **profile.settings} if profile.settings else settings,
            (
                tuple(server.command for server in profile.mcp_servers)
                if profile.mcp_servers
                else mcp_commands
            ),
        )
        token = self._effective_profile.set(effective)
        try:
            yield
        finally:
            self._effective_profile.reset(token)

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
        with self._record_lock:
            self.request_calls.append(template)
            self.model_calls.append(self.last_effective_model)
            self.effort_calls.append(self.last_effective_effort)
            self.settings_calls.append(dict(self.last_effective_settings))
            self.instruction_calls.append(kwargs.get("instructions"))
            self.tool_calls.append(tuple(tool.name for tool in self.scoped_tools))
            self.mcp_calls.append(self.last_effective_mcp_commands)
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

    async def invoke(self, *, prompt: str, **kwargs: Any) -> str:
        return self.invoke_sync(prompt=prompt, **kwargs)

    def invoke_sync(self, *, prompt: str, **kwargs: Any) -> str:
        with self._record_lock:
            self.request_calls.append(prompt)
            self.model_calls.append(self.last_effective_model)
            self.effort_calls.append(self.last_effective_effort)
            self.settings_calls.append(dict(self.last_effective_settings))
            self.instruction_calls.append(kwargs.get("instructions"))
            self.tool_calls.append(tuple(tool.name for tool in self.scoped_tools))
            self.mcp_calls.append(self.last_effective_mcp_commands)
        return f"response:{prompt}"


class WordChainMockAdapter(MockAdapter):
    """Deterministic adapter for ``word_chain.kedi`` without live LLM calls."""

    def produce_sync(
        self,
        *,
        template: str,
        output_schema: dict[str, ResolvedKediType] | None = None,
        **kwargs: Any,
    ) -> AttrDict[str, Any]:
        if output_schema and (output_name := next(iter(output_schema), None)):
            match = _PREV_WORD_RE.search(template)
            if match is not None:
                previous = match.group(1).rstrip(".,;:")
                chain_word = previous[-1] * 2
                self.model_calls.append(self.last_effective_model)
                self.instruction_calls.append(kwargs.get("instructions"))
                return AttrDict({output_name: chain_word})
        return super().produce_sync(
            template=template,
            output_schema=output_schema,
            **kwargs,
        )
