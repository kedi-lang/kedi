from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List

from kedi.agent_adapter import AgentAdapter
from kedi.agent_profile import AgentProfile
from kedi.resolved_types import ResolvedKediType
from kedi.utils import AttrDict

_OUTPUT_KEY_RE = re.compile(r"\[([A-Za-z_]\w*)(?::[^\]]*)?\]")


class MockAdapter(AgentAdapter[AttrDict[str, Any]]):
    def __init__(self, *, default_model: str | None = None) -> None:
        self.default_model_value = default_model
        self.last_effective_model: str | None = default_model
        self.model_calls: list[str | None] = []

    def default_model(self) -> str | None:
        return self.default_model_value

    @contextmanager
    def profile_override(self, profile: AgentProfile) -> Iterator[None]:
        previous = self.last_effective_model
        if profile.model is not None:
            self.last_effective_model = profile.model
        try:
            yield
        finally:
            self.last_effective_model = previous

    async def produce(
        self,
        *,
        template: str,
        output_schema: dict[str, ResolvedKediType],
        **_: Any,
    ) -> AttrDict[str, Any]:
        # Simple async facade around the synchronous implementation
        return self.produce_sync(template=template, output_schema=output_schema)

    def produce_sync(
        self,
        *,
        template: str,
        output_schema: dict[str, ResolvedKediType] | None = None,
        **_: Any,
    ) -> AttrDict[str, Any]:
        self.model_calls.append(self.last_effective_model)
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
