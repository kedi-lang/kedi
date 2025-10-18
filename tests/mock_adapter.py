from __future__ import annotations

from typing import Any, Dict, List, TypeVar
from marker.agent_adapter import AgentAdapter

K = TypeVar('K') # key type
V = TypeVar('V') # value type

class AttrDict(dict[K, V]):
    """
    A class to make dictionary accesible by attribute notation.
    """
    __getattr__ = dict.__getitem__ # forward attrs to items


class MockAdapter(AgentAdapter[Dict[str, Any]]):
    async def produce(
        self,
        *,
        template: str,
        output_schema: Dict[str, type],
        **_: Any,
    ) -> AttrDict[str, Any]:
        # Simple async facade around the synchronous implementation
        return self.produce_sync(template=template, output_schema=output_schema)

    def produce_sync(
        self,
        *,
        template: str,
        output_schema: Dict[str, type] | None = None,
        **_: Any,
    ) -> AttrDict[str, Any]:
        start = template.find("[")
        end = template.find("]", start + 1)
        keys: List[str] = []
        if start != -1 and end != -1:
            keys = [k.strip() for k in template[start + 1 : end].split(",") if k.strip()]
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
