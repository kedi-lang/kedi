from __future__ import annotations

from typing import Any, Dict, List


class MockAdapter:
    def produce_sync(
        self,
        *,
        template: str,
        output_schema: Dict[str, type] | None = None,
        output_type: Any | None = None,
        **_: Any,
    ) -> Dict[str, Any]:
        start = template.find("[")
        end = template.find("]", start + 1)
        keys: List[str] = []
        if start != -1 and end != -1:
            keys = [k.strip() for k in template[start + 1 : end].split(",") if k.strip()]
        annotations: Dict[str, Any] = {}
        # Prefer explicit schema if provided (new adapter contract)
        if isinstance(output_schema, dict) and output_schema:
            annotations = dict(output_schema)
        elif output_type is not None:
            try:
                if isinstance(getattr(output_type, "__annotations__", None), dict):
                    annotations = getattr(output_type, "__annotations__")  # type: ignore[assignment]
            except Exception:
                annotations = {}
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
        return data


