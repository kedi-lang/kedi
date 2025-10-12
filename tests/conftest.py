from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest


def _ensure_src_on_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    src_str = str(src_path)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


_ensure_src_on_path()


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
        if output_type is not None:
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


@pytest.fixture(autouse=True)
def inject_adapters(monkeypatch: Any, request: Any) -> None:
    is_llm = bool(getattr(request.node, "get_closest_marker", None) and request.node.get_closest_marker("llm"))

    from marker.lang import compiler as _compiler
    real_compile = _compiler.compile_program

    if is_llm:
        # Build a real adapter for LLM-marked tests
        try:
            from marker.agent_adapter.adapters import PydanticAdapter, DSPyAdapter
        except Exception:
            PydanticAdapter = None  # type: ignore[assignment]
            DSPyAdapter = None  # type: ignore[assignment]

        adapter_name = os.environ.get("MARKER_ADAPTER", "pydantic").lower()
        model_name = os.environ.get("MODEL_NAME", os.environ.get("GROQ_MODEL", "groq:qwen/qwen3-32b"))
        adapter_obj: Any
        if adapter_name == "dspy" and DSPyAdapter is not None:
            adapter_obj = DSPyAdapter(model=model_name)
        else:
            # default to pydantic
            adapter_obj = PydanticAdapter(model=model_name) if PydanticAdapter is not None else MockAdapter()

        def _wrapped_llm(prog: Any, *args: Any, **kwargs: Any):
            kwargs.setdefault("adapter", adapter_obj)
            return real_compile(prog, *args, **kwargs)

        monkeypatch.setattr(_compiler, "compile_program", _wrapped_llm, raising=True)
        # Also default MarkerRuntime(adapter=None) to the real adapter for direct usage
        import marker.core as _core
        _orig_init_llm = _core.MarkerRuntime.__init__

        def _mr_init_llm(self: Any, *args: Any, **kwargs: Any) -> None:
            if "adapter" not in kwargs or kwargs.get("adapter") is None:
                kwargs["adapter"] = adapter_obj
            return _orig_init_llm(self, *args, **kwargs)

        monkeypatch.setattr(_core.MarkerRuntime, "__init__", _mr_init_llm, raising=True)
        return

    # Non-LLM tests: inject MockAdapter into compile_program
    def _wrapped(prog: Any, *args: Any, **kwargs: Any):
        kwargs.setdefault("adapter", MockAdapter())
        return real_compile(prog, *args, **kwargs)

    monkeypatch.setattr(_compiler, "compile_program", _wrapped, raising=True)

    # Also default MarkerRuntime(adapter=None) to MockAdapter for direct usage in tests
    import marker.core as _core
    _orig_init = _core.MarkerRuntime.__init__

    def _mr_init(self: Any, *args: Any, **kwargs: Any) -> None:
        if "adapter" not in kwargs or kwargs.get("adapter") is None:
            kwargs["adapter"] = MockAdapter()
        return _orig_init(self, *args, **kwargs)

    monkeypatch.setattr(_core.MarkerRuntime, "__init__", _mr_init, raising=True)
