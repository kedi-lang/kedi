#!/usr/bin/env python3
"""Run xd.kedi against real OpenRouter models and print per-call model overrides."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import dotenv

from kedi.agent_adapter.adapters import PydanticAdapter
from kedi.agent_profile import current_block_agent_profile
from kedi.lang import compile_program, parse_program
from kedi.resolved_types import ResolvedKediType
from kedi.source_loading import load_kedi_source_bundle

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADAPTER_MODEL = "openrouter:google/gemini-3.1-flash-lite-preview"


class TrackingPydanticAdapter(PydanticAdapter):
    def __init__(self, model: str | None = None, **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
        self.calls: list[dict[str, str | None]] = []

    def produce_sync(
        self,
        template: str,
        output_schema: dict[str, ResolvedKediType] | None = None,
        output_type: type[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        profile = current_block_agent_profile()
        adapter_default = self.default_model()
        effective = profile.model if profile.model is not None else adapter_default
        label = "unknown"
        for line in template.splitlines():
            if "Scope " in line:
                label = line.split("Scope ", 1)[1].split(".", 1)[0]
                break
        self.calls.append(
            {
                "label": label,
                "profile_model": profile.model,
                "effective_model": effective,
                "adapter_default": adapter_default,
            }
        )
        return super().produce_sync(
            template=template,
            output_schema=output_schema,
            output_type=output_type,
            **kwargs,
        )


def main() -> int:
    dotenv.load_dotenv(ROOT / ".env")
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY missing in .env", file=sys.stderr)
        return 1

    source_path = ROOT / "examples" / "model_test.kedi"
    bundle = load_kedi_source_bundle(source_path, include_val=False, include_cache=True)
    adapter = TrackingPydanticAdapter(model=os.environ.get("MODEL_NAME", DEFAULT_ADAPTER_MODEL))
    rt = compile_program(
        parse_program(bundle.text, source_path=str(source_path), source_map=bundle.source_map),
        adapter=adapter,
    )
    result = rt.run_main()
    print(result)
    print("\n--- adapter model trace ---")
    for index, call in enumerate(adapter.calls, start=1):
        print(
            f"{index}. {call['label']}: "
            f"effective={call['effective_model']} "
            f"(profile={call['profile_model']}, adapter_default={call['adapter_default']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
