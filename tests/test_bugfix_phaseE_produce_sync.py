from __future__ import annotations

import asyncio
from typing import Any

import pytest

from kedi.agent_adapter.utils import run_sync


async def _echo(value: str) -> str:
    await asyncio.sleep(0)
    return value


def test_run_sync_without_running_loop() -> None:
    assert run_sync(_echo("plain")) == "plain"


def test_run_sync_inside_running_loop_returns_cleanly() -> None:
    """BUG-019: a sync produce boundary must work inside a running loop."""

    async def driver() -> str:
        return run_sync(_echo("nested"))

    assert asyncio.run(driver()) == "nested"


def test_run_sync_inside_running_loop_propagates_errors() -> None:
    async def boom() -> None:
        raise RuntimeError("kaboom")

    async def driver() -> None:
        run_sync(boom())

    with pytest.raises(RuntimeError, match="kaboom"):
        asyncio.run(driver())


def test_pydantic_adapter_produce_sync_inside_running_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kedi.agent_adapter.adapters import PydanticAdapter

    adapter = PydanticAdapter(model="openai/gpt-4o-mini")

    async def fake_run(template: str, *, output_type: type[Any], **_: Any) -> Any:
        return type("R", (), {"output": output_type(answer="ok")})()

    monkeypatch.setattr(adapter, "run", fake_run)

    async def driver() -> Any:
        return adapter.produce_sync("template", output_schema={"answer": str})

    result = asyncio.run(driver())
    assert result.answer == "ok"
