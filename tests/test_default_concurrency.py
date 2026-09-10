from __future__ import annotations

import asyncio
from pathlib import Path
from threading import Barrier
from typing import Any

import pytest

import kedi
import kedi.api as api
from kedi.core import shared_thread_engine
from kedi.engine import SequentialEngine
from kedi.lang import compile_program, parse_program
from tests.mock_adapter import MockAdapter


SOURCE = (
    ">> First [first].\n>> Second [second].\n>> Join <first> and <second> as [joined].\n= <joined>"
)


@pytest.mark.parametrize(
    "source", ["[answer] << Say hello\n= <answer>", ">> Say hello", ">> Say [answer]"]
)
@pytest.mark.parametrize("fail", [False, True])
def test_explicit_sequential_calls_preserve_results_and_propagate_errors(source, fail):
    class Adapter(MockAdapter):
        def invoke_sync(self, **kwargs):
            if fail:
                raise ValueError("intentional model failure")
            return "hello"

        def produce_sync(self, **kwargs):
            if fail:
                raise ValueError("intentional model failure")
            return super().produce_sync(**kwargs)

    runtime = compile_program(parse_program(source), adapter=Adapter(), engine=SequentialEngine())
    try:
        if fail:
            with pytest.raises(Exception, match="intentional model failure"):
                runtime.run_main()
        else:
            result = runtime.run_main()
            if source.startswith("[answer]"):
                assert result == "hello"
    finally:
        asyncio.run(runtime.aclose())


class RendezvousAdapter(MockAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.barrier = Barrier(2, timeout=5)

    def produce_sync(self, *, template: str, output_schema: Any = None, **kwargs: Any) -> Any:
        if "joined" not in (output_schema or {}):
            self.barrier.wait()
        else:
            assert "first-value" in template
            assert "second-value" in template
        return super().produce_sync(template=template, output_schema=output_schema, **kwargs)


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("KEDI_PARALLEL", raising=False)
    token = api._config_var.set(api._APIConfig())
    try:
        yield
    finally:
        api._config_var.reset(token)


@pytest.mark.parametrize("surface", ["compiler", "query", "bind"])
def test_default_concurrency_runs_independent_calls_and_preserves_dependencies(
    surface: str, tmp_path: Path
) -> None:
    adapter = RendezvousAdapter()
    if surface == "compiler":
        runtime = compile_program(parse_program(SOURCE), adapter=adapter)
        try:
            result = runtime.run_main()
        finally:
            asyncio.run(runtime.aclose())
    elif surface == "query":

        @kedi.query(adapter=adapter)
        def run() -> str:
            """kedi
            >> First [first].
            >> Second [second].
            >> Join <first> and <second> as [joined].
            = <joined>
            """

        result = run()
    else:
        path = tmp_path / "parallel.kedi"
        path.write_text(SOURCE, encoding="utf-8")

        @kedi.bind(file=path, adapter=adapter)
        def run() -> str: ...

        result = run()
    assert result == "joined-value"
    assert len(adapter.model_calls) == 3


@pytest.mark.parametrize(
    "env,parallel,expected",
    [(None, None, True), ("0", None, False), ("1", False, False), ("0", True, True)],
)
def test_python_api_engine_precedence(
    monkeypatch: pytest.MonkeyPatch, env: str | None, parallel: bool | None, expected: bool
) -> None:
    if env is not None:
        monkeypatch.setenv("KEDI_PARALLEL", env)
    with kedi.context(parallel=parallel):
        assert api._execution_engine(api._config_var.get()).parallel is expected
        with kedi.interactive(adapter=MockAdapter()) as session:
            assert session._compiler.rt._engine.parallel is expected
            assert session.execute(">> Value [answer].\n= <answer>") == "answer-value"


def test_worker_count_and_context_restoration() -> None:
    with kedi.context(max_workers=3):
        assert api._execution_engine(api._config_var.get()) is shared_thread_engine(3)
        with kedi.context(parallel=False):
            assert isinstance(api._execution_engine(api._config_var.get()), SequentialEngine)
        assert api._execution_engine(api._config_var.get()) is shared_thread_engine(3)


def test_explicit_engine_overrides_api_and_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEDI_PARALLEL", "1")
    engine = SequentialEngine()
    with kedi.interactive(adapter=MockAdapter(), parallel=True, engine=engine) as session:
        assert session._compiler.rt._engine is engine


def test_interactive_conversation_order_is_preserved_with_default_concurrency() -> None:
    class OrderedAdapter(MockAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.fields: list[str] = []

        def produce_sync(self, *, template: str, output_schema: Any = None, **kwargs: Any) -> Any:
            self.fields.extend(output_schema or {})
            return super().produce_sync(template=template, output_schema=output_schema, **kwargs)

    adapter = OrderedAdapter()
    with kedi.interactive(adapter=adapter) as session:
        assert session._compiler.rt._engine.parallel is True
        assert session.execute(SOURCE) == "joined-value"
    assert adapter.fields == ["first", "second", "joined"]
