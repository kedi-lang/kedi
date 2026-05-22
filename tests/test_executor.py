from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from kedi import DefaultExecutor
from kedi.core import KediRuntime
from kedi.executors.debug import (
    ExecutorDebugEvent,
    MarkdownDebugExporter,
)
from kedi.lang.compiler import compile_program
from kedi.lang.parser import parse_program


class RecordingExecutor:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self._delegate = DefaultExecutor()

    def set_debug_exporter(self, exporter) -> None:
        self._delegate.set_debug_exporter(exporter)

    def evaluate_inline(self, rt, code: str, local_env: dict[str, Any]) -> Any:
        self.calls.append("evaluate_inline")
        return self._delegate.evaluate_inline(rt, code, local_env)

    def execute_block(self, rt, code: str, local_env: dict[str, Any]) -> Any:
        self.calls.append("execute_block")
        return self._delegate.execute_block(rt, code, local_env)

    def execute_side_effects(self, rt, code: str, env_map: dict[str, Any]) -> None:
        self.calls.append("execute_side_effects")
        self._delegate.execute_side_effects(rt, code, env_map)

    def execute_prelude(self, rt, code: str) -> dict[str, Any]:
        self.calls.append("execute_prelude")
        return self._delegate.execute_prelude(rt, code)

    def create_dynamic_function(
        self, name: str, params: list[str], body: Callable[..., Any]
    ) -> Callable[..., Any]:
        self.calls.append("create_dynamic_function")
        return self._delegate.create_dynamic_function(name, params, body)

    def evaluate_type_expression(self, rt, code: str, env: dict[str, Any]) -> Any:
        self.calls.append("evaluate_type_expression")
        return self._delegate.evaluate_type_expression(rt, code, env)


class MemoryDebugExporter:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def export(self, event) -> None:
        self.events.append(event)


def test_compile_program_uses_injected_executor_end_to_end():
    src = (
        "```\n"
        "seed = 1\n"
        "```\n"
        "@compute(x: `int`) -> `int`:\n"
        "  ```\n"
        "  x = x + seed\n"
        "  ```\n"
        "  `x += 1`\n"
        "  [value: `int`] = ```\n"
        "  return x\n"
        "  ```\n"
        "  = `value`\n"
        "\n"
        "= <compute(`2`)>\n"
    )
    executor = RecordingExecutor()

    runtime = compile_program(parse_program(src), executor=executor)

    assert runtime.run_main() == "4"
    assert runtime.executor is executor
    assert set(executor.calls) >= {
        "execute_prelude",
        "create_dynamic_function",
        "evaluate_type_expression",
        "execute_side_effects",
        "execute_block",
        "evaluate_inline",
    }


def test_markdown_debug_exporter_renders_human_readable_sections(tmp_path: Path):
    path = tmp_path / "executor-debug.md"
    exporter = MarkdownDebugExporter(output_path=path)

    exporter.export(
        ExecutorDebugEvent.create(
            "external_function_call",
            executor="DefaultExecutor",
            step="evaluate_inline",
            execution_kind="inline",
            function_name="fetch",
            code="fetch(url)",
            args=["https://example.com"],
            kwargs={"timeout": 3},
        )
    )

    content = path.read_text(encoding="utf-8")
    assert "## External Function Call" in content
    assert "- Executor: `DefaultExecutor`" in content
    assert "- Step: `evaluate_inline`" in content
    assert "- Execution kind: `inline`" in content
    assert "- Function: `fetch`" in content
    assert "### Code" in content
    assert "fetch(url)" in content
    assert "### Args" in content
    assert "https://example.com" in content


def test_markdown_debug_exporter_generates_default_filename(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    exporter = MarkdownDebugExporter(default_name="sample.kedi")
    path = exporter.output_path

    assert path.parent == tmp_path
    assert path.name.startswith("sample_kedi_debug_export_")
    assert path.suffix == ".md"


def test_default_executor_emits_debug_events_for_successful_step():
    exporter = MemoryDebugExporter()
    executor = DefaultExecutor(debug_exporter=exporter)
    rt = KediRuntime(executor=executor)

    result = executor.evaluate_inline(rt, "x + 1", {"x": 4})

    assert result == 5
    assert [event.event for event in exporter.events] == [
        "execution_started",
        "execution_completed",
    ]
    assert [event.executor for event in exporter.events] == [
        "DefaultExecutor",
        "DefaultExecutor",
    ]
    assert [event.step for event in exporter.events] == [
        "evaluate_inline",
        "evaluate_inline",
    ]
