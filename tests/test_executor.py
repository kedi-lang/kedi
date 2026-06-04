from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from kedi import DefaultExecutor
from kedi.core import KediRuntime
from kedi.executors.debug import (
    ExecutorDebugEvent,
    MarkdownDebugExporter,
    _normalize_base_name,
    _render_markdown_event,
    _sanitize,
    default_debug_export_path,
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

    def execute_side_effects(
        self,
        rt,
        code: str,
        env_map: dict[str, Any],
        *,
        kedi_line_offset: int = 0,
    ) -> None:
        self.calls.append("execute_side_effects")
        self._delegate.execute_side_effects(
            rt,
            code,
            env_map,
            kedi_line_offset=kedi_line_offset,
        )

    def execute_prelude(self, rt, code: str) -> dict[str, Any]:
        self.calls.append("execute_prelude")
        return self._delegate.execute_prelude(rt, code)

    def create_dynamic_function(
        self,
        name: str,
        params: list[str],
        body: Callable[..., Any],
        defaults: dict[str, Any] | None = None,
    ) -> Callable[..., Any]:
        self.calls.append("create_dynamic_function")
        return self._delegate.create_dynamic_function(name, params, body, defaults)

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


def test_markdown_debug_exporter_generates_default_filename(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    exporter = MarkdownDebugExporter(default_name="sample.kedi")
    path = exporter.output_path

    assert path.parent == tmp_path
    assert path.name.startswith("sample_kedi_debug_export_")
    assert path.suffix == ".md"


def test_debug_exporter_normalizes_names_and_sanitizes_payloads(tmp_path: Path) -> None:
    fixed = default_debug_export_path("Report.txt", now=datetime(2026, 3, 3))
    assert fixed.name == "report_debug_export_20260303_000000.md"
    assert _normalize_base_name(" !!! ") == "kedi"

    class CustomValue:
        def __repr__(self) -> str:
            return "<custom>"

    payload = _sanitize(
        {
            "path": tmp_path / "sample.kedi",
            "values": [tmp_path / "x", CustomValue()],
        }
    )

    assert payload == {
        "path": str(tmp_path / "sample.kedi"),
        "values": [str(tmp_path / "x"), "<custom>"],
    }


def test_render_markdown_event_skips_empty_fields_and_renders_all_sections() -> None:
    event = ExecutorDebugEvent.create(
        "execution_failed",
        executor="DefaultExecutor",
        step="execute_block",
        execution_kind="",
        function_name=None,
        fallback_executor="DefaultExecutor",
        mode="fallback",
        error_type="ValueError",
        error="bad value",
        external_functions=["fetch"],
        code="return x",
        inputs={"x": 1},
        local_env={"x": 1},
        env_map={"seed": 1},
        args=[],
        kwargs={},
        result=None,
        output="done",
    )

    rendered = _render_markdown_event(event)

    assert "- Execution kind:" not in rendered
    assert "- Fallback executor: `DefaultExecutor`" in rendered
    assert "- Mode: `fallback`" in rendered
    assert "- Error type: `ValueError`" in rendered
    assert "- External functions: `['fetch']`" in rendered
    assert "### Code" in rendered
    assert "### Inputs" in rendered
    assert "### Local env" in rendered
    assert "### Env map" in rendered
    assert "### Output" in rendered


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


def test_default_executor_setter_and_variable_sync_updates_runtime_env() -> None:
    exporter = MemoryDebugExporter()
    executor = DefaultExecutor()
    rt = KediRuntime(executor=executor)

    executor.set_debug_exporter(exporter)
    executor.evaluate_inline(rt, "1", {})

    runtime_env = {"x": 1, "stale": 2}
    token = rt._current_env.set(runtime_env)
    try:
        target = {"x": 1, "missing": 3}
        executor._sync_existing_variables(rt, target, {"x": 10, "extra": 99})
    finally:
        rt._current_env.reset(token)

    assert exporter.events
    assert target == {"x": 10, "missing": 3}
    assert runtime_env == {"x": 10, "stale": 2}

    target_without_runtime_env = {"x": 1}
    executor._sync_existing_variables(rt, target_without_runtime_env, {"x": 2})
    assert target_without_runtime_env == {"x": 2}
