"""BUG-007: top-level side effects must run exactly once.

``run_program_cli`` used to run the program offline first and, when a template
needed the LLM, re-run the whole program with a real adapter — executing every
top-level side effect twice. With the lazy adapter the program runs once.
"""

from __future__ import annotations

import kedi.app as app
from kedi.app import run_program_cli
from kedi.utils import AttrDict
from tests.mock_adapter import MockAdapter

_COUNTER_BLOCK = (
    "```\n"
    "from pathlib import Path\n"
    "p = Path(args.counter)\n"
    'p.write_text(str(int(p.read_text()) + 1) if p.exists() else "1")\n'
    "```\n"
)


def test_top_level_side_effects_run_once_with_llm_template(tmp_path, monkeypatch) -> None:
    counter = tmp_path / "count.txt"
    prog = tmp_path / "prog.kedi"
    prog.write_text(
        _COUNTER_BLOCK + ">> Produce a greeting. [greeting]\n= <`greeting`>\n",
        encoding="utf-8",
    )

    builds: list[tuple[str, str | None]] = []

    def fake_build(adapter: str, model: str | None) -> MockAdapter:
        builds.append((adapter, model))
        return MockAdapter(default_model=model)

    monkeypatch.setattr(app, "_build_runtime_adapter", fake_build)

    code, out = run_program_cli(
        str(prog),
        adapter="pydantic",
        model=None,
        runtime_args=AttrDict({"counter": str(counter)}, default=None),
    )

    assert code == 0
    # The single top-level side effect must fire exactly once (was twice before).
    assert counter.read_text() == "1"
    # The template needed the LLM, so the adapter was built once, lazily.
    assert len(builds) == 1
    assert "greeting-value" in out


def test_offline_program_never_builds_adapter(tmp_path, monkeypatch) -> None:
    counter = tmp_path / "count.txt"
    prog = tmp_path / "prog.kedi"
    prog.write_text(_COUNTER_BLOCK + "= done\n", encoding="utf-8")

    builds: list[int] = []

    def fake_build(adapter: str, model: str | None) -> MockAdapter:
        builds.append(1)
        return MockAdapter(default_model=model)

    monkeypatch.setattr(app, "_build_runtime_adapter", fake_build)

    code, out = run_program_cli(
        str(prog),
        adapter="pydantic",
        model=None,
        runtime_args=AttrDict({"counter": str(counter)}, default=None),
    )

    assert code == 0
    assert out == "done"
    assert counter.read_text() == "1"
    # A program fully covered without templates never constructs an adapter.
    assert builds == []
