from __future__ import annotations

import asyncio
import os
import sys
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from kedi.cli import main as cli_main
from kedi.codegen.interfaces import CodegenConfig, ProgramContext
from kedi.codegen.manager import CodegenManager
from kedi.codegen.mock_agents import MockPassingAgent
from kedi.codegen.prompts import (
    build_context_section,
    build_implementation_prompt,
    build_test_generation_prompt,
)
from kedi.codegen.translator import (
    kedi_procedure_to_python_signature,
    kedi_signature_to_python_signature,
    kedi_typedef_to_python_class,
    python_impl_to_kedi_impl,
    python_tests_to_kedi_tests,
)
from kedi.lang.ast import ProcedureDef, TypeDef, TypeField
from kedi.reporter import BufferReporter, CliReporter, NullReporter
from kedi.tests_runner import TestCaseResult as KediTestCaseResult
from kedi.tests_runner import TestRunResult as KediTestRunResult
from kedi.tests_runner import run_tests


def test_codegen_cache_paths(tmp_path: Path) -> None:
    src = tmp_path / "prog.kedi"
    src.write_text("= ok\n", encoding="utf-8")
    mgr = CodegenManager(MockPassingAgent())
    cache = mgr.cache_path_for(src)
    assert cache.name.endswith(".cache.kedi")
    assert not mgr.is_cached(src, "foo")
    mgr.write_cache(src, "foo", "", "@foo():\n  = ok\n")
    assert mgr.is_cached(src, "foo")
    assert not CodegenManager(MockPassingAgent(), use_cache=False).is_cached(src, "foo")


def test_buffer_reporter_captures_emit_and_progress() -> None:
    reporter = BufferReporter()
    reporter.emit("GEPA span 1/2")
    reporter.emit("count: 2", end="")
    reporter.progress_char()
    reporter.emit("done")

    output = reporter.getvalue()
    assert "GEPA span 1/2" in output
    assert "count: 2" in output
    assert "." in output
    assert "done" in output


def test_cli_reporter_writes_plain_text() -> None:
    buffer = StringIO()
    reporter = CliReporter(file=buffer)
    reporter.emit("[green]plain[/green]", end="")
    assert buffer.getvalue() == "[green]plain[/green]"


def test_cli_reporter_quiet_mode_suppresses_output() -> None:
    buffer = StringIO()
    reporter = CliReporter(quiet=True, file=buffer)
    reporter.emit("hidden")
    reporter.progress_char()
    assert buffer.getvalue() == ""


def test_null_reporter_discards_emit_and_progress() -> None:
    reporter = NullReporter()
    reporter.emit("hidden")
    reporter.progress_char()


def test_codegen_translator_covers_signature_type_test_and_impl_shapes() -> None:
    assert kedi_typedef_to_python_class(TypeDef(name="Empty", fields=[])) == (
        "class Empty(BaseModel):\n    pass"
    )
    assert (
        kedi_typedef_to_python_class(
            TypeDef(
                name="Person",
                fields=[
                    TypeField(name="name"),
                    TypeField(name="age", type_annotation="int", default_expr="0"),
                ],
            )
        )
        == "class Person(BaseModel):\n    name: str\n    age: int = 0"
    )

    proc = ProcedureDef(
        name="greet",
        params=["name", "suffix", "times"],
        param_type_annotations={"name": "", "times": "int"},
        param_defaults={"times": "1"},
    )
    assert kedi_procedure_to_python_signature(proc) == (
        "def greet(name: str, suffix: str, times: int = 1) -> str:\n"
        "    '''Available procedure: greet'''\n"
        "    ..."
    )

    assert kedi_signature_to_python_signature("@ping()") == "def ping() -> str"
    assert kedi_signature_to_python_signature("@mix(x, y: int) -> list[str]") == (
        "def mix(x: str, y: int) -> list[str]"
    )
    with pytest.raises(ValueError, match="Invalid Kedi signature"):
        kedi_signature_to_python_signature("not a kedi signature")

    assert python_tests_to_kedi_tests("", "empty") == "@test: empty:"
    assert python_tests_to_kedi_tests("assert run() == 'ok'", "run") == (
        "@test: run:\n  > case: generated:\n    ```\n    assert run() == 'ok'\n    ```"
    )
    structured_tests = python_tests_to_kedi_tests(
        "def test_basic():\n"
        '    """ignored"""\n'
        "    with context():\n"
        "        assert build()\n"
        "\n"
        "def test_second():\n"
        "    '''ignored too'''\n"
        "    assert True\n",
        "build",
    )
    assert "> case: basic:" in structured_tests
    assert "    with context():" in structured_tests
    assert "        assert build()" in structured_tests
    assert "> case: second:" in structured_tests
    assert "'''ignored too'''" not in structured_tests
    assert python_tests_to_kedi_tests("def test_blank():\n", "blank") == (
        "@test: blank:\n  > case: blank:\n    ```\n    ```"
    )

    assert python_impl_to_kedi_impl("", "@f():") == "@f():\n  = ```\n  pass\n  ```"
    assert python_impl_to_kedi_impl("    x = 1\n    return x\n", "@f() -> int:") == (
        "@f() -> int:\n  = ```\n  x = 1\n  return x\n  ```"
    )


def test_codegen_translator_rejects_def_wrapper() -> None:
    # B8: a full `def` wrapper would be embedded verbatim and only defined, never called, so the
    # procedure would silently return None. Reject it so the generator retries.
    with pytest.raises(ValueError, match="function body only"):
        python_impl_to_kedi_impl("def foo(x):\n    return x + 1\n", "@foo(x: int) -> int:")
    with pytest.raises(ValueError, match="function body only"):
        python_impl_to_kedi_impl("async def foo(x):\n    return x\n", "@foo(x: int) -> int:")
    # A def wrapper followed by an unrelated trailing statement is still the silent-None bug: the
    # def is never referenced, so it is defined but never called (the single-statement check missed
    # this; the reference check catches it).
    with pytest.raises(ValueError, match="never called"):
        python_impl_to_kedi_impl("def foo(x):\n    return x + 1\ny = 5\n", "@foo(x: int) -> int:")
    # A body that defines a helper AND calls it (helper name referenced) is a valid body.
    assert python_impl_to_kedi_impl(
        "def helper(v):\n    return v * 2\nreturn helper(x)\n", "@f(x: int) -> int:"
    ) == (
        "@f(x: int) -> int:\n  = ```\n  def helper(v):\n      return v * 2\n  return helper(x)\n  ```"
    )


def test_codegen_signature_parser_keeps_generic_commas() -> None:
    # B11: splitting params on every comma tears `dict[str, int]` apart. Split only top-level commas.
    assert kedi_signature_to_python_signature(
        "@f(data: dict[str, int], n) -> list[tuple[int, str]]"
    ) == ("def f(data: dict[str, int], n: str) -> list[tuple[int, str]]")
    # Defensive: empty segments from a malformed double comma are skipped, not turned into params.
    assert kedi_signature_to_python_signature("@f(x,, y)") == "def f(x: str, y: str) -> str"
    # String-aware: a comma or bracket inside a string-literal default must not mis-split the params.
    assert kedi_signature_to_python_signature('@f(sep: str = ",", x: str = "a]b") -> str') == (
        'def f(sep: str = ",", x: str = "a]b") -> str'
    )


def test_codegen_renders_backtick_python_types_cleanly() -> None:
    # B14: codegen must render the structured type_expr, not leak backticks from a `dict[str,int]`
    # python-type annotation into invalid Python.
    from kedi.lang import parse_program
    from kedi.lang.ast import Program

    prog = parse_program(
        "~City(meta: `dict[str, int]`)\n@get(tags: `list[str]`) -> `dict[str, int]`:\n  = nope\n"
    )
    assert isinstance(prog, Program)
    typedef = next(td for td in prog.type_defs if td.name == "City")
    proc = next(pd for pd in prog.procedures if pd.name == "get")

    assert kedi_typedef_to_python_class(typedef) == (
        "class City(BaseModel):\n    meta: dict[str, int]"
    )
    sig = kedi_procedure_to_python_signature(proc)
    assert "tags: list[str]" in sig
    assert "-> dict[str, int]:" in sig
    assert "`" not in sig  # no leaked backticks


def test_codegen_manager_context_and_cache_parsing(tmp_path: Path) -> None:
    src = tmp_path / "context.kedi"
    src.write_text(
        "```\n"
        "GLOBAL = 1\n"
        "```\n"
        "~Person(name: str)\n"
        "@helper(name: str) -> str:\n"
        "  = <name>\n"
        "@generated() -> str:\n"
        "  > auto:\n"
        "    generate this\n"
        "  = nope\n",
        encoding="utf-8",
    )
    mgr = CodegenManager(MockPassingAgent())
    program = mgr.parse_with_cache(src)
    context = mgr.extract_context_from_ast(program)

    assert context.preamble == "GLOBAL = 1"
    assert any("class Person" in definition for definition in context.type_definitions)
    assert len(context.procedure_definitions) == 1
    assert context.procedure_definitions[0].startswith("def helper(name: str) -> str:")

    mgr.write_cache(src, "cached", "", "@cached():\n  = ok\n")
    merged = mgr.parse_with_cache(src)
    assert {proc.name for proc in merged.procedures} == {
        "helper",
        "generated",
        "cached",
    }


def test_codegen_cleanup_failed_cache_removes_failed_block(tmp_path: Path) -> None:
    src = tmp_path / "source.kedi"
    src.write_text("@ok() -> str:\n  = ok\n", encoding="utf-8")
    cache = src.with_suffix(".cache.kedi")
    mgr = CodegenManager(MockPassingAgent())

    # Two procedures cached; cleaning up the failed one preserves the other.
    mgr.write_cache(src, "keep", "", "@keep():\n  = ok\n")
    mgr.write_cache(src, "drop", "", "@drop():\n  = bad\n")

    mgr.cleanup_failed_cache(src, "drop")

    assert cache.exists()
    text = cache.read_text(encoding="utf-8")
    assert "@keep()" in text
    assert "@drop()" not in text

    # Removing the last remaining block deletes the file entirely.
    mgr.cleanup_failed_cache(src, "keep")
    assert not cache.exists()


def test_codegen_cache_write_is_atomic_and_preserves_blocks(tmp_path: Path) -> None:
    src = tmp_path / "prog.kedi"
    src.write_text("= ok\n", encoding="utf-8")
    mgr = CodegenManager(MockPassingAgent())

    mgr.write_cache(src, "first", "", "@first():\n  = one\n")
    mgr.write_cache(src, "second", "", "@second():\n  = two\n")
    mgr.write_cache(src, "first", "", "@first():\n  = changed\n")

    cache_text = mgr.cache_path_for(src).read_text(encoding="utf-8")
    assert "@first():\n  = changed" in cache_text
    assert "@second():\n  = two" in cache_text
    assert not list(tmp_path.glob(".*.tmp"))


def test_codegen_context_for_generation_falls_back_on_context_extraction_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src = tmp_path / "broken.kedi"
    src.write_text("@broken(:\n", encoding="utf-8")
    mgr = CodegenManager(MockPassingAgent())

    assert mgr.context_for_generation(src) == ProgramContext()
    assert mgr.context_for_generation(tmp_path / "missing.kedi") == ProgramContext()

    def raise_unexpected(_program):
        raise RuntimeError("unexpected")

    valid_src = tmp_path / "valid.kedi"
    valid_src.write_text("@ok() -> str:\n  = ok\n", encoding="utf-8")
    monkeypatch.setattr(mgr, "extract_context_from_ast", raise_unexpected)

    assert mgr.context_for_generation(valid_src) == ProgramContext()


def test_codegen_manager_reports_compile_and_test_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kedi.codegen import manager as manager_module

    src = tmp_path / "bad.kedi"
    src.write_text("@broken(:\n", encoding="utf-8")
    mgr = CodegenManager(MockPassingAgent())

    assert mgr.verify_compilation(src) is not None

    monkeypatch.setattr(
        manager_module,
        "run_tests",
        lambda _: [KediTestRunResult("proc", [KediTestCaseResult("case", False, "bad case")])],
    )
    assert mgr.run_tests_and_get_errors(src) == "bad case"

    monkeypatch.setattr(
        manager_module,
        "run_tests",
        lambda _: [KediTestRunResult("proc", [KediTestCaseResult("case", False, None)])],
    )
    assert mgr.run_tests_and_get_errors(src) == "Test case failed"

    def raise_run_tests(_: str) -> list[KediTestRunResult]:
        raise RuntimeError("runner failed")

    monkeypatch.setattr(manager_module, "run_tests", raise_run_tests)
    assert mgr.run_tests_and_get_errors(src) == "runner failed"


def test_codegen_orchestrator_cached_and_failure_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kedi.codegen import manager as manager_module

    src = tmp_path / "orchestrate.kedi"
    src.write_text("@", encoding="utf-8")
    cached = CodegenManager(MockPassingAgent())
    cached.write_cache(src, "f", "", "@f():\n  = ok\n")
    assert cached.orchestrate_generation(src, "@f():", "prompt", "spec")
    cached.cache_path_for(src).unlink()

    class SequencedAgent(MockPassingAgent):
        def __init__(self) -> None:
            self.test_feedbacks: list[str | None] = []
            self.impl_feedbacks: list[str | None] = []

        def generate_tests(
            self,
            python_signature: str,
            prompt: str,
            context: ProgramContext,
            previous_error: str | None = None,
        ) -> str:
            self.test_feedbacks.append(previous_error)
            return "def test_generated():\n    assert True\n"

        def generate_impl(
            self,
            python_signature: str,
            prompt: str,
            context: ProgramContext,
            python_tests: str,
            previous_error: str | None = None,
        ) -> str:
            self.impl_feedbacks.append(previous_error)
            return "result = 'bad'\nreturn result\n"

    monkeypatch.setattr(
        manager_module,
        "kedi_signature_to_python_signature",
        lambda header: "def f() -> str:",
    )
    monkeypatch.setattr(manager_module, "python_tests_to_kedi_tests", lambda tests, name: tests)
    monkeypatch.setattr(manager_module, "python_impl_to_kedi_impl", lambda impl, header: impl)

    agent = SequencedAgent()
    mgr = CodegenManager(agent, CodegenConfig(max_retries=2))
    verify_results = iter(["test compile failed", None, "impl compile failed", None])
    monkeypatch.setattr(mgr, "verify_compilation", lambda _: next(verify_results))
    monkeypatch.setattr(mgr, "run_tests_and_get_errors", lambda _: "test failed")

    assert not mgr.orchestrate_generation(src, "@f():", "prompt", "spec")
    assert agent.test_feedbacks[0] is None
    assert agent.test_feedbacks[1] is not None
    assert agent.test_feedbacks[1].startswith("Previous Python tests:\n")
    assert agent.impl_feedbacks[0] is None
    impl_feedback = agent.impl_feedbacks[1]
    assert impl_feedback is not None
    assert "Compilation error: impl compile failed" in impl_feedback
    assert not mgr.cache_path_for(src).exists()

    no_cache_agent = SequencedAgent()
    no_cache_mgr = CodegenManager(no_cache_agent, CodegenConfig(max_retries=1))
    monkeypatch.setattr(no_cache_mgr, "write_cache", lambda *args: None)
    monkeypatch.setattr(no_cache_mgr, "verify_compilation", lambda _: "compile failed")
    assert not no_cache_mgr.orchestrate_generation(src, "@f():", "prompt", "spec")
    assert not no_cache_mgr.cache_path_for(src).exists()


def test_codegen_orchestration_retries_on_def_wrapper_translation(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # B8 wiring: a `def`-wrapper impl raises in translation; the manager must feed the rejection
    # back and retry, never caching a broken (None-returning) procedure.
    from kedi.codegen import manager as manager_module

    src = tmp_path / "p.kedi"
    src.write_text("= ok\n", encoding="utf-8")

    class DefWrapperAgent(MockPassingAgent):
        def __init__(self) -> None:
            self.impl_feedbacks: list[str | None] = []

        def generate_tests(
            self,
            python_signature: str,
            prompt: str,
            context: ProgramContext,
            previous_error: str | None = None,
        ) -> str:
            return "def test_generated():\n    assert True\n"

        def generate_impl(
            self,
            python_signature: str,
            prompt: str,
            context: ProgramContext,
            python_tests: str,
            previous_error: str | None = None,
        ) -> str:
            self.impl_feedbacks.append(previous_error)
            return "def f():\n    return 'x'\n"  # a def-wrapper → rejected by the real translator

    monkeypatch.setattr(
        manager_module,
        "kedi_signature_to_python_signature",
        lambda header: "def f() -> str:",
    )
    monkeypatch.setattr(manager_module, "python_tests_to_kedi_tests", lambda tests, name: tests)
    # Intentionally do NOT patch python_impl_to_kedi_impl — we want the real def-wrapper rejection.

    agent = DefWrapperAgent()
    mgr = CodegenManager(agent, CodegenConfig(max_retries=2))
    # Tests "compile" so the impl loop is reached; the impl then fails in translation.
    monkeypatch.setattr(mgr, "verify_compilation", lambda _: None)
    monkeypatch.setattr(mgr, "run_tests_and_get_errors", lambda _: None)
    assert not mgr.orchestrate_generation(src, "@f():", "prompt", "spec")
    # A later attempt receives the translation rejection as feedback (and nothing was cached).
    assert any(fb and "Translation error" in fb for fb in agent.impl_feedbacks)
    assert not mgr.cache_path_for(src).exists()


def test_codegen_orchestrator_happy_path(tmp_path: Path) -> None:
    src = tmp_path / "g.kedi"
    # Base program contains a header we want to implement
    src.write_text(
        "@f():\n  > auto: \n    implement a function returning ok\n\n= <f()>\n",
        encoding="utf-8",
    )
    mgr = CodegenManager(MockPassingAgent())
    ok = mgr.orchestrate_generation(src, "@f():", "implement", "spec")
    assert ok
    # Now tests should pass using discovery (includes .cache)
    results = run_tests(str(src))
    assert results and all(c.success for s in results for c in s.cases)


def test_codegen_orchestrator_success_with_no_cache_removes_generated_cache(
    tmp_path: Path,
) -> None:
    src = tmp_path / "g.kedi"
    src.write_text(
        "@f():\n  > auto: \n    implement a function returning ok\n\n= <f()>\n",
        encoding="utf-8",
    )
    mgr = CodegenManager(MockPassingAgent(), use_cache=False)

    assert mgr.orchestrate_generation(src, "@f():", "implement", "spec")
    assert not (tmp_path / "g.cache.kedi").exists()


def test_cli_like_codegen_runner(tmp_path: Path) -> None:
    src = tmp_path / "h.kedi"
    src.write_text(
        "@f():\n  > auto:\n    implement f returning ok\n\n= <f()>\n",
        encoding="utf-8",
    )

    os.environ["KEDI_CODEGEN_AGENT"] = "mock"
    sys.argv = [str(src)]
    try:
        cli_main.main(sys.argv)
    except SystemExit as e:
        assert e.code == 0


def test_codegen_prompt_builders_render_empty_and_full_contexts() -> None:
    empty = ProgramContext()
    assert build_context_section(empty) == (
        "AVAILABLE CONTEXT (use ONLY these):\n\n"
        "# Available types: (none)\n\n"
        "# Available functions: (none)"
    )

    context = ProgramContext(
        preamble="GLOBAL = 1",
        type_definitions=["class City: ...", "class Country: ..."],
        procedure_definitions=["def helper(name: str) -> str: ..."],
    )
    section = build_context_section(context)
    assert "# Preamble code:\nGLOBAL = 1" in section
    assert "class City: ...\n\nclass Country: ..." in section
    assert "def helper(name: str) -> str: ..." in section

    test_prompt = build_test_generation_prompt(
        "def greet(name: str) -> str:",
        "Greet a user.",
        context,
        previous_error="NameError: missing",
    )
    assert "Write pytest-style test code" in test_prompt
    assert "Greet a user." in test_prompt
    assert "Previous attempt failed with:\nNameError: missing" in test_prompt
    assert "try/except blocks, NOT pytest.raises" in test_prompt

    impl_prompt = build_implementation_prompt(
        "def greet(name: str) -> str:",
        "Greet a user.",
        context,
        "def test_greet():\n    assert greet('Ada')",
        previous_error="AssertionError",
    )
    assert "# Tests that must pass:" in impl_prompt
    assert "Provide ONLY the function body" in impl_prompt
    assert "Previous attempt failed with:\nAssertionError" in impl_prompt

    assert "Previous attempt failed" not in build_test_generation_prompt(
        "def greet(name: str) -> str:",
        "Greet a user.",
        context,
    )
    assert "Previous attempt failed" not in build_implementation_prompt(
        "def greet(name: str) -> str:",
        "Greet a user.",
        context,
        "def test_greet():\n    assert greet('Ada')",
    )


def test_mock_passing_agent_generates_tests_for_signature_shapes() -> None:
    agent = MockPassingAgent()
    context = ProgramContext()

    assert agent.generate_tests("not a python signature", "", context) == (
        "def test_basic():\n    assert True\n"
    )

    no_param_tests = agent.generate_tests("def ping() -> str:", "", context)
    assert "result = ping()" in no_param_tests
    assert "isinstance(result, str)" in no_param_tests

    param_tests = agent.generate_tests(
        "def build(items: list[str], count: int, ratio: float, enabled: bool, name: str, raw) -> str:",
        "",
        context,
    )
    assert "build(['test'], 1, 1, True, 'test', 'test')" in param_tests
    assert "build([], 0, 0, False, '', '')" in param_tests


def test_mock_passing_agent_generates_implementations_by_return_type() -> None:
    agent = MockPassingAgent()
    context = ProgramContext()

    assert agent.generate_impl("not a python signature", "", context, "") == (
        "result = 'ok'\nreturn result\n"
    )
    assert "test result" in agent.generate_impl("def f() -> str:", "", context, "")
    assert "result = 0" in agent.generate_impl("def f() -> int:", "", context, "")
    assert "result = True" in agent.generate_impl("def f() -> bool:", "", context, "")
    assert "result = []" in agent.generate_impl("def f() -> list[object]:", "", context, "")
    assert "result = 'ok'" in agent.generate_impl("def f() -> dict:", "", context, "")
    assert "result = 'ok'" in agent.generate_impl("def f():", "", context, "")


def test_pydantic_ai_agent_uses_configured_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kedi.codegen import pydantic_ai_agent as agent_module

    class FakeAdapter:
        instances: list["FakeAdapter"] = []

        def __class_getitem__(cls, item: object) -> type["FakeAdapter"]:
            return cls

        def __init__(self, *, model: str) -> None:
            self.model = model
            self.templates: list[str] = []
            FakeAdapter.instances.append(self)

        def produce_sync(self, *, template: str, output_type: type[Any]) -> Any:
            self.templates.append(template)
            if output_type is agent_module.PythonTestOutput:
                return agent_module.PythonTestOutput(
                    test_code="def test_generated():\n    assert True"
                )
            return agent_module.PythonImplOutput(implementation="result = 1\nreturn result")

    monkeypatch.setattr(agent_module, "PydanticAdapter", FakeAdapter)
    monkeypatch.setenv("MODEL_NAME", "env/model")

    agent = agent_module.PydanticAIAgent()

    assert agent.model == "env/model"
    assert agent.generate_tests("def f() -> int:", "make tests", ProgramContext()) == (
        "def test_generated():\n    assert True"
    )
    assert agent.generate_impl("def f() -> int:", "implement", ProgramContext(), "tests") == (
        "result = 1\nreturn result"
    )
    assert FakeAdapter.instances[-1].model == "env/model"
    assert "Write pytest-style test code" in FakeAdapter.instances[-1].templates[0]
    assert "Provide ONLY the function body" in FakeAdapter.instances[-1].templates[1]

    explicit = agent_module.PydanticAIAgent(model="explicit/model")
    assert explicit.model == "explicit/model"
    assert FakeAdapter.instances[-1].model == "explicit/model"


@pytest.mark.parametrize("async_host", [False, True])
@pytest.mark.parametrize("fail_implementation", [False, True])
def test_codegen_keeps_one_loop_for_tests_and_implementation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    async_host: bool,
    fail_implementation: bool,
) -> None:
    from kedi.agent_adapter.execution import run_sync
    from kedi.codegen import runner as runner_module

    loops: list[asyncio.AbstractEventLoop] = []

    async def check_loop() -> None:
        loop = asyncio.get_running_loop()
        if loops:
            assert loop is loops[0], "codegen reused its adapter on another loop"
        loops.append(loop)

    class LoopBoundAgent(MockPassingAgent):
        def generate_tests(self, *args: Any, **kwargs: Any) -> str:
            run_sync(check_loop())
            return super().generate_tests(*args, **kwargs)

        def generate_impl(self, *args: Any, **kwargs: Any) -> str:
            run_sync(check_loop())
            if fail_implementation:
                raise RuntimeError("implementation request failed")
            return super().generate_impl(*args, **kwargs)

    monkeypatch.setattr(runner_module, "_select_agent", lambda kind, model: LoopBoundAgent())
    source = tmp_path / "loop.kedi"
    source.write_text("@f() -> str:\n  > auto:\n    Return ok.\n", encoding="utf-8")

    def generate() -> bool:
        return runner_module.run_codegen(source, quiet=True, use_cache=False)

    async def async_generate() -> bool:
        return generate()

    if fail_implementation:
        with pytest.raises(RuntimeError, match="implementation request failed"):
            asyncio.run(async_generate()) if async_host else generate()
    else:
        assert asyncio.run(async_generate()) if async_host else generate()
    assert len(loops) == 2
    assert loops[0].is_closed()
    assert not source.with_suffix(".cache.kedi").exists()


def test_codegen_runner_selects_agents_reconstructs_headers_and_runs_auto_specs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kedi.codegen import runner as runner_module

    class FakePydanticAgent:
        def __init__(self, model: str | None) -> None:
            self.model = model

    calls: list[tuple[Path, str, str, str, bool, int]] = []

    class FakeManager:
        def __init__(
            self,
            agent: object,
            config: object,
            *,
            use_cache: bool,
            reporter: object | None = None,
        ) -> None:
            self.agent = agent
            self.config = config
            self.use_cache = use_cache
            self.reporter = reporter

        def orchestrate_generation(
            self,
            source_path: Path,
            header: str,
            prompt: str,
            spec: str,
        ) -> bool:
            calls.append(
                (
                    source_path,
                    header,
                    prompt,
                    spec,
                    self.use_cache,
                    getattr(self.config, "max_retries"),
                )
            )
            return "fail" not in prompt

    monkeypatch.setattr(runner_module, "PydanticAIAgent", FakePydanticAgent)
    monkeypatch.setattr(runner_module, "CodegenManager", FakeManager)

    assert isinstance(runner_module._select_agent("mock", None), MockPassingAgent)
    selected = runner_module._select_agent("", "model/name")
    assert isinstance(selected, FakePydanticAgent)
    assert selected.model == "model/name"
    assert runner_module._reconstruct_header(ProcedureDef(name="plain", params=["x"])) == (
        "@plain(x):"
    )

    src = tmp_path / "auto.kedi"
    src.write_text(
        "@plain(x):\n"
        "  = <x>\n\n"
        "@build(name: str, raw) -> str:\n"
        "  > auto:\n"
        "    implement build\n"
        "  = ok\n\n"
        "@broken() -> str:\n"
        "  > auto:\n"
        "    fail build\n"
        "  = no\n",
        encoding="utf-8",
    )

    assert not runner_module.run_codegen(
        src,
        agent_kind="mock",
        max_retries=2,
        use_cache=False,
    )
    assert calls == [
        (
            src,
            "@build(name: str, raw) -> str:",
            "implement build",
            "",
            False,
            2,
        ),
        (
            src,
            "@broken() -> str:",
            "fail build",
            "",
            False,
            2,
        ),
    ]
