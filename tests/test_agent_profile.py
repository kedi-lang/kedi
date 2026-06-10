from __future__ import annotations

import pytest

from kedi.agent_profile import AgentProfile
from kedi.lang import compile_program, parse_program
from kedi.lang.ast import ModelDirectiveStmt, ProcedureDef, ProfileDefStmt, UseProfileStmt
from kedi.lang.parser.diagnostics import KediParseError
from tests.mock_adapter import MockAdapter


def _adapter() -> MockAdapter:
    return MockAdapter(default_model="default-model")


def _procedure_def(fn: object) -> ProcedureDef:
    proc = getattr(fn, "__kedi_procedure_def__", None)
    assert isinstance(proc, ProcedureDef)
    return proc


def test_parse_model_profile_use_directives() -> None:
    src = """
> model: haiku
> profile: light:
    > model: opus
> use: light
@foo():
    >> answer [x]
"""
    prog = parse_program(src)
    assert len(prog.ordered_toplevel) == 4
    assert isinstance(prog.ordered_toplevel[0], ModelDirectiveStmt)
    assert prog.ordered_toplevel[0].plain_value == "haiku"
    assert isinstance(prog.ordered_toplevel[1], ProfileDefStmt)
    assert prog.ordered_toplevel[1].name == "light"
    assert isinstance(prog.ordered_toplevel[2], UseProfileStmt)
    assert isinstance(prog.ordered_toplevel[3], ProcedureDef)


def test_profile_definition_top_level_only() -> None:
    src = """
@outer():
    > profile: inner:
        > model: opus
"""
    with pytest.raises(KediParseError):
        parse_program(src)


def test_procedure_capture_uses_source_order_model() -> None:
    src = """
> model: gpt-captured
@foo():
    >> value [x]
> model: gpt-later
@bar():
    >> value [x]
"""
    prog = parse_program(src)
    adapter = _adapter()
    rt = compile_program(prog, adapter=adapter)
    foo = rt._procedures["foo"]
    bar = rt._procedures["bar"]
    foo_def = _procedure_def(foo)
    bar_def = _procedure_def(bar)
    assert foo_def.captured_agent_profile.model == "gpt-captured"
    assert bar_def.captured_agent_profile.model == "gpt-later"


def test_call_site_model_does_not_override_captured_procedure_profile() -> None:
    src = """
> model: proc-model
@foo():
    >> value [x]

> model: call-site-model
= `foo()`
"""
    adapter = _adapter()
    rt = compile_program(parse_program(src), adapter=adapter)
    rt.run_main()
    assert adapter.model_calls == ["proc-model"]


def test_block_state_model_within_procedure_body() -> None:
    src = """
@foo():
    >> first [x]
    > model: inner-model
    >> second [y]
"""
    adapter = _adapter()
    rt = compile_program(parse_program(src), adapter=adapter)
    rt._invoke_procedure(rt._procedures["foo"], [])
    assert adapter.model_calls == ["default-model", "inner-model"]


def test_use_profile_applies_model_at_capture_time() -> None:
    src = """
> profile: dynamic:
    > model: from-profile
> use: dynamic
@foo():
    >> value [x]
"""
    adapter = _adapter()
    rt = compile_program(parse_program(src), adapter=adapter)
    foo_def = _procedure_def(rt._procedures["foo"])
    assert foo_def.captured_agent_profile.model == "from-profile"


def test_agent_profile_merge() -> None:
    base = AgentProfile(model="a")
    merged = base.merge(AgentProfile(model="b"))
    assert merged.model == "b"
    assert base.model == "a"
