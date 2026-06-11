from __future__ import annotations

import pytest

from kedi.agent_profile import AgentProfile
from kedi.lang import compile_program, parse_program
from kedi.lang.ast import (
    ModelDirectiveStmt,
    ProcedureDef,
    ProfileDefStmt,
    SystemDirectiveStmt,
    UseDirectiveStmt,
)
from kedi.lang.parser.diagnostics import KediParseError
from kedi.utils import AttrDict
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
> system: top-system
> profile: light:
    > model: opus
    > system: Be concise.
> use: light
@foo():
    >> answer [x]
"""
    prog = parse_program(src)
    assert len(prog.ordered_toplevel) == 5
    assert isinstance(prog.ordered_toplevel[0], ModelDirectiveStmt)
    assert prog.ordered_toplevel[0].plain_value == "haiku"
    assert isinstance(prog.ordered_toplevel[1], SystemDirectiveStmt)
    assert prog.ordered_toplevel[1].plain_value == "top-system"
    assert isinstance(prog.ordered_toplevel[2], ProfileDefStmt)
    assert prog.ordered_toplevel[2].name == "light"
    assert prog.ordered_toplevel[2].system_members[0].plain_value == "Be concise."
    assert isinstance(prog.ordered_toplevel[3], UseDirectiveStmt)
    assert isinstance(prog.ordered_toplevel[4], ProcedureDef)


def test_profile_docstring_comes_from_first_block_comment_only() -> None:
    src = """
> profile: documented:
    ###
    Use this profile for short answers.
    ###
    > model: opus

> profile: ordinary_comment:
    > model: haiku
    ###
    This is a normal comment, not metadata.
    ###
"""
    prog = parse_program(src)

    documented = prog.profile_defs["documented"]
    assert documented.docstring == "Use this profile for short answers."
    assert documented.docstring_loc is not None
    assert documented.model_members[0].plain_value == "opus"

    ordinary = prog.profile_defs["ordinary_comment"]
    assert ordinary.docstring is None
    assert ordinary.docstring_loc is None
    assert ordinary.model_members[0].plain_value == "haiku"


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


def test_procedure_capture_uses_source_order_system() -> None:
    src = """
> system: first-system
@foo():
    >> value [x]
> system: second-system
@bar():
    >> value [x]
"""
    prog = parse_program(src)
    adapter = _adapter()
    rt = compile_program(prog, adapter=adapter)
    foo_def = _procedure_def(rt._procedures["foo"])
    bar_def = _procedure_def(rt._procedures["bar"])
    assert foo_def.captured_agent_profile.system == "first-system"
    assert bar_def.captured_agent_profile.system == "second-system"


def test_runtime_globals_are_available_during_profile_capture() -> None:
    src = """
> profile: personalized:
    > system:
        Address the user as <`args.name or "Guest"`>.

> use: personalized
@foo():
    >> value [x]
"""
    rt = compile_program(
        parse_program(src),
        adapter=_adapter(),
        runtime_globals={"args": AttrDict({"name": "Ada"}, default=None)},
    )
    foo_def = _procedure_def(rt._procedures["foo"])
    assert foo_def.captured_agent_profile.system == "Address the user as Ada."


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
    assert adapter.instruction_calls == [None]


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


def test_block_state_system_within_procedure_body() -> None:
    src = """
@foo():
    >> first [x]
    > system: Answer tersely.
    >> second [y]
"""
    adapter = _adapter()
    rt = compile_program(parse_program(src), adapter=adapter)
    rt._invoke_procedure(rt._procedures["foo"], [])
    assert adapter.instruction_calls == [None, "Answer tersely."]


def test_block_state_system_renders_multiline_substitutions() -> None:
    src = """
@foo(style):
    > system:
        You are <style>.
        Keep it short.
    >> answer [x]
"""
    adapter = _adapter()
    rt = compile_program(parse_program(src), adapter=adapter)
    rt._invoke_procedure(rt._procedures["foo"], ["direct"])
    assert adapter.instruction_calls == ["You are direct.\nKeep it short."]


def test_top_level_system_applies_to_following_prompt() -> None:
    src = """
> system: Top-level instructions.
>> answer [x]
"""
    adapter = _adapter()
    rt = compile_program(parse_program(src), adapter=adapter)
    rt.run_main()
    assert adapter.instruction_calls == ["Top-level instructions."]


def test_top_level_system_renders_multiline_substitutions() -> None:
    src = """
[style: str] = `"concise"`
> system:
    You are <style>.
    No extra text.
>> answer [x]
"""
    adapter = _adapter()
    rt = compile_program(parse_program(src), adapter=adapter)
    rt.run_main()
    assert adapter.instruction_calls == ["You are concise.\nNo extra text."]


def test_top_level_system_renders_inline_python_substitution() -> None:
    src = """
```
from types import SimpleNamespace
args = SimpleNamespace(name="Mert")
```
> system:
    Your name is <`args.name`>.
>> answer [x]
"""
    adapter = _adapter()
    rt = compile_program(parse_program(src), adapter=adapter)
    rt.run_main()
    assert adapter.instruction_calls == ["Your name is Mert."]


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


def test_use_profile_applies_system_at_capture_time() -> None:
    src = """
> profile: dynamic:
    > system: from-profile-system
> use: dynamic
@foo():
    >> value [x]
"""
    adapter = _adapter()
    rt = compile_program(parse_program(src), adapter=adapter)
    foo_def = _procedure_def(rt._procedures["foo"])
    assert foo_def.captured_agent_profile.system == "from-profile-system"
    rt._invoke_procedure(rt._procedures["foo"], [])
    assert adapter.instruction_calls == ["from-profile-system"]


def test_agent_profile_merge() -> None:
    base = AgentProfile(model="a", system="old")
    merged = base.merge(AgentProfile(model="b", system="new"))
    assert merged.model == "b"
    assert merged.system == "new"
    assert base.model == "a"
    assert base.system == "old"
