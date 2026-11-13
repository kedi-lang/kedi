from __future__ import annotations

from typing import Any, cast

import pytest

import kedi.agent_adapter.registry as adapter_registry
from kedi.agent_adapter.meta import AdapterCapabilities, AgentAdapterKind
from kedi.agent_adapter.registry import (
    AdapterFactory,
    adapter_capabilities,
    adapter_factory,
    build_adapter,
    known_adapter_shortnames,
)
from kedi.agent_adapter.request import KediTemplatePromptBuilder
from kedi.agent_profile import (
    A2AConnection,
    AgentProfile,
    AgentProfileOverrideUnsupported,
    CodeModeSettings,
    normalize_codemode_settings,
)
from kedi.agent_profile.directives import (
    eval_agent_command_value,
    eval_agent_directive,
    eval_mcp_field_value,
    eval_settings_field_value,
)
from kedi.agent_profile.mcp import McpServerSpec, mcp_spec_from_mapping
from kedi.agent_profile.settings import parse_plain_setting_value
from kedi.artifacts import artifact_instructions
from kedi.core import KediRuntime
from kedi.lang import compile_program, parse_program
from kedi.lang.ast import (
    AdapterDirectiveStmt,
    AgentDirectiveStmt,
    CodeModeDirectiveStmt,
    EffortDirectiveStmt,
    McpDirectiveStmt,
    ModelDirectiveStmt,
    ProcedureDef,
    ProfileDefStmt,
    SettingsDirectiveStmt,
    SettingsField,
    SystemDirectiveStmt,
    UseDirectiveStmt,
)
from kedi.lang.parser.diagnostics import KediParseError
from kedi.utils import AttrDict
from tests.mock_adapter import MockAdapter

TEMPLATE_INSTRUCTIONS = (
    KediTemplatePromptBuilder(template="", output_fields=()).instruction_parts()[0].content
)


def _adapter() -> MockAdapter:
    return MockAdapter(default_model="default-model")


def _procedure_def(fn: object) -> ProcedureDef:
    proc = getattr(fn, "__kedi_procedure_def__", None)
    assert isinstance(proc, ProcedureDef)
    return proc


def _patch_adapter_factory(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    kind: AgentAdapterKind,
    adapter: MockAdapter,
) -> None:
    adapter.shortname = name
    adapter.kind = kind
    monkeypatch.setitem(
        adapter_registry._ADAPTERS,
        name,
        AdapterFactory(
            shortname=name,
            kind=kind,
            build=lambda _model, _kwargs: adapter,
        ),
    )


def test_parse_model_profile_use_directives() -> None:
    src = """
> agent: codex
> adapter: pydantic
> model: haiku
> effort: low
> instructions: top-system
> profile: light:
    > agent: codex
    > model: opus
    > effort: high
    > instructions: Be concise.
> use: light
@foo():
    >> answer [x]
"""
    prog = parse_program(src)
    assert len(prog.ordered_toplevel) == 8
    assert isinstance(prog.ordered_toplevel[0], AgentDirectiveStmt)
    assert prog.ordered_toplevel[0].plain_value == "codex"
    assert isinstance(prog.ordered_toplevel[1], AdapterDirectiveStmt)
    assert prog.ordered_toplevel[1].plain_value == "pydantic"
    assert isinstance(prog.ordered_toplevel[2], ModelDirectiveStmt)
    assert prog.ordered_toplevel[2].plain_value == "haiku"
    assert isinstance(prog.ordered_toplevel[3], EffortDirectiveStmt)
    assert prog.ordered_toplevel[3].plain_value == "low"
    assert isinstance(prog.ordered_toplevel[4], SystemDirectiveStmt)
    assert prog.ordered_toplevel[4].plain_value == "top-system"
    assert isinstance(prog.ordered_toplevel[5], ProfileDefStmt)
    assert prog.ordered_toplevel[5].name == "light"
    assert prog.ordered_toplevel[5].agent_members[0].plain_value == "codex"
    assert prog.ordered_toplevel[5].effort_members[0].plain_value == "high"
    assert prog.ordered_toplevel[5].system_members[0].plain_value == "Be concise."
    assert isinstance(prog.ordered_toplevel[6], UseDirectiveStmt)
    assert isinstance(prog.ordered_toplevel[7], ProcedureDef)


def test_parse_multiline_agent_directive_with_command() -> None:
    src = """
> agent:
    acp: uv run acp server

> profile: local:
    > agent:
        acp: `command`
"""
    prog = parse_program(src)

    agent = prog.ordered_toplevel[0]
    assert isinstance(agent, AgentDirectiveStmt)
    assert agent.plain_value == "acp"
    assert agent.command_plain_value == "uv run acp server"

    profile_agent = prog.profile_defs["local"].agent_members[0]
    assert profile_agent.plain_value == "acp"
    assert profile_agent.command_backtick_code == "command"


def test_eval_agent_directive_carries_agent_command() -> None:
    stmt = AgentDirectiveStmt(plain_value="acp", command_plain_value="uv run acp server")
    profile = eval_agent_directive(KediRuntime(), stmt, {})

    assert profile.adapter_shortname == "acp"
    assert profile.agent_command == "uv run acp server"


def test_eval_agent_directive_accepts_inline_sequence_command() -> None:
    stmt = AgentDirectiveStmt(
        plain_value="acp",
        command_backtick_code='["uv", "run", "acp", "server"]',
    )
    profile = eval_agent_directive(KediRuntime(), stmt, {})

    assert profile.adapter_shortname == "acp"
    assert profile.agent_command == ("uv", "run", "acp", "server")


def test_eval_agent_command_value_accepts_string_and_rejects_invalid_values() -> None:
    rt = KediRuntime()

    assert (
        eval_agent_command_value(
            rt,
            directive_name="agent command",
            plain_value=None,
            backtick_code='"uv run acp"',
            env={},
        )
        == "uv run acp"
    )
    with pytest.raises(TypeError, match="str or sequence"):
        eval_agent_command_value(
            rt,
            directive_name="agent command",
            plain_value=None,
            backtick_code="[object()]",
            env={},
        )
    with pytest.raises(TypeError, match="str or sequence"):
        eval_agent_command_value(
            rt,
            directive_name="agent command",
            plain_value=None,
            backtick_code="1",
            env={},
        )
    with pytest.raises(TypeError, match="str or sequence"):
        eval_agent_command_value(
            rt,
            directive_name="agent command",
            plain_value=None,
            backtick_code='["uv", 1]',
            env={},
        )
    with pytest.raises(ValueError, match="no value"):
        eval_agent_command_value(
            rt,
            directive_name="agent command",
            plain_value=None,
            backtick_code=None,
            env={},
        )


def test_agent_profile_merge_carries_agent_command() -> None:
    merged = AgentProfile.empty().merge(AgentProfile(agent_command=("uv", "run", "acp")))

    assert merged.agent_command == ("uv", "run", "acp")


def test_agent_profile_merge_replaces_standalone_transport_atomically() -> None:
    connection = A2AConnection(endpoint="http://localhost:8000")
    remote = AgentProfile(agent_command="old").merge(AgentProfile(a2a_connection=connection))
    local = remote.merge(AgentProfile(agent_command=("uv", "run", "acp")))

    assert remote.agent_command is None
    assert remote.a2a_connection == connection
    assert local.agent_command == ("uv", "run", "acp")
    assert local.a2a_connection is None


def test_parse_mcp_directives_top_level_profile_and_procedure() -> None:
    src = """
> mcp:
    command: vsh
    args: `["run", "--mcp"]`

> profile: filesystem:
    > mcp:
        command: fs-mcp

@foo():
    > mcp:
        command: local-mcp
    >> value [x]
"""
    prog = parse_program(src)

    assert isinstance(prog.ordered_toplevel[0], McpDirectiveStmt)
    assert prog.ordered_toplevel[0].fields[0].name == "command"
    assert prog.ordered_toplevel[0].fields[0].plain_value == "vsh"
    assert prog.ordered_toplevel[0].fields[1].backtick_code == '["run", "--mcp"]'

    profile = prog.profile_defs["filesystem"]
    assert profile.mcp_members[0].fields[0].plain_value == "fs-mcp"
    assert isinstance(prog.procedures[0].body[0], McpDirectiveStmt)


def test_parse_settings_directives_top_level_profile_and_procedure() -> None:
    src = """
> settings:
    temperature: 0.2
    max_tokens: 128
    stop_sequences: `["END"]`

> profile: precise:
    > settings:
        parallel_tool_calls: false

@foo():
    > settings:
        max_retries: 1
    >> value [x]
"""
    prog = parse_program(src)

    assert isinstance(prog.ordered_toplevel[0], SettingsDirectiveStmt)
    assert [(field.name, field.plain_value) for field in prog.ordered_toplevel[0].fields] == [
        ("temperature", "0.2"),
        ("max_tokens", "128"),
        ("stop_sequences", None),
    ]
    assert prog.ordered_toplevel[0].fields[2].backtick_code == '["END"]'
    profile = prog.profile_defs["precise"]
    assert profile.settings_members[0].fields[0].name == "parallel_tool_calls"
    assert profile.settings_members[0].fields[0].plain_value == "false"
    assert isinstance(prog.procedures[0].body[0], SettingsDirectiveStmt)
    assert prog.procedures[0].body[0].fields[0].name == "max_retries"


def test_parse_codemode_directive_short_and_expanded_forms() -> None:
    prog = parse_program(
        """
> codemode: enabled
> profile: bounded:
    > codemode:
        max_nested_calls: 12
        max_concurrent_calls: `4`
@inspect():
    > codemode: disabled
    = done
"""
    )

    top = prog.ordered_toplevel[0]
    assert isinstance(top, CodeModeDirectiveStmt)
    assert top.enabled is True
    bounded = prog.profile_defs["bounded"].codemode_members[0]
    assert bounded.enabled is True
    assert [(field.name, field.plain_value, field.backtick_code) for field in bounded.fields] == [
        ("max_nested_calls", "12", None),
        ("max_concurrent_calls", None, "4"),
    ]
    procedure = prog.procedures[0]
    assert isinstance(procedure.body[0], CodeModeDirectiveStmt)
    assert procedure.body[0].enabled is False


def test_codemode_directive_captures_typed_limits_and_rejects_settings_alias() -> None:
    source = """
> codemode:
    default_search_limit: 4
    max_search_limit: 20
    max_nested_calls: 12
    request_timeout: 15.5
@inspect():
    >> inspect [answer]
"""
    rt = compile_program(parse_program(source), adapter=_adapter())
    profile = _procedure_def(rt._procedures["inspect"]).captured_agent_profile

    assert profile.codemode is not None
    assert profile.codemode.enabled is True
    assert profile.codemode.default_search_limit == 4
    assert profile.codemode.max_search_limit == 20
    assert profile.codemode.max_nested_calls == 12
    assert profile.codemode.request_timeout == 15.5
    with pytest.raises(KediParseError, match="model setting must be one of"):
        parse_program("> settings:\n    codemode: true\n")


def test_codemode_profile_and_lexical_procedure_execution() -> None:
    source = """
> codemode: enabled
> profile: bounded:
    > codemode:
        enabled: false
        max_nested_calls: 3
> use: bounded

@local() -> int:
    > codemode: disabled
    = `1`

= `local()`
"""
    rt = compile_program(parse_program(source), adapter=_adapter())

    assert rt.run_main() == 1
    profile = _procedure_def(rt._procedures["local"]).captured_agent_profile
    assert profile.codemode == CodeModeSettings(enabled=False, max_nested_calls=3)


def test_codemode_preload_tools_parse_single_and_block_forms() -> None:
    source = """
> profile: release:
    > codemode:
        preload_tools:
            lookup_release
            list-versions
            lookup_release
> use: release

@local() -> int:
    > codemode:
        preload_tools: lookup_release
    = `1`

= `local()`
"""
    rt = compile_program(parse_program(source), adapter=_adapter())

    assert rt.run_main() == 1
    release_profile = rt._evaluate_profile_definition(rt._profile_registry["release"])
    assert release_profile.codemode == CodeModeSettings(
        preload_tools=("lookup_release", "list-versions")
    )
    profile = _procedure_def(rt._procedures["local"]).captured_agent_profile
    assert profile.codemode == CodeModeSettings(preload_tools=("lookup_release", "list-versions"))


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("> codemode:\n    # comment\n    unknown: 1\n", "codemode setting must be one of"),
        ("> codemode:\n    max_nested_calls: 2\n    max_nested_calls: 3\n", "Duplicate"),
        ("> codemode:\n    enabled: maybe\n", "must be 'true' or 'false'"),
    ],
)
def test_codemode_directive_rejects_invalid_expanded_settings(
    source: str,
    message: str,
) -> None:
    with pytest.raises(KediParseError, match=message):
        parse_program(source)


@pytest.mark.parametrize("value", ["on", "true", "maybe"])
def test_codemode_directive_rejects_noncanonical_short_values(value: str) -> None:
    with pytest.raises(KediParseError, match="codemode directive must be"):
        parse_program(f"> codemode: {value}\n")


def test_codemode_settings_validate_activation_and_limits() -> None:
    configured = CodeModeSettings(
        enabled=False,
        default_search_limit=2,
        max_search_limit=4,
        request_timeout=2.5,
    )

    assert normalize_codemode_settings(None) is None
    assert normalize_codemode_settings(configured) is configured
    assert normalize_codemode_settings(True) == CodeModeSettings(enabled=True)
    assert normalize_codemode_settings({"max_nested_calls": 2}) == CodeModeSettings(
        max_nested_calls=2
    )
    assert normalize_codemode_settings({"preload_tools": "lookup"}) == CodeModeSettings(
        preload_tools=("lookup",)
    )
    assert CodeModeSettings(preload_tools=["lookup", "list", "lookup"]).preload_tools == (
        "lookup",
        "list",
    )
    assert configured.limit_values() == {
        "default_search_limit": 2,
        "max_search_limit": 4,
        "max_hydrated_tools": 64,
        "max_discovery_result_bytes": 256_000,
        "max_code_chars": 20_000,
        "max_nested_calls": 64,
        "max_concurrent_calls": 8,
        "max_tool_result_bytes": 256_000,
        "max_total_tool_result_bytes": 1_000_000,
        "max_print_bytes": 256_000,
        "request_timeout": 2.5,
    }

    with pytest.raises(TypeError, match="enabled must be a boolean"):
        CodeModeSettings(enabled=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="integer settings must be integers"):
        CodeModeSettings(max_nested_calls=True)
    with pytest.raises(ValueError, match="limits must be positive"):
        CodeModeSettings(max_nested_calls=0)
    with pytest.raises(TypeError, match="request_timeout must be a number"):
        CodeModeSettings(request_timeout="slow")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="request_timeout must be positive"):
        CodeModeSettings(request_timeout=0)
    with pytest.raises(ValueError, match="cannot exceed"):
        CodeModeSettings(default_search_limit=2, max_search_limit=1)
    with pytest.raises(ValueError, match="unknown codemode settings"):
        normalize_codemode_settings({"unknown": 1})
    with pytest.raises(TypeError, match="sequence of tool names"):
        CodeModeSettings(preload_tools=42)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="entries must be strings"):
        CodeModeSettings(preload_tools=["lookup", 1])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="cannot be empty"):
        CodeModeSettings(preload_tools=[""])


def test_runtime_rejects_enabled_codemode_on_unsupported_adapter() -> None:
    from kedi.agent_adapter.adapters.pydantic import PydanticAdapter

    adapter = _adapter()

    with pytest.raises(AgentProfileOverrideUnsupported, match="CodeMode is not supported"):
        KediRuntime._assert_profile_override_supported(
            AgentProfile(codemode=True),
            adapter,
        )

    KediRuntime._assert_profile_override_supported(
        AgentProfile(codemode=False),
        adapter,
    )
    KediRuntime._assert_profile_override_supported(
        AgentProfile(codemode=True),
        object.__new__(PydanticAdapter),
    )


@pytest.mark.parametrize("scope", ["top_level", "profile", "procedure"])
def test_mcp_expressions_can_access_prelude_imports(scope: str) -> None:
    import sys
    import textwrap

    prelude = """
```
import sys
from kedi.agent_profile import current_block_agent_profile
```
"""
    directive = """> mcp:
    command: `sys.executable`
    args: `[script_path]`
    env: `{"TEST_PYTHON": sys.executable}`
"""
    result = "= `repr(current_block_agent_profile().mcp_servers[0])`\n"
    if scope == "top_level":
        source = prelude + directive + result
    elif scope == "profile":
        source = prelude + "> profile: local:\n" + textwrap.indent(directive, "    ")
        source += "> use: local\n" + result
    else:
        source = prelude + "@inspect():\n" + textwrap.indent(directive + result, "    ")
        source += "= `inspect()`\n"
    runtime = compile_program(
        parse_program(source), adapter=_adapter(), runtime_globals={"script_path": "server.py"}
    )

    assert runtime.run_main() == repr(
        McpServerSpec(
            command=sys.executable, args=("server.py",), env={"TEST_PYTHON": sys.executable}
        )
    )


def test_mcp_profile_capture_uses_runtime_globals() -> None:
    src = """
> profile: dynamic:
    > mcp:
        command: `mcp_command`
        args: `["serve"]`
> use: dynamic
@foo():
    >> value [x]
"""
    rt = compile_program(
        parse_program(src),
        adapter=_adapter(),
        runtime_globals={"mcp_command": "vsh"},
    )

    foo_def = _procedure_def(rt._procedures["foo"])
    assert len(foo_def.captured_agent_profile.mcp_servers) == 1
    server = foo_def.captured_agent_profile.mcp_servers[0]
    assert server.transport == "stdio"
    assert server.command == "vsh"
    assert server.args == ("serve",)


def test_mcp_fields_support_plain_and_inline_python_strings() -> None:
    src = """
> mcp:
    transport: `transport_name`
    command: `command_name`
    args: `["serve"]`
@foo():
    >> value [x]
"""
    rt = compile_program(
        parse_program(src),
        adapter=_adapter(),
        runtime_globals={
            "transport_name": "stdio",
            "command_name": "uvx",
        },
    )

    proc = _procedure_def(rt._procedures["foo"])
    server = proc.captured_agent_profile.mcp_servers[0]
    assert server.transport == "stdio"
    assert server.command == "uvx"
    assert server.args == ("serve",)


def test_mcp_inside_procedure_body_applies_to_following_prompt() -> None:
    src = """
@foo():
    > mcp:
        command: local-mcp
    >> value [x]
"""
    adapter = _adapter()
    rt = compile_program(parse_program(src), adapter=adapter)
    proc = _procedure_def(rt._procedures["foo"])

    assert proc.captured_agent_profile.mcp_servers == ()
    rt._invoke_procedure(rt._procedures["foo"], [])
    assert adapter.mcp_calls == [("local-mcp",)]


def test_top_level_mcp_applies_to_following_prompt() -> None:
    src = """
> mcp:
    command: top-mcp
>> value [x]
"""
    adapter = _adapter()
    rt = compile_program(parse_program(src), adapter=adapter)
    rt.run_main()
    assert adapter.mcp_calls == [("top-mcp",)]


def test_source_adapter_directive_selects_framework_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src = """
> adapter: secondary
>> value [x]
"""
    default_adapter = _adapter()
    default_adapter.shortname = "default"
    selected_adapter = MockAdapter(default_model="selected-model")
    _patch_adapter_factory(monkeypatch, "secondary", "agent-framework", selected_adapter)

    rt = compile_program(parse_program(src), adapter=default_adapter)
    rt.run_main()

    assert default_adapter.model_calls == []
    assert selected_adapter.model_calls == ["selected-model"]


def test_source_adapter_directive_can_supply_runtime_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src = """
> adapter: secondary
>> value [x]
"""
    selected_adapter = MockAdapter(default_model="selected-model")
    _patch_adapter_factory(monkeypatch, "secondary", "agent-framework", selected_adapter)

    rt = compile_program(parse_program(src))
    rt.run_main()

    assert selected_adapter.model_calls == ["selected-model"]


def test_source_adapter_directive_reuses_matching_runtime_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src = """
> adapter: secondary
>> value [x]
"""
    adapter = _adapter()
    adapter.shortname = "secondary"
    _patch_adapter_factory(monkeypatch, "secondary", "agent-framework", adapter)

    rt = compile_program(parse_program(src), adapter=adapter)
    rt.run_main()

    assert adapter.model_calls == ["default-model"]


def test_selected_adapter_cache_is_keyed_by_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[MockAdapter] = []

    def build(model: str | None, _kwargs: object) -> MockAdapter:
        adapter = MockAdapter(default_model=model or "default")
        adapter.shortname = "secondary"
        built.append(adapter)
        return adapter

    monkeypatch.setitem(
        adapter_registry._ADAPTERS,
        "secondary",
        AdapterFactory(shortname="secondary", kind="agent-framework", build=build),
    )
    src = """
> adapter: secondary
> model: first
>> first [x]
> model: second
>> second [y]
"""
    rt = compile_program(parse_program(src), adapter=None)
    rt.run_main()

    assert [adapter.default_model() for adapter in built] == ["first", "second"]
    assert [adapter.model_calls for adapter in built] == [["first"], ["second"]]


def test_selected_adapter_cache_reuses_same_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[MockAdapter] = []

    def build(model: str | None, _kwargs: object) -> MockAdapter:
        adapter = MockAdapter(default_model=model or "default")
        adapter.shortname = "secondary"
        built.append(adapter)
        return adapter

    monkeypatch.setitem(
        adapter_registry._ADAPTERS,
        "secondary",
        AdapterFactory(shortname="secondary", kind="agent-framework", build=build),
    )
    src = """
> adapter: secondary
> model: same
>> first [x]
>> second [y]
"""
    rt = compile_program(parse_program(src), adapter=None)
    rt.run_main()

    assert len(built) == 1
    assert built[0].model_calls == ["same", "same"]


def test_source_agent_directive_selects_harness_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src = """
> agent: harness
>> value [x]
"""
    default_adapter = _adapter()
    default_adapter.shortname = "default"
    selected_adapter = MockAdapter(default_model="harness-model")
    _patch_adapter_factory(monkeypatch, "harness", "agent-harness", selected_adapter)

    rt = compile_program(parse_program(src), adapter=default_adapter)
    rt.run_main()

    assert default_adapter.model_calls == []
    assert selected_adapter.model_calls == ["harness-model"]


def test_procedure_body_adapter_can_override_captured_top_level_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src = """
> agent: harness
@foo():
    > adapter: secondary
    >> value [x]
"""
    default_adapter = _adapter()
    default_adapter.shortname = "default"
    harness_adapter = MockAdapter(default_model="harness-model")
    secondary_adapter = MockAdapter(default_model="secondary-model")
    _patch_adapter_factory(monkeypatch, "harness", "agent-harness", harness_adapter)
    _patch_adapter_factory(monkeypatch, "secondary", "agent-framework", secondary_adapter)

    rt = compile_program(parse_program(src), adapter=default_adapter)
    rt._invoke_procedure(rt._procedures["foo"], [])

    assert harness_adapter.model_calls == []
    assert secondary_adapter.model_calls == ["secondary-model"]


def test_procedure_body_agent_can_override_captured_top_level_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src = """
> adapter: secondary
@foo():
    > agent: harness
    >> value [x]
"""
    default_adapter = _adapter()
    default_adapter.shortname = "default"
    harness_adapter = MockAdapter(default_model="harness-model")
    secondary_adapter = MockAdapter(default_model="secondary-model")
    _patch_adapter_factory(monkeypatch, "harness", "agent-harness", harness_adapter)
    _patch_adapter_factory(monkeypatch, "secondary", "agent-framework", secondary_adapter)

    rt = compile_program(parse_program(src), adapter=default_adapter)
    rt._invoke_procedure(rt._procedures["foo"], [])

    assert secondary_adapter.model_calls == []
    assert harness_adapter.model_calls == ["harness-model"]


def test_profile_agent_and_adapter_members_select_adapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness_adapter = MockAdapter(default_model="harness-model")
    secondary_adapter = MockAdapter(default_model="secondary-model")
    _patch_adapter_factory(monkeypatch, "harness", "agent-harness", harness_adapter)
    _patch_adapter_factory(monkeypatch, "secondary", "agent-framework", secondary_adapter)

    harness_rt = compile_program(
        parse_program("""
> profile: harness_profile:
    > agent: harness
> use: harness_profile
>> value [x]
"""),
        adapter=None,
    )
    harness_rt.run_main()

    secondary_rt = compile_program(
        parse_program("""
> profile: secondary_profile:
    > adapter: secondary
> use: secondary_profile
>> value [x]
"""),
        adapter=None,
    )
    secondary_rt.run_main()

    assert harness_adapter.model_calls == ["harness-model"]
    assert secondary_adapter.model_calls == ["secondary-model"]


@pytest.mark.parametrize(
    ("src", "message"),
    [
        ("> agent: pydantic\n>> value [x]\n", "cannot select 'pydantic'"),
        ("> adapter: codex\n>> value [x]\n", "cannot select 'codex'"),
        (
            "> profile: mixed:\n    > agent: codex\n    > adapter: pydantic\n> use: mixed\n>> value [x]\n",
            "cannot contain both",
        ),
    ],
)
def test_adapter_directive_validation(
    monkeypatch: pytest.MonkeyPatch,
    src: str,
    message: str,
) -> None:
    _patch_adapter_factory(monkeypatch, "harness", "agent-harness", _adapter())
    _patch_adapter_factory(monkeypatch, "secondary", "agent-framework", _adapter())
    with pytest.raises(Exception, match=message):
        rt = compile_program(parse_program(src), adapter=_adapter())
        if "foo" in rt._procedures:
            rt._invoke_procedure(rt._procedures["foo"], [])
        else:
            rt.run_main()


@pytest.mark.parametrize(
    ("src", "message"),
    [
        ("> agent: foobarbaz\n", "unknown agent harness adapter 'foobarbaz'"),
        ("> adapter: foobarbaz\n", "unknown agent framework adapter 'foobarbaz'"),
        (
            "> profile: invalid:\n    > agent: foobarbaz\n> use: invalid\n",
            "unknown agent harness adapter 'foobarbaz'",
        ),
    ],
)
def test_unknown_agent_adapter_directive_fails_at_runtime(src: str, message: str) -> None:
    rt = compile_program(parse_program(src), adapter=_adapter())

    with pytest.raises(Exception, match=message):
        rt.run_main()


def test_adapter_registry_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    assert "pydantic" in known_adapter_shortnames()
    assert adapter_factory("pydantic").kind == "agent-framework"
    assert adapter_capabilities("pydantic").supports("structured_output")
    assert "dspy" not in known_adapter_shortnames()
    assert adapter_capabilities("langchain").supports("mcp")
    assert adapter_capabilities("claude").supports("mcp")
    assert not adapter_capabilities("codex").supports("mcp")
    assert build_adapter("pydantic").shortname == "pydantic"
    assert build_adapter("claude", "claude-model").shortname == "claude"
    assert build_adapter("codex", "codex-model").shortname == "codex"

    class FakeLangChainAdapter:
        def __init__(self, *, model: str | None, **kwargs: Any) -> None:
            self.model = model
            self.kwargs = kwargs

    monkeypatch.setattr(
        "kedi.agent_adapter.adapters.langchain.LangChainAdapter", FakeLangChainAdapter
    )
    assert build_adapter("langchain", "langchain-model", temperature=0.2).model == "langchain-model"

    with pytest.raises(Exception, match="DSPy runtime adapter moved.*GEPA still uses DSPy"):
        build_adapter("dspy")
    with pytest.raises(Exception, match="Unknown agent adapter"):
        adapter_factory("missing")
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(
            adapter_registry._ADAPTERS,
            "custom",
            AdapterFactory("custom", "agent-framework", lambda *_args: cast(Any, object())),
        )
        assert adapter_capabilities("custom") == AdapterCapabilities()


def test_agent_profile_adapter_selection_helper() -> None:
    assert not AgentProfile.empty().needs_adapter_selection("pydantic")
    assert AgentProfile(adapter_shortname="pydantic").needs_adapter_selection("dspy")
    assert not AgentProfile(adapter_shortname="pydantic").needs_adapter_selection("pydantic")


def test_top_level_mcp_is_captured_by_following_procedure() -> None:
    src = """
> mcp:
    command: captured-mcp
@foo():
    >> value [x]
"""
    rt = compile_program(parse_program(src), adapter=_adapter())
    proc = _procedure_def(rt._procedures["foo"])
    assert tuple(server.command for server in proc.captured_agent_profile.mcp_servers) == (
        "captured-mcp",
    )


def test_mcp_field_without_value_evaluates_to_none() -> None:
    from kedi.lang.ast import McpField

    assert eval_mcp_field_value(cast(Any, None), McpField(name="env"), {}) is None


def test_settings_field_value_and_plain_scalar_parsing() -> None:
    rt = KediRuntime()

    assert parse_plain_setting_value("true") is True
    assert parse_plain_setting_value("null") == "null"
    assert parse_plain_setting_value("none") == "none"
    with pytest.raises(ValueError, match="inline Python"):
        parse_plain_setting_value("'literal'")
    assert parse_plain_setting_value("plain-value") == "plain-value"
    assert (
        eval_settings_field_value(
            rt,
            SettingsField(name="temperature", backtick_code="answer / 100"),
            {"answer": 42},
        )
        == 0.42
    )
    assert eval_settings_field_value(cast(Any, None), SettingsField(name="temperature"), {}) is None


def test_settings_inline_python_supports_non_string_values() -> None:
    src = """
> settings:
    parallel_tool_calls: `False`
    stop_sequences: `["END", "DONE"]`
    extra_body: `{"mode": "json"}`
>> value [x]
"""
    adapter = _adapter()
    rt = compile_program(parse_program(src), adapter=adapter)

    rt.run_main()

    assert adapter.settings_calls == [
        {
            "parallel_tool_calls": False,
            "stop_sequences": ["END", "DONE"],
            "extra_body": {"mode": "json"},
        }
    ]


def test_settings_support_plain_and_inline_python_strings() -> None:
    src = """
> settings:
    tool_choice: required
    service_tier: `"flex"`
>> value [x]
"""
    adapter = _adapter()
    rt = compile_program(parse_program(src), adapter=adapter)

    rt.run_main()

    assert adapter.settings_calls == [
        {
            "tool_choice": "required",
            "service_tier": "flex",
        }
    ]


def test_mcp_spec_validation() -> None:
    stdio = mcp_spec_from_mapping({"command": "vsh", "args": None})
    assert stdio == McpServerSpec(command="vsh")

    http = mcp_spec_from_mapping(
        {"transport": "http", "url": "http://localhost:8000", "headers": {"x": "y"}}
    )
    assert http.transport == "http"
    assert http.url == "http://localhost:8000"
    assert http.command is None

    streamable_http = mcp_spec_from_mapping(
        {"transport": "streamable-http", "url": "http://localhost:8000/mcp"}
    )
    assert streamable_http.transport == "http"
    assert streamable_http.url == "http://localhost:8000/mcp"

    sse = mcp_spec_from_mapping({"transport": "sse", "url": "http://localhost:8001"})
    assert sse.transport == "sse"

    with pytest.raises(ValueError, match="transport"):
        mcp_spec_from_mapping({"transport": "websocket"})
    with pytest.raises(ValueError, match="transport"):
        mcp_spec_from_mapping({})
    with pytest.raises(ValueError, match="requires command"):
        mcp_spec_from_mapping({"transport": "stdio"})
    with pytest.raises(ValueError, match="requires url"):
        mcp_spec_from_mapping({"transport": "streamable-http"})
    with pytest.raises(TypeError, match="args"):
        mcp_spec_from_mapping({"command": "vsh", "args": "run"})
    with pytest.raises(TypeError, match="args"):
        mcp_spec_from_mapping({"command": "vsh", "args": [1]})
    with pytest.raises(TypeError, match="env"):
        mcp_spec_from_mapping({"command": "vsh", "env": []})
    with pytest.raises(TypeError, match="env"):
        mcp_spec_from_mapping({"command": "vsh", "env": {"x": 1}})
    with pytest.raises(TypeError, match="headers"):
        mcp_spec_from_mapping({"command": "vsh", "headers": []})
    with pytest.raises(TypeError, match="command"):
        mcp_spec_from_mapping({"command": 1})
    with pytest.raises(TypeError, match="url"):
        mcp_spec_from_mapping({"transport": "http", "url": 1})
    broken = McpServerSpec(command="vsh")
    object.__setattr__(broken, "transport", "bad")
    with pytest.raises(ValueError, match="Unsupported MCP transport"):
        broken.normalized()


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
> instructions: first-system
@foo():
    >> value [x]
> instructions: second-system
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


def test_procedure_capture_uses_source_order_effort() -> None:
    src = """
> effort: low
@foo():
    >> value [x]
> effort: high
@bar():
    >> value [x]
"""
    prog = parse_program(src)
    adapter = _adapter()
    rt = compile_program(prog, adapter=adapter)
    foo_def = _procedure_def(rt._procedures["foo"])
    bar_def = _procedure_def(rt._procedures["bar"])
    assert foo_def.captured_agent_profile.effort == "low"
    assert bar_def.captured_agent_profile.effort == "high"


def test_procedure_capture_uses_source_order_settings() -> None:
    src = """
> settings:
    temperature: 0.2
@foo():
    >> value [x]
> settings:
    max_tokens: 128
@bar():
    >> value [x]
"""
    adapter = _adapter()
    rt = compile_program(parse_program(src), adapter=adapter)
    foo_def = _procedure_def(rt._procedures["foo"])
    bar_def = _procedure_def(rt._procedures["bar"])
    assert foo_def.captured_agent_profile.settings == {"temperature": 0.2}
    assert bar_def.captured_agent_profile.settings == {
        "temperature": 0.2,
        "max_tokens": 128,
    }


def test_runtime_globals_are_available_during_profile_capture() -> None:
    src = """
> profile: personalized:
    > instructions:
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


def test_runtime_globals_are_available_in_top_level_templates() -> None:
    src = ">> Hello <`args.name`> [x]\n= <x>\n"
    adapter = _adapter()
    rt = compile_program(
        parse_program(src),
        adapter=adapter,
        runtime_globals={"args": AttrDict({"name": "Ada"}, default=None)},
    )

    assert rt.run_main() == "x-value"


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
    assert adapter.instruction_calls == [f"{TEMPLATE_INSTRUCTIONS}\n\n{artifact_instructions()}"]


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
    models_by_request = dict(zip(adapter.request_calls, adapter.model_calls, strict=True))
    assert next(model for request, model in models_by_request.items() if "first" in request) == (
        "default-model"
    )
    assert next(model for request, model in models_by_request.items() if "second" in request) == (
        "inner-model"
    )


def test_block_state_system_within_procedure_body() -> None:
    src = """
@foo():
    >> first [x]
    > instructions: Answer tersely.
    >> second [y]
"""
    adapter = _adapter()
    rt = compile_program(parse_program(src), adapter=adapter)
    rt._invoke_procedure(rt._procedures["foo"], [])
    assert len(adapter.request_calls) == 2
    instructions_by_request = dict(
        zip(adapter.request_calls, adapter.instruction_calls, strict=True)
    )
    assert len(instructions_by_request) == 2
    assert (
        next(
            instructions
            for request, instructions in instructions_by_request.items()
            if "first" in request
        )
        == f"{TEMPLATE_INSTRUCTIONS}\n\n{artifact_instructions()}"
    )
    assert (
        next(
            instructions
            for request, instructions in instructions_by_request.items()
            if "second" in request
        )
        == f"{TEMPLATE_INSTRUCTIONS}\n\nAnswer tersely.\n\n{artifact_instructions()}"
    )


def test_block_state_system_renders_multiline_substitutions() -> None:
    src = """
@foo(style):
    > instructions:
        You are <style>.
        Keep it short.
    >> answer [x]
"""
    adapter = _adapter()
    rt = compile_program(parse_program(src), adapter=adapter)
    rt._invoke_procedure(rt._procedures["foo"], ["direct"])
    assert adapter.instruction_calls == [
        f"{TEMPLATE_INSTRUCTIONS}\n\nYou are direct.\nKeep it short.\n\n{artifact_instructions()}"
    ]


def test_top_level_system_applies_to_following_prompt() -> None:
    src = """
> instructions: Top-level instructions.
>> answer [x]
"""
    adapter = _adapter()
    rt = compile_program(parse_program(src), adapter=adapter)
    rt.run_main()
    assert adapter.instruction_calls == [
        f"{TEMPLATE_INSTRUCTIONS}\n\nTop-level instructions.\n\n{artifact_instructions()}"
    ]


def test_top_level_system_renders_multiline_substitutions() -> None:
    src = """
[style: str] = `"concise"`
> instructions:
    You are <style>.
    No extra text.
>> answer [x]
"""
    adapter = _adapter()
    rt = compile_program(parse_program(src), adapter=adapter)
    rt.run_main()
    assert adapter.instruction_calls == [
        f"{TEMPLATE_INSTRUCTIONS}\n\nYou are concise.\nNo extra text.\n\n{artifact_instructions()}"
    ]


def test_top_level_system_renders_inline_python_substitution() -> None:
    src = """
```
from types import SimpleNamespace
person = SimpleNamespace(name="Mert")
```
> instructions:
    Your name is <`person.name`>.
>> answer [x]
"""
    adapter = _adapter()
    rt = compile_program(parse_program(src), adapter=adapter)
    rt.run_main()
    assert adapter.instruction_calls == [
        f"{TEMPLATE_INSTRUCTIONS}\n\nYour name is Mert.\n\n{artifact_instructions()}"
    ]


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
    > instructions: from-profile-system
> use: dynamic
@foo():
    >> value [x]
"""
    adapter = _adapter()
    rt = compile_program(parse_program(src), adapter=adapter)
    foo_def = _procedure_def(rt._procedures["foo"])
    assert foo_def.captured_agent_profile.system == "from-profile-system"
    rt._invoke_procedure(rt._procedures["foo"], [])
    assert adapter.instruction_calls == [
        f"{TEMPLATE_INSTRUCTIONS}\n\nfrom-profile-system\n\n{artifact_instructions()}"
    ]


def test_use_profile_applies_effort_at_capture_time() -> None:
    src = """
> profile: dynamic:
    > effort: max
> use: dynamic
@foo():
    >> value [x]
"""
    adapter = _adapter()
    rt = compile_program(parse_program(src), adapter=adapter)
    foo_def = _procedure_def(rt._procedures["foo"])
    assert foo_def.captured_agent_profile.effort == "max"
    rt._invoke_procedure(rt._procedures["foo"], [])
    assert adapter.effort_calls == ["max"]


def test_use_profile_applies_settings_at_capture_time() -> None:
    src = """
> profile: dynamic:
    > settings:
        temperature: 0.3
        parallel_tool_calls: false
> use: dynamic
@foo():
    >> value [x]
"""
    adapter = _adapter()
    rt = compile_program(parse_program(src), adapter=adapter)
    foo_def = _procedure_def(rt._procedures["foo"])
    assert foo_def.captured_agent_profile.settings == {
        "temperature": 0.3,
        "parallel_tool_calls": False,
    }
    rt._invoke_procedure(rt._procedures["foo"], [])
    assert adapter.settings_calls == [{"temperature": 0.3, "parallel_tool_calls": False}]


def test_effort_applies_to_top_level_prompt() -> None:
    src = """
> effort: medium
>> answer [x]
"""
    adapter = _adapter()
    rt = compile_program(parse_program(src), adapter=adapter)
    rt.run_main()
    assert adapter.effort_calls == ["medium"]


def test_settings_applies_to_top_level_prompt_and_procedure_body() -> None:
    src = """
> settings:
    temperature: 0.2
>> first [x]

@foo():
    > settings:
        max_tokens: 10
    >> second [y]

= <foo()>
"""
    adapter = _adapter()
    rt = compile_program(parse_program(src), adapter=adapter)
    rt.run_main()
    settings_by_request = dict(zip(adapter.request_calls, adapter.settings_calls, strict=True))
    assert next(
        settings for request, settings in settings_by_request.items() if "first" in request
    ) == {"temperature": 0.2}
    assert next(
        settings for request, settings in settings_by_request.items() if "second" in request
    ) == {"temperature": 0.2, "max_tokens": 10}


def test_backtick_effort_applies_in_profile_and_body() -> None:
    src = """
> effort: `"low"`
> profile: dynamic:
    > effort: `"medium"`
> use: dynamic
@foo():
    > effort: `"high"`
    >> value [x]
"""
    adapter = _adapter()
    rt = compile_program(parse_program(src), adapter=adapter)
    foo_def = _procedure_def(rt._procedures["foo"])
    assert foo_def.captured_agent_profile.effort == "medium"
    rt._invoke_procedure(rt._procedures["foo"], [])
    assert adapter.effort_calls == ["high"]


def test_invalid_plain_effort_rejected_at_top_level() -> None:
    with pytest.raises(KediParseError) as exc_info:
        parse_program("> effort: extreme\n")

    assert any(diag.code == "K-EFFORT-VALUE" for diag in exc_info.value.diagnostics)


def test_invalid_plain_effort_rejected_inside_profile() -> None:
    src = """
> profile: strict:
    > effort: extreme
"""
    with pytest.raises(KediParseError) as exc_info:
        parse_program(src)

    assert any(diag.code == "K-EFFORT-VALUE" for diag in exc_info.value.diagnostics)


def test_invalid_model_setting_key_rejected() -> None:
    with pytest.raises(KediParseError) as exc_info:
        parse_program("> settings:\n    imaginary: 1\n")

    assert any(diag.code == "K-SETTING-KEY" for diag in exc_info.value.diagnostics)


def test_agent_profile_merge() -> None:
    base = AgentProfile(model="a", effort="low", system="old", settings={"temperature": 0.2})
    merged = base.merge(
        AgentProfile(
            model="b",
            effort="high",
            system="new",
            settings={"max_tokens": 100, "temperature": 0.5},
        )
    )
    assert merged.model == "b"
    assert merged.effort == "high"
    assert merged.system == "new"
    assert merged.settings == {"temperature": 0.5, "max_tokens": 100}
    assert base.model == "a"
    assert base.effort == "low"
    assert base.system == "old"
    assert base.settings == {"temperature": 0.2}
