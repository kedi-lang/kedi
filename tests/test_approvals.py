from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest

import kedi
import kedi.api as kedi_api
from kedi.agent_adapter.approval import ApprovalDecision, ApprovalPolicy, ToolApprovalDenied
from kedi.agent_adapter.tool_override import ToolSpec, with_approval
from kedi.agent_adapter.utils import kedi_procedure_tool_spec, python_callable_tool_spec
from kedi.agent_profile.model import eval_approval_directive, eval_profile_definition
from kedi.agent_profile.registry import ProfileApprovalMember, ProfileDefinition
from kedi.core import KediRuntime
from kedi.errors import KediExecutionError
from kedi.lang.ast import ApprovalDirectiveStmt
from kedi.lang import compile_program, parse_program
from kedi.resolved_types import ResolvedKediType
from kedi.utils import AttrDict


@pytest.fixture(autouse=True)
def reset_python_api_config() -> Iterator[None]:
    kedi.reset_config()
    yield
    kedi.reset_config()


def _tool(*, risk: str = "mutating") -> ToolSpec:
    def write_file(*, path: str, content: str) -> str:
        return f"{path}:{content}"

    return ToolSpec(fn=write_file, name="write_file", risk=risk)  # type: ignore[arg-type]


def test_read_only_tools_auto_allow_without_policy() -> None:
    tool = with_approval(_tool(risk="read_only"), policy=None, adapter_shortname="test")

    assert tool.fn(path="notes.txt", content="ignored") == "notes.txt:ignored"


def test_mutating_tools_default_to_denied() -> None:
    tool = with_approval(_tool(), policy=None, adapter_shortname="test")

    with pytest.raises(ToolApprovalDenied, match="no approval policy configured"):
        tool.fn(path="notes.txt", content="blocked")


def test_static_allow_and_deny_policies() -> None:
    allowed = with_approval(_tool(), policy=ApprovalPolicy.allow(), adapter_shortname="test")
    denied = with_approval(_tool(), policy=ApprovalPolicy.deny(), adapter_shortname="test")

    assert allowed.fn(path="notes.txt", content="ok") == "notes.txt:ok"
    with pytest.raises(ToolApprovalDenied, match="static deny policy"):
        denied.fn(path="notes.txt", content="blocked")


def test_dynamic_handler_can_edit_arguments_and_receives_immutable_request() -> None:
    seen = []

    @kedi.approval
    def redirect(request):  # type: ignore[no-untyped-def]
        seen.append(request)
        with pytest.raises(TypeError):
            request.arguments["path"] = "other.txt"
        return ApprovalDecision.edit({"path": "safe.txt", "content": request.arguments["content"]})

    tool = with_approval(_tool(), policy=ApprovalPolicy.dynamic(redirect), adapter_shortname="test")

    assert tool.fn(path="unsafe.txt", content="ok") == "safe.txt:ok"
    assert seen[0].tool_name == "write_file"
    assert seen[0].risk == "mutating"


def test_dynamic_handler_may_be_async() -> None:
    @kedi.approval
    async def permit(_request):  # type: ignore[no-untyped-def]
        await asyncio.sleep(0)
        return ApprovalDecision.allow()

    tool = with_approval(_tool(), policy=ApprovalPolicy.dynamic(permit), adapter_shortname="test")

    assert tool.fn(path="notes.txt", content="ok") == "notes.txt:ok"


def test_approval_decorator_registers_the_default_python_api_policy() -> None:
    @kedi.approval
    def permit(_request: object) -> ApprovalDecision:
        return ApprovalDecision.allow()

    policy = kedi_api._config_var.get().profile.approval

    assert policy is not None
    assert policy.handler is permit


def test_argument_aware_tool_risk_requires_approval_for_sensitive_access() -> None:
    tool = ToolSpec(
        fn=lambda *, secret_files=False: "secret",  # noqa: E731
        name="read_text_file",
        risk="read_only",
        risk_resolver=lambda arguments: "sensitive" if arguments["secret_files"] else "read_only",
    )
    guarded = with_approval(tool, policy=None, adapter_shortname="test")

    assert guarded.fn(secret_files=False) == "secret"
    with pytest.raises(ToolApprovalDenied, match="no approval policy configured"):
        guarded.fn(secret_files=True)


def test_filesystem_secret_read_requires_approval(tmp_path: Path) -> None:
    filesystem_path = Path(kedi.__file__).parent / "modules" / "filesystem.kedi"
    source = filesystem_path.read_text(encoding="utf-8").replace(
        "_ROOT = Path.cwd().resolve()",
        f"_ROOT = Path({str(tmp_path)!r}).resolve()",
    )
    program = parse_program(
        source,
        source_path=str(filesystem_path),
    )
    runtime = compile_program(program)
    read_proc = next(proc for proc in program.procedures if proc.name == "read_text_file")
    spec = kedi_procedure_tool_spec(runtime, read_proc)
    secret_path = tmp_path / ".env"
    secret_path.write_text("TOKEN=hidden\n", encoding="utf-8")

    blocked = with_approval(spec, policy=None, adapter_shortname="test")
    with pytest.raises(ToolApprovalDenied, match="no approval policy configured"):
        blocked.fn(file_path=str(secret_path), secret_files=True)

    allowed = with_approval(spec, policy=ApprovalPolicy.allow(), adapter_shortname="test")
    assert allowed.fn(file_path=str(secret_path), secret_files=True) == "TOKEN=hidden\n"
    with pytest.raises(KediExecutionError, match="secret_files=True"):
        allowed.fn(file_path=str(secret_path), secret_files=False)


def test_non_edit_decisions_reject_edited_arguments() -> None:
    with pytest.raises(ValueError, match="only be provided"):
        ApprovalDecision(decision="allow", edit_arguments={"path": "notes.txt"})

    with pytest.raises(ValueError, match="requires edit_arguments"):
        ApprovalDecision(decision="edit")

    with pytest.raises(ValueError, match="must be 'allow', 'deny', or 'edit'"):
        ApprovalDecision(decision="unsupported")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="must be callable"):
        ApprovalPolicy.dynamic("not-a-handler")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="mutually exclusive"):
        ApprovalPolicy(mode="allow", handler=lambda _request: ApprovalDecision.allow())

    with pytest.raises(ValueError, match="must be 'allow' or 'deny'"):
        ApprovalPolicy(mode="edit")  # type: ignore[arg-type]


def test_approval_handler_must_return_a_decision() -> None:
    tool = with_approval(
        _tool(),
        policy=ApprovalPolicy.dynamic(lambda _request: "allow"),  # type: ignore[arg-type]
        adapter_shortname="test",
    )

    with pytest.raises(ValueError, match="must return ApprovalDecision"):
        tool.fn(path="notes.txt", content="blocked")


def test_tool_wrapper_rejects_positional_arguments_and_invalid_risk() -> None:
    guarded = with_approval(_tool(), policy=ApprovalPolicy.allow(), adapter_shortname="test")
    with pytest.raises(TypeError, match="only accepts keyword"):
        guarded.fn("notes.txt", "blocked")

    invalid_risk = ToolSpec(
        fn=lambda **_kwargs: None,
        name="invalid_risk",
        risk_resolver=lambda _arguments: "unknown",  # type: ignore[return-value]
    )
    with pytest.raises(ValueError, match="invalid Kedi tool risk"):
        with_approval(invalid_risk, policy=ApprovalPolicy.allow(), adapter_shortname="test").fn()


def test_python_tool_risk_and_api_approval_inputs_are_validated() -> None:
    def unsafe() -> None:
        return None

    unsafe.__kedi_tool_risk__ = "unknown"  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="invalid Kedi tool risk"):
        python_callable_tool_spec("unsafe", unsafe)

    assert kedi_api._coerce_approval("allow") == ApprovalPolicy.allow()
    assert kedi_api._coerce_approval("deny") == ApprovalPolicy.deny()
    policy = ApprovalPolicy.allow()
    assert kedi_api._coerce_approval(policy) is policy
    assert kedi_api._approval_identity(policy) == ("allow", None)
    assert kedi_api._approval_identity(None) is None

    def permit(_request: object) -> ApprovalDecision:
        return ApprovalDecision.allow()

    dynamic_policy = kedi_api._coerce_approval(permit)
    assert dynamic_policy is not None
    assert dynamic_policy.handler is permit
    assert kedi_api._approval_identity(dynamic_policy)[:3] == (
        None,
        permit.__module__,
        permit.__qualname__,
    )
    with pytest.raises(TypeError, match="approval must be"):
        kedi_api._coerce_approval("edit")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="approval must be"):
        kedi_api._coerce_approval(1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tool risk"):
        kedi.tool(risk="unknown")  # type: ignore[arg-type]


def test_approval_directive_and_profile_validation_errors() -> None:
    runtime = KediRuntime()
    with pytest.raises(TypeError, match="must evaluate to a callable"):
        eval_approval_directive(
            runtime,
            ApprovalDirectiveStmt(backtick_code="handler"),
            {"handler": "not callable"},
        )
    with pytest.raises(ValueError, match="has no value"):
        eval_approval_directive(runtime, ApprovalDirectiveStmt(), {})
    with pytest.raises(ValueError, match="must be 'allow', 'deny'"):
        eval_approval_directive(runtime, ApprovalDirectiveStmt(plain_value="edit"), {})

    def handler(_request: object) -> ApprovalDecision:
        return ApprovalDecision.allow()

    dynamic = eval_profile_definition(
        runtime,
        ProfileDefinition(
            name="dynamic",
            members=[ProfileApprovalMember(backtick_code="handler")],
        ),
        {"handler": handler},
    )
    assert dynamic.approval == ApprovalPolicy.dynamic(handler)
    denied = eval_profile_definition(
        runtime,
        ProfileDefinition(name="denied", members=[ProfileApprovalMember(plain_value="deny")]),
        {},
    )
    assert denied.approval == ApprovalPolicy.deny()
    with pytest.raises(TypeError, match="must evaluate to a callable"):
        eval_profile_definition(
            runtime,
            ProfileDefinition(
                name="not_callable",
                members=[ProfileApprovalMember(backtick_code="handler")],
            ),
            {"handler": 1},
        )
    with pytest.raises(ValueError, match="must be 'allow' or 'deny'"):
        eval_profile_definition(
            runtime,
            ProfileDefinition(name="invalid", members=[ProfileApprovalMember()]),
            {},
        )


class _ToolCallingAdapter:
    kind = "agent-framework"
    shortname = "test"

    def __init__(self) -> None:
        self._tools: list[ToolSpec] = []

    def tool_scope(self):  # type: ignore[no-untyped-def]
        adapter = self

        class Scope:
            def __enter__(self):  # type: ignore[no-untyped-def]
                adapter._tools = []

            def __exit__(self, *_args):  # type: ignore[no-untyped-def]
                return None

        return Scope()

    def register_tool(self, tool: ToolSpec) -> None:
        self._tools.append(tool)

    async def produce(self, **kwargs):  # type: ignore[no-untyped-def]
        return self.produce_sync(**kwargs)

    def produce_sync(
        self,
        *,
        template: str,
        output_schema: dict[str, ResolvedKediType],
        **_kwargs: object,
    ) -> AttrDict[str, str]:
        assert template
        assert output_schema == {"answer": str}
        return AttrDict({"answer": self._tools[0].fn(path="safe.txt")})


def test_dsl_static_approval_reaches_the_registered_tool() -> None:
    source = """@write_file(path: str) -> str:
  = <path>

> use: write_file
> approval: allow
>> Write [answer]
= <answer>
"""
    runtime = compile_program(parse_program(source), adapter=_ToolCallingAdapter())

    assert runtime.run_main() == "safe.txt"


def test_dsl_static_deny_prevents_the_registered_tool() -> None:
    source = """@write_file(path: str) -> str:
  = <path>

> use: write_file
> approval: deny
>> Write [answer]
= <answer>
"""
    runtime = compile_program(parse_program(source), adapter=_ToolCallingAdapter())

    with pytest.raises(ToolApprovalDenied, match="static deny policy"):
        runtime.run_main()


def test_dsl_dynamic_approval_handler_is_evaluated_from_python_environment() -> None:
    seen: list[str] = []

    @kedi.approval
    def approve(request):  # type: ignore[no-untyped-def]
        seen.append(request.tool_name)
        return ApprovalDecision.allow(reason="test handler")

    source = """@write_file(path: str) -> str:
  = <path>

> use: write_file
> approval: `approve`
>> Write [answer]
= <answer>
"""
    runtime = compile_program(
        parse_program(source),
        adapter=_ToolCallingAdapter(),
        runtime_globals={"approve": approve},
    )

    assert runtime.run_main() == "safe.txt"
    assert seen == ["write_file"]


def test_procedure_approval_directive_is_applied_in_its_call_scope() -> None:
    source = """@write_file(path: str) -> str:
  = <path>

@run() -> str:
  > use: write_file
  > approval: allow
  >> Write [answer]
  = <answer>

= <run()>
"""
    runtime = compile_program(parse_program(source), adapter=_ToolCallingAdapter())

    assert runtime.run_main() == "safe.txt"


def test_approval_edit_is_not_a_valid_static_dsl_mode() -> None:
    with pytest.raises(Exception, match="only valid as ApprovalDecision.edit"):
        parse_program("> approval: edit\n")


def test_profile_approval_is_applied_when_profile_is_used() -> None:
    source = """@write_file(path: str) -> str:
  = <path>

> profile: guarded:
  > approval: allow
  > use: write_file

> use: guarded
>> Write [answer]
= <answer>
"""
    runtime = compile_program(parse_program(source), adapter=_ToolCallingAdapter())

    assert runtime.run_main() == "safe.txt"
