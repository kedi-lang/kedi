"""Regression coverage for the remaining external audit security boundaries."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Barrier, Thread
from typing import cast

import pytest

from kedi.agent_adapter.approvals.policy import ApprovalDecision, ApprovalPolicy
from kedi.agent_adapter.codemode import CodeModeLimits, CodeModeSession, CodeModeTool
from kedi.agent_adapter.codemode.errors import CodeModeResultError
from kedi.agent_adapter.codemode.tool_specs import CodeModeToolSpecBridge
from kedi.agent_adapter.hooks.dispatcher import (
    HookDispatcher,
    HookDispatcherSnapshot,
    bind_hook_dispatcher,
)
from kedi.agent_adapter.hooks.models import PreToolUseDecision
from kedi.agent_adapter.subagents.safety import SubagentSafetyCeiling
from kedi.agent_adapter.tools.specs import ToolSpec
from kedi.agent_profile import AgentProfile, HookSettings
from kedi.agent_profile.codemode import CodeModeSettings
from kedi.errors import KediExecutionError
from kedi.forms import _python_use
from kedi.lang import PackageMetadata, parse_program
from kedi.lang.ast import ProcDefStmt
from kedi.lang.parser import parser
from kedi.package_manifest import validate_package_metadata
from kedi.registry_api import HTTPRegistryAPIClient, RegistryAPIError
from kedi.run_budget import RunBudgetExceeded


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        RunBudgetExceeded("request", 1, 1, 0),
        KediExecutionError("fatal"),
        CodeModeResultError("lost result"),
        ValueError("recoverable"),
    ],
)
async def test_codemode_cannot_suppress_fatal_host_errors(error):
    session = CodeModeSession()
    session.update_tools([CodeModeTool("fail", None, {"type": "object", "properties": {}})])
    session.get_tool_schema(tool_names=["fail"])

    async def invoke(name, arguments):
        raise error

    try:
        code = "try:\n    await fail()\nexcept Exception:\n    pass\n'ok'"
        if isinstance(error, ValueError):
            assert await session.execute_code(code=code, invoke=invoke) == "ok"
        else:
            with pytest.raises(type(error)) as raised:
                await session.execute_code(code=code, invoke=invoke)
            assert raised.value is error
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("first_fatal", [False, True])
async def test_codemode_preserves_first_fatal_even_when_later_calls_fail(first_fatal):
    session = CodeModeSession()
    session.update_tools([CodeModeTool("fail", None, {"type": "object", "properties": {}})])
    session.get_tool_schema(tool_names=["fail"])
    fatal = KediExecutionError("first fatal")
    errors = [
        fatal if first_fatal else ValueError("recoverable"),
        fatal,
        CodeModeResultError("later"),
    ]

    async def invoke(name, arguments):
        raise errors.pop(0)

    try:
        with pytest.raises(KediExecutionError) as raised:
            await session.execute_code(
                code="for i in range(3):\n    try:\n        await fail()\n    except Exception:\n        pass",
                invoke=invoke,
            )
        assert raised.value is fatal
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_codemode_memory_ceiling_is_enforced_and_reset_is_usable():
    session = CodeModeSession(CodeModeLimits(max_memory_bytes=4 * 1024 * 1024))

    async def invoke(name, arguments):
        raise AssertionError("no host calls expected")

    try:
        result = await session.execute_code(code="large = [0] * 10000000", invoke=invoke)
        assert "memory" in result["error"].lower()
        await session.reset()
        assert await session.execute_code(code="2 + 3", invoke=invoke) == 5
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_codemode_cell_cpu_ceiling_is_enforced():
    session = CodeModeSession(CodeModeLimits(max_feed_seconds=0.02))

    async def invoke(name, arguments):
        raise AssertionError("no host calls expected")

    try:
        result = await session.execute_code(code="while True:\n    pass", invoke=invoke)
        assert "error" in result
        await session.reset()
        assert await session.execute_code(code="6", invoke=invoke) == 6
    finally:
        await session.close()


@pytest.mark.parametrize("factory", [CodeModeLimits, CodeModeSettings])
@pytest.mark.parametrize("field", ["max_memory_bytes", "max_feed_seconds"])
def test_codemode_resource_limits_must_be_positive(factory, field):
    with pytest.raises(ValueError, match=field if factory is CodeModeLimits else "positive"):
        factory(**{field: 0})


@pytest.mark.parametrize("value", [True, "one"])
def test_codemode_feed_duration_requires_a_number(value):
    with pytest.raises(TypeError, match="max_feed_seconds"):
        CodeModeSettings(max_feed_seconds=value)


def test_codemode_resource_limits_are_available_in_the_dsl():
    from kedi.agent_profile.compile import profile_stmt_to_definition
    from kedi.agent_profile.evaluation import eval_profile_definition
    from kedi.core.runtime import KediRuntime

    program = parse_program(
        "> profile: bounded:\n  > codemode:\n    max_memory_bytes: 4194304\n    max_feed_seconds: 0.5\n"
    )
    profile = eval_profile_definition(
        KediRuntime(), profile_stmt_to_definition(program.profile_defs["bounded"]), {}
    )
    assert profile.codemode.max_memory_bytes == 4194304
    assert profile.codemode.max_feed_seconds == 0.5


def test_child_without_cwd_inherits_the_parent_boundary(tmp_path):
    ceiling = SubagentSafetyCeiling(approval_policies=(), cwd=tmp_path)
    effective, child = ceiling.constrain_child(AgentProfile())
    assert Path(effective.settings["cwd"]) == tmp_path
    assert child.cwd == tmp_path


def test_default_cwd_does_not_create_a_local_setting_for_remote_children(tmp_path):
    ceiling = SubagentSafetyCeiling.from_parent(AgentProfile(), default_cwd=tmp_path)
    effective, child = ceiling.constrain_child(AgentProfile())
    assert "cwd" not in effective.settings
    assert not child.cwd_explicit


def test_explicit_cwd_is_preserved_for_all_descendants(tmp_path):
    ceiling = SubagentSafetyCeiling.from_parent(
        AgentProfile(settings={"cwd": str(tmp_path)}), default_cwd=tmp_path.parent
    )
    first, child = ceiling.constrain_child(AgentProfile())
    second, grandchild = child.constrain_child(AgentProfile())
    assert first.settings["cwd"] == second.settings["cwd"] == str(tmp_path)
    assert grandchild.cwd_explicit


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_codemode_hook_edits_are_approved_before_body(asynchronous):
    observed = []

    def edit(event):
        observed.append(("hook", event.arguments["value"]))
        return PreToolUseDecision.edit({"value": 2})

    def approve(request):
        observed.append(("approval", request.arguments["value"]))
        return ApprovalDecision.allow()

    def body(*, value):
        observed.append(("body", value))
        return value

    async def async_body(*, value):
        return body(value=value)

    bridge = CodeModeToolSpecBridge(
        [
            ToolSpec(
                fn=async_body if asynchronous else body,
                name="save",
                risk="mutating",
                json_schema={
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                },
            )
        ],
        limits=CodeModeLimits(),
        policies=(ApprovalPolicy.dynamic(approve),),
        adapter_shortname="test",
    )
    dispatcher = HookDispatcher(
        HookDispatcherSnapshot.compose(None, HookSettings(pre_tool_use=(edit,)))
    )
    try:
        with bind_hook_dispatcher(dispatcher):
            assert await bridge._invoke("save", {"value": 1}) == 2
        assert observed == [("hook", 1), ("approval", 2), ("body", 2)]
    finally:
        await bridge.close()


@pytest.mark.parametrize("ceiling", [(), (ApprovalPolicy.deny(),)])
def test_codex_explicit_approval_setting_cannot_bypass_parent(ceiling):
    from kedi.agent_adapter.adapters.codex import CodexAdapter
    from kedi.agent_adapter.adapters.codex.app_server import CodexAppServerClient, CodexTurnResult

    observed = []

    class Client:
        def run_stateless_turn_sync(self, **kwargs):
            observed.append(kwargs["thread_params"]["approvalPolicy"])
            return CodexTurnResult(text="ok", turn={}, thread_id="audit")

    adapter = CodexAdapter(model="model", client=cast(CodexAppServerClient, Client()))
    with (
        adapter.profile_override(AgentProfile(settings={"approval_policy": "never"})),
        adapter.approval_ceiling(ceiling),
    ):
        assert adapter.invoke_sync(prompt="Return ok") == "ok"
    assert observed == ["untrusted" if ceiling else "never"]


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/untrusted",
        "https://example.test/archive.whl",
        "git+https://example.test/repository",
    ],
)
def test_remote_manifest_rejects_direct_url_dependencies(url):
    with pytest.raises(ValueError, match="direct URL"):
        validate_package_metadata(
            PackageMetadata(name="demo", python_dependencies=[f"demo @ {url}"]),
            allow_direct_urls=False,
        )


@pytest.mark.parametrize("source_kind", ["git", "registry"])
def test_remote_package_rejects_url_dependencies_before_installation(
    tmp_path, monkeypatch, source_kind
):
    import kedi.package_registry as packages
    from tests.test_package_dependencies import package_source

    manifest = package_source(tmp_path / "source", "example @ file:///tmp/untrusted.whl")
    installed = []
    monkeypatch.setattr(packages, "install_python_dependencies", installed.append)
    with pytest.raises(ValueError, match="direct URL"):
        packages.install_package(
            manifest, source_kind=source_kind, registry_root=tmp_path / "registry"
        )
    assert installed == []
    assert not (tmp_path / "registry").exists()


def test_nested_procedure_python_is_found_by_form_safety_walk():
    program = parse_program("@helper():\n  = `1 + 1`\n")
    assert "forms cannot contain Python" in _python_use([ProcDefStmt(proc=program.procedures[0])])


def test_tree_sitter_parser_is_owned_by_its_thread():
    barrier = Barrier(4)

    def parse(index):
        barrier.wait(timeout=5)
        instance = parser._get_parser()
        source = f"= worker {index}\n".encode()
        for _ in range(50):
            assert parser.parse(source).root_node.text == source
        return instance

    with ThreadPoolExecutor(max_workers=4) as workers:
        instances = list(workers.map(parse, range(4)))
    assert len({id(instance) for instance in instances}) == 4


def test_registry_rejects_redirect_before_contacting_another_origin():
    seen = []

    class Target(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args):
            pass

    target = ThreadingHTTPServer(("127.0.0.1", 0), Target)

    class Redirect(Target):
        def do_GET(self):
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{target.server_port}/private")
            self.end_headers()

    origin = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
    threads = [Thread(target=server.serve_forever, daemon=True) for server in (target, origin)]
    for thread in threads:
        thread.start()
    try:
        client = HTTPRegistryAPIClient(f"http://127.0.0.1:{origin.server_port}")
        with pytest.raises(RegistryAPIError, match="origin"):
            client._get_json(client._url("start"))
        assert seen == []
    finally:
        for server, thread in zip((target, origin), threads, strict=True):
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


def test_registry_preserves_same_origin_redirects():
    from urllib.request import Request

    from kedi.registry_api import _SameOriginRedirectHandler

    request = _SameOriginRedirectHandler().redirect_request(
        Request("https://registry.example.test/old"),
        None,
        302,
        "Found",
        {},
        "https://registry.example.test/new",
    )
    assert request.full_url == "https://registry.example.test/new"
