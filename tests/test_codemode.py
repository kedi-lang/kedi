from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
from dataclasses import replace
from typing import Any, cast

import pytest
from mcp.server.fastmcp import FastMCP
from pydantic_ai.capabilities import MCP
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import (
    InstructionPart,
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import RunContext, Tool, ToolDenied
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset, ToolsetTool
from pydantic_ai.usage import RunUsage

import kedi.agent_adapter.codemode.pydantic as pydantic_codemode_module
from kedi.agent_adapter.adapters._pydantic_compat import (
    PydanticAICompatibilityError,
    build_nested_tool_manager,
)
from kedi.agent_adapter.adapters.pydantic import (
    PydanticAdapter,
    pydantic_toolset_from_specs,
)
from kedi.agent_adapter.approval import ApprovalDecision, ApprovalPolicy, ApprovalRequest
from kedi.agent_adapter.codemode import (
    CodeModeBoundaryError,
    CodeModeCatalog,
    CodeModeCatalogError,
    CodeModeCursorError,
    CodeModeExecutionError,
    CodeModeHydrationError,
    CodeModeLimits,
    CodeModeSession,
    CodeModeTool,
)
from kedi.agent_adapter.codemode.core import _name_score, _SequentialDispatchGate
from kedi.agent_adapter.codemode.pydantic import PydanticCodeModeToolset
from kedi.agent_adapter.tool_override import (
    ToolSpec,
    current_required_tool_calls,
    with_artifact_result,
)
from kedi.agent_profile import AgentProfile, CodeModeSettings
from kedi.artifacts import ArtifactManager, ArtifactPolicy
from tests.telemetry_helpers import installed_recording_backend


def _integer_tool(name: str, *, sequential: bool = False) -> CodeModeTool:
    return CodeModeTool(
        name=name,
        description=f"Return a value from {name}.",
        parameters_json_schema={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        sequential=sequential,
    )


async def _unused_model(
    _messages: list[ModelMessage],
    _info: AgentInfo,
) -> ModelResponse:
    return ModelResponse(parts=[TextPart("unused")])


def _run_context() -> RunContext[Any]:
    return RunContext(
        deps=None,
        model=FunctionModel(_unused_model),
        usage=RunUsage(),
    )


class _StubToolset(AbstractToolset[Any]):
    def __init__(
        self,
        *,
        tools: dict[str, ToolsetTool[Any]] | None = None,
        instructions: str | InstructionPart | list[str | InstructionPart] | None = None,
        next_step: AbstractToolset[Any] | None = None,
        result: Any = None,
    ) -> None:
        self._tools = tools or {}
        self._instructions = instructions
        self.next_step = next_step
        self.result = result

    @property
    def id(self) -> str | None:
        return "stub"

    async def for_run_step(self, ctx: RunContext[Any]) -> AbstractToolset[Any]:
        del ctx
        return self.next_step or self

    async def get_instructions(
        self,
        ctx: RunContext[Any],
    ) -> str | InstructionPart | list[str | InstructionPart] | None:
        del ctx
        return self._instructions

    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
        del ctx
        return self._tools

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[Any],
        tool: ToolsetTool[Any],
    ) -> Any:
        del name, tool_args, ctx, tool
        return self.result


def _signed_cursor(catalog: CodeModeCatalog, payload: dict[str, Any]) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    signature = hmac.new(catalog._secret, body, hashlib.sha256).digest()[:16]
    return base64.urlsafe_b64encode(body + signature).decode().rstrip("=")


def test_codemode_catalog_paginates_and_binds_cursors_to_snapshot() -> None:
    session = CodeModeSession(CodeModeLimits(default_search_limit=2))
    session.update_tools([_integer_tool(name) for name in ("alpha", "beta", "gamma")])

    first = session.search_tools()
    assert first["tool_names"] == ["alpha", "beta"]
    assert first["next_cursor"] is not None
    assert session.search_tools(cursor=first["next_cursor"])["tool_names"] == ["gamma"]

    with pytest.raises(CodeModeCursorError, match="another search query"):
        session.search_tools(query="alpha", cursor=first["next_cursor"])

    cursor = first["next_cursor"]
    session.update_tools([_integer_tool(name) for name in ("alpha", "delta")])
    with pytest.raises(CodeModeCursorError, match="stale tool catalog"):
        session.search_tools(cursor=cursor)


def test_codemode_catalog_validates_limits_and_authenticated_cursor_fields() -> None:
    catalog = CodeModeCatalog([_integer_tool(name) for name in ("alpha", "beta")])
    first = catalog.search(
        query=None,
        limit=1,
        cursor=None,
        default_limit=1,
        max_limit=10,
    )

    with pytest.raises(CodeModeCursorError, match="another page size"):
        catalog.search(
            query=None,
            limit=2,
            cursor=first["next_cursor"],
            default_limit=1,
            max_limit=10,
        )
    for invalid_limit in (True, "1", 0, 11):
        with pytest.raises(CodeModeCursorError, match="limit"):
            catalog.search(
                query=None,
                limit=invalid_limit,  # type: ignore[arg-type]
                cursor=None,
                default_limit=1,
                max_limit=10,
            )

    cursor = first["next_cursor"]
    assert isinstance(cursor, str)
    midpoint = len(cursor) // 2
    replacement = "A" if cursor[midpoint] != "A" else "B"
    with pytest.raises(CodeModeCursorError, match="Invalid CodeMode search cursor"):
        catalog.search(
            query=None,
            limit=None,
            cursor=f"{cursor[:midpoint]}{replacement}{cursor[midpoint + 1 :]}",
            default_limit=1,
            max_limit=10,
        )

    base = {"version": catalog.version, "query": "", "limit": 1, "offset": 1}
    malformed = (
        {**base, "query": 1},
        {**base, "limit": "1"},
        {**base, "offset": -1},
        {"version": catalog.version},
    )
    for payload in malformed:
        with pytest.raises(CodeModeCursorError, match="Invalid CodeMode search cursor"):
            catalog.search(
                query=None,
                limit=None,
                cursor=_signed_cursor(catalog, payload),
                default_limit=1,
                max_limit=10,
            )


def test_codemode_catalog_search_ranking_and_python_aliases_are_deterministic() -> None:
    tools = [
        _integer_tool("read"),
        _integer_tool("read_file"),
        _integer_tool("filesystem_read_file"),
        _integer_tool("breadth"),
        _integer_tool("other"),
    ]
    catalog = CodeModeCatalog(tools)

    assert catalog.search(query="read", limit=10, cursor=None, default_limit=10, max_limit=10)[
        "tool_names"
    ] == ["read", "filesystem_read_file", "read_file", "breadth"]
    assert catalog.search(query="read file", limit=10, cursor=None, default_limit=10, max_limit=10)[
        "tool_names"
    ] == ["read_file", "filesystem_read_file", "breadth", "read"]
    assert catalog.search(query="file read", limit=10, cursor=None, default_limit=10, max_limit=10)[
        "tool_names"
    ] == ["filesystem_read_file", "read_file", "breadth", "read"]
    assert catalog.search(query="", limit=10, cursor=None, default_limit=10, max_limit=10)[
        "tool_names"
    ] == sorted(tool.name for tool in tools)
    assert CodeModeTool("123-tool", None, {}).callable_name == "_123_tool"
    assert CodeModeTool("class", None, {}).callable_name == "class_"
    assert catalog.search(query="rea", limit=10, cursor=None, default_limit=10, max_limit=10)[
        "tool_names"
    ][:2] == ["read", "read_file"]
    assert _name_score("read", "") == 1


def test_codemode_catalog_rejects_reserved_and_ambiguous_names() -> None:
    with pytest.raises(CodeModeCatalogError, match="reserved"):
        CodeModeCatalog([_integer_tool("execute_code")])
    with pytest.raises(CodeModeCatalogError, match="same Python callable name"):
        CodeModeCatalog([_integer_tool("mcp.read"), _integer_tool("mcp-read")])
    with pytest.raises(CodeModeCatalogError, match="cannot be empty"):
        CodeModeTool("", None, {})
    with pytest.raises(CodeModeCatalogError, match="Duplicate"):
        CodeModeCatalog([_integer_tool("same"), _integer_tool("same")])


def test_codemode_hydration_is_exact_and_exposes_callable_alias() -> None:
    session = CodeModeSession()
    session.update_tools([_integer_tool("mcp__files__read")])

    schema = session.get_tool_schema(tool_names=["mcp__files__read"])
    assert schema["tools"][0]["callable_name"] == "mcp__files__read"
    assert session.hydrated_names == {"mcp__files__read"}
    with pytest.raises(CodeModeHydrationError, match="Unknown CodeMode tool"):
        session.get_tool_schema(tool_names=["missing"])
    with pytest.raises(CodeModeHydrationError, match="duplicate"):
        session.get_tool_schema(tool_names=["mcp__files__read", "mcp__files__read"])


def test_codemode_bounds_discovery_payloads_and_validates_limits() -> None:
    with pytest.raises(ValueError, match="default_search_limit"):
        CodeModeLimits(default_search_limit=2, max_search_limit=1)
    with pytest.raises(ValueError, match="must be positive"):
        CodeModeLimits(max_code_chars=0)
    with pytest.raises(ValueError, match="request_timeout"):
        CodeModeLimits(request_timeout=0)

    session = CodeModeSession(CodeModeLimits(max_discovery_result_bytes=100))
    session.update_tools(
        [
            CodeModeTool(
                name="large_schema",
                description="x" * 200,
                parameters_json_schema={"type": "object", "properties": {}},
            )
        ]
    )
    with pytest.raises(CodeModeBoundaryError, match="get_tool_schema"):
        session.get_tool_schema(tool_names=["large_schema"])

    hydration = CodeModeSession(CodeModeLimits(max_hydrated_tools=1))
    hydration.update_tools([_integer_tool("first"), _integer_tool("second")])
    hydration.get_tool_schema(tool_names=["first"])
    with pytest.raises(CodeModeHydrationError, match="at most 1"):
        hydration.get_tool_schema(tool_names=["second"])
    assert hydration.catalog.tools.keys() == {"first", "second"}


def test_codemode_executes_hydrated_tools_and_preserves_state() -> None:
    async def run() -> None:
        session = CodeModeSession()
        session.update_tools([_integer_tool("math.double")])
        session.get_tool_schema(tool_names=["math.double"])

        async def invoke(name: str, arguments: dict[str, Any]) -> Any:
            assert name == "math.double"
            return arguments["value"] * 2

        first = await session.execute_code(
            code="base = await math_double(value=4)\nbase",
            invoke=invoke,
        )
        second = await session.execute_code(code="base + 1", invoke=invoke)
        assert (first, second) == (8, 9)

        assert await session.execute_code(code="base", invoke=invoke, restart=True) == {
            "output": "",
            "error": "NameError: name 'base' is not defined",
        }
        await session.close()

    asyncio.run(run())


def test_codemode_chains_dependent_tools_and_bounds_parallel_fanout() -> None:
    async def run() -> None:
        session = CodeModeSession(CodeModeLimits(max_nested_calls=8, max_concurrent_calls=2))
        session.update_tools(
            [
                CodeModeTool(
                    name="find_user",
                    description="Find a user by name.",
                    parameters_json_schema={
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                        "additionalProperties": False,
                    },
                ),
                CodeModeTool(
                    name="list_orders",
                    description="List orders for a user.",
                    parameters_json_schema={
                        "type": "object",
                        "properties": {"user_id": {"type": "integer"}},
                        "required": ["user_id"],
                        "additionalProperties": False,
                    },
                ),
                CodeModeTool(
                    name="order_total",
                    description="Return the total for an order.",
                    parameters_json_schema={
                        "type": "object",
                        "properties": {"order_id": {"type": "integer"}},
                        "required": ["order_id"],
                        "additionalProperties": False,
                    },
                ),
            ]
        )
        session.get_tool_schema(tool_names=["find_user", "list_orders", "order_total"])
        calls: list[tuple[str, dict[str, Any]]] = []
        active_totals = 0
        max_active_totals = 0

        async def invoke(name: str, arguments: dict[str, Any]) -> Any:
            nonlocal active_totals, max_active_totals
            calls.append((name, arguments))
            if name == "find_user":
                assert arguments == {"name": "Ada"}
                return {"id": 7, "name": "Ada"}
            if name == "list_orders":
                assert arguments == {"user_id": 7}
                return [{"id": order_id} for order_id in (11, 12, 13, 14)]
            assert name == "order_total"
            active_totals += 1
            max_active_totals = max(max_active_totals, active_totals)
            await asyncio.sleep(0.01)
            active_totals -= 1
            return arguments["order_id"] * 10

        summary = await session.execute_code(
            code=(
                "import asyncio\n"
                'user = await find_user(name="Ada")\n'
                'orders = await list_orders(user_id=user["id"])\n'
                "totals = await asyncio.gather(\n"
                '    order_total(order_id=orders[0]["id"]),\n'
                '    order_total(order_id=orders[1]["id"]),\n'
                '    order_total(order_id=orders[2]["id"]),\n'
                '    order_total(order_id=orders[3]["id"]),\n'
                ")\n"
                'summary = {"user": user["name"], "count": len(totals), "total": sum(totals)}\n'
                "summary"
            ),
            invoke=invoke,
        )

        assert summary == {"user": "Ada", "count": 4, "total": 500}
        assert [name for name, _arguments in calls[:2]] == ["find_user", "list_orders"]
        assert sorted(
            arguments["order_id"] for name, arguments in calls if name == "order_total"
        ) == [
            11,
            12,
            13,
            14,
        ]
        assert max_active_totals == 2

        calls_before_reuse = len(calls)
        assert await session.execute_code(code='summary["total"]', invoke=invoke) == 500
        assert len(calls) == calls_before_reuse
        await session.close()

    asyncio.run(run())


def test_codemode_rejects_unhydrated_and_non_json_tool_results() -> None:
    async def run() -> None:
        session = CodeModeSession()
        session.update_tools([_integer_tool("lookup")])

        async def invoke(_name: str, _arguments: dict[str, Any]) -> Any:
            return object()

        with pytest.raises(CodeModeHydrationError, match="Hydrate"):
            await session.execute_code(code="await lookup(value=1)", invoke=invoke)
        session.get_tool_schema(tool_names=["lookup"])
        with pytest.raises(CodeModeExecutionError, match="JSON-compatible"):
            await session.execute_code(code="await lookup(value=1)", invoke=invoke)
        await session.close()

    asyncio.run(run())


def test_codemode_bounds_calls_and_serializes_sequential_tools() -> None:
    async def run() -> None:
        session = CodeModeSession(CodeModeLimits(max_nested_calls=2, max_concurrent_calls=4))
        session.update_tools([_integer_tool("first", sequential=True), _integer_tool("second")])
        session.get_tool_schema(tool_names=["first", "second"])
        active = 0
        max_active = 0

        async def invoke(_name: str, arguments: dict[str, Any]) -> Any:
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return arguments["value"]

        result = await session.execute_code(
            code=(
                "import asyncio\n"
                "values = await asyncio.gather(first(value=1), second(value=2))\n"
                "sum(values)"
            ),
            invoke=invoke,
        )
        assert result == 3
        assert max_active == 1

        aliased = await session.execute_code(
            code=(
                "import asyncio\n"
                "indirect = first\n"
                "values = await asyncio.gather(indirect(value=3), indirect(value=4))\n"
                "sum(values)"
            ),
            invoke=invoke,
            restart=True,
        )
        assert aliased == 7
        assert max_active == 1

        active = 0
        max_active = 0
        unrelated = await session.execute_code(
            code=(
                "import asyncio\n"
                "values = await asyncio.gather(second(value=5), second(value=6))\n"
                "sum(values)"
            ),
            invoke=invoke,
            restart=True,
        )
        assert unrelated == 11
        assert max_active == 2

        with pytest.raises(CodeModeExecutionError, match="at most 2 nested calls"):
            await session.execute_code(
                code=("await first(value=1)\nawait first(value=2)\nawait first(value=3)"),
                invoke=invoke,
                restart=True,
            )
        await session.close()

    asyncio.run(run())


def test_codemode_sequential_gate_releases_cancelled_waiter() -> None:
    async def run() -> None:
        gate = _SequentialDispatchGate()
        parallel_entered = asyncio.Event()
        release_parallel = asyncio.Event()

        async def hold_parallel() -> None:
            async with gate.enter(sequential=False):
                parallel_entered.set()
                await release_parallel.wait()

        holder = asyncio.create_task(hold_parallel())
        await parallel_entered.wait()

        async def wait_exclusively() -> None:
            async with gate.enter(sequential=True):
                raise AssertionError("cancelled sequential waiter entered the gate")

        waiter = asyncio.create_task(wait_exclusively())
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

        second_parallel_entered = False
        async with gate.enter(sequential=False):
            second_parallel_entered = True
        assert second_parallel_entered

        release_parallel.set()
        await holder

    asyncio.run(run())


def test_codemode_preserves_first_concurrent_host_error() -> None:
    async def run() -> None:
        session = CodeModeSession(CodeModeLimits(max_concurrent_calls=2))
        session.update_tools([_integer_tool("first"), _integer_tool("second")])
        session.get_tool_schema(tool_names=["first", "second"])
        arrived = 0
        release = asyncio.Event()

        async def invoke(name: str, _arguments: dict[str, Any]) -> Any:
            nonlocal arrived
            arrived += 1
            if arrived == 2:
                release.set()
            await release.wait()
            raise RuntimeError(f"{name} host failure")

        with pytest.raises(CodeModeExecutionError, match="host failure"):
            await session.execute_code(
                code=("import asyncio\nawait asyncio.gather(first(value=1), second(value=2))"),
                invoke=invoke,
            )
        assert arrived == 2
        await session.close()

    asyncio.run(run())


def test_codemode_bounds_code_arguments_and_nested_result_bytes() -> None:
    async def run() -> None:
        session = CodeModeSession(
            CodeModeLimits(
                max_code_chars=100,
                max_tool_result_bytes=6,
                max_total_tool_result_bytes=9,
            )
        )
        session.update_tools([_integer_tool("lookup")])
        session.get_tool_schema(tool_names=["lookup"])

        async def small(_name: str, _arguments: dict[str, Any]) -> Any:
            return "abc"

        with pytest.raises(CodeModeExecutionError, match="100 code characters"):
            await session.execute_code(code="x" * 101, invoke=small)
        with pytest.raises(CodeModeExecutionError, match="keyword arguments"):
            await session.execute_code(code="await lookup(1)", invoke=small)

        async def large(_name: str, _arguments: dict[str, Any]) -> Any:
            return "abcde"

        with pytest.raises(CodeModeExecutionError, match="per nested result"):
            await session.execute_code(
                code="await lookup(value=1)",
                invoke=large,
                restart=True,
            )
        with pytest.raises(CodeModeExecutionError, match="aggregate byte budget"):
            await session.execute_code(
                code="await lookup(value=1)\nawait lookup(value=2)",
                invoke=small,
                restart=True,
            )
        syntax_error = await session.execute_code(
            code="await lookup(",
            invoke=small,
            restart=True,
        )
        assert syntax_error["output"] == ""
        assert syntax_error["error"] == "unexpected EOF while parsing"
        await session.close()

    asyncio.run(run())


def test_codemode_returns_print_output_without_losing_expression_result() -> None:
    async def run() -> None:
        session = CodeModeSession()

        async def invoke(_name: str, _arguments: dict[str, Any]) -> Any:
            raise AssertionError("no host tool should be called")

        assert await session.execute_code(code="print('notice')\n42", invoke=invoke) == {
            "output": "notice",
            "result": 42,
        }
        assert await session.execute_code(code="print('notice')", invoke=invoke) == {
            "output": "notice"
        }
        await session.close()

    asyncio.run(run())


def test_codemode_returns_captured_output_separately_from_sandbox_error() -> None:
    async def run() -> None:
        session = CodeModeSession()

        async def invoke(_name: str, _arguments: dict[str, Any]) -> Any:
            raise AssertionError("no host tool should be called")

        result = await session.execute_code(
            code="print('loaded rows')\nmissing_value",
            invoke=invoke,
        )

        assert result == {
            "output": "loaded rows",
            "error": "NameError: name 'missing_value' is not defined",
        }
        await session.close()

    asyncio.run(run())


def test_codemode_direct_cancellation_cancels_active_host_call() -> None:
    async def run() -> None:
        session = CodeModeSession()
        session.update_tools([_integer_tool("slow")])
        session.get_tool_schema(tool_names=["slow"])
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def invoke(_name: str, _arguments: dict[str, Any]) -> Any:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        task = asyncio.create_task(session.execute_code(code="await slow(value=1)", invoke=invoke))
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
        await asyncio.wait_for(cancelled.wait(), timeout=5)
        await session.close()

    asyncio.run(run())


def test_pydantic_codemode_hides_application_tools_and_executes_nested_call() -> None:
    seen_tools: list[list[str]] = []
    calls: list[tuple[int, int]] = []
    step = 0

    async def respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal step
        seen_tools.append([tool.name for tool in info.function_tools])
        step += 1
        if step == 1:
            return ModelResponse(parts=[ToolCallPart("search_tools", {"limit": 10})])
        if step == 2:
            return ModelResponse(parts=[ToolCallPart("get_tool_schema", {"tool_names": ["add"]})])
        if step == 3:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "execute_code",
                        {"code": "value = await add(a=2, b=3)\nvalue * 2"},
                    )
                ]
            )
        return ModelResponse(parts=[TextPart("done")])

    def add(*, a: int, b: int) -> int:
        calls.append((a, b))
        return a + b

    adapter = PydanticAdapter(FunctionModel(respond), tools=[Tool(add)], codemode=True)
    with adapter.profile_override(AgentProfile(approval=ApprovalPolicy.allow())):
        result = adapter.run_sync("Compute the value.")

    assert result.output == "done"
    assert calls == [(2, 3)]
    assert seen_tools == [["search_tools", "get_tool_schema", "execute_code"]] * 4


def test_pydantic_codemode_wrapper_preserves_lifecycle_instructions_and_native_tools() -> None:
    async def run() -> None:
        ctx = _run_context()
        function_toolset = FunctionToolset([Tool(lambda: "ok", name="native_search")])
        function_tools = await function_toolset.get_tools(ctx)
        native = replace(
            function_tools["native_search"],
            tool_def=replace(
                function_tools["native_search"].tool_def,
                tool_kind="tool-search",
            ),
        )
        replacement = _StubToolset(tools={"native_search": native})
        wrapped = _StubToolset(
            tools={"native_search": native},
            instructions=InstructionPart("upstream"),
            next_step=replacement,
            result="passthrough",
        )
        projected = PydanticCodeModeToolset(wrapped=wrapped)

        stepped = await projected.for_run_step(ctx)
        assert isinstance(stepped, PydanticCodeModeToolset)
        assert stepped is not projected
        assert stepped.wrapped is replacement
        assert stepped._session is projected._session
        instructions = await projected.get_instructions(ctx)
        assert isinstance(instructions, list)
        assert instructions[0] == InstructionPart("upstream")

        wrapped._instructions = ["one", InstructionPart("two")]
        instructions = await projected.get_instructions(ctx)
        assert isinstance(instructions, list)
        assert instructions[:2] == ["one", InstructionPart("two")]

        tools = await projected.get_tools(ctx)
        assert "native_search" in tools
        assert (
            await projected.call_tool("native_search", {}, ctx, tools["native_search"])
            == "passthrough"
        )
        await projected.__aenter__()
        assert await projected.__aexit__(None, None, None) is None

    asyncio.run(run())


def test_pydantic_nested_tool_manager_reports_missing_private_contracts() -> None:
    ctx = _run_context()
    toolset = FunctionToolset()
    with pytest.raises(PydanticAICompatibilityError, match="did not attach a tool manager"):
        build_nested_tool_manager(ctx=ctx, toolset=toolset, tools={})

    ctx.tool_manager = cast(Any, object())
    with pytest.raises(PydanticAICompatibilityError, match="unsupported Agent tool layout"):
        build_nested_tool_manager(ctx=ctx, toolset=toolset, tools={})


def test_pydantic_codemode_reports_reserved_and_ambiguous_tool_names() -> None:
    async def run() -> None:
        ctx = _run_context()
        reserved_set = FunctionToolset([Tool(lambda: "x", name="search_tools")])
        reserved = PydanticCodeModeToolset(wrapped=reserved_set)
        with pytest.raises(UserError, match="reserves"):
            await reserved.get_tools(ctx)

        base = FunctionToolset([Tool(lambda: "x", name="source")])
        source = (await base.get_tools(ctx))["source"]
        ambiguous = _StubToolset(
            tools={
                "mcp.read": source,
                "mcp-read": replace(
                    source,
                    tool_def=replace(source.tool_def, name="mcp-read"),
                ),
            }
        )
        projected = PydanticCodeModeToolset(wrapped=ambiguous)
        with pytest.raises(UserError, match="same Python callable name"):
            await projected.get_tools(ctx)

    asyncio.run(run())


@pytest.mark.parametrize("outcome", ["approval_required", "denied"])
def test_pydantic_codemode_normalizes_unresolved_nested_call_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    class Manager:
        async def handle_call(
            self,
            _call: ToolCallPart,
            *,
            wrap_validation_errors: bool,
        ) -> Any:
            assert wrap_validation_errors is False
            if outcome == "approval_required":
                from pydantic_ai.exceptions import ApprovalRequired

                raise ApprovalRequired()
            return ToolDenied("policy rejected the call")

    async def run() -> None:
        ctx = _run_context()
        wrapped = FunctionToolset([Tool(lambda: "ok", name="target")])
        projected = PydanticCodeModeToolset(wrapped=wrapped)
        tools = await projected.get_tools(ctx)
        await projected.call_tool(
            "get_tool_schema",
            {"tool_names": ["target"]},
            ctx,
            tools["get_tool_schema"],
        )
        monkeypatch.setattr(
            pydantic_codemode_module,
            "build_nested_tool_manager",
            lambda **_kwargs: Manager(),
        )
        expected = "inline approval" if outcome == "approval_required" else "denied"
        with pytest.raises(ModelRetry, match=expected):
            await projected.call_tool(
                "execute_code",
                {"code": "await target()"},
                ctx,
                tools["execute_code"],
            )
        await projected._session.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    ("code", "artifacted"),
    [
        ("value = await large_result()\nlen(value)", False),
        ("await large_result()", True),
    ],
)
def test_pydantic_codemode_artifactizes_only_final_result(
    code: str,
    artifacted: bool,
) -> None:
    manager = ArtifactManager()
    policy = ArtifactPolicy(threshold_bytes=100)
    spec = with_artifact_result(
        ToolSpec(fn=lambda: "x" * 10_000, name="large_result"),
        manager=manager,
        policy=policy,
    )
    observed_results: list[Any] = []
    step = 0

    async def respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal step
        step += 1
        if step == 1:
            return ModelResponse(
                parts=[ToolCallPart("get_tool_schema", {"tool_names": ["large_result"]})]
            )
        if step == 2:
            return ModelResponse(parts=[ToolCallPart("execute_code", {"code": code})])
        observed_results.extend(
            part.content
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == "execute_code"
        )
        return ModelResponse(parts=[TextPart("done")])

    adapter = PydanticAdapter(FunctionModel(respond), codemode=True)
    try:
        with adapter.profile_override(AgentProfile(approval=ApprovalPolicy.allow())):
            result = adapter.run_sync(
                "Reduce the large result.",
                toolsets=[pydantic_toolset_from_specs([spec])],
            )

        assert result.output == "done"
        assert len(observed_results) == 1
        if artifacted:
            assert observed_results[0]["ref_id"] == "tool_call_result_1"
        else:
            assert observed_results[0] == 10_000
        tool_events = [
            event
            for event in manager.history.snapshot()
            if event.kind in {"tool_call", "tool_result"}
        ]
        assert [event.data["tool_name"] for event in tool_events] == [
            "execute_code",
            "execute_code",
        ]
        assert tool_events[-1].data["artifacted"] is artifacted
    finally:
        manager.close()


def test_pydantic_codemode_preserves_kedi_approval_edit() -> None:
    calls: list[str] = []
    approvals: list[ApprovalRequest] = []
    step = 0

    async def respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal step
        step += 1
        if step == 1:
            return ModelResponse(
                parts=[ToolCallPart("get_tool_schema", {"tool_names": ["read_file"]})]
            )
        if step == 2:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "execute_code",
                        {"code": "await read_file(path=' .env ')"},
                    )
                ]
            )
        return ModelResponse(parts=[TextPart("done")])

    def read_file(*, path: str) -> str:
        calls.append(path)
        return path

    def approve(request: ApprovalRequest) -> ApprovalDecision:
        approvals.append(request)
        return ApprovalDecision.edit({"path": " README.md "})

    spec = ToolSpec(
        fn=read_file,
        name="read_file",
        description="Read a project file.",
        json_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        risk="sensitive",
        argument_validator=lambda arguments: {"path": str(arguments["path"]).strip()},
    )
    adapter = PydanticAdapter(FunctionModel(respond), codemode=True)
    with adapter.profile_override(AgentProfile(approval=ApprovalPolicy.dynamic(approve))):
        result = adapter.run_sync(
            "Read the file.",
            toolsets=[pydantic_toolset_from_specs([spec])],
        )

    assert result.output == "done"
    assert calls == ["README.md"]
    assert approvals[0].arguments == {"path": ".env"}


def test_pydantic_codemode_preserves_denial_and_nested_tool_metadata() -> None:
    denied_calls = 0
    denial_step = 0

    async def deny_respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal denial_step
        denial_step += 1
        if denial_step == 1:
            return ModelResponse(
                parts=[ToolCallPart("get_tool_schema", {"tool_names": ["dangerous"]})]
            )
        if denial_step == 2:
            return ModelResponse(
                parts=[ToolCallPart("execute_code", {"code": "await dangerous()"})]
            )
        return ModelResponse(parts=[TextPart("denied safely")])

    def dangerous() -> str:
        nonlocal denied_calls
        denied_calls += 1
        return "unexpected"

    denied_adapter = PydanticAdapter(FunctionModel(deny_respond), codemode=True)
    with denied_adapter.profile_override(AgentProfile(approval=ApprovalPolicy.deny())):
        denied = denied_adapter.run_sync(
            "Do not run it.",
            toolsets=[
                pydantic_toolset_from_specs([ToolSpec(dangerous, "dangerous", risk="destructive")])
            ],
        )
    assert denied.output == "denied safely"
    assert denied_calls == 0

    metadata_step = 0
    observed: list[ToolReturnPart] = []

    async def metadata_respond(
        messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> ModelResponse:
        nonlocal metadata_step
        metadata_step += 1
        if metadata_step == 1:
            return ModelResponse(
                parts=[ToolCallPart("get_tool_schema", {"tool_names": ["annotated"]})]
            )
        if metadata_step == 2:
            return ModelResponse(
                parts=[ToolCallPart("execute_code", {"code": "await annotated()"})]
            )
        observed.extend(
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == "execute_code"
        )
        return ModelResponse(parts=[TextPart("metadata retained")])

    def annotated() -> ToolReturn:
        return ToolReturn(return_value={"value": 1}, metadata={"source": "nested"})

    metadata_adapter = PydanticAdapter(
        FunctionModel(metadata_respond),
        tools=[Tool(annotated)],
        codemode=True,
    )
    with metadata_adapter.profile_override(AgentProfile(approval=ApprovalPolicy.allow())):
        result = metadata_adapter.run_sync("Keep metadata.")
    assert result.output == "metadata retained"
    nested_return = next(iter(observed[0].metadata["tool_returns"].values()))
    assert nested_return.metadata == {"source": "nested"}


def test_pydantic_codemode_rejects_multiple_artifact_boundaries() -> None:
    first_manager = ArtifactManager()
    second_manager = ArtifactManager()
    policy = ArtifactPolicy(threshold_bytes=1)
    first = with_artifact_result(
        ToolSpec(lambda: "first", "first"),
        manager=first_manager,
        policy=policy,
    )
    second = with_artifact_result(
        ToolSpec(lambda: "second", "second"),
        manager=second_manager,
        policy=policy,
    )

    async def respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("unreachable")])

    adapter = PydanticAdapter(FunctionModel(respond), codemode=True)
    try:
        with (
            adapter.profile_override(AgentProfile(approval=ApprovalPolicy.allow())),
            pytest.raises(UserError, match="multiple artifact result boundaries"),
        ):
            adapter.run_sync(
                "Reject split stores.",
                toolsets=[
                    pydantic_toolset_from_specs([first]),
                    pydantic_toolset_from_specs([second]),
                ],
            )
    finally:
        first_manager.close()
        second_manager.close()


def test_pydantic_codemode_satisfies_required_tool_tracking() -> None:
    calls = 0
    step = 0

    async def respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal step
        step += 1
        if step == 1:
            return ModelResponse(
                parts=[ToolCallPart("get_tool_schema", {"tool_names": ["required_lookup"]})]
            )
        if step == 2:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "execute_code",
                        {"code": "await required_lookup()"},
                    )
                ]
            )
        return ModelResponse(parts=[TextPart("verified")])

    def required_lookup() -> str:
        nonlocal calls
        calls += 1
        return "source"

    spec = ToolSpec(
        fn=required_lookup,
        name="required_lookup",
        metadata={"required_before_output": True},
    )
    adapter = PydanticAdapter(FunctionModel(respond), codemode=True, retries=2)
    with adapter.profile_override(AgentProfile(approval=ApprovalPolicy.allow())):
        result = adapter.run_sync(
            "Use the required lookup.",
            toolsets=[pydantic_toolset_from_specs([spec])],
        )

    assert result.output == "verified"
    assert calls == 1
    assert current_required_tool_calls() is None


def test_pydantic_codemode_cancellation_unwinds_nested_tool() -> None:
    async def run() -> None:
        step = 0
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            nonlocal step
            step += 1
            if step == 1:
                return ModelResponse(
                    parts=[ToolCallPart("get_tool_schema", {"tool_names": ["slow_tool"]})]
                )
            return ModelResponse(
                parts=[ToolCallPart("execute_code", {"code": "await slow_tool()"})]
            )

        async def slow_tool() -> str:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
            return "unreachable"

        adapter = PydanticAdapter(FunctionModel(respond), codemode=True)

        async def execute() -> None:
            with adapter.profile_override(AgentProfile(approval=ApprovalPolicy.allow())):
                await adapter.run(
                    "Start the slow tool.",
                    toolsets=[pydantic_toolset_from_specs([ToolSpec(slow_tool, "slow_tool")])],
                )

        task = asyncio.create_task(execute())
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
        await asyncio.wait_for(cancelled.wait(), timeout=5)

    asyncio.run(run())


def test_pydantic_codemode_discovers_and_invokes_local_mcp_tool() -> None:
    server = FastMCP("kedi-codemode-test")

    @server.tool()
    def mcp_lookup(query: str) -> dict[str, str]:
        return {"query": query, "source": "mcp"}

    step = 0

    async def respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal step
        assert [tool.name for tool in info.function_tools] == [
            "search_tools",
            "get_tool_schema",
            "execute_code",
        ]
        step += 1
        if step == 1:
            return ModelResponse(parts=[ToolCallPart("search_tools", {"query": "mcp"})])
        if step == 2:
            return ModelResponse(
                parts=[ToolCallPart("get_tool_schema", {"tool_names": ["mcp_lookup"]})]
            )
        if step == 3:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "execute_code",
                        {"code": ("item = await mcp_lookup(query='kedi')\nitem['source']")},
                    )
                ]
            )
        return ModelResponse(parts=[TextPart("done")])

    adapter = PydanticAdapter(FunctionModel(respond), codemode=True)
    with adapter.profile_override(AgentProfile(approval=ApprovalPolicy.allow())):
        result = adapter.run_sync(
            "Use the MCP tool.",
            toolsets=[MCPToolset(server)],
        )

    assert result.output == "done"


def test_pydantic_codemode_rejects_duplicate_mcp_tool_names() -> None:
    first = FastMCP("first")
    second = FastMCP("second")

    @first.tool(name="shared_lookup")
    def first_lookup() -> str:
        return "first"

    @second.tool(name="shared_lookup")
    def second_lookup() -> str:
        return "second"

    async def respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("unreachable")])

    adapter = PydanticAdapter(FunctionModel(respond), codemode=True)
    with (
        adapter.profile_override(AgentProfile(approval=ApprovalPolicy.allow())),
        pytest.raises(UserError, match="name conflicts with existing tool"),
    ):
        adapter.run_sync(
            "Reject ambiguous tools.",
            toolsets=[MCPToolset(first), MCPToolset(second)],
        )


def test_pydantic_codemode_rejects_provider_native_mcp() -> None:
    async def respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("unreachable")])

    adapter = PydanticAdapter(
        FunctionModel(respond),
        capabilities=[MCP("https://example.com/mcp", native=True, local=False)],
        codemode=True,
    )

    with pytest.raises(UserError, match="provider-native MCP"):
        adapter.run_sync("Do not bypass CodeMode.")


def test_pydantic_codemode_rejects_external_approval_resolution() -> None:
    adapter = PydanticAdapter(
        FunctionModel(_unused_model),
        codemode=True,
        approval_resolution="external",
    )
    with pytest.raises(ValueError, match="inline Kedi approval resolution"):
        adapter.run_sync("Approval must stay inline.")


def test_profile_codemode_directive_state_is_not_forwarded_to_model() -> None:
    observed: list[tuple[list[str], dict[str, Any]]] = []

    async def respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        observed.append(
            ([tool.name for tool in info.function_tools], dict(info.model_settings or {}))
        )
        return ModelResponse(parts=[TextPart("done")])

    adapter = PydanticAdapter(FunctionModel(respond), tools=[Tool(lambda: "unused", name="raw")])
    with adapter.profile_override(AgentProfile(codemode=True, settings={"temperature": 0.2})):
        result = adapter.run_sync("Inspect settings.")

    assert result.output == "done"
    assert observed == [(["search_tools", "get_tool_schema", "execute_code"], {"temperature": 0.2})]


def test_profile_codemode_limits_reach_pydantic_control_tools() -> None:
    search_result: dict[str, Any] | None = None
    step = 0

    async def respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal search_result, step
        step += 1
        if step == 1:
            return ModelResponse(parts=[ToolCallPart("search_tools", {})])
        for message in messages:
            for part in message.parts:
                if isinstance(part, ToolReturnPart) and part.tool_name == "search_tools":
                    search_result = cast(dict[str, Any], part.content)
        return ModelResponse(parts=[TextPart("done")])

    adapter = PydanticAdapter(
        FunctionModel(respond),
        tools=[
            Tool(lambda: "alpha", name="alpha"),
            Tool(lambda: "beta", name="beta"),
        ],
    )
    profile = AgentProfile(
        approval=ApprovalPolicy.allow(),
        codemode=CodeModeSettings(default_search_limit=1),
    )
    with adapter.profile_override(profile):
        result = adapter.run_sync("Browse one tool.")

    assert result.output == "done"
    assert search_result is not None
    assert search_result["tool_names"] == ["alpha"]
    assert search_result["next_cursor"] is not None


def test_profile_codemode_disabled_overrides_adapter_default() -> None:
    observed: list[list[str]] = []

    async def respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        observed.append([tool.name for tool in info.function_tools])
        return ModelResponse(parts=[TextPart("done")])

    adapter = PydanticAdapter(
        FunctionModel(respond),
        tools=[Tool(lambda: "raw", name="raw")],
        codemode=True,
    )
    with adapter.profile_override(AgentProfile(codemode=False)):
        result = adapter.run_sync("Use the direct surface.")

    assert result.output == "done"
    assert observed == [["raw"]]


def test_codemode_boundary_error_is_a_codemode_execution_error() -> None:
    assert issubclass(CodeModeBoundaryError, CodeModeExecutionError)


def test_codemode_telemetry_is_payload_free() -> None:
    async def run() -> Any:
        session = CodeModeSession()
        session.update_tools([_integer_tool("echo")])
        session.search_tools(query="private search text")
        session.get_tool_schema(tool_names=["echo"])

        async def invoke(_name: str, arguments: dict[str, Any]) -> Any:
            return arguments["value"]

        try:
            return await session.execute_code(
                code="private_value = await echo(value=1)\nprivate_value",
                invoke=invoke,
            )
        finally:
            await session.close()

    with installed_recording_backend() as backend:
        assert asyncio.run(run()) == 1

    spans = {span.operation: span for span in backend.finished_spans}
    assert {"search_tools", "get_tool_schema", "execute_code"} <= spans.keys()
    assert spans["execute_code"].attributes["kedi.codemode.execute.nested_calls"] == 1
    rendered = repr(backend.finished_spans)
    assert "private search text" not in rendered
    assert "private_value" not in rendered
    assert {metric.name for metric in backend.metrics} >= {
        "kedi.codemode.operations",
        "kedi.codemode.duration",
    }
