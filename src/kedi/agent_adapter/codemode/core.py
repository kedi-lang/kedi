from __future__ import annotations

import ast
import asyncio
import base64
import hashlib
import hmac
import json
import keyword
import re
import secrets
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from pydantic_core import to_jsonable_python
from pydantic_monty import AsyncMonty, AsyncMontySession, CollectString, MontyError

from .errors import (
    CodeModeBoundaryError,
    CodeModeCatalogError,
    CodeModeCursorError,
    CodeModeExecutionError,
    CodeModeHydrationError,
)
from .telemetry import codemode_span

CONTROL_TOOL_NAMES = frozenset({"search_tools", "get_tool_schema", "execute_code"})
_INVALID_IDENTIFIER = re.compile(r"\W+")
_QUERY_TOKEN = re.compile(r"[a-z0-9]+")

CODEMODE_INSTRUCTIONS = """\
CodeMode is enabled. Application tools are available only through three control tools:

1. Call `search_tools` to browse tool names. `query` is optional; use `cursor` for another page.
2. Call `get_tool_schema` with the exact names you intend to use. Do not guess arguments.
3. Call `execute_code` with small, task-focused Python that invokes only hydrated functions.

The sandbox supports ordinary literals, indexing and slicing, arithmetic, comparisons, boolean
logic, if/elif/else, for/while, range/enumerate/zip, comprehensions, sorting, filtering, grouping,
f-strings, and helper functions. Read mapping values with `mapping[key]`; mapping methods such as
`.get()` are unavailable. Tool functions are async; call them with `await`. Chain dependent calls by
passing prior results directly: `user = await find_user(name="Ada")`, then
`orders = await list_orders(user_id=user["id"])`. For independent calls, first `import asyncio`, then
use `await asyncio.gather(...)`. No third-party packages, host filesystem, environment, network, eval,
exec, generators, or unrestricted OS access are available. Return only the value needed by the agent.
Treat `execute_code` calls as persistent notebook cells: assign reusable outputs to variables and
continue from those variables in later calls. Never repeat a successful application tool call merely
to inspect, reformat, or recompute its result; printing or returning a variable does not consume it.
On failure, `execute_code` returns separate `output` and `error` fields. Use the captured output to
locate progress, correct the error, and continue from variables that were already assigned.
"""

ToolInvoker = Callable[[str, dict[str, Any]], Awaitable[Any]]


@dataclass(frozen=True, slots=True)
class CodeModeLimits:
    default_search_limit: int = 10
    max_search_limit: int = 50
    max_hydrated_tools: int = 64
    max_discovery_result_bytes: int = 256_000
    max_code_chars: int = 20_000
    max_nested_calls: int = 64
    max_concurrent_calls: int = 8
    max_tool_result_bytes: int = 256_000
    max_total_tool_result_bytes: int = 1_000_000
    max_print_bytes: int = 256_000
    request_timeout: float = 60.0

    def __post_init__(self) -> None:
        integer_limits = {
            "default_search_limit": self.default_search_limit,
            "max_search_limit": self.max_search_limit,
            "max_hydrated_tools": self.max_hydrated_tools,
            "max_discovery_result_bytes": self.max_discovery_result_bytes,
            "max_code_chars": self.max_code_chars,
            "max_nested_calls": self.max_nested_calls,
            "max_concurrent_calls": self.max_concurrent_calls,
            "max_tool_result_bytes": self.max_tool_result_bytes,
            "max_total_tool_result_bytes": self.max_total_tool_result_bytes,
            "max_print_bytes": self.max_print_bytes,
        }
        invalid = [name for name, value in integer_limits.items() if value <= 0]
        if invalid:
            raise ValueError(f"CodeMode limits must be positive: {', '.join(invalid)}")
        if self.default_search_limit > self.max_search_limit:
            raise ValueError("default_search_limit cannot exceed max_search_limit")
        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be positive")


@dataclass(frozen=True, slots=True)
class CodeModeTool:
    name: str
    description: str | None
    parameters_json_schema: dict[str, Any]
    return_schema: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    sequential: bool = False
    callable_name: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise CodeModeCatalogError("CodeMode tool names cannot be empty")
        object.__setattr__(self, "callable_name", _python_identifier(self.name))

    def schema_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "callable_name": self.callable_name,
            "description": self.description,
            "parameters_json_schema": self.parameters_json_schema,
            "return_schema": self.return_schema,
        }
        if self.sequential:
            payload["sequential"] = True
        return payload

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "callable_name": self.callable_name,
            "description": self.description,
            "parameters": self.parameters_json_schema,
            "returns": self.return_schema,
            "sequential": self.sequential,
        }


class CodeModeCatalog:
    """Immutable tool snapshot with authenticated, snapshot-bound pagination."""

    def __init__(self, tools: Sequence[CodeModeTool], *, secret: bytes | None = None) -> None:
        by_name: dict[str, CodeModeTool] = {}
        by_callable_name: dict[str, CodeModeTool] = {}
        for tool in tools:
            if tool.name in CONTROL_TOOL_NAMES:
                raise CodeModeCatalogError(
                    f"Tool name {tool.name!r} is reserved while CodeMode is enabled"
                )
            if tool.name in by_name:
                raise CodeModeCatalogError(f"Duplicate CodeMode tool name: {tool.name!r}")
            previous = by_callable_name.get(tool.callable_name)
            if previous is not None:
                raise CodeModeCatalogError(
                    "CodeMode tools resolve to the same Python callable name: "
                    f"{previous.name!r} and {tool.name!r} -> {tool.callable_name!r}"
                )
            by_name[tool.name] = tool
            by_callable_name[tool.callable_name] = tool

        self._by_name = by_name
        self._by_callable_name = by_callable_name
        self._ordered_names = tuple(sorted(by_name))
        self._secret = secret or secrets.token_bytes(32)
        encoded = json.dumps(
            [by_name[name].fingerprint_payload() for name in self._ordered_names],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
        self.version = hashlib.sha256(encoded).hexdigest()[:20]

    @property
    def tools(self) -> Mapping[str, CodeModeTool]:
        return self._by_name

    def tool_for_callable(self, name: str) -> CodeModeTool | None:
        return self._by_callable_name.get(name)

    def search(
        self,
        *,
        query: str | None,
        limit: int | None,
        cursor: str | None,
        default_limit: int,
        max_limit: int,
    ) -> dict[str, Any]:
        offset = 0
        effective_query = _normalize_query(query)
        effective_limit = default_limit if limit is None else limit
        if cursor is not None:
            payload = self._decode_cursor(cursor)
            cursor_query = payload["query"]
            if query is not None and effective_query != cursor_query:
                raise CodeModeCursorError("CodeMode cursor belongs to another search query")
            if limit is not None and effective_limit != payload["limit"]:
                raise CodeModeCursorError("CodeMode cursor belongs to another page size")
            effective_query = cursor_query
            effective_limit = payload["limit"]
            offset = payload["offset"]

        if not isinstance(effective_limit, int) or isinstance(effective_limit, bool):
            raise CodeModeCursorError("search_tools limit must be an integer")
        if not 1 <= effective_limit <= max_limit:
            raise CodeModeCursorError(f"search_tools limit must be between 1 and {max_limit}")

        names = self._matching_names(effective_query)
        page = names[offset : offset + effective_limit]
        next_offset = offset + len(page)
        next_cursor = None
        if next_offset < len(names):
            next_cursor = self._encode_cursor(
                query=effective_query,
                limit=effective_limit,
                offset=next_offset,
            )
        return {"tool_names": page, "next_cursor": next_cursor}

    def resolve_exact(self, names: Sequence[str]) -> list[CodeModeTool]:
        if len(set(names)) != len(names):
            raise CodeModeHydrationError("get_tool_schema received duplicate tool names")
        missing = [name for name in names if name not in self._by_name]
        if missing:
            rendered = ", ".join(repr(name) for name in missing)
            raise CodeModeHydrationError(f"Unknown CodeMode tool name(s): {rendered}")
        return [self._by_name[name] for name in names]

    def _matching_names(self, query: str) -> list[str]:
        if not query:
            return list(self._ordered_names)
        scored = [
            (score, name) for name in self._ordered_names if (score := _name_score(name, query)) > 0
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [name for _, name in scored]

    def _encode_cursor(self, *, query: str, limit: int, offset: int) -> str:
        body = json.dumps(
            {
                "version": self.version,
                "query": query,
                "limit": limit,
                "offset": offset,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        signature = hmac.new(self._secret, body, hashlib.sha256).digest()[:16]
        return base64.urlsafe_b64encode(body + signature).decode().rstrip("=")

    def _decode_cursor(self, cursor: str) -> dict[str, Any]:
        try:
            padded = cursor + "=" * (-len(cursor) % 4)
            raw = base64.urlsafe_b64decode(padded.encode())
            body, signature = raw[:-16], raw[-16:]
            expected = hmac.new(self._secret, body, hashlib.sha256).digest()[:16]
            if not hmac.compare_digest(signature, expected):
                raise ValueError("signature mismatch")
            payload = json.loads(body)
            if payload["version"] != self.version:
                raise CodeModeCursorError("CodeMode cursor belongs to a stale tool catalog")
            if not isinstance(payload["query"], str):
                raise ValueError("invalid query")
            if not isinstance(payload["limit"], int) or not isinstance(payload["offset"], int):
                raise ValueError("invalid pagination")
            if payload["offset"] < 0:
                raise ValueError("invalid offset")
            return payload
        except CodeModeCursorError:
            raise
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise CodeModeCursorError("Invalid CodeMode search cursor") from exc


class CodeModeSession:
    """Per-agent-run catalog, hydration, and stateful Monty execution state."""

    def __init__(self, limits: CodeModeLimits | None = None) -> None:
        self.limits = limits or CodeModeLimits()
        self._secret = secrets.token_bytes(32)
        self._catalog = CodeModeCatalog((), secret=self._secret)
        self._hydrated: set[str] = set()
        self._pool: AsyncMonty | None = None
        self._monty_session: AsyncMontySession | None = None
        self._pool_stack = AsyncExitStack()
        self._session_stack = AsyncExitStack()
        self._execution_lock = asyncio.Lock()
        self._active_host_tasks: set[asyncio.Task[Any]] = set()

    @property
    def catalog(self) -> CodeModeCatalog:
        return self._catalog

    @property
    def hydrated_names(self) -> frozenset[str]:
        return frozenset(self._hydrated)

    def update_tools(self, tools: Sequence[CodeModeTool]) -> None:
        catalog = CodeModeCatalog(tools, secret=self._secret)
        if catalog.version == self._catalog.version:
            return
        self._catalog = catalog
        self._hydrated.intersection_update(catalog.tools)

    def search_tools(
        self,
        *,
        query: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        with codemode_span(
            "search tools",
            "search_tools",
            {
                "kedi.codemode.catalog.size": len(self._catalog.tools),
                "kedi.codemode.search.query_provided": bool(query),
                "kedi.codemode.search.cursor_provided": cursor is not None,
            },
        ) as active:
            result = self._catalog.search(
                query=query,
                limit=limit,
                cursor=cursor,
                default_limit=self.limits.default_search_limit,
                max_limit=self.limits.max_search_limit,
            )
            bounded = self._bounded_discovery_result(result, operation="search_tools")
            active.set_attribute("kedi.codemode.search.result_count", len(result["tool_names"]))
            active.set_attribute(
                "kedi.codemode.search.has_next_page",
                result["next_cursor"] is not None,
            )
            return bounded

    def get_tool_schema(self, *, tool_names: Sequence[str]) -> dict[str, Any]:
        with codemode_span(
            "get tool schema",
            "get_tool_schema",
            {"kedi.codemode.schema.requested_count": len(tool_names)},
        ) as active:
            tools = self._catalog.resolve_exact(tool_names)
            hydrated = self._hydrated | {tool.name for tool in tools}
            if len(hydrated) > self.limits.max_hydrated_tools:
                raise CodeModeHydrationError(
                    f"CodeMode allows at most {self.limits.max_hydrated_tools} hydrated tools per run"
                )
            self._hydrated = hydrated
            result = {"tools": [tool.schema_payload() for tool in tools]}
            bounded = self._bounded_discovery_result(result, operation="get_tool_schema")
            active.set_attribute("kedi.codemode.schema.hydrated_count", len(self._hydrated))
            return bounded

    async def execute_code(
        self,
        *,
        code: str,
        invoke: ToolInvoker,
        restart: bool = False,
    ) -> Any:
        if len(code) > self.limits.max_code_chars:
            raise CodeModeExecutionError(
                f"CodeMode accepts at most {self.limits.max_code_chars} code characters"
            )
        if restart:
            await self.reset()

        called_tools = self._called_catalog_tools(code)
        unhydrated = sorted(tool.name for tool in called_tools if tool.name not in self._hydrated)
        if unhydrated:
            rendered = ", ".join(repr(name) for name in unhydrated)
            raise CodeModeHydrationError(
                f"Hydrate these tools with get_tool_schema before execute_code: {rendered}"
            )

        hydrated_tools = [
            self._catalog.tools[name]
            for name in sorted(self._hydrated)
            if name in self._catalog.tools
        ]
        # Tool functions are first-class Monty values, so an indirect call can evade
        # AST name matching. Hydrating a sequential tool therefore serializes the
        # execution's complete dispatch lane.
        serialize_calls = any(tool.sequential for tool in hydrated_tools)
        dispatch_lock = asyncio.Lock()
        dispatch_slots = asyncio.Semaphore(self.limits.max_concurrent_calls)
        call_count = 0
        total_result_bytes = 0
        host_error: Exception | None = None

        async def dispatch(
            tool: CodeModeTool,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> Any:
            nonlocal call_count, total_result_bytes
            if args:
                raise TypeError(f"CodeMode tool {tool.name!r} accepts keyword arguments only")
            call_count += 1
            if call_count > self.limits.max_nested_calls:
                raise CodeModeBoundaryError(
                    f"CodeMode allows at most {self.limits.max_nested_calls} nested calls per execution"
                )

            async def perform() -> Any:
                task = asyncio.current_task()
                if task is not None:  # pragma: no branch - coroutine always has a running task
                    self._active_host_tasks.add(task)
                try:
                    return await invoke(tool.name, kwargs)
                finally:
                    if task is not None:
                        self._active_host_tasks.discard(task)

            if serialize_calls:
                result = await _locked(dispatch_lock, perform)
            else:
                result = await _locked(dispatch_slots, perform)
            jsonable, size = _json_boundary_value(result)
            if size > self.limits.max_tool_result_bytes:
                raise CodeModeBoundaryError(
                    f"Tool {tool.name!r} returned {size} bytes; CodeMode allows "
                    f"{self.limits.max_tool_result_bytes} bytes per nested result"
                )
            total_result_bytes += size
            if total_result_bytes > self.limits.max_total_tool_result_bytes:
                raise CodeModeBoundaryError(
                    "CodeMode nested tool results exceeded the aggregate byte budget"
                )
            return jsonable

        async def call(tool: CodeModeTool, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
            nonlocal host_error
            try:
                return await dispatch(tool, args, kwargs)
            except Exception as exc:
                if host_error is None:
                    host_error = exc
                raise

        external_lookup: dict[str, Any] = {}
        for tool in hydrated_tools:
            external_lookup[tool.callable_name] = _external_tool(tool, call)

        with codemode_span(
            "execute code",
            "execute_code",
            {
                "kedi.codemode.execute.code_chars": len(code),
                "kedi.codemode.execute.restart": restart,
                "kedi.codemode.execute.hydrated_count": len(hydrated_tools),
            },
        ) as active:
            async with self._execution_lock:
                session = await self._get_monty_session()
                capture = CollectString(max_bytes=self.limits.max_print_bytes)
                error: str | None = None
                try:
                    result = await session.feed_run(
                        code,
                        external_lookup=external_lookup,
                        print_callback=capture,
                    )
                except asyncio.CancelledError:
                    await self._cancel_host_tasks()
                    raise
                except MontyError as exc:
                    await self._cancel_host_tasks()
                    if host_error is not None:
                        raise CodeModeExecutionError(str(exc)) from host_error
                    result = None
                    error = str(exc)
            active.set_attribute("kedi.codemode.execute.nested_calls", call_count)
            active.set_attribute(
                "kedi.codemode.execute.nested_result_bytes",
                total_result_bytes,
            )

        output = capture.output.rstrip("\n")
        if error is not None:
            return {"output": output, "error": error}
        if output and result is not None:
            return {"output": output, "result": result}
        if output:
            return {"output": output}
        return result

    async def reset(self) -> None:
        await self._session_stack.aclose()
        self._session_stack = AsyncExitStack()
        self._monty_session = None

    async def close(self) -> None:
        await self._cancel_host_tasks()
        await self.reset()
        await self._pool_stack.aclose()
        self._pool_stack = AsyncExitStack()
        self._pool = None

    async def _get_monty_session(self) -> AsyncMontySession:
        if self._pool is None:
            self._pool = await self._pool_stack.enter_async_context(
                AsyncMonty(request_timeout=self.limits.request_timeout)
            )
        if self._monty_session is None:
            self._monty_session = await self._session_stack.enter_async_context(
                self._pool.checkout(script_name="kedi_codemode.py")
            )
        return self._monty_session

    async def _cancel_host_tasks(self) -> None:
        current = asyncio.current_task()
        tasks = [
            task for task in self._active_host_tasks if task is not current and not task.done()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _called_catalog_tools(self, code: str) -> list[CodeModeTool]:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []
        tools: dict[str, CodeModeTool] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                tool = self._catalog.tool_for_callable(node.func.id)
                if tool is not None:
                    tools[tool.name] = tool
        return list(tools.values())

    def _bounded_discovery_result(
        self,
        result: dict[str, Any],
        *,
        operation: str,
    ) -> dict[str, Any]:
        jsonable, size = _json_boundary_value(result)
        if size > self.limits.max_discovery_result_bytes:
            raise CodeModeBoundaryError(
                f"{operation} returned {size} bytes; CodeMode allows "
                f"{self.limits.max_discovery_result_bytes} discovery bytes"
            )
        return jsonable


def _external_tool(
    tool: CodeModeTool,
    call: Callable[[CodeModeTool, tuple[Any, ...], dict[str, Any]], Awaitable[Any]],
) -> Callable[..., Awaitable[Any]]:
    async def invoke(*args: Any, **kwargs: Any) -> Any:
        return await call(tool, args, kwargs)

    invoke.__name__ = tool.callable_name
    invoke.__doc__ = tool.description
    return invoke


async def _locked(
    lock: asyncio.Lock | asyncio.Semaphore,
    fn: Callable[[], Awaitable[Any]],
) -> Any:
    async with lock:
        return await fn()


def _json_boundary_value(value: Any) -> tuple[Any, int]:
    def unsupported(item: Any) -> Any:
        raise TypeError(f"unsupported value {type(item).__name__}")

    try:
        jsonable = to_jsonable_python(value, fallback=unsupported)
        encoded = json.dumps(
            jsonable,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as exc:
        raise CodeModeBoundaryError(
            "Nested tool results must be JSON-compatible before entering CodeMode"
        ) from exc
    return jsonable, len(encoded)


def _python_identifier(name: str) -> str:
    candidate = _INVALID_IDENTIFIER.sub("_", name)
    if candidate and candidate[0].isdigit():
        candidate = f"_{candidate}"
    if keyword.iskeyword(candidate):
        candidate = f"{candidate}_"
    return candidate or "_"


def _normalize_query(query: str | None) -> str:
    return " ".join(_QUERY_TOKEN.findall((query or "").casefold()))


def _name_score(name: str, query: str) -> int:
    normalized_name = " ".join(_QUERY_TOKEN.findall(name.casefold()))
    if normalized_name == query:
        return 100
    name_tokens = normalized_name.split()
    query_tokens = query.split()
    if not query_tokens:
        return 1
    if all(token in name_tokens for token in query_tokens):
        return 80
    if normalized_name.startswith(query):
        return 60
    if query in normalized_name:
        return 40
    overlap = sum(1 for token in query_tokens if any(token in part for part in name_tokens))
    return overlap * 10
