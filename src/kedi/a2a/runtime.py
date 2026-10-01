from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Awaitable, Callable, cast

from kedi.agent_adapter.conversation import ConversationState
from kedi.agent_adapter.execution import run_sync
from kedi.agent_adapter.meta import AgentAdapter
from kedi.agent_adapter.registry import build_adapter
from kedi.agent_adapter.subagents import (
    SubagentUsageLimits,
    SupportsSubagentExecution,
    validate_final_schema,
    validate_structured_result,
)
from kedi.agent_profile import AgentProfile
from kedi.agent_surface import AgentSurface
from kedi.artifacts import ArtifactManager
from kedi.lang import compile_program, parse_program
from kedi.module_registry import resolve_module_path
from kedi.source_loading import load_kedi_source_bundle

from ._storage_io import finish_storage_operation
from .protocol import DEFAULT_A2A_MAX_RESPONSE_BYTES, response_size

if TYPE_CHECKING:
    from kedi.core import KediRuntime

    from .persistence import A2APersistence


DEFAULT_A2A_MAX_INPUT_CHARS = 128_000
DEFAULT_A2A_MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024
DEFAULT_A2A_MAX_PENDING_TASKS = 64
DEFAULT_A2A_MAX_SESSIONS = 1_024


@dataclass(slots=True)
class _ServedSession:
    surface: AgentSurface
    lock: asyncio.Lock
    artifacts: ArtifactManager
    conversation: ConversationState | None
    borrowers: int = 0
    loaded: bool = False
    poisoned: bool = False

    def close(self) -> None:
        self.surface.close()
        if self.conversation is not None:
            self.conversation.close()
        if self.artifacts is not None:
            self.artifacts.close()


class A2AServedProgram:
    """A compiled, exported Kedi agent profile exposed as an A2A service."""

    def __init__(
        self,
        *,
        runtime: KediRuntime,
        entry: str,
        description: str,
        profile: AgentProfile,
        adapter: AgentAdapter[Any],
        tools: tuple[Any, ...],
        max_concurrency: int,
        max_pending_tasks: int = DEFAULT_A2A_MAX_PENDING_TASKS,
        max_sessions: int = DEFAULT_A2A_MAX_SESSIONS,
        max_input_chars: int = DEFAULT_A2A_MAX_INPUT_CHARS,
        max_response_bytes: int = DEFAULT_A2A_MAX_RESPONSE_BYTES,
        owns_adapter: bool = True,
        source_fingerprint: str = "",
        model_identity: str = "",
    ) -> None:
        if max_concurrency < 1 or max_pending_tasks < 1 or max_sessions < 1:
            raise ValueError("A2A concurrency, pending-task, and session limits must be positive")
        if max_pending_tasks < max_concurrency:
            raise ValueError("A2A pending-task limit cannot be below max concurrency")
        if max_input_chars < 1 or max_response_bytes < 1:
            raise ValueError("A2A input and response limits must be positive")
        if not isinstance(adapter, SupportsSubagentExecution):
            raise TypeError("Served A2A adapters must implement Kedi child execution")
        self.runtime = runtime
        self.entry = entry
        self.description = description
        self.profile = profile
        self.adapter = adapter
        self._execution_adapter = cast(SupportsSubagentExecution, adapter)
        self.tools = tools
        self.max_input_chars = max_input_chars
        self.max_response_bytes = max_response_bytes
        self._owns_adapter = owns_adapter
        self.source_fingerprint = source_fingerprint
        self.model_identity = model_identity
        self._persistence: A2APersistence | None = None
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._max_pending_tasks = max_pending_tasks
        self._pending_tasks = 0
        self._pending_by_principal: dict[str, int] = {}
        self._pending_lock = asyncio.Lock()
        self._max_sessions = max_sessions
        self.max_sessions = max_sessions
        self._sessions: OrderedDict[tuple[str, str], _ServedSession] = OrderedDict()
        self._sessions_lock = Lock()
        self._closed = False
        self._invocations: set[asyncio.Task[Any]] = set()
        self._close_task: asyncio.Task[None] | None = None
        self._limits = SubagentUsageLimits().capped()

    async def invoke(
        self,
        *,
        principal: str,
        context_id: str,
        prompt: str,
        output_schema: dict[str, Any] | None,
        on_complete: Callable[[Any, dict[str, Any]], Awaitable[None]] | None = None,
    ) -> str | dict[str, Any]:
        if not prompt.strip():
            raise ValueError("A2A task text cannot be empty")
        if len(prompt) > self.max_input_chars:
            raise ValueError(f"A2A task exceeds the {self.max_input_chars}-character input limit")
        if output_schema is not None:
            validate_final_schema(output_schema)
        await self._reserve_task(principal)
        session: _ServedSession | None = None
        try:
            session = self._session(principal, context_id)
            try:
                async with session.lock:
                    if self._persistence is not None and (not session.loaded or session.poisoned):
                        await self._restore_session(session, principal, context_id)
                    async with self._semaphore:
                        run_kwargs = dict(self._execution_adapter.subagent_run_kwargs(self._limits))
                        with session.surface.run_scope():
                            if output_schema is None:
                                result: Any = await self._execution_adapter.invoke(
                                    prompt=prompt,
                                    **run_kwargs,
                                )
                                if not isinstance(result, str):
                                    result = str(result)
                            else:
                                result = await self._execution_adapter.produce_subagent(
                                    template=prompt,
                                    output_schema=output_schema,
                                    **run_kwargs,
                                )
                                result = validate_structured_result(result, output_schema)
                        if response_size(result) > self.max_response_bytes:
                            raise ValueError(
                                f"A2A result exceeds the {self.max_response_bytes}-byte response limit"
                            )
                        if self._persistence is not None:
                            if on_complete is None:
                                raise RuntimeError(
                                    "Durable A2A execution requires a commit callback"
                                )
                            checkpoint = await finish_storage_operation(
                                asyncio.to_thread(self._session_checkpoint, session)
                            )
                            await on_complete(result, checkpoint)
                            try:
                                await finish_storage_operation(
                                    asyncio.to_thread(
                                        session.artifacts.prune_inactive_payloads,
                                        checkpoint["artifacts"],
                                    )
                                )
                            except Exception:
                                logging.getLogger(__name__).warning(
                                    "Committed A2A task could not prune inactive artifacts",
                                    exc_info=True,
                                )
            except BaseException:
                if self._persistence is not None:
                    session.poisoned = True
                raise
        finally:
            if session is not None:
                with self._sessions_lock:
                    session.borrowers -= 1
            await self._release_task(principal)
        return result

    def configure_persistence(self, state: A2APersistence) -> None:
        if self._persistence is not None:
            raise RuntimeError("A2A persistence is already configured")
        if self.profile.history is True and self.adapter.shortname not in {"pydantic", "langchain"}:
            raise ValueError(
                f"A2A durable history does not support adapter {self.adapter.shortname!r}"
            )
        self._persistence = state

    def _session_checkpoint(self, session: _ServedSession) -> dict[str, Any]:
        checkpoint = {
            "version": 1,
            "artifacts": session.artifacts.export_checkpoint(),
            "conversation": (
                session.conversation.export_checkpoint(self.adapter.shortname)
                if session.conversation is not None
                else None
            ),
        }
        if (
            len(json.dumps(checkpoint, allow_nan=False).encode("utf-8"))
            > DEFAULT_A2A_MAX_CHECKPOINT_BYTES
        ):
            raise ValueError("A2A session checkpoint exceeds the 64 MiB limit")
        return checkpoint

    async def _restore_session(
        self, session: _ServedSession, principal: str, context_id: str
    ) -> None:
        assert self._persistence is not None
        checkpoint = await self._persistence.load_checkpoint(principal, context_id)
        if checkpoint is None and not session.poisoned:
            session.loaded = True
            return
        if checkpoint is not None and checkpoint.get("version") != 1:
            raise RuntimeError("Unsupported A2A session checkpoint version")

        def hydrate() -> tuple[ArtifactManager, ConversationState | None, AgentSurface]:
            assert self._persistence is not None
            artifacts = (
                ArtifactManager.from_checkpoint(
                    checkpoint["artifacts"], durable_root=self._persistence.artifact_root
                )
                if checkpoint is not None
                else ArtifactManager(durable_root=self._persistence.artifact_root)
            )
            try:
                conversation = (
                    ConversationState.from_checkpoint(
                        checkpoint["conversation"],
                        adapter_shortname=self.adapter.shortname,
                        artifacts=artifacts,
                    )
                    if checkpoint is not None and self.profile.history is True
                    else ConversationState(artifacts=artifacts)
                    if self.profile.history is True
                    else None
                )
                surface = AgentSurface(
                    agent=self.adapter,
                    profile=self.profile,
                    tool_specs=self.tools,
                    conversation_state=conversation,
                    artifact_manager=artifacts,
                )
            except BaseException:
                artifacts.close()
                raise
            return artifacts, conversation, surface

        async def restore() -> None:
            artifacts, conversation, surface = await asyncio.to_thread(hydrate)
            try:
                await asyncio.to_thread(session.close)
            except BaseException:
                surface.close()
                if conversation is not None:
                    conversation.close()
                artifacts.close()
                raise
            session.surface = surface
            session.artifacts = artifacts
            session.conversation = conversation
            session.loaded = True
            session.poisoned = False

        await finish_storage_operation(restore())

    async def _reserve_task(self, principal: str) -> None:
        async with self._pending_lock:
            if self._closed:
                raise RuntimeError("A2A served program is closed")
            task = asyncio.current_task()
            assert task is not None
            principal_count = self._pending_by_principal.get(principal, 0)
            if self._pending_tasks >= self._max_pending_tasks:
                raise RuntimeError("A2A task capacity is exhausted; retry later")
            if principal_count >= self._max_pending_tasks:
                raise RuntimeError("A2A principal task capacity is exhausted; retry later")
            self._pending_tasks += 1
            self._pending_by_principal[principal] = principal_count + 1
            self._invocations.add(task)

    async def _release_task(self, principal: str) -> None:
        async with self._pending_lock:
            task = asyncio.current_task()
            assert task is not None
            self._invocations.discard(task)
            self._pending_tasks -= 1
            remaining = self._pending_by_principal[principal] - 1
            if remaining:
                self._pending_by_principal[principal] = remaining
            else:
                del self._pending_by_principal[principal]

    async def aclose(self) -> None:
        if asyncio.current_task() in self._invocations:
            raise RuntimeError("Cannot close A2A served program from an active invocation")
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._close())
        # A disconnected shutdown caller must not interrupt resource cleanup.
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        invocations = tuple(self._invocations)
        for task in invocations:
            task.cancel()
        if invocations:
            await asyncio.gather(*invocations, return_exceptions=True)
        with self._sessions_lock:
            sessions = tuple(self._sessions.values())
            self._sessions.clear()
        cleanup_error: Exception | None = None
        for session in sessions:
            try:
                session.close()
            except Exception as exc:
                cleanup_error = cleanup_error or exc
        try:
            await self.runtime.aclose()
        except Exception as exc:
            cleanup_error = cleanup_error or exc
        if self._owns_adapter:
            try:
                await _aclose_adapter(self.adapter)
            except Exception as exc:
                cleanup_error = cleanup_error or exc
        if cleanup_error is not None:
            raise cleanup_error

    def _session(self, principal: str, context_id: str) -> _ServedSession:
        key = (principal, context_id)
        evicted: _ServedSession | None = None
        with self._sessions_lock:
            if self._closed:
                raise RuntimeError("A2A served program is closed")
            session = self._sessions.get(key)
            if session is not None:
                self._sessions.move_to_end(key)
                session.borrowers += 1
                return session
            if len(self._sessions) >= self._max_sessions:
                for old_key, candidate in self._sessions.items():
                    if candidate.borrowers == 0:
                        evicted = candidate
                        del self._sessions[old_key]
                        break
                else:
                    raise RuntimeError("A2A session capacity is exhausted")
            session = self._new_session()
            session.borrowers = 1
            self._sessions[key] = session
        if evicted is not None:
            evicted.close()
        return session

    def _new_session(self) -> _ServedSession:
        artifacts = ArtifactManager(
            durable_root=(self._persistence.artifact_root if self._persistence else None)
        )
        conversation = (
            ConversationState(artifacts=artifacts) if self.profile.history is True else None
        )
        surface = AgentSurface(
            agent=self.adapter,
            profile=self.profile,
            tool_specs=self.tools,
            conversation_state=conversation,
            artifact_manager=artifacts,
        )
        return _ServedSession(
            surface=surface,
            lock=asyncio.Lock(),
            artifacts=artifacts,
            conversation=conversation,
        )


def prepare_a2a_program(
    source: str | Path,
    *,
    entry: str,
    adapter_shortname: str = "pydantic",
    model: str | None = None,
    adapter: AgentAdapter[Any] | None = None,
    max_concurrency: int = 4,
    max_pending_tasks: int = DEFAULT_A2A_MAX_PENDING_TASKS,
    max_sessions: int = DEFAULT_A2A_MAX_SESSIONS,
    max_input_chars: int = DEFAULT_A2A_MAX_INPUT_CHARS,
    max_response_bytes: int = DEFAULT_A2A_MAX_RESPONSE_BYTES,
) -> A2AServedProgram:
    source_path = Path(source).expanduser().resolve()
    loaded = load_kedi_source_bundle(source_path)
    program = parse_program(
        loaded.text,
        source_path=str(source_path),
        source_map=loaded.source_map,
    )
    source_fingerprint = _source_fingerprint(source_path, program, loaded.text)
    definition = program.profile_defs.get(entry)
    if definition is None:
        raise ValueError(f"A2A entry {entry!r} is not an agent profile")
    if not _is_exported(program, entry):
        raise ValueError(f"A2A entry profile {entry!r} must be explicitly exported")

    runtime = compile_program(
        program,
        adapter=adapter,
        subagent_timeout_seconds=None,
    )
    owns_adapter = adapter is None
    runtime_adapter: AgentAdapter[Any] | None = None
    try:
        profile = runtime._evaluate_profile_definition(runtime._profile_registry[entry])
        _validate_headless_profile(profile)
        selected_shortname = profile.adapter_shortname or (
            adapter.shortname if adapter is not None else adapter_shortname
        )
        if selected_shortname == "a2a":
            raise ValueError("An A2A server entry cannot delegate its own primary run to A2A")
        selected_model = profile.model or model
        if selected_model is None:
            raise ValueError(
                "A2A serving requires a model in the exported profile, --model, or KEDI_A2A_MODEL"
            )
        if (
            adapter is not None
            and profile.adapter_shortname is not None
            and profile.adapter_shortname != adapter.shortname
        ):
            raise ValueError(
                f"Injected adapter {adapter.shortname!r} does not match entry adapter "
                f"{profile.adapter_shortname!r}"
            )
        runtime_adapter = adapter or build_adapter(selected_shortname, selected_model)
        runtime.adapter = runtime_adapter
        prototype = runtime.bind_agent(profile=profile)
        try:
            concrete_adapter = prototype.agent
            if not isinstance(concrete_adapter, SupportsSubagentExecution):
                raise TypeError(
                    f"Adapter {concrete_adapter.shortname!r} cannot execute a served Kedi "
                    "agent profile"
                )
            effective_profile = replace(prototype.profile, name=entry)
            tools = prototype.tool_specs
        finally:
            prototype.close()
        compiled_definition = runtime._profile_registry[entry]
        description = compiled_definition.description or f"Kedi agent profile {entry}"
        return A2AServedProgram(
            runtime=runtime,
            entry=entry,
            description=description,
            profile=effective_profile,
            adapter=concrete_adapter,
            tools=tools,
            max_concurrency=max_concurrency,
            max_pending_tasks=max_pending_tasks,
            max_sessions=max_sessions,
            max_input_chars=max_input_chars,
            max_response_bytes=max_response_bytes,
            owns_adapter=owns_adapter,
            source_fingerprint=source_fingerprint,
            model_identity=selected_model,
        )
    except BaseException as exc:
        _cleanup_failed_prepare(
            runtime,
            owned_adapter=runtime_adapter if owns_adapter else None,
            original_error=exc,
        )
        raise


def _source_fingerprint(source_path: Path, program: Any, source_text: str) -> str:
    digest = hashlib.sha256()
    visited: set[Path] = set()

    def visit(path: Path, parsed: Any, text: str) -> None:
        if path in visited:
            return
        visited.add(path)
        digest.update(len(text.encode("utf-8")).to_bytes(8, "big"))
        digest.update(text.encode("utf-8"))
        for module_import in parsed.imports:
            module_path = resolve_module_path(module_import.module, source_path=path)
            imported = load_kedi_source_bundle(
                module_path,
                include_val=False,
                include_cache=True,
            )
            digest.update(module_import.module.encode("utf-8"))
            visit(
                module_path,
                parse_program(
                    imported.text,
                    source_path=str(module_path),
                    source_map=imported.source_map,
                ),
                imported.text,
            )

    visit(source_path, program, source_text)
    return digest.hexdigest()


async def _aclose_adapter(adapter: AgentAdapter[Any]) -> None:
    close_async = getattr(adapter, "aclose", None)
    if callable(close_async):
        result = close_async()
        if inspect.isawaitable(result):
            await result
        return
    close = getattr(adapter, "close", None)
    if callable(close):
        close()


def _cleanup_failed_prepare(
    runtime: KediRuntime,
    *,
    owned_adapter: AgentAdapter[Any] | None,
    original_error: BaseException,
) -> None:
    cleanups = [runtime.aclose()]
    if owned_adapter is not None:
        cleanups.append(_aclose_adapter(owned_adapter))
    for cleanup in cleanups:
        try:
            run_sync(cleanup)
        except Exception as cleanup_error:
            add_note = getattr(original_error, "add_note", None)
            if callable(add_note):
                add_note(f"A2A preparation cleanup failed: {cleanup_error}")


def _is_exported(program: Any, name: str) -> bool:
    for export in program.exports:
        if name in export.names:
            return True
        if export.export_all and not name.startswith("_"):
            return True
    return False


def _validate_headless_profile(profile: AgentProfile) -> None:
    approval = profile.approval
    handler = None if approval is None else approval.handler
    if handler is not None and getattr(handler, "telemetry_mechanism", None) == "cli":
        raise ValueError(
            "A2A serving cannot use an interactive CLI approval handler; configure an "
            "explicit noninteractive approval policy"
        )


__all__ = [
    "A2AServedProgram",
    "DEFAULT_A2A_MAX_INPUT_CHARS",
    "DEFAULT_A2A_MAX_PENDING_TASKS",
    "DEFAULT_A2A_MAX_SESSIONS",
    "prepare_a2a_program",
]
