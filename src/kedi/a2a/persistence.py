from __future__ import annotations

import asyncio
import os
import re
import secrets
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from a2a.server.models import Base, TaskModel
from a2a.server.tasks.database_task_store import DatabaseTaskStore
from a2a.types import Message, Part, Role, Task, TaskState
from filelock import FileLock, SoftFileLock, Timeout
from google.protobuf.json_format import MessageToDict, ParseDict
from sqlalchemy import JSON, ForeignKey, Integer, String, delete, event, func, select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Mapped, mapped_column

from ._storage_io import finish_storage_operation

_SCHEMA_VERSION = 1
_TERMINAL = frozenset(
    {
        TaskState.TASK_STATE_COMPLETED,
        TaskState.TASK_STATE_FAILED,
        TaskState.TASK_STATE_CANCELED,
        TaskState.TASK_STATE_REJECTED,
    }
)


class _Service(Base):
    __tablename__ = "kedi_a2a_service"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False)
    identity: Mapped[str] = mapped_column(String(64), nullable=False)


class _Session(Base):
    __tablename__ = "kedi_a2a_sessions"

    owner: Mapped[str] = mapped_column(String(255), primary_key=True)
    context_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    checkpoint: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class _Execution(Base):
    __tablename__ = "kedi_a2a_executions"

    task_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tasks.id", ondelete="CASCADE"), primary_key=True
    )
    owner: Mapped[str] = mapped_column(String(255), nullable=False)
    context_id: Mapped[str] = mapped_column(String(36), nullable=False)
    boot_id: Mapped[str] = mapped_column(String(48), nullable=False)
    disposition: Mapped[str] = mapped_column(String(24), nullable=False)


@dataclass(slots=True)
class _PendingCommit:
    owner: str
    context_id: str
    checkpoint: dict[str, Any]
    future: asyncio.Future[None]


def _task_to_model(task: Task, owner: str) -> TaskModel:
    return TaskModel(
        id=task.id,
        context_id=task.context_id,
        kind="task",
        owner=owner,
        last_updated=(
            task.status.timestamp.ToDatetime()
            if task.status.HasField("timestamp")
            else datetime.now(timezone.utc).replace(tzinfo=None)
        ),
        status=MessageToDict(task.status),
        artifacts=[MessageToDict(value) for value in task.artifacts],
        history=[MessageToDict(value) for value in task.history],
        task_metadata=MessageToDict(task.metadata) if task.metadata.fields else None,
        protocol_version="1.0",
    )


def _model_to_task(row: TaskModel) -> Task:
    if row.protocol_version != "1.0":
        raise RuntimeError("Unsupported stored A2A protocol version")
    task = Task(id=row.id, context_id=row.context_id)
    ParseDict(row.status, task.status)
    for value in row.artifacts or ():
        ParseDict(value, task.artifacts.add())
    for value in row.history or ():
        ParseDict(value, task.history.add())
    if row.task_metadata:
        task.metadata.update(row.task_metadata)
    return task


def _configure_sqlite(connection: Any, _record: Any) -> None:
    cursor = connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=FULL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


class A2APersistence:
    """One-owner SQLite A2A task store and atomic session checkpoint boundary."""

    def __init__(
        self,
        root: Path,
        *,
        identity: str,
        max_tasks_per_principal: int,
        max_sessions: int = 1_024,
        adapter_shortname: str = "pydantic",
        history_enabled: bool = False,
        require_admission: bool = True,
    ) -> None:
        root = root.expanduser()
        if root.is_symlink():
            raise ValueError("A2A state directory cannot be a symlink")
        self.root = root.resolve()
        self.identity = identity
        self.max_tasks_per_principal = max_tasks_per_principal
        self.max_sessions = max_sessions
        self.adapter_shortname = adapter_shortname
        self.history_enabled = history_enabled
        self.require_admission = require_admission
        self.boot_id = secrets.token_urlsafe(18)
        self.artifact_root = self.root / "artifacts"
        self._lock = FileLock(
            str(self.root / "state.lock"),
            thread_local=False,
            fallback_to_soft=False,
            preserve_lock_file=True,
        )
        self._engine: AsyncEngine | None = None
        self._database: DatabaseTaskStore | None = None
        self._sessions: async_sessionmaker[Any] | None = None
        self._pending: dict[str, _PendingCommit] = {}
        self._admissions: dict[str, asyncio.Future[None]] = {}
        self._write_lock = asyncio.Lock()

    async def open(self) -> None:
        if self._engine is not None:
            return
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if os.name == "posix" and stat.S_IMODE(self.root.stat().st_mode) & 0o077:
            raise PermissionError("A2A state directory must not be accessible by other users")
        if self.artifact_root.is_symlink():
            raise ValueError("A2A artifact directory cannot be a symlink")
        try:
            try:
                await finish_storage_operation(asyncio.to_thread(self._lock.acquire, timeout=0))
            except Timeout as exc:
                raise RuntimeError(f"A2A state directory is already in use: {self.root}") from exc
            if isinstance(self._lock, SoftFileLock):
                raise RuntimeError("A2A state directory requires an OS-backed file lock")
            database_path = self.root / "state.sqlite3"
            if database_path.is_symlink():
                raise ValueError("A2A database file cannot be a symlink")
            if database_path.exists():
                info = database_path.stat()
                if not stat.S_ISREG(info.st_mode) or (
                    os.name == "posix" and stat.S_IMODE(info.st_mode) & 0o077
                ):
                    raise PermissionError("A2A database file must be a private regular file")
            else:
                descriptor = os.open(database_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.close(descriptor)
            self._engine = create_async_engine(
                f"sqlite+aiosqlite:///{database_path}",
                connect_args={"timeout": 5},
            )

            event.listen(self._engine.sync_engine, "connect", _configure_sqlite)
            self._sessions = async_sessionmaker(self._engine, expire_on_commit=False)
            self._database = DatabaseTaskStore(
                self._engine,
                core_to_model_conversion=_task_to_model,
                model_to_core_conversion=_model_to_task,
            )
            async with self._engine.begin() as connection:
                await connection.run_sync(Base.metadata.create_all)
            await self._check_identity()
            await self._validate_checkpoints()
            await self._reconcile_interrupted_tasks()
        except BaseException:
            await self.close()
            raise

    async def close(self) -> None:
        await finish_storage_operation(self._close())

    async def _close(self) -> None:
        async with self._write_lock:
            await self._close_locked()

    async def _close_locked(self) -> None:
        for pending in (*self._pending.values(), *self._admissions.values()):
            future = pending.future if isinstance(pending, _PendingCommit) else pending
            if not future.done():
                future.set_exception(RuntimeError("A2A state store closed before task commit"))
        self._pending.clear()
        self._admissions.clear()
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._database = None
            self._sessions = None
        if self._lock.is_locked:
            await asyncio.to_thread(self._lock.release)

    def expect_admission(self, task_id: str) -> asyncio.Future[None]:
        future = asyncio.get_running_loop().create_future()
        if task_id in self._admissions:
            raise RuntimeError("A2A task is already being admitted")
        self._admissions[task_id] = future
        return future

    def abort_admission(self, task_id: str) -> None:
        admission = self._admissions.pop(task_id, None)
        if admission is not None and not admission.done():
            admission.cancel()

    def stage_completion(
        self,
        *,
        task_id: str,
        owner: str,
        context_id: str,
        checkpoint: dict[str, Any],
    ) -> asyncio.Future[None]:
        future = asyncio.get_running_loop().create_future()
        if task_id in self._pending:
            raise RuntimeError("A2A task already has a pending completion")
        self._pending[task_id] = _PendingCommit(owner, context_id, checkpoint, future)
        return future

    async def abort_completion(self, task_id: str) -> None:
        async def abort() -> None:
            async with self._write_lock:
                pending = self._pending.pop(task_id, None)
                if pending is not None and not pending.future.done():
                    pending.future.cancel()

        await finish_storage_operation(abort())

    async def load_checkpoint(self, owner: str, context_id: str) -> dict[str, Any] | None:
        sessions = self._require_sessions()
        async with sessions() as connection:
            row = await connection.get(_Session, (owner, context_id))
            return None if row is None else dict(row.checkpoint)

    async def save(self, task: Task, context: Any) -> None:
        async def write() -> None:
            async with self._write_lock:
                await self._save_locked(task, context)

        await finish_storage_operation(write())

    async def _save_locked(self, task: Task, context: Any) -> None:
        owner = context.user.user_name
        pending = self._pending.get(task.id)
        admitted = self._admissions.get(task.id)
        try:
            if pending is not None and (pending.owner, pending.context_id) != (
                owner,
                task.context_id,
            ):
                raise RuntimeError("A2A checkpoint owner or context does not match task")
            sessions = self._require_sessions()
            async with sessions.begin() as connection:
                stored = await connection.get(TaskModel, task.id)
                if stored is not None and stored.owner != owner:
                    raise RuntimeError("A2A task identifier belongs to another principal")
                if stored is not None and stored.context_id != task.context_id:
                    raise RuntimeError("A2A task identifier changed context")
                if stored is not None:
                    previous = _model_to_task(stored)
                    if previous.status.state in _TERMINAL:
                        unchanged = task == previous and pending is None and admitted is None
                        # The SDK mutates its cached Task before calling save().
                        task.CopyFrom(previous)
                        if unchanged:
                            return
                        raise RuntimeError("A2A terminal task cannot be changed")
                if task.status.state == TaskState.TASK_STATE_COMPLETED and pending is None:
                    raise RuntimeError("Completed A2A task has no prepared session checkpoint")
                if stored is None:
                    if self.require_admission and admitted is None:
                        raise RuntimeError("A2A task was not admitted for durable execution")
                    await self._reserve_capacity(connection, owner)
                    await self._reserve_session_capacity(connection, owner, task.context_id)
                await connection.merge(_task_to_model(task, owner))
                execution = await connection.get(_Execution, task.id)
                if execution is None:
                    execution = _Execution(
                        task_id=task.id,
                        owner=owner,
                        context_id=task.context_id,
                        boot_id=self.boot_id,
                        disposition="active",
                    )
                    connection.add(execution)
                if task.status.state in _TERMINAL:
                    execution.disposition = (
                        TaskState.Name(task.status.state).removeprefix("TASK_STATE_").lower()
                    )
                if pending is not None and task.status.state == TaskState.TASK_STATE_COMPLETED:
                    row = await connection.get(_Session, (owner, task.context_id))
                    if row is None:
                        count = await connection.scalar(select(func.count()).select_from(_Session))
                        if count >= self.max_sessions:
                            raise RuntimeError("A2A durable session capacity is exhausted")
                        connection.add(
                            _Session(
                                owner=owner,
                                context_id=task.context_id,
                                revision=1,
                                checkpoint=pending.checkpoint,
                            )
                        )
                    else:
                        row.revision += 1
                        row.checkpoint = pending.checkpoint
        except BaseException as exc:
            if pending is not None:
                self._pending.pop(task.id, None)
                if not pending.future.done():
                    pending.future.set_exception(exc)
            if admitted is not None:
                self._admissions.pop(task.id, None)
                if not admitted.done():
                    admitted.set_exception(exc)
            raise
        if pending is not None and task.status.state == TaskState.TASK_STATE_COMPLETED:
            self._pending.pop(task.id, None)
            if not pending.future.done():
                pending.future.set_result(None)
        elif pending is not None and task.status.state in _TERMINAL:
            self._pending.pop(task.id, None)
            if not pending.future.done():
                pending.future.set_exception(
                    RuntimeError("A2A task ended before checkpoint commit")
                )
        if admitted is not None and task.status.state in {
            TaskState.TASK_STATE_SUBMITTED,
            TaskState.TASK_STATE_WORKING,
        }:
            self._admissions.pop(task.id, None)
            if not admitted.done():
                admitted.set_result(None)

    async def get(self, task_id: str, context: Any) -> Task | None:
        return await self._require_database().get(task_id, context)

    async def list(self, params: Any, context: Any) -> Any:
        return await self._require_database().list(params, context)

    async def delete(self, task_id: str, context: Any) -> None:
        owner = context.user.user_name
        sessions = self._require_sessions()
        async with self._write_lock:
            async with sessions.begin() as connection:
                await connection.execute(
                    delete(_Execution).where(
                        _Execution.task_id == task_id, _Execution.owner == owner
                    )
                )
                await connection.execute(
                    delete(TaskModel).where(TaskModel.id == task_id, TaskModel.owner == owner)
                )

    async def _reserve_capacity(self, connection: Any, owner: str) -> None:
        count = await connection.scalar(
            select(func.count()).select_from(TaskModel).where(TaskModel.owner == owner)
        )
        if count < self.max_tasks_per_principal:
            return
        needed = count - self.max_tasks_per_principal + 1
        candidates = await connection.scalars(
            select(TaskModel)
            .join(_Execution, _Execution.task_id == TaskModel.id)
            .where(
                TaskModel.owner == owner,
                _Execution.disposition.in_(
                    ("completed", "failed", "canceled", "rejected", "interrupted")
                ),
            )
            .order_by(TaskModel.last_updated.asc(), TaskModel.id.asc())
            .limit(needed)
        )
        terminal = list(candidates)
        if len(terminal) < needed:
            raise RuntimeError("A2A task retention capacity is exhausted")
        for task in terminal:
            await connection.delete(task)

    async def _reserve_session_capacity(self, connection: Any, owner: str, context_id: str) -> None:
        if await connection.get(_Session, (owner, context_id)) is not None:
            return
        active = (
            await connection.execute(
                select(_Execution.owner, _Execution.context_id)
                .where(_Execution.disposition == "active")
                .distinct()
            )
        ).all()
        active_contexts = {(row.owner, row.context_id) for row in active}
        if (owner, context_id) in active_contexts:
            return
        saved = (await connection.execute(select(_Session.owner, _Session.context_id))).all()
        reserved = {(row.owner, row.context_id) for row in saved} | active_contexts
        if len(reserved) >= self.max_sessions:
            raise RuntimeError("A2A durable session capacity is exhausted")

    async def _check_identity(self) -> None:
        sessions = self._require_sessions()
        async with sessions.begin() as connection:
            service = await connection.get(_Service, 1)
            if service is None:
                connection.add(
                    _Service(id=1, schema_version=_SCHEMA_VERSION, identity=self.identity)
                )
            elif service.schema_version != _SCHEMA_VERSION or service.identity != self.identity:
                raise RuntimeError(
                    "A2A state directory belongs to an incompatible program or schema; "
                    "select a new --state-dir"
                )

    async def _reconcile_interrupted_tasks(self) -> None:
        sessions = self._require_sessions()
        async with sessions.begin() as connection:
            active = (
                await connection.scalars(
                    select(_Execution).where(_Execution.disposition.in_(("active", "suspended")))
                )
            ).all()
            for execution in active:
                task_model = await connection.get(TaskModel, execution.task_id)
                if task_model is None:
                    raise RuntimeError("A2A state has an execution without its task")
                task = _model_to_task(task_model)
                if task.status.state in _TERMINAL:
                    execution.disposition = (
                        TaskState.Name(task.status.state).removeprefix("TASK_STATE_").lower()
                    )
                    continue
                task.status.state = TaskState.TASK_STATE_FAILED
                task.status.timestamp.FromDatetime(datetime.now(timezone.utc))
                task.status.message.CopyFrom(
                    Message(
                        role=Role.ROLE_AGENT,
                        task_id=task.id,
                        context_id=task.context_id,
                        message_id=f"server-interrupted-{task.id}",
                        parts=[
                            Part(
                                text=(
                                    "Server restarted before this task completed. External effects "
                                    "may have occurred; start a new attempt explicitly."
                                )
                            )
                        ],
                    )
                )
                task.metadata.update(
                    {
                        "kedi_interruption": {
                            "reason": "server_interrupted",
                            "previous_boot_id": execution.boot_id,
                            "detected_at": datetime.now(timezone.utc).isoformat(),
                        }
                    }
                )
                await connection.merge(_task_to_model(task, execution.owner))
                execution.disposition = "interrupted"

    async def _validate_checkpoints(self) -> None:
        from kedi.agent_adapter.conversation import ConversationState
        from kedi.artifacts import ArtifactManager

        sessions = self._require_sessions()
        async with sessions() as connection:
            checkpoints = (await connection.scalars(select(_Session.checkpoint))).all()
        committed_sessions: set[str] = set()
        for checkpoint in checkpoints:
            if not isinstance(checkpoint, dict) or checkpoint.get("version") != 1:
                raise RuntimeError("A2A state contains an unsupported session checkpoint")
            if (checkpoint.get("conversation") is not None) != self.history_enabled:
                raise RuntimeError("A2A state contains an incompatible conversation checkpoint")
            artifact_checkpoint = checkpoint.get("artifacts")
            if not isinstance(artifact_checkpoint, dict):
                raise RuntimeError("A2A state contains an invalid artifact checkpoint")
            session_id = artifact_checkpoint.get("session_id")
            if not isinstance(session_id, str):
                raise RuntimeError("A2A state contains an invalid artifact session")
            committed_sessions.add(session_id)

            def validate() -> None:
                artifacts = ArtifactManager.from_checkpoint(
                    checkpoint["artifacts"], durable_root=self.artifact_root
                )
                try:
                    if checkpoint.get("conversation") is not None:
                        conversation = ConversationState.from_checkpoint(
                            checkpoint["conversation"],
                            adapter_shortname=self.adapter_shortname,
                            artifacts=artifacts,
                        )
                        conversation.close()
                    artifacts.prune_inactive_payloads(artifact_checkpoint)
                finally:
                    artifacts.close()

            await finish_storage_operation(asyncio.to_thread(validate))
        await finish_storage_operation(
            asyncio.to_thread(self._cleanup_orphan_artifact_sessions, committed_sessions)
        )

    def _cleanup_orphan_artifact_sessions(self, committed_sessions: set[str]) -> None:
        if not self.artifact_root.exists():
            return
        for directory in self.artifact_root.iterdir():
            if directory.is_symlink():
                raise RuntimeError("A2A artifact store contains a symlink")
            if not directory.is_dir() or directory.name in committed_sessions:
                continue
            if not re.fullmatch(r"session_[A-Za-z0-9_-]{24}", directory.name):
                continue
            children = tuple(directory.iterdir())
            if any(
                child.is_symlink()
                or not child.is_file()
                or not (
                    child.suffix in {".json", ".payload", ".index"}
                    or (child.name.startswith(".") and child.name.endswith(".tmp"))
                )
                for child in children
            ):
                continue
            for child in children:
                child.unlink()
            directory.rmdir()

    def _require_sessions(self) -> async_sessionmaker[Any]:
        if self._sessions is None:
            raise RuntimeError("A2A state store is not open")
        return self._sessions

    def _require_database(self) -> DatabaseTaskStore:
        if self._database is None:
            raise RuntimeError("A2A state store is not open")
        return self._database


__all__ = ["A2APersistence"]
