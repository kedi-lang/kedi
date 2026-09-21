from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .protocol import (
    KEDI_STRUCTURED_OUTPUT_ARTIFACT_NAME,
    KEDI_STRUCTURED_OUTPUT_EXTENSION_URI,
    A2ARemoteResult,
    A2ARemoteTaskError,
    A2ARemoteTaskSnapshot,
    A2ATransportError,
)


_TERMINAL_TASK_STATES = frozenset(
    {"completed", "failed", "canceled", "rejected", "input_required", "auth_required"}
)


class _ResponseAccumulator:
    """Assemble bounded A2A messages, task updates, and artifact chunks."""

    def __init__(self, *, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self.task_id: str | None = None
        self.context_id: str | None = None
        self.state: str | None = None
        self.latest_message_text: str | None = None
        self._artifacts: dict[str, Any] = {}
        self._artifact_order: list[str] = []
        self._direct_message = False

    def add(self, response: Any) -> None:
        payload = response.WhichOneof("payload")
        if payload == "message":
            self._direct_message = True
            self._capture_message(response.message)
        elif payload == "task":
            self._capture_task(response.task)
        elif payload == "status_update":
            update = response.status_update
            self.task_id = update.task_id or self.task_id
            self.context_id = update.context_id or self.context_id
            self.state = _task_state_name(update.status.state)
            if update.status.HasField("message"):
                self._capture_message(update.status.message)
        elif payload == "artifact_update":
            update = response.artifact_update
            self.task_id = update.task_id or self.task_id
            self.context_id = update.context_id or self.context_id
            self._capture_artifact(update.artifact, append=update.append)
        self._assert_size()

    def add_task(self, task: Any) -> None:
        self._capture_task(task)
        self._assert_size()

    @property
    def is_complete(self) -> bool:
        return self._direct_message or self.state in _TERMINAL_TASK_STATES

    def snapshot(self) -> A2ARemoteTaskSnapshot:
        if self.task_id is None:
            raise A2ATransportError("Remote A2A task response did not contain a task ID")
        return A2ARemoteTaskSnapshot(
            task_id=self.task_id,
            context_id=self.context_id,
            state=self.state,
            text=self._artifact_text() or self.latest_message_text,
        )

    def result(self, *, structured: bool) -> A2ARemoteResult:
        if self.state in {"failed", "canceled", "rejected", "input_required", "auth_required"}:
            detail = self.latest_message_text or f"Remote A2A task ended in state {self.state}"
            raise A2ARemoteTaskError(
                detail,
                task_id=self.task_id,
                context_id=self.context_id,
                state=self.state,
            )
        if self.task_id is not None and self.state != "completed":
            raise A2ATransportError(
                "Remote A2A task observation ended before a terminal state",
                task_id=self.task_id,
                context_id=self.context_id,
                state=self.state,
            )
        data = self._structured_data() if structured else None
        if structured and data is None:
            raise A2ARemoteTaskError(
                "Remote Kedi A2A agent completed without its structured-output artifact",
                task_id=self.task_id,
                context_id=self.context_id,
                state=self.state,
            )
        text = self._artifact_text() or self.latest_message_text
        return A2ARemoteResult(
            text=text,
            data=data,
            task_id=self.task_id,
            context_id=self.context_id,
            state=self.state,
        )

    def _capture_task(self, task: Any) -> None:
        self.task_id = task.id or self.task_id
        self.context_id = task.context_id or self.context_id
        if task.HasField("status"):
            self.state = _task_state_name(task.status.state)
            if task.status.HasField("message"):
                self._capture_message(task.status.message)
        for artifact in task.artifacts:
            self._capture_artifact(artifact, append=False)
        for message in reversed(task.history):
            if _is_agent_message(message):
                self._capture_message(message)
                break

    def _capture_message(self, message: Any) -> None:
        self.task_id = message.task_id or self.task_id
        self.context_id = message.context_id or self.context_id
        if not _is_agent_message(message):
            return
        text = _parts_text(message.parts)
        if text:
            self.latest_message_text = text

    def _capture_artifact(self, artifact: Any, *, append: bool) -> None:
        artifact_id = artifact.artifact_id or f"artifact-{len(self._artifact_order)}"
        if artifact_id not in self._artifact_order:
            self._artifact_order.append(artifact_id)
        if append and artifact_id in self._artifacts:
            self._artifacts[artifact_id].parts.extend(artifact.parts)
            return
        clone = type(artifact)()
        clone.CopyFrom(artifact)
        self._artifacts[artifact_id] = clone

    def _structured_data(self) -> Any | None:
        from google.protobuf.json_format import MessageToDict

        for artifact_id in self._artifact_order:
            artifact = self._artifacts[artifact_id]
            if (
                artifact.name != KEDI_STRUCTURED_OUTPUT_ARTIFACT_NAME
                and KEDI_STRUCTURED_OUTPUT_EXTENSION_URI not in artifact.extensions
            ):
                continue
            for part in artifact.parts:
                if part.WhichOneof("content") == "data":
                    return MessageToDict(part.data)
        return None

    def _artifact_text(self) -> str | None:
        values = [
            _parts_text(self._artifacts[artifact_id].parts) for artifact_id in self._artifact_order
        ]
        text = "\n".join(value for value in values if value)
        return text or None

    def _assert_size(self) -> None:
        total = len((self.latest_message_text or "").encode("utf-8"))
        total += sum(artifact.ByteSize() for artifact in self._artifacts.values())
        if total > self.max_bytes:
            raise A2ARemoteTaskError(
                f"Remote A2A response exceeded {self.max_bytes} bytes",
                task_id=self.task_id,
                context_id=self.context_id,
                state=self.state,
            )


def _parts_text(parts: Sequence[Any]) -> str | None:
    text = "\n".join(part.text for part in parts if part.WhichOneof("content") == "text")
    return text.strip() or None


def _is_agent_message(message: Any) -> bool:
    try:
        from a2a.types import Role

        return message.role == Role.ROLE_AGENT
    except ImportError:
        return False


def _task_state_name(value: int) -> str:
    try:
        from a2a.types import TaskState

        name = TaskState.Name(value)
    except (ImportError, ValueError):
        return str(value)
    return name.removeprefix("TASK_STATE_").lower()


__all__ = ["_ResponseAccumulator"]
