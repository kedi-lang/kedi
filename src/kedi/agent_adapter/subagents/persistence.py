"""Optional durable storage for terminal subagent conversations."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import Any

from kedi.agent_adapter.request_budget import decode_history_usage_state, encode_history_usage_state

from . import DelegationResult, SubagentTurn
from .runs import SubagentRun

STATE_VERSION = 1
_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "timed_out"})


@dataclass(slots=True, kw_only=True)
class StoredConversation:
    conversation_id: str
    profile_name: str
    active_run_id: str | None
    turns: list[SubagentTurn]
    adapter_state: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LoadedSubagentState:
    runs: tuple[SubagentRun, ...]
    results: dict[str, DelegationResult]
    conversations: dict[str, StoredConversation]


class SubagentStateStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self._lock = Lock()

    def load(self) -> LoadedSubagentState:
        with self._lock:
            if not self.path.exists():
                return LoadedSubagentState((), {}, {})
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid subagent state file {self.path}: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("version") != STATE_VERSION:
            raise ValueError(
                f"Unsupported subagent state format in {self.path}; expected version "
                f"{STATE_VERSION}"
            )
        try:
            run_values = payload.get("runs", ())
            result_values = payload.get("results", {})
            conversation_values = payload.get("conversations", {})
            if not isinstance(run_values, list):
                raise TypeError("runs must be an array")
            if not isinstance(result_values, dict):
                raise TypeError("results must be an object")
            if not isinstance(conversation_values, dict):
                raise TypeError("conversations must be an object")
            runs = tuple(self._load_run(item) for item in run_values)
            results = {run_id: DelegationResult(**value) for run_id, value in result_values.items()}
            conversations = {
                conversation_id: StoredConversation(
                    conversation_id=conversation_id,
                    profile_name=value["profile_name"],
                    active_run_id=value.get("active_run_id"),
                    turns=[SubagentTurn(**turn) for turn in value.get("turns", ())],
                    adapter_state=decode_history_usage_state(value.get("adapter_state", {})),
                )
                for conversation_id, value in conversation_values.items()
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid subagent state data in {self.path}: {exc}") from exc
        return LoadedSubagentState(runs, results, conversations)

    def save(
        self,
        *,
        runs: tuple[SubagentRun, ...],
        results: dict[str, DelegationResult],
        conversations: dict[str, StoredConversation],
    ) -> None:
        payload = {
            "version": STATE_VERSION,
            "runs": [asdict(run) for run in runs],
            "results": {run_id: asdict(result) for run_id, result in results.items()},
            "conversations": {
                conversation_id: {
                    "profile_name": conversation.profile_name,
                    "active_run_id": conversation.active_run_id,
                    "turns": [asdict(turn) for turn in conversation.turns],
                    "adapter_state": encode_history_usage_state(conversation.adapter_state),
                }
                for conversation_id, conversation in conversations.items()
            },
        }
        try:
            encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise ValueError("Subagent state must be JSON serializable") from exc

        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path: Path | None = None
            try:
                with NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=self.path.parent,
                    prefix=f".{self.path.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as temporary:
                    temporary_path = Path(temporary.name)
                    os.chmod(temporary_path, 0o600)
                    temporary.write(encoded)
                    temporary.flush()
                    os.fsync(temporary.fileno())
                os.replace(temporary_path, self.path)
            finally:
                if temporary_path is not None and temporary_path.exists():
                    temporary_path.unlink()

    @staticmethod
    def _load_run(value: Any) -> SubagentRun:
        if not isinstance(value, dict):
            raise TypeError("run records must be objects")
        run = SubagentRun(**value)
        if run.status in {"pending", "running"}:
            return replace(
                run,
                status="failed",
                finished_at=run.finished_at or run.started_at or run.created_at,
                error_type="InterruptedError",
                error_message="Subagent run was interrupted by a process restart",
            )
        if run.status not in _TERMINAL_STATUSES:
            raise ValueError(f"unknown subagent status {run.status!r}")
        return run
