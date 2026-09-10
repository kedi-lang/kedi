from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping

from ..._version import __version__
from ...agent_adapter.invocation import InvocationUsage

TrialState = Literal[
    "completed",
    "budget_exhausted",
    "agent_failure",
    "integration_failure",
    "timed_out",
    "cancelled",
]
VerificationState = Literal["not_attempted", "passed", "failed", "unavailable", "stale"]

_TOKEN_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{12,}\b"),
    re.compile(r"(?i)\b(?:bearer|token)\s+[A-Za-z0-9._~+/-]{12,}"),
)
_SECRET_NAME_PARTS = (
    "ACCESS_KEY",
    "API_KEY",
    "AUTH_TOKEN",
    "CREDENTIAL",
    "PRIVATE_KEY",
    "PASSWORD",
    "SECRET",
)
_SECRET_NAME_SUFFIXES = ("_KEY", "_PASSWORD", "_SECRET", "_TOKEN")
DEFAULT_HARBOR_RETRY_EXCLUSIONS = (
    "AgentTimeoutError",
    "VerifierTimeoutError",
    "RewardFileNotFoundError",
    "RewardFileEmptyError",
    "VerifierOutputParseError",
    "ApiUsageLimitError",
    "AgentSafetyRefusalError",
    "AgentAuthenticationError",
    "ModelNotFoundError",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def redact_text(value: str, *, environment: Mapping[str, str] | None = None) -> str:
    redacted = value
    source = os.environ if environment is None else environment
    for name, secret in source.items():
        if secret and len(secret) >= 8 and is_secret_name(name):
            redacted = redacted.replace(secret, "[REDACTED]")
    for pattern in _TOKEN_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True, slots=True, kw_only=True)
class HarnessPolicy:
    artifacts: bool = True
    history: bool = True
    codemode: bool = False
    batching_guidance: bool = False
    request_observation: bool = False
    prefix_cache: bool = True
    compaction_mode: Literal["disabled", "native"] = "disabled"
    compaction_threshold: int | None = None
    history_archive_threshold_tokens: int | None = None
    history_archive_preserve_tokens: int = 12_000
    subagents: bool = False
    dynamic_workflows: bool = False
    artifact_threshold_bytes: int = 16_384
    artifact_quota_bytes: int = 256_000_000
    command_timeout_seconds: float = 600.0
    terminal_preview_bytes: int = 32_768
    terminal_output_limit_bytes: int = 512_000_000
    retained_process_timeout_seconds: float = 3_600.0
    model_request_limit: int = 512
    model_cumulative_input_tokens: int = 50_000_000

    def __post_init__(self) -> None:
        if self.compaction_mode not in {"disabled", "native"}:
            raise ValueError("compaction_mode must be disabled or native")
        if not self.history and self.prefix_cache:
            raise ValueError("prefix_cache requires history")
        if self.history and not self.prefix_cache:
            raise ValueError(
                "stateful Kedi history always uses the selected provider's native prefix "
                "cache policy; disable history to disable Kedi-managed cache placement"
            )
        if not self.history and self.compaction_mode != "disabled":
            raise ValueError("native compaction requires history")
        if self.compaction_threshold is not None and self.compaction_threshold <= 0:
            raise ValueError("compaction_threshold must be positive")
        if self.compaction_mode == "disabled" and self.compaction_threshold is not None:
            raise ValueError("compaction_threshold requires native compaction")
        if self.history_archive_threshold_tokens is not None:
            if self.history_archive_threshold_tokens <= 0:
                raise ValueError("history_archive_threshold_tokens must be positive")
            if not self.history:
                raise ValueError("history archival requires history")
            if not self.artifacts:
                raise ValueError("history archival requires artifacts")
            if self.compaction_mode != "disabled":
                raise ValueError("history archival and native compaction are mutually exclusive")
        if self.history_archive_preserve_tokens <= 0:
            raise ValueError("history_archive_preserve_tokens must be positive")
        if (
            self.history_archive_threshold_tokens is not None
            and self.history_archive_preserve_tokens >= self.history_archive_threshold_tokens
        ):
            raise ValueError(
                "history_archive_preserve_tokens must be lower than "
                "history_archive_threshold_tokens"
            )
        if self.subagents or self.dynamic_workflows:
            raise ValueError(
                "the Terminal-Bench profile does not define subagents or dynamic workflows"
            )
        if self.artifact_threshold_bytes <= 0:
            raise ValueError("artifact_threshold_bytes must be positive")
        if self.artifact_quota_bytes <= 0:
            raise ValueError("artifact_quota_bytes must be positive")
        if not math.isfinite(self.command_timeout_seconds) or self.command_timeout_seconds <= 0:
            raise ValueError("command_timeout_seconds must be positive")
        if self.terminal_preview_bytes <= 0:
            raise ValueError("terminal_preview_bytes must be positive")
        if self.terminal_output_limit_bytes <= 0:
            raise ValueError("terminal_output_limit_bytes must be positive")
        if (
            not math.isfinite(self.retained_process_timeout_seconds)
            or self.retained_process_timeout_seconds <= 0
        ):
            raise ValueError("retained_process_timeout_seconds must be positive")
        if self.model_request_limit <= 0:
            raise ValueError("model_request_limit must be positive")
        if self.model_cumulative_input_tokens <= 0:
            raise ValueError("model_cumulative_input_tokens must be positive")


@dataclass(frozen=True, slots=True, kw_only=True)
class HarborRunPolicy:
    """Harbor-owned execution controls that materially affect a benchmark run."""

    timeout_multiplier: float = 1.0
    agent_timeout_multiplier: float = 1.0
    verifier_timeout_multiplier: float = 1.0
    agent_setup_timeout_multiplier: float = 1.0
    environment_build_timeout_multiplier: float = 1.0
    max_retries: int = 0
    retry_include_exceptions: tuple[str, ...] = ()
    retry_exclude_exceptions: tuple[str, ...] = DEFAULT_HARBOR_RETRY_EXCLUSIONS

    def __post_init__(self) -> None:
        multipliers = (
            self.timeout_multiplier,
            self.agent_timeout_multiplier,
            self.verifier_timeout_multiplier,
            self.agent_setup_timeout_multiplier,
            self.environment_build_timeout_multiplier,
        )
        if any(not math.isfinite(value) or value <= 0 for value in multipliers):
            raise ValueError("Harbor timeout multipliers must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must not be negative")
        include = _normalized_exception_names(
            self.retry_include_exceptions,
            field_name="retry_include_exceptions",
        )
        exclude = _normalized_exception_names(
            self.retry_exclude_exceptions,
            field_name="retry_exclude_exceptions",
        )
        overlap = set(include).intersection(exclude)
        if overlap:
            raise ValueError("retry include/exclude lists overlap: " + ", ".join(sorted(overlap)))
        object.__setattr__(self, "retry_include_exceptions", include)
        object.__setattr__(self, "retry_exclude_exceptions", exclude)


@dataclass(frozen=True, slots=True, kw_only=True)
class RunManifest:
    schema_version: int = 1
    dataset: str = "terminal-bench/terminal-bench-2-1"
    dataset_version: str = "6"
    harbor_version: str = "0.22.0"
    harbor_revision: str
    kedi_version: str = __version__
    kedi_commit: str
    kedi_dirty: bool
    kedi_package: str | None = None
    kedi_wheel_sha256: str | None = None
    codex_auth_helper_distribution_sha256: str | None = None
    kedi_otel_distribution_sha256: str | None = None
    tree_sitter_kedi_revision: str | None = None
    tree_sitter_kedi_distribution_sha256: str | None = None
    model: str
    adapter: str = "pydantic"
    effort: str | None = None
    model_settings: Mapping[str, Any] = field(default_factory=dict)
    environment: str = "daytona"
    task_names: tuple[str, ...] = ()
    attempts: int = 1
    concurrency: int = 1
    harbor_policy: HarborRunPolicy = field(default_factory=HarborRunPolicy)
    policy: HarnessPolicy = field(default_factory=HarnessPolicy)
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported Terminal-Bench manifest schema")
        if not self.harbor_revision.strip() or not self.kedi_commit.strip():
            raise ValueError("Harbor and Kedi revisions must be pinned")
        if self.kedi_package is not None and not self.kedi_package.strip():
            raise ValueError("kedi_package must not be empty")
        if self.kedi_package is not None and self.kedi_wheel_sha256 is not None:
            raise ValueError("kedi_package and kedi_wheel_sha256 are mutually exclusive")
        if self.kedi_wheel_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.kedi_wheel_sha256
        ):
            raise ValueError("kedi_wheel_sha256 must be a lowercase SHA-256 digest")
        if self.codex_auth_helper_distribution_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.codex_auth_helper_distribution_sha256
        ):
            raise ValueError(
                "codex_auth_helper_distribution_sha256 must be a lowercase SHA-256 digest"
            )
        if self.codex_auth_helper_distribution_sha256 is not None and not self.model.startswith(
            "codex/"
        ):
            raise ValueError("codex_auth_helper_distribution_sha256 requires a codex/ model")
        if self.kedi_otel_distribution_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.kedi_otel_distribution_sha256
        ):
            raise ValueError("kedi_otel_distribution_sha256 must be a lowercase SHA-256 digest")
        if self.tree_sitter_kedi_revision is not None and not re.fullmatch(
            r"[0-9a-f]{40}", self.tree_sitter_kedi_revision
        ):
            raise ValueError("tree_sitter_kedi_revision must be a lowercase Git revision")
        if self.tree_sitter_kedi_distribution_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.tree_sitter_kedi_distribution_sha256
        ):
            raise ValueError(
                "tree_sitter_kedi_distribution_sha256 must be a lowercase SHA-256 digest"
            )
        if (
            self.tree_sitter_kedi_distribution_sha256 is not None
            and self.tree_sitter_kedi_revision is None
        ):
            raise ValueError(
                "tree_sitter_kedi_distribution_sha256 requires a pinned tree_sitter_kedi_revision"
            )
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.effort is not None and self.effort not in {
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        }:
            raise ValueError("unsupported effort value")
        try:
            encoded_settings = json.dumps(
                self.model_settings,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
            )
            normalized_settings = json.loads(encoded_settings)
        except (TypeError, ValueError) as exc:
            raise ValueError("model_settings must be a JSON object") from exc
        if not isinstance(normalized_settings, dict):
            raise ValueError("model_settings must be a JSON object")
        if _contains_secret_material(normalized_settings):
            raise ValueError("model_settings must not contain credentials or secret values")
        object.__setattr__(self, "model_settings", normalized_settings)
        if self.attempts <= 0 or self.concurrency <= 0:
            raise ValueError("attempts and concurrency must be positive")
        if len(set(self.task_names)) != len(self.task_names):
            raise ValueError("task_names must not contain duplicates")

    @property
    def material_digest(self) -> str:
        material = self.to_dict()
        material.pop("created_at", None)
        if material.get("tree_sitter_kedi_distribution_sha256") is None:
            material.pop("tree_sitter_kedi_distribution_sha256", None)
        if material.get("kedi_otel_distribution_sha256") is None:
            material.pop("kedi_otel_distribution_sha256", None)
        if material.get("codex_auth_helper_distribution_sha256") is None:
            material.pop("codex_auth_helper_distribution_sha256", None)
        policy = material.get("policy")
        if isinstance(policy, dict) and policy.get("batching_guidance") is False:
            policy.pop("batching_guidance", None)
        return hashlib.sha256(
            json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path) -> None:
        payload = {**self.to_dict(), "material_digest": self.material_digest}
        if path.exists():
            existing = load_manifest(path)
            if existing.material_digest != self.material_digest:
                raise FileExistsError(f"refusing to replace a different run manifest: {path}")
            return
        atomic_write_json(path, payload)


def load_manifest(path: Path) -> RunManifest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected_digest = raw.pop("material_digest", None)
    raw["task_names"] = tuple(raw.get("task_names", ()))
    raw["harbor_policy"] = HarborRunPolicy(**raw.get("harbor_policy", {}))
    raw["policy"] = HarnessPolicy(**raw.get("policy", {}))
    manifest = RunManifest(**raw)
    if expected_digest is not None and expected_digest != manifest.material_digest:
        raise ValueError(f"run manifest digest does not match its contents: {path}")
    return manifest


def current_git_state(root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return commit, dirty


def current_gitlink_revision(root: Path, path: str) -> str:
    output = subprocess.run(
        ["git", "ls-tree", "HEAD", path],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    parts = output.split()
    if len(parts) != 4 or parts[0] != "160000" or parts[1] != "commit":
        raise ValueError(f"{path!r} is not a pinned Git submodule")
    revision = parts[2]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"{path!r} has an invalid Git revision")
    return revision


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


def is_secret_name(name: object) -> bool:
    normalized = str(name).upper()
    return (
        normalized in {"KEY", "PASSWORD", "SECRET", "TOKEN"}
        or normalized.endswith(_SECRET_NAME_SUFFIXES)
        or any(part in normalized for part in _SECRET_NAME_PARTS)
    )


def _normalized_exception_names(
    values: tuple[str, ...],
    *,
    field_name: str,
) -> tuple[str, ...]:
    normalized = tuple(value.strip() for value in values)
    if any(not value for value in normalized):
        raise ValueError(f"{field_name} must not contain empty names")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must not contain duplicates")
    return normalized


@dataclass(frozen=True, slots=True, kw_only=True)
class TrialRecord:
    schema_version: int
    trial_id: str
    state: TrialState
    started_at: str
    finished_at: str
    duration_seconds: float
    instruction_digest: str
    workspace: str
    model: str
    adapter: str
    manifest_digest: str | None
    usage: Mapping[str, Any]
    verification: Mapping[str, Any]
    terminal: Mapping[str, Any]
    policy: Mapping[str, Any]
    request_observation: Mapping[str, Any] = field(default_factory=lambda: {"enabled": False})
    result_summary: str | None = None
    failure_phase: str | None = None
    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def invocation_usage_dict(usage: InvocationUsage) -> dict[str, Any]:
    return {
        "requests": usage.requests,
        "tool_calls": usage.tool_calls,
        "input_tokens": usage.input_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "uncached_input_tokens": usage.uncached_input_tokens,
        "cost_usd": float(usage.cost_usd) if usage.cost_usd is not None else None,
    }


def _contains_secret_material(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            is_secret_name(key) or _contains_secret_material(item) for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_secret_material(item) for item in value)
    return isinstance(value, str) and redact_text(value, environment={}) != value


__all__ = [
    "DEFAULT_HARBOR_RETRY_EXCLUSIONS",
    "HarnessPolicy",
    "HarborRunPolicy",
    "RunManifest",
    "TrialRecord",
    "TrialState",
    "VerificationState",
    "atomic_write_json",
    "current_gitlink_revision",
    "current_git_state",
    "is_secret_name",
    "invocation_usage_dict",
    "load_manifest",
    "redact_text",
    "sha256_file",
    "utc_now",
]
