from __future__ import annotations

# pyright: reportMissingImports=false
import asyncio
import json
import math
import os
import re
import shlex
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised by the Python 3.10 compatibility job.
    import tomli as tomllib  # ty: ignore[unresolved-import]

from harbor.agents.installed.base import BaseInstalledAgent  # ty: ignore[unresolved-import]
from harbor.agents.model_connection import ModelConnectionSpec  # ty: ignore[unresolved-import]
from harbor.environments.base import BaseEnvironment  # ty: ignore[unresolved-import]
from harbor.models.agent.context import AgentContext  # ty: ignore[unresolved-import]

from ..._version import __version__
from .records import HarnessPolicy, invocation_usage_dict

_LOGFIRE_CREDENTIALS_FILENAME = "logfire_credentials.json"
_RUNNER_PID = PurePosixPath("/installed-agent/kedi-runner.pid")
_RUNNER_SHUTDOWN_GRACE_SECONDS = 20
_RUNNER_FINALIZATION_TIMEOUT_SECONDS = _RUNNER_SHUTDOWN_GRACE_SECONDS + 5


def _setup_command(steps: tuple[tuple[str, str], ...], log_path: PurePosixPath) -> str:
    commands = ["set -e"]
    for phase, command in steps:
        for state, operation in (("started", command), ("completed", None)):
            marker = shlex.quote(f"[kedi-setup] phase={phase} state={state} at=%s\\n")
            commands.append(f'printf {marker} "$(date -u +%Y-%m-%dT%H:%M:%SZ)"')
            if operation is not None:
                commands.append(f"({operation})")
    quoted_log = shlex.quote(log_path.as_posix())
    return (
        f"mkdir -p -- {shlex.quote(log_path.parent.as_posix())} || exit; "
        "(" + "; ".join(commands) + f") > {quoted_log} 2>&1; "
        'setup_status="$?"; '
        f'cat -- {quoted_log}; exit "$setup_status"'
    )


def _resolve_logfire_token() -> str | None:
    if token := os.getenv("LOGFIRE_TOKEN"):
        return token
    credentials_dir = Path(os.getenv("LOGFIRE_CREDENTIALS_DIR", ".logfire")).expanduser()
    try:
        token = json.loads(
            (credentials_dir / _LOGFIRE_CREDENTIALS_FILENAME).read_text(encoding="utf-8")
        )["token"]
    except (KeyError, OSError, TypeError, ValueError):
        return None
    return token if isinstance(token, str) else None


class KediAgent(BaseInstalledAgent):
    """Official Harbor custom-agent bridge for the Kedi coding harness."""

    MODEL_CONNECTION = ModelConnectionSpec(passthrough=True)

    _VENV = PurePosixPath("/installed-agent/kedi")
    _INSTRUCTION = PurePosixPath("/installed-agent/kedi-instruction.txt")
    _MCP_CONFIG = PurePosixPath("/installed-agent/kedi-mcp.json")
    _MODEL_SETTINGS = PurePosixPath("/installed-agent/kedi-model-settings.json")
    _CODEX_AUTH = PurePosixPath("/installed-agent/kedi-codex-auth.json")
    _KEDI_HOME = PurePosixPath("/installed-agent/kedi-home")
    _PYDANTIC_RUNTIME = "pydantic-ai-slim[anthropic,google,openai,openrouter]==2.36.0"
    _CODEX_RUNTIME = "codex-auth-helper[websocket]==1.8.0"
    _LOGFIRE_RUNTIME = "logfire[httpx]==4.41.0"
    _KEDI_OTEL_RUNTIME = "opentelemetry-instrumentation-kedi==0.1.0"
    _UV_RUNTIME = "uv==0.11.21"
    _PIP_NETWORK_OPTIONS = "--retries 20 --timeout 120"
    _MANAGED_PYTHON = "3.11"
    _BOOTSTRAP_VENV = PurePosixPath("/installed-agent/kedi-bootstrap")
    _TREE_SITTER_KEDI_REPOSITORY = "https://github.com/kedi-lang/tree-sitter-kedi.git"

    def __init__(
        self,
        *args: Any,
        kedi_package: str | None = None,
        kedi_wheel: str | Path | None = None,
        codex_auth_helper_distribution: str | Path | None = None,
        kedi_otel_distribution: str | Path | None = None,
        adapter: str = "pydantic",
        effort: str | None = None,
        model_settings: dict[str, Any] | None = None,
        history_enabled: bool = True,
        artifacts_enabled: bool = True,
        codemode_enabled: bool = False,
        batching_guidance_enabled: bool = False,
        request_observation: bool = False,
        prefix_cache_enabled: bool | None = None,
        compaction_mode: str = "disabled",
        compaction_threshold: int | None = None,
        history_archive_threshold_tokens: int | None = None,
        history_archive_preserve_tokens: int = 12_000,
        artifact_threshold_bytes: int = 16_384,
        artifact_quota_bytes: int = 256_000_000,
        command_timeout_seconds: float = 600.0,
        terminal_preview_bytes: int = 32_768,
        terminal_output_limit_bytes: int = 512_000_000,
        retained_process_timeout_seconds: float = 3_600.0,
        model_request_limit: int = 512,
        model_cumulative_input_tokens: int = 50_000_000,
        runner_timeout_seconds: int | None = None,
        task_root: str | None = None,
        manifest_digest: str | None = None,
        codex_auth_file: str | Path | None = None,
        tree_sitter_kedi_revision: str | None = None,
        tree_sitter_kedi_distribution: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if kedi_package is not None and kedi_wheel is not None:
            raise ValueError("kedi_package and kedi_wheel are mutually exclusive")
        self._kedi_package = kedi_package or f"kedi=={__version__}"
        self._kedi_wheel = Path(kedi_wheel).expanduser().resolve() if kedi_wheel else None
        if self._kedi_wheel is not None and not self._kedi_wheel.is_file():
            raise FileNotFoundError(self._kedi_wheel)
        self._codex_auth_helper_distribution = (
            Path(codex_auth_helper_distribution).expanduser().resolve()
            if codex_auth_helper_distribution
            else None
        )
        if (
            self._codex_auth_helper_distribution is not None
            and not self._codex_auth_helper_distribution.is_file()
        ):
            raise FileNotFoundError(self._codex_auth_helper_distribution)
        self._kedi_otel_distribution = (
            Path(kedi_otel_distribution).expanduser().resolve() if kedi_otel_distribution else None
        )
        if self._kedi_otel_distribution is not None and not self._kedi_otel_distribution.is_file():
            raise FileNotFoundError(self._kedi_otel_distribution)
        policy = HarnessPolicy(
            history=history_enabled,
            artifacts=artifacts_enabled,
            codemode=codemode_enabled,
            batching_guidance=batching_guidance_enabled,
            request_observation=request_observation,
            prefix_cache=(
                history_enabled if prefix_cache_enabled is None else prefix_cache_enabled
            ),
            compaction_mode=cast(Literal["disabled", "native"], compaction_mode),
            compaction_threshold=compaction_threshold,
            history_archive_threshold_tokens=history_archive_threshold_tokens,
            history_archive_preserve_tokens=history_archive_preserve_tokens,
            artifact_threshold_bytes=artifact_threshold_bytes,
            artifact_quota_bytes=artifact_quota_bytes,
            command_timeout_seconds=command_timeout_seconds,
            terminal_preview_bytes=terminal_preview_bytes,
            terminal_output_limit_bytes=terminal_output_limit_bytes,
            retained_process_timeout_seconds=retained_process_timeout_seconds,
            model_request_limit=model_request_limit,
            model_cumulative_input_tokens=model_cumulative_input_tokens,
        )
        if runner_timeout_seconds is not None and (
            not math.isfinite(runner_timeout_seconds) or runner_timeout_seconds <= 0
        ):
            raise ValueError("runner_timeout_seconds must be positive")
        self._adapter = adapter
        self._effort = effort
        self._model_settings = dict(model_settings or {})
        self._history_enabled = policy.history
        self._artifacts_enabled = policy.artifacts
        self._codemode_enabled = policy.codemode
        self._batching_guidance_enabled = policy.batching_guidance
        self._request_observation = policy.request_observation
        self._compaction_mode = policy.compaction_mode
        self._compaction_threshold = policy.compaction_threshold
        self._history_archive_threshold_tokens = policy.history_archive_threshold_tokens
        self._history_archive_preserve_tokens = policy.history_archive_preserve_tokens
        self._artifact_threshold_bytes = policy.artifact_threshold_bytes
        self._artifact_quota_bytes = policy.artifact_quota_bytes
        self._command_timeout_seconds = policy.command_timeout_seconds
        self._terminal_preview_bytes = policy.terminal_preview_bytes
        self._terminal_output_limit_bytes = policy.terminal_output_limit_bytes
        self._retained_process_timeout_seconds = policy.retained_process_timeout_seconds
        self._model_request_limit = policy.model_request_limit
        self._model_cumulative_input_tokens = policy.model_cumulative_input_tokens
        self._runner_timeout_seconds = runner_timeout_seconds
        self._task_root = task_root
        self._manifest_digest = manifest_digest
        self._codex_auth_file = self._resolve_codex_auth_file(codex_auth_file)
        if self._codex_auth_helper_distribution is not None and self._codex_auth_file is None:
            raise ValueError("codex_auth_helper_distribution requires a codex/ model")
        if tree_sitter_kedi_revision is not None and not re.fullmatch(
            r"[0-9a-f]{40}", tree_sitter_kedi_revision
        ):
            raise ValueError("tree_sitter_kedi_revision must be a lowercase Git revision")
        self._tree_sitter_kedi_revision = tree_sitter_kedi_revision
        self._tree_sitter_kedi_distribution = (
            Path(tree_sitter_kedi_distribution).expanduser().resolve()
            if tree_sitter_kedi_distribution
            else None
        )
        if (
            self._tree_sitter_kedi_distribution is not None
            and not self._tree_sitter_kedi_distribution.is_file()
        ):
            raise FileNotFoundError(self._tree_sitter_kedi_distribution)

    @staticmethod
    def name() -> str:
        return "kedi"

    def get_version_command(self) -> str | None:
        return f"{self._VENV}/bin/python -c " + shlex.quote(
            "from kedi import __version__; print(__version__)"
        )

    async def install(self, environment: BaseEnvironment) -> None:
        system_dependencies = [
            "bash",
            "git",
            "python3",
            "python_pip",
            "python_venv",
            "ripgrep",
        ]
        if (
            self._tree_sitter_kedi_revision is not None
            or self._tree_sitter_kedi_distribution is not None
        ):
            system_dependencies.append("build_tools")
        await self.ensure_system_dependencies(
            environment,
            tuple(system_dependencies),
        )
        install_target = self._kedi_package
        if self._kedi_wheel is not None:
            remote_wheel = PurePosixPath("/installed-agent") / self._kedi_wheel.name
            await environment.upload_file(self._kedi_wheel, remote_wheel.as_posix())
            install_target = remote_wheel.as_posix()
        runtime_packages = [install_target, self._PYDANTIC_RUNTIME]
        if self._tree_sitter_kedi_distribution is not None:
            remote_distribution = (
                PurePosixPath("/installed-agent") / self._tree_sitter_kedi_distribution.name
            )
            await environment.upload_file(
                self._tree_sitter_kedi_distribution,
                remote_distribution.as_posix(),
            )
            runtime_packages.append(remote_distribution.as_posix())
        elif self._tree_sitter_kedi_revision is not None:
            runtime_packages.append(
                "tree-sitter-kedi @ git+"
                f"{self._TREE_SITTER_KEDI_REPOSITORY}@{self._tree_sitter_kedi_revision}"
            )
        if self._codex_auth_file is not None:
            if self._codex_auth_helper_distribution is None:
                runtime_packages.append(self._CODEX_RUNTIME)
            else:
                remote_codex_helper = (
                    PurePosixPath("/installed-agent") / self._codex_auth_helper_distribution.name
                )
                await environment.upload_file(
                    self._codex_auth_helper_distribution,
                    remote_codex_helper.as_posix(),
                )
                runtime_packages.append(remote_codex_helper.as_posix())
        logfire_enabled = os.getenv("KEDI_HARBOR_LOGFIRE") == "1"
        if logfire_enabled:
            runtime_packages.append(self._LOGFIRE_RUNTIME)
            if self._kedi_otel_distribution is None:
                runtime_packages.append(self._KEDI_OTEL_RUNTIME)
            else:
                remote_kedi_otel = (
                    PurePosixPath("/installed-agent") / self._kedi_otel_distribution.name
                )
                await environment.upload_file(
                    self._kedi_otel_distribution,
                    remote_kedi_otel.as_posix(),
                )
                runtime_packages.append(remote_kedi_otel.as_posix())
        bootstrap = (
            f"rm -rf {shlex.quote(self._BOOTSTRAP_VENV.as_posix())} && "
            f"python3 -m venv {shlex.quote(self._BOOTSTRAP_VENV.as_posix())} && "
            f"{self._BOOTSTRAP_VENV}/bin/python -m pip install "
            f"--disable-pip-version-check {self._PIP_NETWORK_OPTIONS} "
            f"{shlex.quote(self._UV_RUNTIME)}"
        )
        create_runtime = (
            f"rm -rf {shlex.quote(self._VENV.as_posix())} && "
            f"{self._BOOTSTRAP_VENV}/bin/uv venv "
            f"--python {shlex.quote(self._MANAGED_PYTHON)} --managed-python --seed "
            f"{shlex.quote(self._VENV.as_posix())}"
        )
        await self.exec_as_root(
            environment,
            command=_setup_command(
                (
                    ("bootstrap", bootstrap),
                    ("managed-python", create_runtime),
                    (
                        "runtime-packages",
                        f"{self._VENV}/bin/python -m pip install --disable-pip-version-check "
                        f"{self._PIP_NETWORK_OPTIONS} "
                        + " ".join(shlex.quote(package) for package in runtime_packages),
                    ),
                ),
                self.environment_logs_dir / "setup-runtime.log",
            ),
        )

    def _execution_timeout(self, environment: BaseEnvironment) -> float | None:
        """Read single-step Harbor timing policy without exposing task metadata to the agent."""
        timeout = self._runner_timeout_seconds
        config_path = self.logs_dir.parent / "config.json"
        if not config_path.is_file():
            return timeout
        config = json.loads(config_path.read_text(encoding="utf-8"))
        task_path = environment.environment_dir.parent / "task.toml"
        with task_path.open("rb") as source:
            task = tomllib.load(source)
        if task.get("steps"):
            # A phase-specific budget cannot be inferred from the whole-task budget.
            self.logger.warning("Automatic runner deadlines only support single-step tasks")
            return timeout
        agent = config.get("agent", {})
        base = agent.get("override_timeout_sec") or task.get("agent", {}).get("timeout_sec")
        if base is None:
            return timeout
        multiplier = config.get("agent_timeout_multiplier")
        if multiplier is None:
            multiplier = config.get("timeout_multiplier", 1.0)
        resolved = min(base, agent.get("max_timeout_sec") or math.inf) * multiplier
        if not math.isfinite(resolved) or resolved <= 0:
            raise ValueError("Harbor agent timeout must be positive and finite")
        return min(timeout, resolved) if timeout is not None else resolved

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if not self.model_name:
            raise ValueError("KediAgent requires a Harbor model name")
        timeout_seconds = self._execution_timeout(environment)
        deadline_unix = time.time() + timeout_seconds if timeout_seconds is not None else None
        await self._upload_config_text(
            environment,
            content=self.render_instruction(instruction),
            remote_path=self._INSTRUCTION.as_posix(),
            filename="instruction.txt",
        )
        if self.mcp_servers:
            await self._upload_config_text(
                environment,
                content=json.dumps(
                    [server.model_dump(mode="json") for server in self.mcp_servers],
                    sort_keys=True,
                ),
                remote_path=self._MCP_CONFIG.as_posix(),
                filename="mcp.json",
            )
        if self._model_settings:
            await self._upload_config_text(
                environment,
                content=json.dumps(self._model_settings, sort_keys=True),
                remote_path=self._MODEL_SETTINGS.as_posix(),
                filename="model-settings.json",
            )
        env = dict(self.model_connection.env)
        env["KEDI_HOME"] = self._KEDI_HOME.as_posix()
        if workspace_policy := os.getenv("KEDI_WORKSPACE_POLICY"):
            env["KEDI_WORKSPACE_POLICY"] = workspace_policy
        if logfire_enabled := os.getenv("KEDI_HARBOR_LOGFIRE") == "1":
            env["KEDI_HARBOR_LOGFIRE"] = "1"
        if logfire_enabled and (logfire_token := _resolve_logfire_token()):
            env["LOGFIRE_TOKEN"] = logfire_token
        if self._codex_auth_file is not None:
            await self._upload_ephemeral_codex_auth(environment)
            env["KEDI_CODEX_AUTH_FILE"] = self._CODEX_AUTH.as_posix()
        await self._install_skills(environment, env)

        result_path = self.environment_logs_dir / "kedi-result.json"
        output_path = self.environment_logs_dir / "kedi.txt"
        trial_id = str(self.context_id or self.session_id or "unknown")
        command = [
            f"{self._VENV}/bin/python",
            "-m",
            "kedi.integrations.harbor.runner",
            "--instruction-file",
            self._INSTRUCTION.as_posix(),
            "--model",
            self.model_name,
            "--adapter",
            self._adapter,
            "--workspace",
            ".",
            "--result-file",
            result_path.as_posix(),
            "--trial-id",
            trial_id,
            "--compaction-mode",
            self._compaction_mode,
            "--history-enabled" if self._history_enabled else "--no-history-enabled",
            "--artifacts-enabled" if self._artifacts_enabled else "--no-artifacts-enabled",
            "--codemode-enabled" if self._codemode_enabled else "--no-codemode-enabled",
            (
                "--batching-guidance-enabled"
                if self._batching_guidance_enabled
                else "--no-batching-guidance-enabled"
            ),
            ("--request-observation" if self._request_observation else "--no-request-observation"),
            "--artifact-threshold-bytes",
            str(self._artifact_threshold_bytes),
            "--artifact-quota-bytes",
            str(self._artifact_quota_bytes),
            "--command-timeout-seconds",
            str(self._command_timeout_seconds),
            "--terminal-preview-bytes",
            str(self._terminal_preview_bytes),
            "--terminal-output-limit-bytes",
            str(self._terminal_output_limit_bytes),
            "--retained-process-timeout-seconds",
            str(self._retained_process_timeout_seconds),
            "--model-request-limit",
            str(self._model_request_limit),
            "--model-cumulative-input-tokens",
            str(self._model_cumulative_input_tokens),
        ]
        if self.mcp_servers:
            command.extend(("--mcp-config", self._MCP_CONFIG.as_posix()))
        if self._model_settings:
            command.extend(("--model-settings", self._MODEL_SETTINGS.as_posix()))
        if self._effort is not None:
            command.extend(("--effort", self._effort))
        if self._compaction_threshold is not None:
            command.extend(("--compaction-threshold", str(self._compaction_threshold)))
        if self._history_archive_threshold_tokens is not None:
            command.extend(
                (
                    "--history-archive-threshold-tokens",
                    str(self._history_archive_threshold_tokens),
                    "--history-archive-preserve-tokens",
                    str(self._history_archive_preserve_tokens),
                )
            )
        if timeout_seconds is not None:
            command.extend(("--trial-timeout-seconds", str(timeout_seconds)))
            command.extend(("--trial-deadline-unix", str(deadline_unix)))
        if self._manifest_digest is not None:
            command.extend(("--manifest-digest", self._manifest_digest))
        shell_prefix = "set -o pipefail; "
        if self._codex_auth_file is not None:
            cleanup = f"rm -f -- {shlex.quote(self._CODEX_AUTH.as_posix())}"
            shell_prefix += f"trap {shlex.quote(cleanup)} EXIT; "
        quoted_pid_path = shlex.quote(_RUNNER_PID.as_posix())
        exit_path = shlex.quote((self.environment_logs_dir / "runner-exit.json").as_posix())
        exit_temp_path = shlex.quote((self.environment_logs_dir / ".runner-exit.tmp").as_posix())
        shell_command = (
            shell_prefix
            + f"rm -f -- {quoted_pid_path} {exit_path} {exit_temp_path}; "
            + f"{shlex.join(command)} > {shlex.quote(output_path.as_posix())} 2>&1 & "
            + 'runner_pid="$!"; '
            + f"printf '%s\\n' \"$runner_pid\" > {quoted_pid_path}; "
            + 'wait "$runner_pid"; runner_status="$?"; '
            + 'printf \'{"schema_version":1,"exit_code":%s,"finished_at":"%s"}\\n\' '
            + f'"$runner_status" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > {exit_temp_path} && '
            + f"mv -- {exit_temp_path} {exit_path}; "
            + f'rm -f -- {quoted_pid_path}; exit "$runner_status"'
        )
        try:
            await self.exec_as_agent(
                environment,
                command=shell_command,
                env=env,
                cwd=self._task_root,
                timeout_sec=(
                    math.ceil(max(0.0, deadline_unix - time.time())) + 30
                    if deadline_unix is not None
                    else None
                ),
            )
        except asyncio.CancelledError:
            try:
                await self._finalize_cancelled_runner(environment, trial_id)
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                self.logger.warning(
                    "Remote runner cleanup did not finish (%s); Harbor still owns sandbox teardown",
                    type(exc).__name__,
                )
            raise

    async def _finalize_cancelled_runner(
        self,
        environment: BaseEnvironment,
        trial_id: str,
    ) -> None:
        pid_path = shlex.quote(_RUNNER_PID.as_posix())
        expected_trial = shlex.quote(trial_id)
        command = (
            f"pid=$(cat {pid_path} 2>/dev/null || true); "
            "case \"$pid\" in ''|*[!0-9]*) exit 0;; esac; "
            '[ -r "/proc/$pid/cmdline" ] || exit 0; '
            "tr '\\000' '\\n' < \"/proc/$pid/cmdline\" "
            "| grep -Fx -- 'kedi.integrations.harbor.runner' >/dev/null || exit 0; "
            f"tr '\\000' '\\n' < \"/proc/$pid/cmdline\" | grep -Fx -- {expected_trial} "
            ">/dev/null || exit 0; "
            'kill -TERM "$pid" 2>/dev/null || true; '
            'remaining=200; while kill -0 "$pid" 2>/dev/null && '
            '[ "$remaining" -gt 0 ]; do sleep 0.1; remaining=$((remaining - 1)); done; '
            'kill -KILL "$pid" 2>/dev/null || true; '
            f"rm -f -- {pid_path}"
        )
        # The remote command timeout cannot bound a stalled transport or log poll.
        await asyncio.wait_for(
            self.exec_as_agent(
                environment,
                command=command,
                cwd=self._task_root,
                timeout_sec=_RUNNER_FINALIZATION_TIMEOUT_SECONDS,
            ),
            timeout=_RUNNER_FINALIZATION_TIMEOUT_SECONDS,
        )

    def _resolve_codex_auth_file(self, configured: str | Path | None) -> Path | None:
        if not (self.model_name or "").startswith("codex/"):
            return None
        candidate = configured or os.environ.get("KEDI_CODEX_AUTH_FILE")
        path = (
            Path(candidate).expanduser()
            if candidate is not None
            else Path.home() / ".codex" / "auth.json"
        )
        if not path.is_file():
            raise FileNotFoundError(
                "Codex authentication is unavailable. Run `codex login`, then retry."
            )
        return path.resolve()

    async def _upload_private_file(
        self,
        environment: BaseEnvironment,
        *,
        source: Path,
        destination: PurePosixPath,
    ) -> None:
        remote_path = destination.as_posix()
        await environment.upload_file(source, remote_path)
        quoted_path = shlex.quote(remote_path)
        default_user = getattr(environment, "default_user", None)
        ownership = ""
        if default_user is not None:
            ownership = f"chown {shlex.quote(str(default_user))} {quoted_path} && "
        await self.exec_as_root(
            environment,
            command=f"{ownership}chmod 600 {quoted_path}",
        )

    async def _upload_ephemeral_codex_auth(self, environment: BaseEnvironment) -> None:
        if self._codex_auth_file is None:  # pragma: no cover - guarded by caller
            raise RuntimeError("Codex auth source is unavailable")

        from codex_auth_helper import (  # ty: ignore[unresolved-import, unused-ignore-comment]  # Optional TB extra.
            CodexAuthConfig,
            CodexAuthState,
            CodexAuthStore,
            create_codex_async_openai,
        )

        config = CodexAuthConfig(
            auth_path=self._codex_auth_file,
            refresh_margin=timedelta(minutes=30),
        )
        client = create_codex_async_openai(config=config)
        try:
            access_token = await client.token_manager.get_access_token()
            state = client.token_manager.current_state
        finally:
            await client.close()

        ephemeral_state = CodexAuthState(
            access_token=access_token,
            refresh_token="kedi-ephemeral-no-refresh",
            account_id=state.account_id,
            auth_mode=state.auth_mode,
            expires_at=state.expires_at,
            last_refresh=state.last_refresh,
        )
        with tempfile.TemporaryDirectory(prefix="kedi-codex-auth-") as temp_dir:
            path = Path(temp_dir) / "auth.json"
            CodexAuthStore(path).write_state(ephemeral_state)
            await self._upload_private_file(
                environment,
                source=path,
                destination=self._CODEX_AUTH,
            )

    def populate_context_post_run(self, context: AgentContext) -> None:
        result_path = self.logs_dir / "kedi-result.json"
        payload: dict[str, Any] | None = None
        metadata = {**(context.metadata or {})}
        if not result_path.is_file():
            metadata.update(kedi_state="integration_failure", kedi_result_present=False)
        else:
            try:
                value = json.loads(result_path.read_text(encoding="utf-8"))
                if not isinstance(value, dict):
                    raise TypeError("Kedi result must be a JSON object")
                payload = value
            except (OSError, TypeError, ValueError) as exc:
                metadata.update(
                    kedi_state="integration_failure",
                    kedi_result_present=True,
                    kedi_result_error=type(exc).__name__,
                )

        usage = payload.get("usage") if payload is not None else None
        if not isinstance(usage, dict):
            usage = {}
        from .request_observer import recover_request_usage

        recovery = recover_request_usage(self.logs_dir / "model-requests.jsonl")
        if payload is None and recovery is None:
            context.metadata = metadata
            return
        recorded_requests = _optional_int(usage.get("requests")) or 0
        if recovery is not None and recovery.usage.requests > recorded_requests:
            usage = invocation_usage_dict(recovery.usage)
            metadata.update(
                kedi_usage_source="model_request_evidence",
                kedi_usage_recovered=True,
                kedi_usage_recovery_invalid_events=recovery.invalid_events,
                kedi_usage_recovery_events_without_cost=recovery.events_without_cost,
            )

        context.n_input_tokens = _optional_int(usage.get("input_tokens"))
        context.n_cache_tokens = _optional_int(usage.get("cache_read_tokens"))
        context.n_output_tokens = _optional_int(usage.get("output_tokens"))
        context.cost_usd = _optional_float(usage.get("cost_usd"))
        if payload is not None:
            metadata.update(
                kedi_state=payload.get("state"),
                kedi_result_present=True,
                kedi_adapter=payload.get("adapter"),
                kedi_verification=payload.get("verification"),
            )
        metadata.update(
            kedi_requests=usage.get("requests"),
            kedi_tool_calls=usage.get("tool_calls"),
            kedi_cache_write_tokens=usage.get("cache_write_tokens"),
            kedi_uncached_input_tokens=usage.get("uncached_input_tokens"),
        )
        context.metadata = metadata

    async def _install_skills(
        self,
        environment: BaseEnvironment,
        env: dict[str, str],
    ) -> None:
        if not self.skills_dir:
            return
        destination = self._KEDI_HOME / "registry" / "skills"
        await self.exec_as_agent(
            environment,
            command=(
                f"mkdir -p {shlex.quote(destination.as_posix())} && "
                f"cp -R {shlex.quote(self.skills_dir.rstrip('/'))}/. "
                f"{shlex.quote(destination.as_posix())}/"
            ),
            env=env,
        )


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


__all__ = ["KediAgent"]
