from __future__ import annotations

# pyright: reportMissingImports=false

import json
import os
import re
import shlex
import tempfile
from datetime import timedelta
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

from harbor.agents.installed.base import BaseInstalledAgent  # ty: ignore[unresolved-import]
from harbor.agents.model_connection import ModelConnectionSpec  # ty: ignore[unresolved-import]
from harbor.environments.base import BaseEnvironment  # ty: ignore[unresolved-import]
from harbor.models.agent.context import AgentContext  # ty: ignore[unresolved-import]

from ..._version import __version__
from .records import HarnessPolicy


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
    _CODEX_RUNTIME = "codex-auth-helper==1.6.1"
    _LOGFIRE_RUNTIME = "logfire[httpx]==4.41.0"
    _UV_RUNTIME = "uv==0.11.21"
    _MANAGED_PYTHON = "3.11"
    _BOOTSTRAP_VENV = PurePosixPath("/installed-agent/kedi-bootstrap")
    _TREE_SITTER_KEDI_REPOSITORY = "https://github.com/kedi-lang/tree-sitter-kedi.git"

    def __init__(
        self,
        *args: Any,
        kedi_package: str | None = None,
        kedi_wheel: str | Path | None = None,
        adapter: str = "pydantic",
        effort: str | None = None,
        model_settings: dict[str, Any] | None = None,
        history_enabled: bool = True,
        artifacts_enabled: bool = True,
        prefix_cache_enabled: bool | None = None,
        compaction_mode: str = "disabled",
        compaction_threshold: int | None = None,
        artifact_threshold_bytes: int = 16_384,
        artifact_quota_bytes: int = 256_000_000,
        command_timeout_seconds: float = 120.0,
        terminal_preview_bytes: int = 32_768,
        terminal_output_limit_bytes: int = 512_000_000,
        runner_timeout_seconds: int | None = None,
        task_root: str | None = None,
        manifest_digest: str | None = None,
        codex_auth_file: str | Path | None = None,
        tree_sitter_kedi_revision: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if kedi_package is not None and kedi_wheel is not None:
            raise ValueError("kedi_package and kedi_wheel are mutually exclusive")
        self._kedi_package = kedi_package or f"kedi=={__version__}"
        self._kedi_wheel = Path(kedi_wheel).expanduser().resolve() if kedi_wheel else None
        if self._kedi_wheel is not None and not self._kedi_wheel.is_file():
            raise FileNotFoundError(self._kedi_wheel)
        policy = HarnessPolicy(
            history=history_enabled,
            artifacts=artifacts_enabled,
            prefix_cache=(
                history_enabled if prefix_cache_enabled is None else prefix_cache_enabled
            ),
            compaction_mode=cast(Literal["disabled", "native"], compaction_mode),
            compaction_threshold=compaction_threshold,
            artifact_threshold_bytes=artifact_threshold_bytes,
            artifact_quota_bytes=artifact_quota_bytes,
            command_timeout_seconds=command_timeout_seconds,
            terminal_preview_bytes=terminal_preview_bytes,
            terminal_output_limit_bytes=terminal_output_limit_bytes,
        )
        if runner_timeout_seconds is not None and runner_timeout_seconds <= 0:
            raise ValueError("runner_timeout_seconds must be positive")
        self._adapter = adapter
        self._effort = effort
        self._model_settings = dict(model_settings or {})
        self._history_enabled = policy.history
        self._artifacts_enabled = policy.artifacts
        self._compaction_mode = policy.compaction_mode
        self._compaction_threshold = policy.compaction_threshold
        self._artifact_threshold_bytes = policy.artifact_threshold_bytes
        self._artifact_quota_bytes = policy.artifact_quota_bytes
        self._command_timeout_seconds = policy.command_timeout_seconds
        self._terminal_preview_bytes = policy.terminal_preview_bytes
        self._terminal_output_limit_bytes = policy.terminal_output_limit_bytes
        self._runner_timeout_seconds = runner_timeout_seconds
        self._task_root = task_root
        self._manifest_digest = manifest_digest
        self._codex_auth_file = self._resolve_codex_auth_file(codex_auth_file)
        if tree_sitter_kedi_revision is not None and not re.fullmatch(
            r"[0-9a-f]{40}", tree_sitter_kedi_revision
        ):
            raise ValueError("tree_sitter_kedi_revision must be a lowercase Git revision")
        self._tree_sitter_kedi_revision = tree_sitter_kedi_revision

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
        if self._tree_sitter_kedi_revision is not None:
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
        if self._tree_sitter_kedi_revision is not None:
            runtime_packages.append(
                "tree-sitter-kedi @ git+"
                f"{self._TREE_SITTER_KEDI_REPOSITORY}@{self._tree_sitter_kedi_revision}"
            )
        if self._codex_auth_file is not None:
            runtime_packages.append(self._CODEX_RUNTIME)
        logfire_enabled = os.getenv("KEDI_HARBOR_LOGFIRE") == "1"
        if logfire_enabled:
            runtime_packages.append(self._LOGFIRE_RUNTIME)
        supported_python = shlex.quote("import sys; raise SystemExit(sys.version_info < (3, 10))")
        create_runtime = (
            f"if python3 -c {supported_python}; then "
            f"python3 -m venv {shlex.quote(self._VENV.as_posix())}; "
            "else "
            f"rm -rf {shlex.quote(self._BOOTSTRAP_VENV.as_posix())} && "
            f"python3 -m venv {shlex.quote(self._BOOTSTRAP_VENV.as_posix())} && "
            f"{self._BOOTSTRAP_VENV}/bin/python -m pip install "
            f"--disable-pip-version-check {shlex.quote(self._UV_RUNTIME)} && "
            f"{self._BOOTSTRAP_VENV}/bin/uv venv "
            f"--python {shlex.quote(self._MANAGED_PYTHON)} --managed-python --seed "
            f"{shlex.quote(self._VENV.as_posix())}; "
            "fi"
        )
        await self.exec_as_root(
            environment,
            command=(
                f"rm -rf {shlex.quote(self._VENV.as_posix())} && "
                f"{create_runtime} && "
                f"{self._VENV}/bin/python -m pip install --disable-pip-version-check "
                + " ".join(shlex.quote(package) for package in runtime_packages)
            ),
        )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if not self.model_name:
            raise ValueError("KediAgent requires a Harbor model name")
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
        if logfire_enabled := os.getenv("KEDI_HARBOR_LOGFIRE") == "1":
            env["KEDI_HARBOR_LOGFIRE"] = "1"
        if logfire_enabled and (logfire_token := os.getenv("LOGFIRE_TOKEN")):
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
        ]
        if self.mcp_servers:
            command.extend(("--mcp-config", self._MCP_CONFIG.as_posix()))
        if self._model_settings:
            command.extend(("--model-settings", self._MODEL_SETTINGS.as_posix()))
        if self._effort is not None:
            command.extend(("--effort", self._effort))
        if self._compaction_threshold is not None:
            command.extend(("--compaction-threshold", str(self._compaction_threshold)))
        if self._runner_timeout_seconds is not None:
            command.extend(("--trial-timeout-seconds", str(self._runner_timeout_seconds)))
        if self._manifest_digest is not None:
            command.extend(("--manifest-digest", self._manifest_digest))
        shell_prefix = "set -o pipefail; "
        if self._codex_auth_file is not None:
            cleanup = f"rm -f -- {shlex.quote(self._CODEX_AUTH.as_posix())}"
            shell_prefix += f"trap {shlex.quote(cleanup)} EXIT; "
        shell_command = (
            shell_prefix + f"{shlex.join(command)} 2>&1 | tee {shlex.quote(output_path.as_posix())}"
        )
        await self.exec_as_agent(
            environment,
            command=shell_command,
            env=env,
            cwd=self._task_root,
            timeout_sec=(
                self._runner_timeout_seconds + 30
                if self._runner_timeout_seconds is not None
                else None
            ),
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

        from codex_auth_helper import (  # ty: ignore[unresolved-import]
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
        if not result_path.is_file():
            context.metadata = {
                **(context.metadata or {}),
                "kedi_state": "integration_failure",
                "kedi_result_present": False,
            }
            return
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            context.metadata = {
                **(context.metadata or {}),
                "kedi_state": "integration_failure",
                "kedi_result_present": True,
                "kedi_result_error": type(exc).__name__,
            }
            return
        usage = payload.get("usage") or {}
        context.n_input_tokens = _optional_int(usage.get("input_tokens"))
        context.n_cache_tokens = _optional_int(usage.get("cache_read_tokens"))
        context.n_output_tokens = _optional_int(usage.get("output_tokens"))
        context.cost_usd = _optional_float(usage.get("cost_usd"))
        context.metadata = {
            **(context.metadata or {}),
            "kedi_state": payload.get("state"),
            "kedi_result_present": True,
            "kedi_adapter": payload.get("adapter"),
            "kedi_requests": usage.get("requests"),
            "kedi_tool_calls": usage.get("tool_calls"),
            "kedi_cache_write_tokens": usage.get("cache_write_tokens"),
            "kedi_uncached_input_tokens": usage.get("uncached_input_tokens"),
            "kedi_verification": payload.get("verification"),
        }

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
