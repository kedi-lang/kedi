from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Literal, Sequence, cast

import click
import dotenv

from ...effort import EFFORT_VALUES
from .records import (
    DEFAULT_HARBOR_RETRY_EXCLUSIONS,
    HarborRunPolicy,
    HarnessPolicy,
    RunManifest,
    current_gitlink_revision,
    current_git_state,
    load_manifest,
    sha256_file,
)


@click.group()
def cli() -> None:
    """Prepare and run reproducible Kedi Terminal-Bench jobs through Harbor."""


@cli.command("manifest")
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--harbor-revision", required=True)
@click.option("--model", required=True)
@click.option("--task", "tasks", multiple=True, required=True)
@click.option("--kedi-wheel", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--kedi-package")
@click.option(
    "--codex-auth-helper-distribution",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--kedi-otel-distribution",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--tree-sitter-kedi-distribution",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--allow-dirty", is_flag=True)
@click.option("--environment", default="daytona", show_default=True)
@click.option("--adapter", default="pydantic", show_default=True)
@click.option("--effort", type=click.Choice(EFFORT_VALUES))
@click.option(
    "--model-settings",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--concurrency", type=click.IntRange(min=1), default=1, show_default=True)
@click.option("--attempts", type=click.IntRange(min=1), default=1, show_default=True)
@click.option(
    "--timeout-multiplier",
    type=click.FloatRange(min=0, min_open=True),
    default=1.0,
    show_default=True,
)
@click.option(
    "--agent-timeout-multiplier",
    type=click.FloatRange(min=0, min_open=True),
    default=1.0,
    show_default=True,
)
@click.option(
    "--verifier-timeout-multiplier",
    type=click.FloatRange(min=0, min_open=True),
    default=1.0,
    show_default=True,
)
@click.option(
    "--agent-setup-timeout-multiplier",
    type=click.FloatRange(min=0, min_open=True),
    default=1.0,
    show_default=True,
)
@click.option(
    "--environment-build-timeout-multiplier",
    type=click.FloatRange(min=0, min_open=True),
    default=1.0,
    show_default=True,
)
@click.option("--max-retries", type=click.IntRange(min=0), default=0, show_default=True)
@click.option("--retry-include", "retry_include_exceptions", multiple=True)
@click.option(
    "--retry-exclude",
    "retry_exclude_exceptions",
    multiple=True,
    default=DEFAULT_HARBOR_RETRY_EXCLUSIONS,
)
@click.option(
    "--retry-all-exceptions",
    is_flag=True,
    help="Clear Harbor's default retry exclusions; only meaningful with retries enabled.",
)
@click.option(
    "--compaction-mode",
    type=click.Choice(("disabled", "native")),
    default="disabled",
    show_default=True,
)
@click.option("--compaction-threshold", type=click.IntRange(min=1))
@click.option(
    "--history-archive-threshold-tokens",
    type=click.IntRange(min=1),
)
@click.option(
    "--history-archive-preserve-tokens",
    type=click.IntRange(min=1),
    default=12_000,
    show_default=True,
)
@click.option("--history/--no-history", default=True, show_default=True)
@click.option("--artifacts/--no-artifacts", default=True, show_default=True)
@click.option("--codemode/--no-codemode", default=False, show_default=True)
@click.option("--batching-guidance/--no-batching-guidance", default=False, show_default=True)
@click.option(
    "--request-observation/--no-request-observation",
    default=False,
    show_default=True,
)
@click.option("--artifact-threshold-bytes", type=click.IntRange(min=1), default=16_384)
@click.option("--artifact-quota-bytes", type=click.IntRange(min=1), default=256_000_000)
@click.option("--command-timeout-seconds", type=float, default=600.0, show_default=True)
@click.option(
    "--retained-process-timeout-seconds",
    type=click.FloatRange(min=0, min_open=True),
    default=3_600.0,
    show_default=True,
)
@click.option("--model-request-limit", type=click.IntRange(min=1), default=512, show_default=True)
@click.option(
    "--model-cumulative-input-tokens",
    type=click.IntRange(min=1),
    default=50_000_000,
    show_default=True,
)
@click.option("--kedi-root", type=click.Path(path_type=Path), default=Path("."))
def create_manifest(
    *,
    output: Path,
    harbor_revision: str,
    model: str,
    tasks: tuple[str, ...],
    kedi_wheel: Path | None,
    kedi_package: str | None,
    codex_auth_helper_distribution: Path | None,
    kedi_otel_distribution: Path | None,
    tree_sitter_kedi_distribution: Path | None,
    allow_dirty: bool,
    environment: str,
    adapter: str,
    effort: str | None,
    model_settings: Path | None,
    concurrency: int,
    attempts: int,
    timeout_multiplier: float,
    agent_timeout_multiplier: float,
    verifier_timeout_multiplier: float,
    agent_setup_timeout_multiplier: float,
    environment_build_timeout_multiplier: float,
    max_retries: int,
    retry_include_exceptions: tuple[str, ...],
    retry_exclude_exceptions: tuple[str, ...],
    retry_all_exceptions: bool,
    compaction_mode: str,
    compaction_threshold: int | None,
    history_archive_threshold_tokens: int | None,
    history_archive_preserve_tokens: int,
    history: bool,
    artifacts: bool,
    codemode: bool,
    batching_guidance: bool,
    request_observation: bool,
    artifact_threshold_bytes: int,
    artifact_quota_bytes: int,
    command_timeout_seconds: float,
    retained_process_timeout_seconds: float,
    model_request_limit: int,
    model_cumulative_input_tokens: int,
    kedi_root: Path,
) -> None:
    if kedi_wheel is not None and kedi_package is not None:
        raise click.UsageError("--kedi-wheel and --kedi-package are mutually exclusive")
    commit, dirty = current_git_state(kedi_root.resolve())
    if dirty and not allow_dirty:
        raise click.ClickException(
            "Kedi worktree is dirty; commit the run source or pass --allow-dirty explicitly"
        )
    settings_value: object = {}
    if model_settings is not None:
        try:
            settings_value = json.loads(model_settings.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise click.ClickException(f"could not read model settings: {exc}") from exc
    manifest = RunManifest(
        harbor_revision=harbor_revision,
        kedi_commit=commit,
        kedi_dirty=dirty,
        kedi_package=kedi_package,
        kedi_wheel_sha256=sha256_file(kedi_wheel) if kedi_wheel is not None else None,
        codex_auth_helper_distribution_sha256=(
            sha256_file(codex_auth_helper_distribution)
            if codex_auth_helper_distribution is not None
            else None
        ),
        kedi_otel_distribution_sha256=(
            sha256_file(kedi_otel_distribution) if kedi_otel_distribution is not None else None
        ),
        tree_sitter_kedi_revision=(
            current_gitlink_revision(kedi_root.resolve(), "tree-sitter-kedi")
            if kedi_wheel is not None or tree_sitter_kedi_distribution is not None
            else None
        ),
        tree_sitter_kedi_distribution_sha256=(
            sha256_file(tree_sitter_kedi_distribution)
            if tree_sitter_kedi_distribution is not None
            else None
        ),
        model=model,
        adapter=adapter,
        effort=effort,
        model_settings=cast(dict[str, object], settings_value),
        environment=environment,
        task_names=tuple(tasks),
        attempts=attempts,
        concurrency=concurrency,
        harbor_policy=HarborRunPolicy(
            timeout_multiplier=timeout_multiplier,
            agent_timeout_multiplier=agent_timeout_multiplier,
            verifier_timeout_multiplier=verifier_timeout_multiplier,
            agent_setup_timeout_multiplier=agent_setup_timeout_multiplier,
            environment_build_timeout_multiplier=environment_build_timeout_multiplier,
            max_retries=max_retries,
            retry_include_exceptions=retry_include_exceptions,
            retry_exclude_exceptions=(() if retry_all_exceptions else retry_exclude_exceptions),
        ),
        policy=HarnessPolicy(
            artifacts=artifacts,
            history=history,
            codemode=codemode,
            batching_guidance=batching_guidance,
            request_observation=request_observation,
            prefix_cache=history,
            compaction_mode=cast(Literal["disabled", "native"], compaction_mode),
            compaction_threshold=compaction_threshold,
            history_archive_threshold_tokens=history_archive_threshold_tokens,
            history_archive_preserve_tokens=history_archive_preserve_tokens,
            artifact_threshold_bytes=artifact_threshold_bytes,
            artifact_quota_bytes=artifact_quota_bytes,
            command_timeout_seconds=command_timeout_seconds,
            retained_process_timeout_seconds=retained_process_timeout_seconds,
            model_request_limit=model_request_limit,
            model_cumulative_input_tokens=model_cumulative_input_tokens,
        ),
    )
    manifest.write(output.resolve())
    click.echo(f"Wrote immutable manifest {output} ({manifest.material_digest})")


@cli.command("run")
@click.argument("manifest_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--kedi-wheel", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--kedi-package")
@click.option(
    "--codex-auth-helper-distribution",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--kedi-otel-distribution",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--tree-sitter-kedi-distribution",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--jobs-dir", type=click.Path(path_type=Path))
@click.option("--job-name")
@click.option("--harbor-command", default="harbor", show_default=True)
@click.option("--dry-run", is_flag=True)
def run_manifest(
    manifest_path: Path,
    *,
    kedi_wheel: Path | None,
    kedi_package: str | None,
    codex_auth_helper_distribution: Path | None,
    kedi_otel_distribution: Path | None,
    tree_sitter_kedi_distribution: Path | None,
    jobs_dir: Path | None,
    job_name: str | None,
    harbor_command: str,
    dry_run: bool,
) -> None:
    if kedi_wheel is not None and kedi_package is not None:
        raise click.UsageError("--kedi-wheel and --kedi-package are mutually exclusive")
    manifest = load_manifest(manifest_path.resolve())
    resolved_wheel, resolved_package = _resolve_kedi_distribution(
        manifest,
        kedi_wheel=kedi_wheel,
        kedi_package=kedi_package,
    )
    resolved_tree_sitter_distribution = _resolve_tree_sitter_kedi_distribution(
        manifest,
        distribution=tree_sitter_kedi_distribution,
    )
    resolved_codex_auth_helper_distribution = _resolve_codex_auth_helper_distribution(
        manifest,
        distribution=codex_auth_helper_distribution,
    )
    resolved_kedi_otel_distribution = _resolve_kedi_otel_distribution(
        manifest,
        distribution=kedi_otel_distribution,
    )
    executable = _resolve_executable(harbor_command, allow_missing=dry_run)
    effective_jobs_dir = (jobs_dir or Path("jobs")).resolve()
    effective_job_name = job_name or f"kedi-{manifest.material_digest[:12]}"
    command = _harbor_run_command(
        manifest,
        harbor_command=executable,
        kedi_wheel=resolved_wheel,
        kedi_package=resolved_package,
        codex_auth_helper_distribution=resolved_codex_auth_helper_distribution,
        kedi_otel_distribution=resolved_kedi_otel_distribution,
        jobs_dir=effective_jobs_dir,
        job_name=effective_job_name,
        tree_sitter_kedi_distribution=resolved_tree_sitter_distribution,
    )
    if dry_run:
        click.echo(shlex.join(command))
        return
    _validate_harbor_version(executable, expected=manifest.harbor_version)
    job_dir = effective_jobs_dir / effective_job_name
    job_dir.mkdir(parents=True, exist_ok=True)
    manifest.write(job_dir / "kedi-manifest.json")
    if _job_is_complete(job_dir):
        click.echo(f"Harbor job is already complete: {job_dir}")
        return
    resuming = _job_has_started(job_dir)
    if resuming:
        command = [executable, "job", "resume", "-p", str(job_dir)]
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        message = "Harbor resume" if resuming else "Harbor"
        raise click.ClickException(f"{message} exited with status {completed.returncode}")


@cli.command("resume")
@click.argument("job_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--harbor-command", default="harbor", show_default=True)
def resume_job(job_dir: Path, *, harbor_command: str) -> None:
    executable = _resolve_executable(harbor_command)
    manifest_path = job_dir.resolve() / "kedi-manifest.json"
    if not manifest_path.is_file():
        raise click.ClickException(f"Kedi run manifest is missing from Harbor job: {manifest_path}")
    manifest = load_manifest(manifest_path)
    _validate_harbor_version(executable, expected=manifest.harbor_version)
    if _job_is_complete(job_dir):
        click.echo(f"Harbor job is already complete: {job_dir.resolve()}")
        return
    completed = subprocess.run(
        [executable, "job", "resume", "-p", str(job_dir.resolve())],
        check=False,
    )
    if completed.returncode:
        raise click.ClickException(f"Harbor resume exited with status {completed.returncode}")


def _job_has_started(job_dir: Path) -> bool:
    return (job_dir / "lock.json").is_file() or (job_dir / "config.json").is_file()


def _job_is_complete(job_dir: Path) -> bool:
    result_path = job_dir / "result.json"
    if not result_path.is_file():
        return False
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        stats = result["stats"]
        total = result["n_total_trials"]
        return (
            isinstance(total, int)
            and not isinstance(total, bool)
            and result.get("finished_at") is not None
            and stats["n_pending_trials"] == 0
            and stats["n_running_trials"] == 0
            and stats["n_completed_trials"] == total
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _harbor_run_command(
    manifest: RunManifest,
    *,
    harbor_command: str,
    kedi_wheel: Path | None,
    kedi_package: str | None,
    codex_auth_helper_distribution: Path | None = None,
    kedi_otel_distribution: Path | None = None,
    jobs_dir: Path | None,
    job_name: str | None,
    tree_sitter_kedi_distribution: Path | None = None,
) -> list[str]:
    command = [
        harbor_command,
        "run",
        "--dataset",
        f"{manifest.dataset}@{manifest.dataset_version}",
        "--agent",
        "kedi.integrations.harbor:KediAgent",
        "--model",
        manifest.model,
        "--env",
        manifest.environment,
        "--n-attempts",
        str(manifest.attempts),
        "--n-concurrent",
        str(manifest.concurrency),
        "--timeout-multiplier",
        str(manifest.harbor_policy.timeout_multiplier),
        "--agent-timeout-multiplier",
        str(manifest.harbor_policy.agent_timeout_multiplier),
        "--verifier-timeout-multiplier",
        str(manifest.harbor_policy.verifier_timeout_multiplier),
        "--agent-setup-timeout-multiplier",
        str(manifest.harbor_policy.agent_setup_timeout_multiplier),
        "--environment-build-timeout-multiplier",
        str(manifest.harbor_policy.environment_build_timeout_multiplier),
        "--max-retries",
        str(manifest.harbor_policy.max_retries),
        "--yes",
        "--agent-kwarg",
        f"adapter={manifest.adapter}",
        "--agent-kwarg",
        f"compaction_mode={manifest.policy.compaction_mode}",
        "--agent-kwarg",
        f"history_enabled={json.dumps(manifest.policy.history)}",
        "--agent-kwarg",
        f"artifacts_enabled={json.dumps(manifest.policy.artifacts)}",
        "--agent-kwarg",
        f"codemode_enabled={json.dumps(manifest.policy.codemode)}",
        "--agent-kwarg",
        f"batching_guidance_enabled={json.dumps(manifest.policy.batching_guidance)}",
        "--agent-kwarg",
        f"request_observation={json.dumps(manifest.policy.request_observation)}",
        "--agent-kwarg",
        f"prefix_cache_enabled={json.dumps(manifest.policy.prefix_cache)}",
        "--agent-kwarg",
        f"artifact_threshold_bytes={manifest.policy.artifact_threshold_bytes}",
        "--agent-kwarg",
        f"artifact_quota_bytes={manifest.policy.artifact_quota_bytes}",
        "--agent-kwarg",
        f"command_timeout_seconds={manifest.policy.command_timeout_seconds}",
        "--agent-kwarg",
        f"terminal_preview_bytes={manifest.policy.terminal_preview_bytes}",
        "--agent-kwarg",
        f"terminal_output_limit_bytes={manifest.policy.terminal_output_limit_bytes}",
        "--agent-kwarg",
        f"retained_process_timeout_seconds={manifest.policy.retained_process_timeout_seconds}",
        "--agent-kwarg",
        f"model_request_limit={manifest.policy.model_request_limit}",
        "--agent-kwarg",
        f"model_cumulative_input_tokens={manifest.policy.model_cumulative_input_tokens}",
        "--agent-kwarg",
        f"manifest_digest={manifest.material_digest}",
    ]
    if kedi_otel_distribution is not None:
        command.extend(
            (
                "--agent-kwarg",
                f"kedi_otel_distribution={kedi_otel_distribution.resolve()}",
            )
        )
    if codex_auth_helper_distribution is not None:
        command.extend(
            (
                "--agent-kwarg",
                (f"codex_auth_helper_distribution={codex_auth_helper_distribution.resolve()}"),
            )
        )
    if manifest.policy.history_archive_threshold_tokens is not None:
        command.extend(
            (
                "--agent-kwarg",
                (
                    "history_archive_threshold_tokens="
                    f"{manifest.policy.history_archive_threshold_tokens}"
                ),
                "--agent-kwarg",
                (
                    "history_archive_preserve_tokens="
                    f"{manifest.policy.history_archive_preserve_tokens}"
                ),
            )
        )
    for exception_name in manifest.harbor_policy.retry_include_exceptions:
        command.extend(("--retry-include", exception_name))
    for exception_name in manifest.harbor_policy.retry_exclude_exceptions:
        command.extend(("--retry-exclude", exception_name))
    if manifest.effort is not None:
        command.extend(("--agent-kwarg", f"effort={manifest.effort}"))
    if manifest.tree_sitter_kedi_revision is not None:
        command.extend(
            (
                "--agent-kwarg",
                f"tree_sitter_kedi_revision={manifest.tree_sitter_kedi_revision}",
            )
        )
    if tree_sitter_kedi_distribution is not None:
        command.extend(
            (
                "--agent-kwarg",
                f"tree_sitter_kedi_distribution={tree_sitter_kedi_distribution.resolve()}",
            )
        )
    if manifest.model_settings:
        command.extend(
            (
                "--agent-kwarg",
                "model_settings="
                + json.dumps(
                    manifest.model_settings,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            )
        )
    if manifest.policy.compaction_threshold is not None:
        command.extend(
            (
                "--agent-kwarg",
                f"compaction_threshold={manifest.policy.compaction_threshold}",
            )
        )
    if kedi_wheel is not None:
        command.extend(("--agent-kwarg", f"kedi_wheel={kedi_wheel.resolve()}"))
    elif kedi_package is not None:
        command.extend(("--agent-kwarg", f"kedi_package={kedi_package}"))
    for task in manifest.task_names:
        task_name = task if "/" in task else f"terminal-bench/{task}"
        command.extend(("--include-task-name", task_name))
    if jobs_dir is not None:
        command.extend(("--jobs-dir", str(jobs_dir.resolve())))
    if job_name is not None:
        command.extend(("--job-name", job_name))
    return command


def _resolve_executable(command: str, *, allow_missing: bool = False) -> str:
    if Path(command).is_absolute():
        if not allow_missing and not Path(command).is_file():
            raise click.ClickException(f"Harbor executable not found: {command}")
        return command
    resolved = shutil.which(command)
    if resolved is None:
        if allow_missing:
            return command
        raise click.ClickException(
            "Harbor is not installed; use Python 3.12+ and install kedi[terminal-bench]"
        )
    return resolved


def _validate_harbor_version(executable: str, *, expected: str) -> None:
    try:
        completed = subprocess.run(
            [executable, "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise click.ClickException(f"could not determine Harbor version: {exc}") from exc
    actual = completed.stdout.strip()
    if actual != expected:
        raise click.ClickException(
            f"Harbor version {actual!r} does not match manifest version {expected!r}"
        )


def _resolve_kedi_distribution(
    manifest: RunManifest,
    *,
    kedi_wheel: Path | None,
    kedi_package: str | None,
) -> tuple[Path | None, str | None]:
    if manifest.kedi_wheel_sha256 is not None:
        if kedi_package is not None:
            raise click.UsageError("this manifest requires its recorded Kedi wheel")
        if kedi_wheel is None:
            raise click.UsageError("this manifest requires --kedi-wheel")
        resolved = kedi_wheel.expanduser().resolve()
        actual = sha256_file(resolved)
        if actual != manifest.kedi_wheel_sha256:
            raise click.ClickException(
                "Kedi wheel SHA-256 does not match the immutable run manifest"
            )
        return resolved, None

    if kedi_wheel is not None:
        raise click.UsageError(
            "the manifest does not record a wheel digest; recreate it with --kedi-wheel"
        )
    expected = manifest.kedi_package or f"kedi=={manifest.kedi_version}"
    if kedi_package is not None and kedi_package != expected:
        raise click.ClickException(
            f"Kedi package {kedi_package!r} does not match manifest package {expected!r}"
        )
    return None, expected


def _resolve_tree_sitter_kedi_distribution(
    manifest: RunManifest,
    *,
    distribution: Path | None,
) -> Path | None:
    expected = manifest.tree_sitter_kedi_distribution_sha256
    if expected is None:
        if distribution is not None:
            raise click.UsageError(
                "the manifest does not record a tree-sitter-kedi distribution digest; "
                "recreate it with --tree-sitter-kedi-distribution"
            )
        return None
    if distribution is None:
        raise click.UsageError("this manifest requires --tree-sitter-kedi-distribution")
    resolved = distribution.expanduser().resolve()
    if sha256_file(resolved) != expected:
        raise click.ClickException(
            "tree-sitter-kedi distribution SHA-256 does not match the immutable run manifest"
        )
    return resolved


def _resolve_codex_auth_helper_distribution(
    manifest: RunManifest,
    *,
    distribution: Path | None,
) -> Path | None:
    expected = manifest.codex_auth_helper_distribution_sha256
    if expected is None:
        if distribution is not None:
            raise click.UsageError(
                "the manifest does not record a codex-auth-helper distribution digest; "
                "recreate it with --codex-auth-helper-distribution"
            )
        return None
    if distribution is None:
        raise click.UsageError("this manifest requires --codex-auth-helper-distribution")
    resolved = distribution.expanduser().resolve()
    if sha256_file(resolved) != expected:
        raise click.ClickException(
            "codex-auth-helper distribution SHA-256 does not match the immutable run manifest"
        )
    return resolved


def _resolve_kedi_otel_distribution(
    manifest: RunManifest,
    *,
    distribution: Path | None,
) -> Path | None:
    expected = manifest.kedi_otel_distribution_sha256
    if expected is None:
        if distribution is not None:
            raise click.UsageError(
                "the manifest does not record a kedi-otel distribution digest; "
                "recreate it with --kedi-otel-distribution"
            )
        return None
    if distribution is None:
        raise click.UsageError("this manifest requires --kedi-otel-distribution")
    resolved = distribution.expanduser().resolve()
    if sha256_file(resolved) != expected:
        raise click.ClickException(
            "kedi-otel distribution SHA-256 does not match the immutable run manifest"
        )
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    dotenv.load_dotenv()
    cli.main(args=list(argv) if argv is not None else None, prog_name="kedi-terminal-bench")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["cli", "main"]
