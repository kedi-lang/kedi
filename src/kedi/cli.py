from __future__ import annotations

import sys
from pathlib import Path

import click
from click.core import ParameterSource

from kedi._interactive import SessionPersistenceError
from kedi.app import (
    maybe_run_codegen,
    run_evals_cli,
    run_optimization_cli,
    run_program_cli,
    run_source_cli,
    run_tests_cli,
)
from kedi.error_rendering import render_execution_error, render_parse_error
from kedi.errors import KediExecutionError
from kedi.idle import run_idle
from kedi.lang import parse_program
from kedi.lang.parser.diagnostics import KediParseError
from kedi.package_audit import PackageAuditResult, audit_installed_packages
from kedi.package_registry import add_package, install_package
from kedi.registry_api import RegistryAPIError
from kedi.skill_registry import install_skill_path, install_skill_repo
from kedi.source_loading import load_kedi_source_bundle
from kedi.utils import parse_args

context_settings = {
    "ignore_unknown_options": True,
    "allow_extra_args": True,
    # The default eager help option runs before the source positional argument
    # is available, so `kedi install --help` cannot show package-command help.
    "help_option_names": [],
}
_PACKAGE_COMMANDS = frozenset({"install", "add", "audit"})


def _package_command_help(command: str) -> str:
    if command == "install":
        return (
            "Usage: kedi install [PACKAGE.KEDI]\n\n"
            "Install package.kedi from the current directory or the supplied manifest path."
        )
    if command == "add":
        return (
            "Usage: kedi add <PACKAGE-NAME | git+https://github.com/OWNER/REPOSITORY.git>\n\n"
            "Install a package from the Kedi registry or an explicit GitHub source."
        )
    return (
        "Usage: kedi audit\n\n"
        "Check installed package receipts for yanked or revoked registry commits."
    )


def _run_package_command(command: str, arguments: list[str]) -> int:
    if any(argument.startswith("-") for argument in arguments):
        raise click.UsageError(f"Unexpected option for 'kedi {command}'.")

    if command == "audit":
        if arguments:
            raise click.UsageError("'kedi audit' does not accept arguments.")
        return _run_package_audit()
    if command == "install":
        if len(arguments) > 1:
            raise click.UsageError("'kedi install' accepts at most one manifest path.")
        try:
            installed = install_package(Path(arguments[0]) if arguments else Path("package.kedi"))
        except (FileNotFoundError, ModuleNotFoundError, RuntimeError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
    else:
        if not arguments:
            raise click.UsageError("Missing argument for 'kedi add'.")
        if len(arguments) > 1:
            raise click.UsageError("'kedi add' accepts exactly one package source.")
        try:
            installed = add_package(arguments[0])
        except (FileNotFoundError, ModuleNotFoundError, RuntimeError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc

    message = f"Installed {installed.name} at {installed.root}"
    if installed.source_revision is not None:
        message += f" ({installed.source_revision})"
    click.secho(message, fg="green")
    return 0


def _run_package_audit() -> int:
    try:
        report = audit_installed_packages()
    except (RegistryAPIError, OSError, ValueError) as exc:
        click.echo(f"Registry audit could not complete: {exc}", err=True)
        raise click.exceptions.Exit(2) from exc

    visible_statuses = {"yanked", "revoked", "unverified", "unknown", "malformed"}
    for result in report.results:
        if result.status in visible_statuses:
            click.echo(_format_package_audit_result(result))

    counts = report.counts
    ordered = ("active", "superseded", "yanked", "revoked", "unverified", "unknown", "malformed")
    summary = " ".join(f"{status}={counts[status]}" for status in ordered)
    click.echo(f"Checked {len(report.results)} installed package(s): {summary}")
    if report.exit_code:
        raise click.exceptions.Exit(report.exit_code)
    return 0


def _format_package_audit_result(result: PackageAuditResult) -> str:
    identity = result.package
    if result.version is not None:
        identity += f" {result.version}"
    if result.commit is not None:
        identity += f" {result.commit[:12]}"
    line = f"{result.status.upper()} {identity}"
    if result.source_kind is not None and result.status == "unverified":
        line += f" ({result.source_kind})"
    if result.detail:
        line += f": {result.detail}"
    if result.advisory_url:
        line += f" [{result.advisory_url}]"
    return line


def _run_skills_command(arguments: list[str]) -> int:
    if not arguments or arguments[0] != "add":
        raise click.UsageError("Usage: kedi skills add (--repo OWNER/REPOSITORY | --path PATH)")
    repo: str | None = None
    path: str | None = None
    rest = arguments[1:]
    while rest:
        option = rest.pop(0)
        if option not in {"--repo", "--path"} or not rest:
            raise click.UsageError("Usage: kedi skills add (--repo OWNER/REPOSITORY | --path PATH)")
        value = rest.pop(0)
        if option == "--repo":
            if repo is not None:
                raise click.UsageError("--repo may be provided only once.")
            repo = value
        else:
            if path is not None:
                raise click.UsageError("--path may be provided only once.")
            path = value
    if (repo is None) == (path is None):
        raise click.UsageError("Choose exactly one of --repo or --path.")
    try:
        installed = (
            install_skill_repo(repo) if repo is not None else install_skill_path(Path(path or ""))
        )
    except (FileNotFoundError, ModuleNotFoundError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    message = f"Installed skill {installed.name} at {installed.root}"
    if installed.source_revision is not None:
        message += f" ({installed.source_revision})"
    click.secho(message, fg="green")
    return 0


def _skills_command_help() -> str:
    return (
        "Usage: kedi skills add (--repo OWNER/REPOSITORY | --path PATH)\n\n"
        "Install one root SKILL.md into ~/.kedi/registry/skills."
    )


def _run_notebook_command(arguments: list[str]) -> int:
    try:
        from kedi_notebook.server import serve_cli
    except ImportError as exc:
        raise click.ClickException(
            "Kedi notebook is not installed. Install kedi-notebook, then run 'kedi notebook' again."
        ) from exc
    return serve_cli(arguments)


@click.command(context_settings=context_settings, help="Run a .kedi program with KediRuntime")
@click.argument("source", required=False)
@click.option("-h", "--help", "show_help", is_flag=True, help="Show this message and exit.")
@click.option(
    "-c",
    "--command",
    "command_source",
    help="Run Kedi source from the command string instead of a file",
)
@click.option(
    "--adapter",
    type=click.Choice(["pydantic", "dspy", "langchain", "claude", "codex", "acp"]),
    envvar="KEDI_ADAPTER",
    default="pydantic",
    help="Backend adapter for template execution",
)
@click.option(
    "--adapter-model",
    envvar=["KEDI_ADAPTER_MODEL", "MODEL_NAME"],
    default="groq:qwen/qwen3-32b",
    help="Model for template adapter",
)
@click.option(
    "--codegen-agent",
    type=click.Choice(["pydantic_ai", "mock"]),
    envvar="KEDI_CODEGEN_AGENT",
    default="pydantic_ai",
    help="Agent for AI code generation",
)
@click.option(
    "--codegen-model",
    envvar="KEDI_CODEGEN_MODEL",
    default="openrouter/minimax/minimax-m2.7",
    help="Model for codegen agent",
)
@click.option(
    "--codegen-retries",
    type=int,
    envvar="KEDI_CODEGEN_RETRIES",
    default=5,
    help="Number of retry stages for codegen",
)
@click.option("--test", is_flag=True, help="Run tests defined with @test: blocks")
@click.option("--eval", "do_eval", is_flag=True, help="Run evals defined with @eval: blocks")
@click.option(
    "--optimizer",
    type=click.Choice(["mock", "gepa"]),
    envvar="KEDI_OPTIMIZER",
    default="mock",
    help="Optimizer to use for prompt optimization",
)
@click.option(
    "--optimizer-model",
    envvar="KEDI_OPTIMIZER_MODEL",
    default="openrouter/minimax/minimax-m2.7",
    help="Model for optimizer (main model for GEPA)",
)
@click.option(
    "--optimizer-reflection-model",
    envvar="KEDI_OPTIMIZER_REFLECTION_MODEL",
    default="openrouter/minimax/minimax-m2.7",
    help="Reflection model for GEPA optimizer",
)
@click.option(
    "--optimizer-max-metric-calls",
    type=int,
    envvar="KEDI_OPTIMIZER_MAX_METRIC_CALLS",
    default=100,
    help="Maximum number of metric evaluations during optimization",
)
@click.option(
    "--optimizer-model-max-tokens",
    type=int,
    envvar="KEDI_OPTIMIZER_MODEL_MAX_TOKENS",
    default=8000,
    help="Max tokens for the main optimizer model",
)
@click.option(
    "--optimizer-reflection-model-max-tokens",
    type=int,
    envvar="KEDI_OPTIMIZER_REFLECTION_MODEL_MAX_TOKENS",
    default=32000,
    help="Max tokens for the reflection model (GEPA)",
)
@click.option(
    "--optimizer-reflection-minibatch-size",
    type=int,
    envvar="KEDI_OPTIMIZER_REFLECTION_MINIBATCH_SIZE",
    default=3,
    help="Number of examples to evaluate before triggering reflection (GEPA)",
)
@click.option(
    "--optimizer-max-validation-examples",
    type=int,
    envvar="KEDI_OPTIMIZER_MAX_VALIDATION_EXAMPLES",
    default=None,
    help="Maximum validation examples for baseline evaluation (default: use entire test set)",
)
@click.option(
    "--optimize",
    is_flag=True,
    help="Run optimizer on procedures with > optimize: blocks before evaluation",
)
@click.option(
    "--optimizer-fresh",
    is_flag=True,
    help="For GEPA: delete .optimized.json, scores, and checkpoint dir so optimization starts from the source template only",
)
@click.option("--no-cache", is_flag=True, help="Disable reading/writing codegen cache")
@click.option(
    "--idle",
    "idle_mode",
    is_flag=True,
    help="Start an interactive Kedi terminal session",
)
@click.option(
    "--record",
    "record_idle",
    is_flag=True,
    help="Persist an interactive session before exit",
)
@click.option(
    "--load",
    "load_session_path",
    type=click.Path(path_type=Path, dir_okay=False),
    help="Resume an interactive session snapshot",
)
@click.option(
    "--highlight",
    "highlight_idle",
    is_flag=True,
    help="Enable syntax highlighting in interactive mode",
)
@click.option(
    "--quiet",
    is_flag=True,
    help="Suppress progress output from codegen and GEPA optimization",
)
@click.option(
    "-p",
    "--parse",
    "parse_only",
    is_flag=True,
    help="Parse the source without compiling",
)
@click.pass_context
def main(
    ctx: click.Context,
    source: str | None,
    command_source: str | None,
    adapter: str,
    adapter_model: str,
    codegen_agent: str,
    codegen_model: str,
    codegen_retries: int,
    test: bool = False,
    do_eval: bool = False,
    optimizer: str = "mock",
    optimizer_model: str = "openrouter/minimax/minimax-m2.7",
    optimizer_reflection_model: str = "openrouter/minimax/minimax-m2.7",
    optimizer_max_metric_calls: int = 100,
    optimizer_model_max_tokens: int = 8000,
    optimizer_reflection_model_max_tokens: int = 32000,
    optimizer_reflection_minibatch_size: int = 3,
    optimizer_max_validation_examples: int | None = None,
    optimize: bool = False,
    optimizer_fresh: bool = False,
    no_cache: bool = False,
    idle_mode: bool = False,
    record_idle: bool = False,
    load_session_path: Path | None = None,
    highlight_idle: bool = False,
    quiet: bool = False,
    parse_only: bool = False,
    show_help: bool = False,
):
    if show_help:
        if source == "debug" and command_source is None:
            click.echo(
                "Usage: kedi debug --stdio\n\n"
                "Start the optional Kedi Debug Adapter Protocol backend for an editor."
            )
        elif source == "a2a" and command_source is None:
            from kedi.a2a.cli import run_a2a_cli

            model_source = ctx.get_parameter_source("adapter_model")
            explicit_model = adapter_model if model_source is not ParameterSource.DEFAULT else None
            return run_a2a_cli(
                [*ctx.args, "--help"],
                adapter=adapter,
                model=explicit_model,
            )
        elif source == "notebook" and command_source is None:
            click.echo(
                "Usage: kedi notebook [--host HOST] [--port PORT] [--python PATH] "
                "[--cwd PATH] [--no-open]\n\nRun the local Kedi notebook."
            )
        elif source == "skills" and command_source is None:
            click.echo(_skills_command_help())
        elif source in _PACKAGE_COMMANDS and command_source is None:
            click.echo(_package_command_help(source))
        elif source == "explain" and command_source is None:
            click.echo(
                "Usage: kedi explain SOURCE\n\nInspect model calls without running the program."
            )
        else:
            click.echo(ctx.get_help())
        return 0
    if idle_mode:
        if source is not None or command_source is not None:
            raise click.UsageError("--idle does not accept SOURCE or -c/--command.")
        if test or do_eval or optimize or parse_only:
            raise click.UsageError("--idle cannot be combined with execution modes.")
        try:
            return run_idle(
                adapter=adapter,
                model=adapter_model,
                record=record_idle,
                load_path=load_session_path,
                highlight=highlight_idle,
            )
        except SessionPersistenceError as exc:
            raise click.ClickException(str(exc)) from exc
    if record_idle or load_session_path is not None or highlight_idle:
        raise click.UsageError("--record, --load, and --highlight require --idle.")
    if source == "debug" and command_source is None:
        if list(ctx.args) != ["--stdio"]:
            raise click.UsageError("Usage: kedi debug --stdio (launch a program from your editor)")
        try:
            from importlib import import_module

            debugger_main = import_module("kedi_debugger.__main__").main
        except ModuleNotFoundError as exc:
            if exc.name not in {"kedi_debugger", "kedi_debugger.__main__"}:
                raise
            raise click.ClickException(
                "The selected Python needs kedi-debugger. For a source checkout, run "
                "'python -m pip install -e /path/to/kedi/debugger' in this environment."
            ) from exc
        return debugger_main(["--stdio"])
    if source == "skills" and command_source is None:
        return _run_skills_command(list(ctx.args))
    if source == "a2a" and command_source is None:
        from kedi.a2a.cli import run_a2a_cli

        model_source = ctx.get_parameter_source("adapter_model")
        explicit_model = adapter_model if model_source is not ParameterSource.DEFAULT else None
        return run_a2a_cli(
            list(ctx.args),
            adapter=adapter,
            model=explicit_model,
        )
    if source == "notebook" and command_source is None:
        return _run_notebook_command(list(ctx.args))
    if source in _PACKAGE_COMMANDS and command_source is None:
        return _run_package_command(source, list(ctx.args))
    if source == "explain" and command_source is None:
        if len(ctx.args) != 1:
            raise click.UsageError("Usage: kedi explain SOURCE")
        from kedi.explain import explain_file, render_explain

        try:
            report = explain_file(ctx.args[0], adapter=adapter, model=adapter_model)
        except (OSError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(render_explain(report))
        if report.diagnostics:
            ctx.exit(1)
        return 0
    if source == "parse":
        if command_source is not None:
            parse_only = True
            source = None
        elif not ctx.args:
            raise click.UsageError("Missing argument 'SOURCE'.")
        else:
            source = ctx.args[0]
            parse_only = True
            ctx.args = ctx.args[1:]
    if command_source is None and source is None:
        raise click.UsageError("Missing argument 'SOURCE'.")
    if command_source is not None and source is not None:
        ctx.args = [source, *ctx.args]
        source = None
    if command_source is None:
        assert source is not None
        try:
            with open(source, "r", encoding="utf-8") as f:
                f.read()
        except Exception as e:
            click.echo(f"Error reading {source}: {e}", err=True)
            ctx.exit(1)
    elif optimize or test or do_eval:
        raise click.UsageError("-c/--command only supports direct run or --parse.")
    runtime_args = parse_args(ctx.args)
    try:
        if parse_only:
            if command_source is not None:
                parse_program(command_source, source_path="<command>")
            else:
                assert source is not None
                source_path = Path(source).resolve()
                bundle = load_kedi_source_bundle(source_path)
                parse_program(
                    bundle.text,
                    source_path=str(source_path),
                    source_map=bundle.source_map,
                )
            click.secho("Parsed successfully.", fg="green")
            return 0

        if command_source is not None:
            code, out = run_source_cli(
                command_source,
                adapter=adapter,
                model=adapter_model,
                runtime_args=runtime_args,
            )
            if out:
                click.echo(out)
            if code != 0:
                ctx.exit(code)
            return code

        # Run optimization if requested (independent of eval flag)
        if optimize:
            assert source is not None
            code, msg = run_optimization_cli(
                source,
                optimizer=optimizer,
                optimizer_model=optimizer_model,
                reflection_model=(optimizer_reflection_model if optimizer == "gepa" else None),
                max_metric_calls=optimizer_max_metric_calls,
                optimizer_model_max_tokens=optimizer_model_max_tokens,
                reflection_model_max_tokens=optimizer_reflection_model_max_tokens,
                reflection_minibatch_size=optimizer_reflection_minibatch_size,
                max_validation_examples=optimizer_max_validation_examples,
                adapter=adapter,
                adapter_model=adapter_model,
                optimizer_fresh=optimizer_fresh,
                runtime_args=runtime_args,
                quiet=quiet,
            )
            if msg:
                if code != 0:
                    click.echo(msg, err=True)
                else:
                    click.echo(msg)
            if code != 0:
                ctx.exit(code)
            # Continue after optimization even if --eval is not set

        if test or do_eval:
            # Run validations
            if test:
                try:
                    assert source is not None
                    code, out = run_tests_cli(
                        source,
                        adapter=adapter,
                        model=adapter_model,
                        use_cache=not no_cache,
                        runtime_args=runtime_args,
                    )
                finally:
                    if no_cache:
                        Path(source).with_suffix(".cache.kedi").unlink(missing_ok=True)
                click.echo(out)
                if code != 0:
                    ctx.exit(code)
                return code
            else:
                # Optimization is handled above by run_optimization_cli when --optimize is set.
                # Do not pass optimize=True here or GEPA would run twice with default budget/model.
                try:
                    assert source is not None
                    code, out = run_evals_cli(
                        source,
                        adapter=adapter,
                        adapter_model=adapter_model,
                        codegen_model=codegen_model,
                        optimizer=optimizer,
                        optimizer_model=optimizer_model,
                        reflection_model=(
                            optimizer_reflection_model if optimizer == "gepa" else None
                        ),
                        optimize=False,
                        use_cache=not no_cache,
                        runtime_args=runtime_args,
                        quiet=quiet,
                    )
                finally:
                    if no_cache:
                        Path(source).with_suffix(".cache.kedi").unlink(missing_ok=True)
                if out:
                    if code != 0:
                        click.echo(out, err=True)
                    else:
                        click.echo(out)
                if code != 0:
                    ctx.exit(code)
                return code

        # Run codegen if needed
        assert source is not None
        ok, err = maybe_run_codegen(
            source,
            agent=codegen_agent,
            model=codegen_model,
            retries=codegen_retries,
            use_cache=not no_cache,
            quiet=quiet,
        )
        if not ok:
            click.echo(err or "Code generation failed.", err=True)
            ctx.exit(1)

        # Execute program
        try:
            code, out = run_program_cli(
                source,
                adapter=adapter,
                model=adapter_model,
                runtime_args=runtime_args,
            )
            if out:
                click.echo(out)
            if code != 0:
                ctx.exit(code)
            return code
        finally:
            # Clean up cache if --no-cache was used
            if no_cache:
                cache_path = Path(source).with_suffix(".cache.kedi")
                cache_path.unlink(missing_ok=True)
    except click.exceptions.Exit:
        # Re-raise click Exit exceptions (from ctx.exit())
        raise
    except KediParseError as e:
        click.echo(render_parse_error(e, use_color=sys.stderr.isatty()), err=True, nl=False)
        ctx.exit(1)
    except KediExecutionError as e:
        click.echo(render_execution_error(e, use_color=sys.stderr.isatty()), err=True, nl=False)
        ctx.exit(1)
    except Exception as e:
        click.echo(
            render_execution_error(
                KediExecutionError(message=str(e) or type(e).__name__, original=e),
                use_color=sys.stderr.isatty(),
            ),
            err=True,
            nl=False,
        )
        ctx.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
