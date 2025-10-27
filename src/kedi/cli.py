from __future__ import annotations

import os
import sys
from pathlib import Path

import click
import dotenv

from kedi.app import maybe_run_codegen, run_evals_cli, run_program_cli, run_tests_cli
from kedi.utils import parse_args

dotenv.load_dotenv()

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(THIS_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

context_settings = {"ignore_unknown_options": True, "allow_extra_args": True}


@click.command(context_settings=context_settings, help="Run a .kedi program with KediRuntime")
@click.argument("source")
@click.option(
    "--adapter",
    type=click.Choice(["pydantic", "dspy"]),
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
    default="openai:gpt-4o",
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
@click.option("--no-cache", is_flag=True, help="Disable reading/writing codegen cache")
@click.pass_context
def main(
    ctx: click.Context,
    source: str,
    adapter: str,
    adapter_model: str,
    codegen_agent: str,
    codegen_model: str,
    codegen_retries: int,
    test: bool = False,
    do_eval: bool = False,
    no_cache: bool = False,
):
    try:
        with open(source, "r", encoding="utf-8") as f:
            f.read()
    except Exception as e:
        click.echo(f"Error reading {source}: {e}", err=True)
        return 1
    data = parse_args(ctx.args)
    # the data comes from command line arguments
    # should be passed into runtime environment
    # TL;DR: pass `data` into runtime
    try:
        if test or do_eval:
            # Run validations
            if test:
                code, out = run_tests_cli(source, model=adapter_model)
                click.echo(out)
                return code
            if do_eval:
                code, out = run_evals_cli(source, model=adapter_model)
                click.echo(out)
                return code

        # Run codegen if needed
        ok, err = maybe_run_codegen(
            source,
            agent=codegen_agent,
            model=codegen_model,
            retries=codegen_retries,
            use_cache=not no_cache,
        )
        if not ok:
            click.echo(err or "Code generation failed.", err=True)
            return 1

        # Execute program
        try:
            code, out = run_program_cli(source, adapter=adapter, model=adapter_model)
            click.echo(out)
            return code
        finally:
            # Clean up cache if --no-cache was used
            if no_cache:
                cache_path = Path(source).with_suffix(".cache.kedi")
                cache_path.unlink(missing_ok=True)
    except Exception as e:
        click.echo(f"Execution error: {e}", err=True)
        return 1


if __name__ == "__main__":
    main()
