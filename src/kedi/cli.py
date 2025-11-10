from __future__ import annotations

import os
import sys
from pathlib import Path

import click
import dotenv

from kedi.app import maybe_run_codegen, run_evals_cli, run_optimization_cli, run_program_cli, run_tests_cli
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
    default="openai/gpt-4.1-mini",
    help="Model for optimizer (main model for GEPA)",
)
@click.option(
    "--optimizer-reflection-model",
    envvar="KEDI_OPTIMIZER_REFLECTION_MODEL",
    default="openai/gpt-5",
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
    optimizer: str = "mock",
    optimizer_model: str = "openai/gpt-4o",
    optimizer_reflection_model: str = "openai/gpt-5",
    optimizer_max_metric_calls: int = 30,
    optimizer_model_max_tokens: int = 2000,
    optimizer_reflection_model_max_tokens: int = 32000,
    optimizer_reflection_minibatch_size: int = 3,
    optimizer_max_validation_examples: int | None = None,
    optimize: bool = False,
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
        # Run optimization if requested (independent of eval flag)
        if optimize:
            code, msg = run_optimization_cli(
                source,
                optimizer=optimizer,
                optimizer_model=optimizer_model,
                reflection_model=optimizer_reflection_model if optimizer == "gepa" else None,
                max_metric_calls=optimizer_max_metric_calls,
                optimizer_model_max_tokens=optimizer_model_max_tokens,
                reflection_model_max_tokens=optimizer_reflection_model_max_tokens,
                reflection_minibatch_size=optimizer_reflection_minibatch_size,
                max_validation_examples=optimizer_max_validation_examples,
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
                code, out = run_tests_cli(source, model=adapter_model)
                click.echo(out)
                if code != 0:
                    ctx.exit(code)
                return code
            if do_eval:
                code, out = run_evals_cli(
                    source, 
                    model=adapter_model, 
                    optimizer=optimizer, 
                    optimizer_model=optimizer_model,
                    reflection_model=optimizer_reflection_model if optimizer == "gepa" else None,
                    optimize=optimize, 
                    use_cache=not no_cache
                )
                if out:
                    if code != 0:
                        click.echo(out, err=True)
                    else:
                        click.echo(out)
                if code != 0:
                    ctx.exit(code)
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
    except click.exceptions.Exit:
        # Re-raise click Exit exceptions (from ctx.exit())
        raise
    except Exception as e:
        click.echo(f"Execution error: {e}", err=True)
        return 1


if __name__ == "__main__":
    main()
