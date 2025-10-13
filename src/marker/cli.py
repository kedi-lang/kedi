from __future__ import annotations

import argparse
import os
import sys

import dotenv

dotenv.load_dotenv()

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(THIS_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from marker.app import run_tests_cli, run_evals_cli, maybe_run_codegen, run_program_cli
from pathlib import Path
from marker.agent_adapter.adapters import PydanticAdapter, DSPyAdapter


def get_config_value(cli_value, env_var: str, default: str) -> str:
    """Get configuration value with priority: CLI > ENV > DEFAULT"""
    if cli_value is not None:
        return cli_value
    env_value = os.environ.get(env_var)
    if env_value is not None:
        return env_value
    return default


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a .marker program with MarkerRuntime"
    )
    parser.add_argument("source", help="Path to .marker source file")
    
    # Adapter configuration (for template execution)
    parser.add_argument(
        "--adapter",
        choices=["pydantic", "dspy"],
        default=None,
        help="Backend adapter for template execution (default: pydantic, env: MARKER_ADAPTER)"
    )
    parser.add_argument(
        "--adapter-model",
        default=None,
        help="Model for template adapter (default: groq:qwen/qwen3-32b, env: MARKER_ADAPTER_MODEL or MODEL_NAME)"
    )
    
    # Codegen configuration
    parser.add_argument(
        "--codegen-agent",
        choices=["pydantic_ai", "mock"],
        default=None,
        help="Agent for AI code generation (default: pydantic_ai, env: MARKER_CODEGEN_AGENT)"
    )
    parser.add_argument(
        "--codegen-model",
        default=None,
        help="Model for codegen agent (default: openai:gpt-4o, env: MARKER_CODEGEN_MODEL)"
    )
    parser.add_argument(
        "--codegen-retries",
        type=int,
        default=None,
        help="Number of retry stages for codegen (default: 5, env: MARKER_CODEGEN_RETRIES)"
    )
    
    # Actions
    parser.add_argument("--test", action="store_true", help="Run tests defined with @test: blocks")
    parser.add_argument("--eval", dest="do_eval", action="store_true", help="Run evals defined with @eval: blocks")
    parser.add_argument("--no-cache", action="store_true", help="Disable reading/writing codegen cache")
    
    args = parser.parse_args()

    # Resolve configuration with priority: CLI > ENV > DEFAULT
    adapter = get_config_value(args.adapter, "MARKER_ADAPTER", "pydantic")
    adapter_model = get_config_value(
        args.adapter_model, 
        "MARKER_ADAPTER_MODEL",
        os.environ.get("MODEL_NAME", "groq:qwen/qwen3-32b")
    )
    codegen_agent = get_config_value(args.codegen_agent, "MARKER_CODEGEN_AGENT", "pydantic_ai")
    codegen_model = get_config_value(args.codegen_model, "MARKER_CODEGEN_MODEL", "openai:gpt-4o")
    codegen_retries = args.codegen_retries or int(os.environ.get("MARKER_CODEGEN_RETRIES", "5"))

    try:
        with open(args.source, "r", encoding="utf-8") as f:
            src = f.read()
    except Exception as e:
        print(f"Error reading {args.source}: {e}", file=sys.stderr)
        return 1

    try:
        if args.test or args.do_eval:
            # Run validations
            if args.test:
                code, out = run_tests_cli(args.source, model=adapter_model)
                print(out)
                return code
            if args.do_eval:
                code, out = run_evals_cli(args.source, model=adapter_model)
                print(out)
                return code

        # Run codegen if needed
        ok, err = maybe_run_codegen(
            args.source, 
            agent=codegen_agent,
            model=codegen_model,
            retries=codegen_retries,
            use_cache=not args.no_cache
        )
        if not ok:
            print(err or "Code generation failed.", file=sys.stderr)
            return 1
        
        # Execute program
        try:
            code, out = run_program_cli(args.source, adapter=adapter, model=adapter_model)
            print(out)
            return code
        finally:
            # Clean up cache if --no-cache was used
            if args.no_cache:
                cache_path = Path(args.source).with_suffix(".cache.marker")
                if cache_path.exists():
                    cache_path.unlink()
    except Exception as e:
        print(f"Execution error: {e}", file=sys.stderr)
        return 1
