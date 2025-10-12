from __future__ import annotations

import argparse
import os
import sys

import dotenv

dotenv.load_dotenv()

# Ensure local 'src' is on sys.path when running from repo root
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(THIS_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from marker.lang import compile_program, parse_program
from marker.agent_adapter.adapters import PydanticAdapter, DSPyAdapter


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a .marker program with MarkerRuntime (real LLM)"
    )
    parser.add_argument("source", help="Path to .marker source file")
    parser.add_argument("--adapter", choices=["pydantic", "dspy"], default=os.environ.get("MARKER_ADAPTER", "pydantic"), help="Backend adapter to use")
    parser.add_argument("--model", dest="model", default=os.environ.get("MODEL_NAME", os.environ.get("GROQ_MODEL", None)), help="Model id for the chosen adapter (if applicable)")
    args = parser.parse_args()

    try:
        with open(args.source, "r", encoding="utf-8") as f:
            src = f.read()
    except Exception as e:
        print(f"Error reading {args.source}: {e}", file=sys.stderr)
        return 1

    try:
        program = parse_program(src)
        # Instantiate adapter per flags
        adapter = (
            DSPyAdapter(model=args.model or "groq:qwen/qwen3-32b")
            if args.adapter == "dspy"
            else PydanticAdapter(model=args.model or "groq:qwen/qwen3-32b")
        )
        runtime = compile_program(program, adapter=adapter)
        result = runtime.run_main()
        if result is None:
            print("")
        else:
            print(str(result))
        return 0
    except Exception as e:
        print(f"Execution error: {e}", file=sys.stderr)
        return 1
