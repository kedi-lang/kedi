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

from marker_pattern.lang import compile_program, parse_program


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a .marker program with MarkerRuntime (real LLM)"
    )
    parser.add_argument("source", help="Path to .marker source file")
    parser.add_argument(
        "--model",
        dest="model",
        default=None,
        help="Override model id (e.g. groq:qwen/qwen3-32b)",
    )
    args = parser.parse_args()

    try:
        with open(args.source, "r", encoding="utf-8") as f:
            src = f.read()
    except Exception as e:
        print(f"Error reading {args.source}: {e}", file=sys.stderr)
        return 1

    try:
        program = parse_program(src)
        runtime = compile_program(program)
        if args.model:
            # Override the model after compilation; procedures will use it at call time
            runtime.model = args.model
        result = runtime.run_main()
        if result is None:
            print("")
        else:
            print(str(result))
        return 0
    except Exception as e:
        print(f"Execution error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
