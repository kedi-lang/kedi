#!/usr/bin/env python3
"""Execute bundled Kedi examples with assertions and no live model requests.

Requires the target Kedi environment, including pydantic-ai-slim. No pytest.
Optional --report writes a JSON result to the caller-selected path.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import redirect_stdout
import importlib.metadata
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "assets" / "examples"


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--report", type=Path)
    args = cli.parse_args()
    try:
        import kedi
        from kedi.agent_adapter import PydanticAdapter
        from kedi.lang import compile_program, parse_program
        import pydantic_ai.models
        from pydantic_ai.models.test import TestModel
        from pydantic_ai.models.function import FunctionModel
        from pydantic_ai.messages import (
            ModelRequest,
            ModelResponse,
            ToolCallPart,
            ToolReturnPart,
            TextPart,
        )
    except ImportError as exc:
        cli.exit(2, f"Missing validation dependency: {exc}\nUse the Kedi project interpreter.\n")

    # A fixture accidentally selecting a real provider must fail rather than use credentials.
    pydantic_ai.models.ALLOW_MODEL_REQUESTS = False
    for name in ("KEDI_ADAPTER", "KEDI_AGENT", "KEDI_ADAPTER_MODEL"):
        os.environ.pop(name, None)
    results = []

    def record(name, fn):
        try:
            fn()
            results.append({"name": name, "ok": True})
            print(f"PASS {name}")
        except Exception as exc:
            results.append({"name": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
            print(f"FAIL {name}: {exc}", file=sys.stderr)
            traceback.print_exc()

    def equal(value, expected):
        if value != expected:
            raise AssertionError(f"Expected {expected!r}, received {value!r}")

    def execute(name, model=None, *, source=None, **options):
        path = (EXAMPLES / name).resolve()
        program = parse_program(
            source if source is not None else path.read_text(encoding="utf-8"),
            source_path=str(path),
        )
        adapter = PydanticAdapter(model) if model is not None else None
        runtime = compile_program(program, adapter=adapter, **options)
        try:
            return runtime.run_main()
        finally:
            asyncio.run(runtime.aclose())

    def command(*arguments, env=None):
        result = subprocess.run(
            [sys.executable, "-m", "kedi", *map(str, arguments)],
            text=True,
            capture_output=True,
            env=env,
            timeout=45,
        )
        if result.returncode:
            raise AssertionError(
                f"CLI exited {result.returncode}: {result.stderr}\n{result.stdout}"
            )
        return result.stdout

    def expect_error(fn, expected):
        try:
            fn()
        except Exception as exc:
            chain = [exc, getattr(exc, "original", None), exc.__cause__]
            if not any(isinstance(item, expected) for item in chain if item is not None):
                raise AssertionError(f"Expected {expected}, received {type(exc)}: {exc}") from exc
        else:
            raise AssertionError(f"Expected {expected}; operation unexpectedly succeeded")

    def all_parse():
        from check_kedi import check

        for path in sorted(EXAMPLES.rglob("*.kedi")):
            for tree_only in (True, False):
                row = check(path.resolve(), tree_only=tree_only)
                if not row["ok"]:
                    raise AssertionError(row)

    record("all example files: CST and Kedi AST", all_parse)
    deterministic = {
        "hello.kedi": "Hello, Kedi.",
        "control_flow.kedi": {"status": "pending", "count": 3, "total": 6},
        "map_batch.kedi": [4, 9, 25],
        "python_scope.kedi": {"count": 2, "fixture": "source-relative fixture"},
        "constraints.kedi": {
            "first": {"score": 0.75, "tags": ["checked"]},
            "second": {"score": 1.0, "tags": []},
        },
        "public_api.kedi": "kedi",
    }
    for name, expected in deterministic.items():
        record(name, lambda name=name, expected=expected: equal(execute(name), expected))
    record(
        "native models and sole call RHS",
        lambda: equal(
            execute("native_values.kedi").model_dump(), {"total": 36, "labels": ["book", "pen"]}
        ),
    )
    record(
        "relative module import and positional call",
        lambda: equal(
            execute("modules/main.kedi").model_dump(),
            {"subtotal": 1000, "discount": 100, "total": 900},
        ),
    )
    record(
        "native return vs rendered variable RHS",
        lambda: equal(
            execute("probe.kedi", source="[n: int] = `5`\n[copy] = <n>\n= `(n, copy)`\n"), (5, "5")
        ),
    )

    def cli_checks():
        equal(
            command(EXAMPLES / "cli_args.kedi", "--name", "Ada", "--count", "2").strip(),
            "Hello, Ada. | Hello, Ada.",
        )
        equal(command(EXAMPLES / "cli_args.kedi").strip(), "Hello, Kedi.")

    record("CLI typed argument conversion and defaults", cli_checks)

    def tests_and_eval():
        out = command(EXAMPLES / "public_api.kedi", "--test")
        if out.count("OK") != 2:
            raise AssertionError(out)
        out = command(EXAMPLES / "modules/pricing.kedi", "--test")
        if out.count("OK") != 2:
            raise AssertionError(out)
        out = command(EXAMPLES / "public_api.kedi", "--eval")
        if "1.0000" not in out:
            raise AssertionError(out)

    record("native @test suites and held-out @eval", tests_and_eval)

    def package():
        with tempfile.TemporaryDirectory(prefix="kedi-skill-registry-") as folder:
            env = dict(os.environ, KEDI_HOME=str(Path(folder).resolve()))
            command("install", EXAMPLES / "package/package.kedi", env=env)
            equal(
                command(EXAMPLES / "package_client.kedi", env=env).strip(), "Package hello, Kedi."
            )

    record("package install and import in isolated registry", package)

    def python_api():
        with kedi.context(adapter=PydanticAdapter(TestModel(call_tools=[])), model=None):
            with redirect_stdout(io.StringIO()) as output:
                runpy.run_path(str(EXAMPLES / "python_api.py"), run_name="__main__")
        equal(output.getvalue().strip(), "query=2, bind=10")

    record("Python query and file bind", python_api)

    actions = [
        {"task": "Update parser tests", "owner": "Ada"},
        {"task": "Update migration guide", "owner": None},
    ]
    record(
        "typed nested extraction",
        lambda: equal(
            [
                item.model_dump()
                for item in execute(
                    "extraction.kedi",
                    TestModel(call_tools=[], custom_output_args={"actions": actions}),
                )
            ],
            actions,
        ),
    )

    def pipeline():
        calls = []

        def respond(messages, info):
            tool = info.output_tools[0]
            fields = tool.parameters_json_schema["properties"]
            prompt = str(messages)
            calls.append(prompt)
            if "category" in fields:
                payload = {"category": "access"}
            else:
                if "access" not in prompt:
                    raise AssertionError("Dependent template did not receive capture")
                payload = {"instruction": "Route to account support."}
            return ModelResponse(parts=[ToolCallPart(tool.name, payload)])

        equal(
            execute("pipeline.kedi", FunctionModel(respond)),
            {"category": "access", "instruction": "Route to account support."},
        )
        equal(len(calls), 2)

    record("dependent structured model pipeline", pipeline)

    def history():
        calls = []

        def respond(messages, info):
            if calls and "atlas" not in str(messages):
                raise AssertionError("Previous turn was absent from history")
            calls.append(messages)
            return ModelResponse(parts=[TextPart("Remembered." if len(calls) == 1 else "atlas")])

        equal(execute("raw_history.kedi", FunctionModel(respond)), "atlas")
        equal(len(calls), 2)

    record("raw invokes and ordered stateful history", history)

    def claim():
        def respond(messages, info):
            tool = info.output_tools[0]
            fields = tool.parameters_json_schema["properties"]
            return ModelResponse(parts=[ToolCallPart(tool.name, {name: False for name in fields})])

        equal(execute("claim_branch.kedi", FunctionModel(respond)), "hold")

    record("model-classified branch", claim)

    def model_map():
        def respond(messages, info):
            label = "first" if "first record" in str(messages) else "second"
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"label": label})])

        equal(execute("model_map.kedi", FunctionModel(respond)), ["first", "second"])

    record("model map preserves iteration association", model_map)

    def tool_profile():
        seen = []

        def respond(messages, info):
            returns = [
                p
                for m in messages
                if isinstance(m, ModelRequest)
                for p in m.parts
                if isinstance(p, ToolReturnPart)
            ]
            if not returns:
                if "lookup_release" not in {t.name for t in info.function_tools}:
                    raise AssertionError("Tool was not registered")
                return ModelResponse(parts=[ToolCallPart("lookup_release", {"version": "1.4.0"})])
            equal(returns[-1].content, "Typed child results and cancellation cleanup.")
            seen.append(returns[-1].tool_name)
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        info.output_tools[0].name, {"summary": "Typed children; cleanup fixed."}
                    )
                ]
            )

        equal(
            execute("tool_profile.kedi", FunctionModel(respond)), "Typed children; cleanup fixed."
        )
        equal(seen, ["lookup_release"])

    record("profile uses real read-only tool body", tool_profile)

    def subagents():
        calls = []

        def respond(messages, info):
            security = "authorization and data exposure" in str(messages)
            value = {
                "issue": "No account ownership check" if security else "Connection can leak",
                "recommendation": "Check ownership" if security else "Use finally",
            }
            if "any account id" not in str(messages):
                raise AssertionError("Child missing explicit task context")
            calls.append(value)
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        info.output_tools[0].name,
                        {"task_summary": "Reviewed.", "final_result": {"review": value}},
                    )
                ]
            )

        result = execute("parallel_review.kedi", FunctionModel(respond))
        equal(result.security.recommendation, "Check ownership")
        equal(result.reliability.recommendation, "Use finally")
        equal(len(calls), 2)

    record("two native tasks and typed await results", subagents)

    def capability():
        def respond(messages, info):
            instructions = str(getattr(info, "instructions", "")) + str(messages)
            if "Distinguish observed facts" not in instructions:
                raise AssertionError("Capability instruction missing")
            return ModelResponse(
                parts=[ToolCallPart(info.output_tools[0].name, {"readiness": "pending"})]
            )

        equal(execute("capability.kedi", FunctionModel(respond)), "pending")

    record("version-gated eager capability composition", capability)

    def budget():
        from kedi import RunBudgetExceeded

        calls = []

        def respond(messages, info):
            calls.append(messages)
            return ModelResponse(parts=[TextPart("Tests need review.")])

        expect_error(lambda: execute("budget.kedi", FunctionModel(respond)), RunBudgetExceeded)
        equal(len(calls), 1)

    record("request budget rejects dependent second generation", budget)
    record(
        "optimize span executes original typed template",
        lambda: equal(
            execute("optimize.kedi", TestModel(call_tools=[], custom_output_args={"owner": "Ada"})),
            "Ada",
        ),
    )

    from kedi.lang.parser import KediParseError

    invalid_sources = {
        "removed system directive": "> system: old\n",
        "bare template": "Fill [value].\n",
        "typed assignment": "[n: int] = `1`\n[n: int] := `2`\n",
        "orphan map": "> map:\n  > show: orphan\n",
        "unreachable statement": "= done\n> show: unreachable\n",
        "raw capture with output fields": "[text] << Produce [field].\n",
    }
    for name, source in invalid_sources.items():
        record(
            "reject " + name,
            lambda source=source: expect_error(lambda: parse_program(source), KediParseError),
        )
    for name, source in {
        "string supplied to int parameter": "@f(n: int) -> int:\n  = `n`\n= <f(5)>\n",
        "unannotated non-string return": "@f():\n  = `5`\n= `f()`\n",
        "truthy non-bool condition": "> if: `1`:\n  > show: invalid\n",
        "assignment violates type": "[n: int] = `1`\n[n] := text\n= `n`\n",
    }.items():
        record(
            "reject " + name,
            lambda source=source: expect_error(
                lambda: execute("negative.kedi", source=source), TypeError
            ),
        )

    versions = {
        name: importlib.metadata.version(name)
        for name in ("kedi", "tree-sitter", "tree-sitter-kedi", "pydantic", "pydantic-ai-slim")
    }
    report = {
        "versions": versions,
        "python": sys.version.split()[0],
        "live_provider_calls": False,
        "checks": results,
        "passed": sum(row["ok"] for row in results),
        "total": len(results),
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{report['passed']}/{report['total']} checks passed. Live provider calls disabled.")
    return 0 if all(row["ok"] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
