"""Compare Kedi template prompt modes on one local, already-downloaded Laya model."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from kedi_laya import LayaClient

from kedi.agent_adapter import LangChainAdapter, PydanticAdapter
from kedi.lang import compile_program, parse_program

CASES = [
    ("A customer was charged twice and requests a refund.", "billing", True),
    ("The export button crashes. Please fix it. No refund is requested.", "technical", False),
    (
        "Please send an invoice for last month. Everything works. No refund is needed.",
        "billing",
        False,
    ),
    ("The login page crashes and nobody can sign in. Please repair it.", "technical", False),
]
CAPTURES = """
The responsible team is [team: Annotated[
  Literal["billing", "technical"],
  "Which team should handle the customer request?",
  `ChoiceCriteria({"billing": "Payments, duplicate charges and invoices", "technical": "Broken software and unavailable services"})`
]].
The probability that a refund is requested is [refund: Annotated[
  Probability,
  "Does the customer request a refund for a duplicate payment?"
]].

= ```
department, probability = team, refund
return {"team": department, "refund_probability": probability}
```
"""


class CompletionPydanticAdapter(PydanticAdapter):
    def template_prompt_mode(self, profile):
        return "completion"


class CompletionLangChainAdapter(LangChainAdapter):
    def template_prompt_mode(self, profile):
        return "completion"


class RecordingClient(LayaClient):
    def decide(self, state, questions, *, model=None):
        result = super().decide(state, questions, model=model)
        self.last_call = {
            "state": state,
            "questions": {key: asdict(value) for key, value in questions.items()},
            "usage": asdict(result.usage),
            "answers": {key: asdict(value) for key, value in result.answers.items()},
        }
        return result


def measure(checkpoint: str, repeats: int) -> dict:
    rows = []
    with RecordingClient(checkpoint) as client:
        for repeat in range(repeats):
            for case_id, (text, team, refund) in enumerate(CASES):
                source = (
                    "> import: decisions:\n  Probability\n  ChoiceCriteria\n\n>> " + text + CAPTURES
                )
                program = parse_program(source)
                modes = (
                    ("completion", "decision")
                    if (repeat + case_id) % 2 == 0
                    else ("decision", "completion")
                )
                for surface in ("pydantic", "langchain"):
                    questions = []
                    for mode in modes:
                        if surface == "pydantic":
                            cls = (
                                PydanticAdapter if mode == "decision" else CompletionPydanticAdapter
                            )
                            adapter = cls(client.as_pydantic_model())
                        else:
                            cls = (
                                LangChainAdapter
                                if mode == "decision"
                                else CompletionLangChainAdapter
                            )
                            adapter = cls(client.as_langchain_model())
                        runtime = compile_program(program, adapter=adapter)
                        start = time.perf_counter()
                        output = runtime.run_main()
                        seconds = time.perf_counter() - start
                        questions.append(client.last_call["questions"])
                        rows.append(
                            {
                                "case": case_id,
                                "repeat": repeat,
                                "surface": surface,
                                "mode": mode,
                                "cold": len(rows) == 0,
                                "seconds": seconds,
                                "output": output,
                                "expected_team": team,
                                "expected_refund": refund,
                                "team_correct": output["team"] == team,
                                "refund_correct_at_085": (output["refund_probability"] > 0.85)
                                == refund,
                                **client.last_call,
                            }
                        )
                    assert questions[0] == questions[1], "Only the prompt may differ within a pair"
    summaries = {}
    for surface in ("pydantic", "langchain"):
        for mode in ("completion", "decision"):
            selected = [r for r in rows if r["surface"] == surface and r["mode"] == mode]
            summaries[f"{surface}/{mode}"] = {
                "calls": len(selected),
                "team_correct": sum(r["team_correct"] for r in selected),
                "refund_correct_at_085": sum(r["refund_correct_at_085"] for r in selected),
                "warm_median_seconds": statistics.median(
                    r["seconds"] for r in selected if not r["cold"]
                ),
                "mean_input_tokens": statistics.mean(r["usage"]["input_tokens"] for r in selected),
            }
    return {
        "checkpoint": str(Path(checkpoint).resolve()),
        "versions": {
            name: version(name)
            for name in (
                "laya-mlx",
                "mlx",
                "kedi-decisions",
                "kedi-laya",
                "pydantic-ai-slim",
                "langchain",
            )
        },
        "notes": "Four fixed cases; repeats are not independent tasks. No calibrated confidence or general accuracy claim. Cold inference is excluded from warm timing.",
        "summary": summaries,
        "rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.checkpoint.is_dir():
        parser.error(
            "checkpoint must be an existing local directory; this validation does not download weights"
        )
    if not 1 <= args.repeats <= 10:
        parser.error("--repeats must be between 1 and 10")
    report = measure(str(args.checkpoint), args.repeats)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
