"""The adapter conformance kit: Kedi renders, the adapter transports.

``check_adapter(factory)`` runs canary Kedi programs through an adapter that ``factory``
points at a :class:`RecordingEndpoint`, captures the request Kedi rendered for each call, and
checks the recorded wire requests against it:

- every instruction part reaches the system text exactly once, and the prompt the user turn;
- the output schema's and the tools' descriptions arrive unchanged;
- the returned value passes Kedi's validation;
- text Kedi did not write does not change with the request (two variants of each scenario
  are compared); request-independent harness text passes and is listed;
- a run budget stops a request before it reaches the endpoint, SDK retries included.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from typing import Any

from ..request import AgentRunRequest, RenderedRequest
from ..request.context import observe_requests
from .endpoint import Exchange, RecordingEndpoint, Script, texts

PIXEL = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
TOOL_MARKER = "CANARY-TOOL-RESULT-7"


@dataclass(frozen=True)
class Variant:
    source: str
    script: Script
    expected: Any
    error: str | None = None
    """The program must fail: with this exception type, or with any error for ``ANY_ERROR``."""
    requests: int | None = None
    """Exactly this many requests reach the endpoint."""


ANY_ERROR = "*"


@dataclass(frozen=True)
class Scenario:
    name: str
    variants: tuple[Variant, Variant]
    requires: str | None = None
    tool_result: bool | None = None
    foreign_text: bool = True


def _tool_program(head: str, prompt: str, approval: str, budget: str = "") -> str:
    return (
        f"{head}\n{budget}> approval: {approval}\n> use: canary_tool\n"
        f"[reply] << {prompt}\n= <reply>\n"
    )


ONE_REQUEST = "> budget:\n    request_limit: 1\n"


def _tool(name_line: str, doc: str) -> str:
    return f"@canary_tool({name_line}) -> str:\n  ###\n  {doc}\n  ###\n  = {TOOL_MARKER}\n"


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "invoke",
        (
            Variant(
                "> instructions: Keep every answer to one line.\n"
                "[reply] << Greet the reviewers of the release.\n= <reply>\n",
                Script(final="Hello, reviewers.", text=True),
                "Hello, reviewers.",
            ),
            Variant(
                "> instructions: Use plain words a child knows.\n"
                "[reply] << Name the largest ocean on Earth.\n= <reply>\n",
                Script(final="The Pacific.", text=True),
                "The Pacific.",
            ),
        ),
    ),
    Scenario(
        "produce",
        (
            Variant(
                '~Address(city: Annotated[str, "City where the person lives"], code: str)\n'
                '~Person(name: Annotated[str, "Full legal name"], home: Address)\n'
                ">> Ada Lovelace lived in London, W1. Her record is [person: Person].\n"
                "= `person.home.city`\n",
                Script(
                    final={
                        "person": {"name": "Ada Lovelace", "home": {"city": "London", "code": "W1"}}
                    },
                    text=False,
                ),
                "London",
            ),
            Variant(
                '~Engine(kind: Annotated[str, "Engine type, such as electric"], power: int)\n'
                '~Car(model: Annotated[str, "Model name as sold"], engine: Engine)\n'
                ">> The Zoe has an electric 100 kW motor. The car is [car: Car].\n"
                "= `car.engine.power`\n",
                Script(
                    final={"car": {"model": "Zoe", "engine": {"kind": "electric", "power": 100}}},
                    text=False,
                ),
                100,
            ),
        ),
    ),
    Scenario(
        "claim",
        (
            Variant(
                "[r] = unknown\n> if: The number seven is a prime number\n  [r] := yes\n> else:\n  [r] := no\n= <r>\n",
                Script(final=True, text=False),
                "yes",
            ),
            Variant(
                "[r] = unknown\n> if: A triangle has four corners\n  [r] := yes\n> else:\n  [r] := no\n= <r>\n",
                Script(final=False, text=False),
                "no",
            ),
        ),
    ),
    Scenario(
        "tool-allow",
        (
            Variant(
                _tool_program(
                    _tool("city: str", "Weather forecast for one city."),
                    "Use canary_tool for Ankara and report what it says.",
                    "allow",
                ),
                Script(
                    final="It is sunny.",
                    text=True,
                    tool="canary_tool",
                    arguments={"city": "Ankara"},
                ),
                "It is sunny.",
            ),
            Variant(
                _tool_program(
                    _tool("country: str", "Population count of one country."),
                    "Use canary_tool for Norway and report what it says.",
                    "allow",
                ),
                Script(
                    final="About five million.",
                    text=True,
                    tool="canary_tool",
                    arguments={"country": "Norway"},
                ),
                "About five million.",
            ),
        ),
        tool_result=True,
    ),
    Scenario(
        "tool-deny",
        (
            Variant(
                _tool_program(
                    _tool("city: str", "Weather forecast for one city."),
                    "Use canary_tool for Ankara and report what it says.",
                    "deny",
                ),
                Script(
                    final="It was denied.",
                    text=True,
                    tool="canary_tool",
                    arguments={"city": "Ankara"},
                ),
                "It was denied.",
            ),
            Variant(
                _tool_program(
                    _tool("country: str", "Population count of one country."),
                    "Use canary_tool for Norway and report what it says.",
                    "deny",
                ),
                Script(
                    final="The tool was denied.",
                    text=True,
                    tool="canary_tool",
                    arguments={"country": "Norway"},
                ),
                "The tool was denied.",
            ),
        ),
        tool_result=False,
    ),
    Scenario(
        "media",
        (
            Variant(
                f'> import: media\n[chart: Image] = `image("{PIXEL}")`\n'
                "[reply] << Describe <<chart>> in one sentence.\n= <reply>\n",
                Script(final="A single pixel.", text=True),
                "A single pixel.",
            ),
            Variant(
                f'> import: media\n[icon: Image] = `image("{PIXEL}")`\n'
                "[reply] << What color is <<icon>>?\n= <reply>\n",
                Script(final="Red.", text=True),
                "Red.",
            ),
        ),
        requires="media_input",
        # Kedi's attachment framing is not in render().user, and its ids change per variant.
        foreign_text=False,
    ),
    Scenario(
        "request-budget",
        (
            # The tool call takes a second request, which the budget refuses before it is sent.
            Variant(
                _tool_program(
                    _tool("city: str", "Weather forecast for one city."),
                    "Use canary_tool for Ankara and report what it says.",
                    "allow",
                    ONE_REQUEST,
                ),
                Script(
                    final="It is sunny.",
                    text=True,
                    tool="canary_tool",
                    arguments={"city": "Ankara"},
                ),
                None,
                error="RunBudgetExceeded",
                requests=1,
            ),
            # The first request fails; a retry must be admitted too, so none reaches the wire.
            Variant(
                f"{ONE_REQUEST}[reply] << Name the largest ocean on Earth.\n= <reply>\n",
                Script(final="The Pacific.", text=True, fail_first=1),
                None,
                error=ANY_ERROR,
                requests=1,
            ),
        ),
        requires="request_budget",
        foreign_text=False,
    ),
)

NOT_COVERED = ("subagents",)


@dataclass
class ScenarioResult:
    name: str
    status: str
    problems: list[str] = field(default_factory=list)


@dataclass
class ConformanceReport:
    adapter: str
    results: list[ScenarioResult] = field(default_factory=list)
    harness_text: list[str] = field(default_factory=list)
    not_covered: tuple[str, ...] = NOT_COVERED

    @property
    def passed(self) -> bool:
        return all(result.status != "failed" for result in self.results)

    def as_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "passed": self.passed,
            "scenarios": [
                {"name": result.name, "status": result.status, "problems": result.problems}
                for result in self.results
            ],
            "harness_text": self.harness_text,
            "not_covered": list(self.not_covered),
        }

    def render(self) -> str:
        lines = [f"Conformance check for {self.adapter}: {'passed' if self.passed else 'failed'}"]
        for result in self.results:
            lines.append(f"  {result.status:8} {result.name}")
            lines.extend(f"           - {problem}" for problem in result.problems)
        if self.harness_text:
            lines.append("  Request-independent harness text (allowed):")
            lines.extend(f"    {text!r}" for text in self.harness_text)
        lines.append(f"  Not covered yet: {', '.join(self.not_covered)}")
        return "\n".join(lines)


@dataclass
class _Run:
    rendered: list[RenderedRequest]
    exchanges: list[Exchange]
    value: Any = None
    error: BaseException | None = None


def check_adapter(factory: Callable[[str], Any]) -> ConformanceReport:
    """Run the canary scenarios through ``factory(base_url)``'s adapter and check the wire."""

    with RecordingEndpoint() as endpoint:
        probe = factory(endpoint.base_url)
        report = ConformanceReport(adapter=type(probe).__name__)
        capabilities = getattr(probe, "capabilities", None)
        _close(probe)
        for scenario in SCENARIOS:
            if scenario.requires is not None and not (
                capabilities is not None and capabilities.supports(scenario.requires)
            ):
                report.results.append(ScenarioResult(scenario.name, "skipped"))
                continue
            runs = [_run(endpoint, factory, variant) for variant in scenario.variants]
            result = ScenarioResult(scenario.name, "passed")
            for variant, run in zip(scenario.variants, runs, strict=True):
                result.problems.extend(_check_run(scenario, variant, run))
            if scenario.foreign_text and not result.problems:
                foreign = [_foreign(run.rendered[0], run.exchanges[0]) for run in runs]
                if foreign[0] != foreign[1]:
                    result.problems.append(
                        "text the adapter wrote changes with the request: "
                        f"{list(foreign[0])!r} vs {list(foreign[1])!r}"
                    )
                else:
                    report.harness_text.extend(
                        text for text in foreign[0] if text not in report.harness_text
                    )
            if result.problems:
                result.status = "failed"
            report.results.append(result)
    return report


def _run(endpoint: RecordingEndpoint, factory: Callable[[str], Any], variant: Variant) -> _Run:
    from ...lang import compile_program, parse_program

    run = _Run(rendered=[], exchanges=[])

    def observe(request: AgentRunRequest) -> None:
        rendered = request.render()
        run.rendered.append(rendered)
        endpoint.script = replace(
            variant.script, kedi_tools=frozenset(tool.name for tool in rendered.tools)
        )

    endpoint.script = variant.script
    endpoint.take()
    adapter = factory(endpoint.base_url)
    runtime = compile_program(parse_program(variant.source), adapter=adapter)
    try:
        with observe_requests(observe):
            run.value = runtime.run_main()
    except Exception as error:  # noqa: BLE001 - the report says what failed
        run.error = error
    finally:
        asyncio.run(runtime.aclose())
        _close(adapter)
        run.exchanges = endpoint.take()
    return run


def _close(adapter: Any) -> None:
    """Close an adapter the kit created, through ``aclose()`` or ``close()`` when it has one."""

    close = getattr(adapter, "aclose", None) or getattr(adapter, "close", None)
    if close is None:
        return
    result = close()
    if inspect.iscoroutine(result):
        asyncio.run(result)


def _check_run(scenario: Scenario, variant: Variant, run: _Run) -> list[str]:
    problems: list[str] = []
    if variant.requests is not None and len(run.exchanges) != variant.requests:
        problems.append(
            f"{len(run.exchanges)} requests reached the model endpoint, expected {variant.requests}"
        )
    if run.error is not None:
        error = getattr(run.error, "original", None) or run.error
        if variant.error in (ANY_ERROR, type(error).__name__):
            return problems
        message = getattr(run.error, "message", None) or str(error)
        return [*problems, f"the program failed: {type(error).__name__}: {message.strip()[:600]}"]
    if variant.error is not None:
        return [*problems, f"the program returned {run.value!r}; it must fail with {variant.error}"]
    if run.value != variant.expected:
        problems.append(f"the program returned {run.value!r}, expected {variant.expected!r}")
    if not run.exchanges:
        return [*problems, "no request reached the model endpoint"]
    (rendered,) = run.rendered  # every scenario makes one Kedi call
    for index, exchange in enumerate(run.exchanges, start=1):
        problems.extend(
            f"request {index}: {problem}" for problem in _check_exchange(rendered, exchange)
        )
    if scenario.tool_result is not None:
        seen = any(TOOL_MARKER in json.dumps(exchange.body) for exchange in run.exchanges)
        if seen != scenario.tool_result:
            problems.append(
                "the tool result reached the model although approval denied it"
                if seen
                else "the tool's result never reached the model"
            )
    return problems


@dataclass(frozen=True)
class _Wire:
    system: tuple[str, ...]
    user: tuple[str, ...]
    tool_descriptions: tuple[str, ...]
    body: str


def _wire(exchange: Exchange) -> _Wire:
    body = exchange.body
    system: list[str] = []
    user: list[str] = []
    if exchange.protocol == "openai":
        for message in body.get("messages") or []:
            role = message.get("role")
            if role in {"system", "developer"}:
                system.append(texts(message.get("content")))
            elif role == "user":
                user.append(texts(message.get("content")))
        descriptions = [
            tool.get("function", {}).get("description") or "" for tool in body.get("tools") or []
        ]
    elif exchange.protocol == "anthropic":
        system.append(texts(body.get("system")))
        for message in body.get("messages") or []:
            if message.get("role") == "user":
                user.append(texts(message.get("content")))
        descriptions = [tool.get("description") or "" for tool in body.get("tools") or []]
    else:
        system.append(body.get("instructions") or "")
        user.extend(
            message.get("content") or ""
            for message in body.get("messages") or []
            if message.get("role") == "user"
        )
        descriptions = [tool.get("description") or "" for tool in body.get("tools") or []]
    return _Wire(
        tuple(text for text in system if text),
        tuple(text for text in user if text),
        tuple(text for text in descriptions if text),
        json.dumps(body, ensure_ascii=False),
    )


def _check_exchange(rendered: RenderedRequest, exchange: Exchange) -> Iterator[str]:
    wire = _wire(exchange)
    system, user = "\n".join(wire.system), "\n".join(wire.user)
    for part in (part for part in rendered.instruction_parts if part.content):
        in_system, in_user = system.count(part.content), user.count(part.content)
        if in_system + in_user != 1:
            yield f"instruction part {part.identity} reached the wire {in_system + in_user} times"
        elif in_system != 1:
            yield f"instruction part {part.identity} is not in the system text"
    if user.count(rendered.user) != 1:
        yield f"the prompt reached the user turn {user.count(rendered.user)} times"
    for description in _descriptions(rendered.output_json_schema):
        if _escaped(description) not in wire.body:
            yield f"the output schema description {description!r} did not arrive"
    for tool in rendered.tools:
        if tool.description and _escaped(tool.description) not in wire.body:
            yield f"the description of tool {tool.name} did not arrive"


def _foreign(rendered: RenderedRequest, exchange: Exchange) -> tuple[str, ...]:
    """Text in the request the adapter wrote: everything but Kedi's strings, normalized."""

    wire = _wire(exchange)
    text = "\n".join((*wire.system, *wire.user, *wire.tool_descriptions))
    kedi = {part.content for part in rendered.instruction_parts if part.content}
    kedi |= {rendered.user, *(tool.description for tool in rendered.tools if tool.description)}
    kedi |= set(_descriptions(rendered.output_json_schema))
    kedi |= {text for tool in rendered.tools for text in _descriptions(tool.parameters)}
    for string in sorted(kedi, key=len, reverse=True):
        text = text.replace(string, "\x00")
    pieces = (re.sub(r"\s+", " ", piece).strip() for piece in text.split("\x00"))
    return tuple(piece for piece in pieces if piece)


def _descriptions(schema: Any) -> Iterator[str]:
    if isinstance(schema, dict):
        description = schema.get("description")
        if isinstance(description, str):
            yield description
        for value in schema.values():
            yield from _descriptions(value)
    elif isinstance(schema, (list, tuple)):
        for item in schema:
            yield from _descriptions(item)


def _escaped(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)[1:-1]


__all__ = ["ANY_ERROR", "ConformanceReport", "SCENARIOS", "ScenarioResult", "check_adapter"]
