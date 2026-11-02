# Tools and Approvals

This program exposes a read-only lookup and a mutating report writer. A dynamic
approval handler redirects writes into a controlled directory.

## Complete Program

````kedi
```
from pathlib import Path

import kedi
from kedi import ApprovalDecision


@kedi.tool(risk="read_only")
def lookup_incident(incident_id: int) -> dict[str, object]:
    """Return one incident from the local demonstration index."""
    return {
        "id": incident_id,
        "title": "Checkout latency",
        "severity": "high",
    }


@kedi.tool(risk="mutating")
def write_report(path: str, content: str) -> str:
    """Write a UTF-8 incident report and return its final path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    return str(destination)


def review_tool(request):
    if request.tool_name != "write_report":
        return ApprovalDecision.allow()

    return ApprovalDecision.edit(
        {**request.arguments, "path": "reports/incident-42.md"},
        reason="use the application's fixed report destination",
    )
```

> adapter: pydantic
> model: openai:gpt-6-luna
> instructions: Fetch incident 42, summarize it, and save reports/incident-42.md.
> use:
    lookup_incident
    write_report
> approval: `review_tool`

>> The incident report was saved by write_report at [saved_path: str].
= <saved_path>
````

The Python callables are converted to model-facing tool schemas from their
names, signatures, annotations, and docstrings. Tool calls use keyword
arguments.

## Why the Approval Handler Edits

The model may request `/tmp/report.md` or `../report.md`. The handler receives
an immutable, deep-copied `ApprovalRequest` and returns a complete replacement
argument mapping. Kedi then:

1. recalculates argument-sensitive risk;
2. validates edited arguments against the tool signature;
3. applies remaining approval ceilings;
4. invokes the tool only if the edited call is permitted.

Editing is safer here than trusting a path instruction in the prompt. Prompts
guide the model; approval policies enforce the call boundary.

This example assumes an application-owned working directory without hostile
symlinks. Replacing a path argument is not an OS sandbox: Python I/O and a
symlink at the destination remain outside that guarantee. The writer can
overwrite the fixed report. Inspect the actual file, not just `saved_path`,
before treating the workflow as successful.

## Procedure Tools

A Kedi procedure can be exposed without Python:

The following fragment assumes a `release_index` mapping supplied by a prelude
or the embedding application.

```kedi
@release_notes(version: str) -> str:
  ###
  Return release notes for one exact version.
  ###
  = `release_index[version]`

> use: release_notes
```

Procedure tools default to `mutating`, because Kedi cannot infer side effects
from a procedure body. Use Python `@kedi.tool(risk="read_only")` when a precise
risk declaration matters.

## Default and Static Policies

Without an explicit runtime policy, read-only tools are allowed and mutating or
sensitive tools are denied. The interactive CLI asks for risky calls; an
unanswered non-interactive prompt is denied.

```kedi
> approval: deny
```

`deny` still allows read-only calls. `allow` permits registered mutating and
sensitive calls in that scope and should be reserved for an already trusted
tool surface. There is no `> approval: edit`; argument edits require a handler.

## Sensitive Files

Bundled filesystem tools refuse `.env` and `.env.*` through ordinary reads.
Their explicit `secret_files=True` path elevates the call to `sensitive`; it
does not grant authorization by itself. Approval does not intercept arbitrary
Python file I/O, so embedded Python and imported packages remain trusted code.

## Continue With a Complete Workflow

[Reviewed Evidence](../agentic-engineering/reviewed-evidence.md) adds typed child
delegation, a separately checked write destination, hooks, and Python embedding.
Its offline tests inspect actual reads and writes and verify that a denied
write has no side effect. For opt-in tool-call explanations, see
[Tool Reasons](../agentic-engineering/tool-reasons.md).
