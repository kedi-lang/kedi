# Approvals and Sensitive Operations

Approvals mediate individual tool calls after arguments are produced and before
the tool executes. They are not model-output validation and do not sandbox
ordinary embedded Python.

## Risk Classes

Kedi uses three ordered risk classes:

- `read_only`: observes non-sensitive state;
- `mutating`: changes state or performs an effect;
- `sensitive`: accesses secrets or otherwise requires stronger review.

Custom Kedi and Python tools default to `mutating`. A Python `@kedi.tool` can
declare a different risk:

```python
import kedi


@kedi.tool(risk="read_only")
def list_public_files() -> list[str]:
    return ["README.md", "CHANGELOG.md"]
```

An argument-aware risk resolver may elevate a call, but cannot downgrade a
tool's static risk.

## Default Policy

Without an explicit policy, `read_only` calls are allowed and `mutating` or
`sensitive` calls are denied. This is the Python/runtime fail-closed default.

The interactive `kedi` CLI installs a prompt policy for risky calls. It offers
Allow once, Deny, and Allow always for this run. “Always” is scoped to the
current process, tool, and risk level; it is never persisted.

In non-interactive execution, an unanswered prompt is denied.

## Static Policies

```kedi
> approval: allow
```

`allow` permits registered mutating and sensitive tools in that scope. Use it
only when the complete tool surface and arguments are already trusted.

```kedi
> approval: deny
```

`deny` refuses risky calls. Read-only calls remain automatically allowed.
`edit` is not a static mode.

## Dynamic Policy

Define a sync or async handler in Python and select it:

````kedi
```
from pathlib import Path
from kedi import ApprovalDecision

def review_tool(request):
    if request.tool_name != "write_report":
        return ApprovalDecision.deny(reason="tool is outside this workflow")

    path = Path(request.arguments["path"])
    safe_path = Path("reports") / path.name
    return ApprovalDecision.edit(
        {**request.arguments, "path": str(safe_path)},
        reason="redirected to the reports directory",
    )
```

> approval: `review_tool`
````

The handler must return `ApprovalDecision.allow()`,
`ApprovalDecision.deny()`, or `ApprovalDecision.edit(arguments)`. Returning a
string or arbitrary mapping is an error.

### LLM-backed approval helper

`llm_approval` requires enabled tool reasons. Its judgment is fallible and does
not replace a deterministic authorization boundary. See
[Tool Reasons and LLM Approval](tool-reasons.md#llm-backed-approval-helper).

### Optional Tool Reasons

Enable `tool_reason` in settings or the supported adapter constructor to collect
an optional justification for approval-applicable tools. Kedi removes the
synthetic argument before the real callable or MCP server receives the call.
See [configuration and failure rules](tool-reasons.md#optional-tool-reasons).

## Approval Request

The immutable request contains:

| Field | Meaning |
| --- | --- |
| `tool_name` | Registered tool name |
| `arguments` | Deep-copied, read-only argument mapping |
| `risk` | Effective risk after argument-aware elevation |
| `adapter_shortname` | Active adapter when known |
| `description` | Tool description when available |
| `metadata` | Deep-copied tool metadata when available |
| `reason` | Optional model-provided action justification, separate from tool arguments |
| `tool_reason_enabled` | Whether the calling adapter enabled justification collection |

The handler cannot mutate the request in place. It must return `edit` with a new
complete argument mapping.

## Edited Arguments

Edited arguments are reclassified and revalidated against the tool signature,
custom Kedi models, and schema before invocation. An edit that redirects a
normal path to `.env` can therefore become `sensitive` and face subsequent
policy checks. Invalid edited types fail before the tool runs.

Only an `edit` decision may carry replacement arguments. Allow and deny cannot
smuggle an edit payload.

## Sensitive Files

The bundled filesystem tools classify `.env` and `.env.*` as secrets. Normal
`read_text_file(path)` refuses them. The caller must set
`secret_files=True`, which elevates the call to `sensitive`, and approval must
then permit it.

Writes and patches targeting secret filenames are likewise elevated before
mutation. The opt-in flag requests review; it does not itself authorize access.

## Scope and Descendants

Approval is lexical and restored when a scope exits. A procedure directive
applies to its following tool calls; a profile carries its policy when used.

Subagents have their own policy, but an ancestor's approval ceiling remains in
force. A child cannot replace a parent deny with allow. Parent edits are
reclassified and checked by child policy. This makes delegation monotonically
restrictive rather than a route around approval.

## Limits of Approval

Approval wraps registered tool calls. It does not intercept arbitrary I/O in a
Python block, a package prelude, or third-party code. Treat embedded Python as
trusted executable code and use host isolation when that boundary matters.
