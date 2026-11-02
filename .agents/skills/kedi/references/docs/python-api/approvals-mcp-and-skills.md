# Approvals, MCP, and Skills


## `@kedi.approval`

Decorate a handler to install it as the current Python API configuration's
default dynamic policy:

```python
import kedi


@kedi.approval
def review_call(request: kedi.ApprovalRequest) -> kedi.ApprovalDecision:
    if request.tool_name == "write_report":
        return kedi.ApprovalDecision.edit(
            {**request.arguments, "path": "reports/latest.md"},
            reason="confine writes to the report path",
        )
    return kedi.ApprovalDecision.deny(reason="tool is outside this workflow")
```

The decorator returns the original handler, so it can also be passed explicitly
to `context`, `query`, or `bind`.


## Approval Policies

The `approval=` parameter accepts:

```python
kedi.configure(approval="allow")
kedi.configure(approval="deny")
kedi.configure(approval=kedi.ApprovalPolicy.allow())
kedi.configure(approval=kedi.ApprovalPolicy.dynamic(review_call))
```

Only `"allow"` and `"deny"` are valid strings. With no policy, mutating and
sensitive tools are denied. Read-only tools are always allowed, even under a
static deny policy.


## Dynamic Approval Handlers

Handlers may be synchronous or asynchronous:

```python
async def approve_from_service(
    request: kedi.ApprovalRequest,
) -> kedi.ApprovalDecision:
    allowed = await policy_service.check(request.tool_name, request.arguments)
    if allowed:
        return kedi.ApprovalDecision.allow(reason="approved by policy service")
    return kedi.ApprovalDecision.deny(reason="rejected by policy service")
```

An async handler requires an async-capable tool path when an event loop is
already running. Returning any object other than `ApprovalDecision` is an
invalid decision error.

On the async approval path, synchronous handlers run in an owned worker thread
with the current configuration and trace context. A Kedi approval procedure may
therefore make a nested model call without blocking the model client's event
loop. Async handlers remain on that loop. Cancellation interrupts nested Kedi
model work and waits for the handler worker to unwind before returning; arbitrary
blocking Python code cannot be forcibly terminated or have its effects rolled back.


## Approval Requests and Decisions

`ApprovalRequest` is frozen and contains:

| Field | Meaning |
| --- | --- |
| `tool_name` | Registered tool name |
| `arguments` | Deep-copied, read-only argument mapping |
| `risk` | `read_only`, `mutating`, or `sensitive` |
| `adapter_shortname` | Active adapter when known |
| `description` | Tool description when known |
| `metadata` | Optional deep-copied adapter/runtime metadata |
| `reason` | Model-supplied explanation when tool reasons are enabled; otherwise `None` |
| `tool_reason_enabled` | Whether the calling tool surface has optional reasons enabled |

Return one explicit decision:

```python
kedi.ApprovalDecision.allow(reason="safe for this task")
kedi.ApprovalDecision.deny(reason="outside allowed scope")
kedi.ApprovalDecision.edit(
    {"path": "reports/output.md", "content": "approved content"},
    reason="rewrote destination",
)
```

Only `edit` may contain replacement arguments. Edited arguments are
revalidated and reclassified before execution; an edit is not a bypass around
tool schemas or risk policy.


## `McpServerSpec`

See [`McpServerSpec`](mcp.md).

## Configure MCP Servers

See [Configure MCP Servers](mcp.md).

## Enable Skills

See [Enable Skills](skills.md).

## Per-Callable Overrides

```python
@kedi.query(
    approval=review_call,
    mcp_servers=[docs_mcp],
    skills=True,
)
def investigate(question: str) -> str:
    """kedi
    >> The evidence-based answer to <question> is [answer: str].
    = `answer`
    """
    ...
```

An explicit per-callable approval overrides inherited approval for that
callable. `skills=False` can disable inherited skills. MCP servers append
because their merge model is additive.


## Scope Precedence

For profile values, effective precedence is:

1. defaults from `configure()`;
2. active nested `context()` scopes;
3. `query()` or `bind()` decorator overrides;
4. lexical DSL directives and selected named profiles.

Later explicit values win. Settings maps merge by key, tool names merge with
later definitions taking ownership, and MCP server lists append. Approval and
skills replace when explicitly provided.
