# Capability Definitions

`kedi.Capability` packages tools, guidance, and owned tool policies for reuse in
Kedi programs. It is a contribution to an agent, not an agent constructor or a
replacement for a profile.

!!! note "Unreleased"
    This page describes the next Kedi release. Pydantic AI and LangChain support
    eager composition; Pydantic AI also supports deferred loading.

## Define and Apply

Put the descriptor in the query environment, then apply it with `> use:`:

```python
import kedi


def read_revision() -> int:
    """Read the source revision from this example's local release record."""
    return 7


reader = kedi.Capability(
    name="reader",
    identity="example.release.reader",
    instructions="Read the current revision.",
    tools=(read_revision,),
    approval=kedi.ApprovalPolicy.allow(),
)


@kedi.query(env={"reader": reader}, approval="deny")
def current_revision() -> int:
    """kedi
    > use: reader
    >> The current source revision is [revision: int].
    = `revision`
    """
    ...
```

Configure a supporting adapter and model through the normal
[configuration API](configuration-and-context.md) before calling
`current_revision()`. The provider dependencies and credentials remain the
application's responsibility.

The capability owns `read_revision`. Its local allow policy does not change the
query's deny policy for unrelated tools. The capture is validated as an integer,
and the backtick return preserves that native type.

Neither constructing `reader` nor executing `> use: reader` calls its tool.
Default eager use makes the contribution available to subsequent model calls.

## Constructor Contract

All arguments are keyword-only:

| Argument | Contract |
| --- | --- |
| `name` | Required identifier used in diagnostics and discovery. |
| `identity` | Optional stable qualified identity; defaults to `python:<name>`. Use a package-qualified value for published bundles. |
| `description` | Optional discovery/catalog description. |
| `instructions` | Optional additive user guidance. |
| `tools` | Sequence of typed callables or `ToolSpec` objects. Callable metadata and retries use normal Kedi tool conversion. Duplicate exposed names are rejected. |
| `mcp_servers` | Sequence of `McpServerSpec` definitions. Transport configuration stays separate from model instructions. |
| `capabilities` | Nested Kedi descriptors, composed in order after this descriptor's own guidance. |
| `subagents` | Mapping from child names to existing `ProfileDefinition` objects; names must match the definitions. This is an advanced binding surface, not a model override. |
| `approval` | `ApprovalPolicy` or `None`, scoped to owned tools. Unlike query configuration, this constructor does not accept string policy names. |
| `hooks` | `HookSettings` or event-handler mapping for owned tool lifecycle events only. |
| `requires` | Sequence of canonical adapter requirements. Invalid names fail at construction. |
| `defer` | A boolean, default `False`; no truthiness conversion from strings or numbers. |

Callable names and descriptions honor `@kedi.tool` metadata. A `ToolSpec` can
add an explicit tool policy, which takes precedence over the capability policy.
Capability policies do not bypass independent mandatory enforcement.

Host model, effort, settings, history, budgets, artifacts, and tool-reason
configuration are not constructor parameters. Use an agent/profile or a separate
child invocation for those concerns. `user_prompt_submit` and disabling inherited
hooks are not capability operations.

## Composition and Reset

The [language capability guide](../agentic-engineering/capabilities.md) shows the
same semantics with Kedi definitions, profile composition, and exported tools.

Capability instructions append to the current user guidance. Applying the same
capability again does not duplicate its guidance or tools. An explicit
`> instructions:` replaces all earlier user guidance in that scope, including
capability contributions, even when the replacement text equals the old base.

The reset does not unregister tools or discard their policy. Explicit reuse
restores the capability guidance once. Runtime-owned template, tool, and artifact
contracts are separate and survive user-instruction resets.

Profiles retain their replacement semantics. A procedure retains its
definition-site configuration, and a local instruction override does not change
the outer agent. See [Profiles](../agentic-engineering/profiles.md) and
[Agent Scope](../agentic-engineering/scoping-and-capabilities.md).

## Deferred Loading

`defer=True` requires the adapter's `deferred_tools` support. It registers a
discoverable capability whose tools and guidance activate through native loading;
it does not create a hidden subagent or switch models. The initial mapping targets
Pydantic AI. LangChain deferred use is rejected before inference rather than
silently becoming eager.

Native loading may deliver guidance in a tool result. A later instruction reset
does not erase that historical message. Deferral also does not promise delayed
MCP connection establishment. Nested descriptors retain independent deferred
boundaries: after loading the parent, its deferred children become discoverable
and require separate loads. A child shared by multiple parents is registered
once; any loaded parent or explicit direct use makes it discoverable.

Descriptors may contribute `subagents` while deferred. Their targets become
available only after native activation, including shared delegation, workflow,
background, continuation, and lifecycle controls. CodeMode schema discovery does
not reveal inactive targets. See [nested capability and subagent examples](../agentic-engineering/capabilities.md#nested-capabilities).

## Reuse and Native Integration

Definitions snapshot input collections and mutable schema/MCP data while
preserving executable tool identity. Each lowering receives its own data copies;
activation and suppression state belong to a run, not to the shared descriptor.
Do not mutate nested descriptor data as an alternative to defining a new bundle.

Imported Kedi profiles retain privately referenced Python capabilities, including
across re-exports. A same-named object in the caller does not replace that private
dependency.

`as_profile()` returns a fresh additive `AgentProfile` patch for low-level runtime
integration. It performs no inference or activation. It is not a shortcut that
registers tools on a standalone adapter; normal runtime tool binding is still
required.

Native Pydantic AI capability objects are not portable Kedi descriptors. The
native `capabilities=` option passed through `kedi.configure` or `kedi.context`
still belongs to Pydantic AI. `@kedi.query` does not gain a `capabilities=` option;
use its existing `env` plus `> use:` for Kedi descriptors.
