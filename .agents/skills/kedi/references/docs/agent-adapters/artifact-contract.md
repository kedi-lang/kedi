# Artifact Integration Contract

An adapter must preserve the distinction between application values and
model-visible artifact references. Ordinary Kedi/Python callers still receive
native typed results; a model receives bounded metadata and retrieval tools.

## Request and Tool Boundaries

Use Kedi's runtime-owned artifact session and scoped policy. Publish management
tools and instructions before the first request, not only when the first large
payload appears. Apply admission after tool execution and return validation;
do not run the source tool again to retrieve an admitted value.

Keep tool-call IDs and their result messages paired. Bounded reads are new tool
results, not edits to old messages. Release and expiry change storage access,
not conversation history. A failed admission must not fall back to inserting
the full payload into the request.

## Lifecycle and Errors

Do not create a new manager per continuation turn or share references between
unrelated sessions. Close only resources owned by the run/session. Correctable
retrieval argument errors may be returned through the adapter's tool-error path;
configuration and unexpected storage failures must not be disguised as empty
content. Preserve cancellation and existing retry budgets.

Artifact support does not imply native continuation, compaction, or streaming
support. Declare only implemented capabilities and test native returns, bounded
model substitutions, invalid references, release, expiry, and cleanup. See
[Artifact Safety](../runtime/artifact-safety.md) and
[Custom Adapters](custom-adapters.md).
