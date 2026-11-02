# Jev Tool Selection

Tool selection is a finite decision. Argument generation and execution remain separate operations with their own validation and approval boundaries.

## Explicit Tool Routing

Both framework integrations support opt-in selection of registered tools. Jev may
select a zero-argument tool for framework execution. A tool requiring arguments
produces a `ToolCallProposed` exception for an explicit argument-producing handler;
it does not invent arguments or silently delegate to an LLM.

Tool selection has its own `typesafe_tool_call_threshold` (default 0.6, inclusive).
It is not the claim threshold. Selection is not authorization: retain Kedi hooks,
approval policy, and framework execution limits. Review or fallback behavior must
be explicitly written by the application.
