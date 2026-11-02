# MCP Servers

`> mcp:` attaches one Model Context Protocol server to the active agent state.
The selected adapter translates the normalized specification into its MCP
client/toolset surface.

MCP supplies tools, not a new model or a child-agent conversation. A connected
server's tools still need an adapter that can expose them and an approval
policy that permits their effects. The commands and `example.com` URLs below
illustrate configuration shapes; they are not bundled Kedi services.

## Stdio Transport

```kedi
> mcp:
    transport: stdio
    command: uv
    args: `["run", "project-mcp"]`
    env: `{"LOG_LEVEL": "WARNING"}`
```

`stdio` requires `command`. `args` must be a list or tuple of strings; `env`
must be a dictionary whose keys and values are strings.

When `command` is present and `transport` is omitted, Kedi infers `stdio`:

```kedi
> mcp:
    command: project-mcp
```

The process inherits adapter/runtime environment according to that adapter;
the directive's `env` supplies explicit server variables.

## Streamable HTTP

````kedi
```
import os
token = os.environ["PROJECT_MCP_TOKEN"]
```

> mcp:
    transport: streamable-http
    url: https://tools.example.com/mcp
    headers: `{"Authorization": "Bearer " + token}`
````

`http` is an alias for `streamable-http`. Both normalize to the same transport
contract and require `url`. `headers` must be a string dictionary.

Do not hard-code production tokens in `.kedi` source. Read them from a trusted
runtime environment and avoid rendering them into prompts or logs.

## SSE

```kedi
> mcp:
    transport: sse
    url: https://tools.example.com/events
    headers: `{"X-Tenant": "example-team"}`
```

SSE also requires `url` and supports string headers. It remains a separate
transport from streamable HTTP.

## Dynamic Fields

String fields can be plain Kedi strings or Python expressions:

````kedi
```
import os
```

> mcp:
    transport: `os.getenv("MCP_TRANSPORT", "stdio")`
    command: `os.getenv("MCP_COMMAND", "project-mcp")`
    url: `os.getenv("MCP_URL")`
````

Use dynamic transport only when deployment configuration truly selects among
valid shapes. Kedi validates the evaluated specification: stdio cannot omit a
command, HTTP/SSE cannot omit a URL, and collection element types must be
strings.

Transport normalization removes irrelevant fields. A stdio spec does not carry
URL/headers; a remote spec does not carry command/args/env.

## Scoping and Profiles

MCP follows agent-state scope:

- a top-level server is captured by following procedures;
- a profile member is attached when that profile is applied;
- a procedure-body directive affects following calls in that invocation;
- leaving an inner scope restores the outer server list.

MCP server lists append in declaration/merge order. Applying another profile
does not erase existing servers unless a separate API creates a new outer
configuration.

## Adapter Support

Pydantic and LangChain map specs through their MCP integrations. DSPy currently
supports the stdio path using `dspy.Tool.from_mcp_tool` and `ReAct.acall`.
Harness and provider support varies.

The LSP emits capability diagnostics when the selected adapter cannot attach
MCP tools. Treat that as an unmet profile contract. Kedi does not emulate an MCP
server by inserting tool descriptions into prompt text.

## Failure Behavior

Invalid transport names, wrong collection types, missing required fields,
connection failures, protocol errors, and missing optional adapter packages
surface explicitly. Server lifecycle is adapter-owned; consult the adapter page
for startup, cleanup, and concurrency behavior.

## Permission Boundary

Do not assume a remote server's self-description makes its tools harmless.
Kedi's local MCP integration uses its approval boundary; the optional
[tool reason](tool-reasons.md) is projected locally and removed before the
server call. CodeMode can include MCP only where the adapter can materialize
and intercept the local tool path. Provider-hosted MCP that bypasses this path
has a different contract; see [CodeMode MCP](codemode.md#mcp-tools).

Keep server credentials in configuration, not in the task text. Tool output
from a server is data, not authorization to enable additional capabilities.
