# MCP from Python

Configure external tool servers through typed specifications. The following construction snippets require your installed server or running endpoint; they do not start a bundled Kedi service.

## `McpServerSpec`

Import the typed specification from `kedi`:

```python
from kedi import McpServerSpec
```

For stdio:

```python
filesystem_mcp = McpServerSpec(
    transport="stdio",
    command="npx",
    args=("-y", "@modelcontextprotocol/server-filesystem", "/workspace"),
    env={"LOG_LEVEL": "warning"},
).normalized()
```

For streamable HTTP:

```python
docs_mcp = McpServerSpec(
    transport="http",
    url="http://127.0.0.1:8000/mcp",
    headers={"Authorization": "Bearer token"},
).normalized()
```

For SSE, use `transport="sse"` and `url=...`. The Python dataclass accepts the
canonical values `stdio`, `http`, and `sse`; DSL spelling
`streamable-http` normalizes to `http`.

Calling `.normalized()` is recommended when constructing specs directly. It
validates that stdio has a command, remote transports have a URL, and clears
fields that do not apply to the selected transport.


## Configure MCP Servers

Pass a sequence at any configuration scope:

```python
kedi.configure(mcp_servers=[docs_mcp])

with kedi.context(mcp_servers=[filesystem_mcp]):
    result = investigate("...")
```

Context and per-callable MCP servers append to inherited servers; they do not
replace or deduplicate them. Adapter capability differs: unsupported adapters
must fail or report the capability limitation rather than silently emulate an
MCP server.

MCP servers are external code or services. Stdio commands run with the host
process's authority, and remote headers may carry credentials.
