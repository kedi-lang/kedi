"""A stdio MCP server for the JSON-RPC bridge tests."""

from mcp.server.fastmcp import FastMCP

server = FastMCP("kedi-bridge-tests")


@server.tool()
def shout(text: str) -> str:
    """Return the text in capitals."""

    return text.upper()


@server.tool()
def dump(lines: int) -> str:
    """Return a long log."""

    return "\n".join(f"log line {number}" for number in range(lines))


@server.tool()
def broken() -> str:
    """Always fails."""

    raise RuntimeError("the disk is full")


if __name__ == "__main__":
    server.run("stdio")
