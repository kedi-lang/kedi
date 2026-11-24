"""A stdio MCP server whose tools record their side effects in the log file given as argv[1]."""

import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

LOG = Path(sys.argv[1])
server = FastMCP("wiki")


@server.tool()
def delete_page(page: str) -> str:
    """Delete a wiki page."""
    with LOG.open("a", encoding="utf-8") as log:
        log.write(f"deleted {page}\n")
    return f"deleted {page}"


@server.tool()
def broken_page(page: str) -> str:
    """Fail without a side effect."""
    raise ValueError(f"cannot open {page}")


server.run()
