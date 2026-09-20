#!/usr/bin/env python3
"""Serve generated Kedi registry JSON from a local directory."""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory containing the generated v1 tree")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not (root / "v1").is_dir():
        parser.error(f"{root} must contain a v1 directory")

    handler = partial(SimpleHTTPRequestHandler, directory=str(root))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Kedi registry: http://{args.host}:{server.server_port}", flush=True)
    print(
        f"Use KEDI_REGISTRY_URL=http://{args.host}:{server.server_port}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
