"""``kedi-lsp`` server entry point — a ``pygls`` LanguageServer.

Wires textDocument lifecycle events to the feature functions in
:mod:`kedi.lsp.features`. Designed to be small and inspectable — see
``compute_diagnostics`` / ``compute_hover`` / ``compute_definition``
for the actual logic.
"""

from __future__ import annotations

import logging
from typing import Optional
from urllib.parse import urlparse, unquote

from lsprotocol import types as lsp
from pygls.lsp.server import LanguageServer

from . import features


_LOGGER = logging.getLogger(__name__)


def _uri_to_path(uri: str) -> Optional[str]:
    """Convert a file:// URI to a local path; return None for non-file URIs."""
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return None
    return unquote(parsed.path)


def make_server() -> LanguageServer:
    """Build the configured ``LanguageServer`` for Kedi."""
    server = LanguageServer(name="kedi-lsp", version="0.1.0")

    @server.feature(lsp.TEXT_DOCUMENT_DID_OPEN)
    def _did_open(params: lsp.DidOpenTextDocumentParams) -> None:
        _publish_diagnostics(server, params.text_document.uri)

    @server.feature(lsp.TEXT_DOCUMENT_DID_CHANGE)
    def _did_change(params: lsp.DidChangeTextDocumentParams) -> None:
        _publish_diagnostics(server, params.text_document.uri)

    @server.feature(lsp.TEXT_DOCUMENT_DID_CLOSE)
    def _did_close(params: lsp.DidCloseTextDocumentParams) -> None:
        # Clear diagnostics when the document is closed.
        server.text_document_publish_diagnostics(
            lsp.PublishDiagnosticsParams(uri=params.text_document.uri, diagnostics=[])
        )

    @server.feature(lsp.TEXT_DOCUMENT_HOVER)
    def _hover(params: lsp.HoverParams) -> Optional[lsp.Hover]:
        doc = server.workspace.get_text_document(params.text_document.uri)
        return features.compute_hover(
            source=doc.source,
            source_path=_uri_to_path(params.text_document.uri),
            pos=params.position,
        )

    @server.feature(lsp.TEXT_DOCUMENT_DEFINITION)
    def _definition(params: lsp.DefinitionParams) -> Optional[lsp.Location]:
        doc = server.workspace.get_text_document(params.text_document.uri)
        return features.compute_definition(
            source=doc.source,
            uri=params.text_document.uri,
            source_path=_uri_to_path(params.text_document.uri),
            pos=params.position,
        )

    return server


def _publish_diagnostics(server: LanguageServer, uri: str) -> None:
    """Re-parse the document at ``uri`` and publish its diagnostics."""
    doc = server.workspace.get_text_document(uri)
    diagnostics = features.compute_diagnostics(
        source=doc.source, source_path=_uri_to_path(uri)
    )
    server.text_document_publish_diagnostics(
        lsp.PublishDiagnosticsParams(uri=uri, diagnostics=diagnostics)
    )


def main() -> None:
    """Console-script entry point for ``kedi-lsp``."""
    logging.basicConfig(level=logging.INFO)
    server = make_server()
    server.start_io()


if __name__ == "__main__":
    main()
