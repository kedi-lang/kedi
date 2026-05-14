"""``kedi-lsp`` server entry point.

Thin protocol wrapper over the functions in :mod:`kedi.lsp.features`.
Wires every supported LSP feature plus the custom ``kedi/pythonRanges``
request used by the VS Code extension's embedded-Python forwarding.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional
from urllib.parse import unquote, urlparse

from lsprotocol import types as lsp
from pygls.lsp.server import LanguageServer

from . import features


_LOGGER = logging.getLogger(__name__)


def _uri_to_path(uri: str) -> Optional[str]:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return None
    return unquote(parsed.path)


def make_server() -> LanguageServer:
    server = LanguageServer(name="kedi-lsp", version="0.1.0")

    # ----- Document lifecycle: re-parse + publish diagnostics -----

    @server.feature(lsp.TEXT_DOCUMENT_DID_OPEN)
    def _did_open(params: lsp.DidOpenTextDocumentParams) -> None:
        _publish_diagnostics(server, params.text_document.uri)

    @server.feature(lsp.TEXT_DOCUMENT_DID_CHANGE)
    def _did_change(params: lsp.DidChangeTextDocumentParams) -> None:
        _publish_diagnostics(server, params.text_document.uri)

    @server.feature(lsp.TEXT_DOCUMENT_DID_CLOSE)
    def _did_close(params: lsp.DidCloseTextDocumentParams) -> None:
        server.text_document_publish_diagnostics(
            lsp.PublishDiagnosticsParams(
                uri=params.text_document.uri, diagnostics=[]
            )
        )

    # ----- Hover -----

    @server.feature(lsp.TEXT_DOCUMENT_HOVER)
    def _hover(params: lsp.HoverParams) -> Optional[lsp.Hover]:
        doc = server.workspace.get_text_document(params.text_document.uri)
        return features.compute_hover(
            source=doc.source,
            source_path=_uri_to_path(params.text_document.uri),
            pos=params.position,
        )

    # ----- Definition -----

    @server.feature(lsp.TEXT_DOCUMENT_DEFINITION)
    def _definition(params: lsp.DefinitionParams) -> Optional[lsp.Location]:
        doc = server.workspace.get_text_document(params.text_document.uri)
        return features.compute_definition(
            source=doc.source,
            uri=params.text_document.uri,
            source_path=_uri_to_path(params.text_document.uri),
            pos=params.position,
        )

    # ----- References -----

    @server.feature(lsp.TEXT_DOCUMENT_REFERENCES)
    def _references(params: lsp.ReferenceParams) -> List[lsp.Location]:
        doc = server.workspace.get_text_document(params.text_document.uri)
        return features.compute_references(
            source=doc.source,
            uri=params.text_document.uri,
            source_path=_uri_to_path(params.text_document.uri),
            pos=params.position,
            include_declaration=params.context.include_declaration,
        )

    # ----- Document symbols -----

    @server.feature(lsp.TEXT_DOCUMENT_DOCUMENT_SYMBOL)
    def _document_symbol(
        params: lsp.DocumentSymbolParams,
    ) -> List[lsp.DocumentSymbol]:
        doc = server.workspace.get_text_document(params.text_document.uri)
        return features.compute_document_symbols(
            source=doc.source,
            source_path=_uri_to_path(params.text_document.uri),
        )

    # ----- Rename (prepare + apply) -----

    @server.feature(lsp.TEXT_DOCUMENT_PREPARE_RENAME)
    def _prepare_rename(params: lsp.PrepareRenameParams) -> Optional[lsp.Range]:
        doc = server.workspace.get_text_document(params.text_document.uri)
        return features.compute_prepare_rename(
            source=doc.source,
            source_path=_uri_to_path(params.text_document.uri),
            pos=params.position,
        )

    @server.feature(lsp.TEXT_DOCUMENT_RENAME)
    def _rename(params: lsp.RenameParams) -> Optional[lsp.WorkspaceEdit]:
        doc = server.workspace.get_text_document(params.text_document.uri)
        return features.compute_rename(
            source=doc.source,
            uri=params.text_document.uri,
            source_path=_uri_to_path(params.text_document.uri),
            pos=params.position,
            new_name=params.new_name,
        )

    # ----- Formatting -----

    @server.feature(lsp.TEXT_DOCUMENT_FORMATTING)
    def _formatting(params: lsp.DocumentFormattingParams) -> List[lsp.TextEdit]:
        doc = server.workspace.get_text_document(params.text_document.uri)
        return features.compute_formatting(doc.source)

    # ----- Signature help -----

    @server.feature(
        lsp.TEXT_DOCUMENT_SIGNATURE_HELP,
        lsp.SignatureHelpOptions(trigger_characters=["(", ","]),
    )
    def _signature_help(
        params: lsp.SignatureHelpParams,
    ) -> Optional[lsp.SignatureHelp]:
        doc = server.workspace.get_text_document(params.text_document.uri)
        return features.compute_signature_help(
            source=doc.source,
            source_path=_uri_to_path(params.text_document.uri),
            pos=params.position,
        )

    # ----- Inlay hints -----

    @server.feature(lsp.TEXT_DOCUMENT_INLAY_HINT)
    def _inlay_hint(params: lsp.InlayHintParams) -> List[lsp.InlayHint]:
        doc = server.workspace.get_text_document(params.text_document.uri)
        return features.compute_inlay_hints(
            source=doc.source,
            source_path=_uri_to_path(params.text_document.uri),
            rng=params.range,
        )

    # ----- Semantic tokens (full document) -----

    @server.feature(
        lsp.TEXT_DOCUMENT_SEMANTIC_TOKENS_FULL,
        lsp.SemanticTokensRegistrationOptions(
            legend=lsp.SemanticTokensLegend(
                token_types=features.SEMANTIC_TOKEN_TYPES,
                token_modifiers=features.SEMANTIC_TOKEN_MODIFIERS,
            ),
            full=True,
            range=False,
            document_selector=None,
        ),
    )
    def _semantic_tokens_full(
        params: lsp.SemanticTokensParams,
    ) -> lsp.SemanticTokens:
        doc = server.workspace.get_text_document(params.text_document.uri)
        return features.compute_semantic_tokens_full(
            source=doc.source,
            source_path=_uri_to_path(params.text_document.uri),
        )

    # ----- Custom: kedi/pythonRanges -----

    @server.feature("kedi/pythonRanges")
    def _python_ranges(params: Any) -> dict:
        """Return all Python regions in a Kedi document.

        ``params`` is a dict-like with ``textDocument.uri``. We return
        ``{"uri": ..., "ranges": [{kind, range, text}, ...]}``.
        """
        # pygls/lsprotocol may hand us either a raw dict or a parsed
        # structure depending on the registration; accept both.
        uri = None
        if isinstance(params, dict):
            td = params.get("textDocument", {})
            uri = td.get("uri") if isinstance(td, dict) else None
        else:
            td = getattr(params, "text_document", None) or getattr(
                params, "textDocument", None
            )
            if td is not None:
                uri = getattr(td, "uri", None)
        if uri is None:
            return {"uri": "", "ranges": []}
        doc = server.workspace.get_text_document(uri)
        ranges = features.compute_python_ranges(doc.source)
        return {"uri": uri, "ranges": ranges}

    return server


def _publish_diagnostics(server: LanguageServer, uri: str) -> None:
    doc = server.workspace.get_text_document(uri)
    diagnostics = features.compute_diagnostics(
        source=doc.source, source_path=_uri_to_path(uri)
    )
    server.text_document_publish_diagnostics(
        lsp.PublishDiagnosticsParams(uri=uri, diagnostics=diagnostics)
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    server = make_server()
    server.start_io()


if __name__ == "__main__":
    main()
