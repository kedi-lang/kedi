"""``kedi-lsp`` server entry point.

Thin protocol wrapper over the functions in :mod:`kedi.lsp.features`.
Wires every supported LSP feature plus the custom embedded-Python
requests used by editor extension forwarding.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, List, Optional, cast
from urllib.parse import unquote, urlparse

from lsprotocol import types as lsp
from pygls.lsp.server import LanguageServer

from .._version import __version__
from . import features, kedi_docstrings
from .python_virtual import compute_python_virtual_document

_LOGGER = logging.getLogger(__name__)


def _uri_to_path(uri: str) -> Optional[str]:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return None
    return unquote(parsed.path)


def _is_python_uri(uri: str) -> bool:
    path = _uri_to_path(uri)
    return path is not None and path.endswith(".py")


def make_server() -> LanguageServer:
    server = LanguageServer(name="kedi-lsp", version=__version__)
    diagnostic_settings = {"codemode_preload_warning_threshold": 5}

    @server.feature(lsp.INITIALIZE)
    def _initialize(params: lsp.InitializeParams) -> None:
        diagnostic_settings["codemode_preload_warning_threshold"] = (
            _codemode_preload_warning_threshold(params.initialization_options)
        )

    @server.feature(lsp.WORKSPACE_DID_CHANGE_CONFIGURATION)
    def _did_change_configuration(params: lsp.DidChangeConfigurationParams) -> None:
        diagnostic_settings["codemode_preload_warning_threshold"] = (
            _codemode_preload_warning_threshold(
                params.settings,
                default=diagnostic_settings["codemode_preload_warning_threshold"],
            )
        )
        for uri in tuple(server.workspace.text_documents):
            _publish_diagnostics(
                server,
                uri,
                codemode_preload_warning_threshold=diagnostic_settings[
                    "codemode_preload_warning_threshold"
                ],
            )

    # ----- Document lifecycle: re-parse + publish diagnostics -----

    @server.feature(lsp.TEXT_DOCUMENT_DID_OPEN)
    def _did_open(params: lsp.DidOpenTextDocumentParams) -> None:
        _publish_diagnostics(
            server,
            params.text_document.uri,
            codemode_preload_warning_threshold=diagnostic_settings[
                "codemode_preload_warning_threshold"
            ],
        )

    @server.feature(lsp.TEXT_DOCUMENT_DID_CHANGE)
    def _did_change(params: lsp.DidChangeTextDocumentParams) -> None:
        _publish_diagnostics(
            server,
            params.text_document.uri,
            codemode_preload_warning_threshold=diagnostic_settings[
                "codemode_preload_warning_threshold"
            ],
        )

    @server.feature(lsp.TEXT_DOCUMENT_DID_CLOSE)
    def _did_close(params: lsp.DidCloseTextDocumentParams) -> None:
        server.text_document_publish_diagnostics(
            lsp.PublishDiagnosticsParams(uri=params.text_document.uri, diagnostics=[])
        )

    # Some clients send watched-file notifications whenever *.kedi files
    # change. We do not currently need them because document changes are
    # driven by open/change/close, but registering a no-op handler keeps
    # pygls from logging noisy "unknown method" warnings.
    @server.feature("workspace/didChangeWatchedFiles")
    def _did_change_watched_files(params: lsp.DidChangeWatchedFilesParams) -> None:
        return None

    # ----- Hover -----

    @server.feature(lsp.TEXT_DOCUMENT_HOVER)
    def _hover(params: lsp.HoverParams) -> Optional[lsp.Hover]:
        doc = server.workspace.get_text_document(params.text_document.uri)
        if _is_python_uri(params.text_document.uri):
            return cast(
                Optional[lsp.Hover],
                kedi_docstrings.compute_kedi_docstring_hover(
                    doc.source,
                    _uri_to_path(params.text_document.uri),
                    params.position,
                ),
            )
        return features.compute_hover(
            source=doc.source,
            source_path=_uri_to_path(params.text_document.uri),
            pos=params.position,
        )

    # ----- Completion -----

    @server.feature(
        lsp.TEXT_DOCUMENT_COMPLETION,
        lsp.CompletionOptions(trigger_characters=[">", "<", "["]),
    )
    def _completion(params: lsp.CompletionParams) -> List[lsp.CompletionItem]:
        doc = server.workspace.get_text_document(params.text_document.uri)
        if _is_python_uri(params.text_document.uri):
            return []
        return features.compute_completion(
            source=doc.source,
            source_path=_uri_to_path(params.text_document.uri),
            pos=params.position,
            explicit=(
                params.context is None
                or params.context.trigger_kind == lsp.CompletionTriggerKind.Invoked
            ),
        )

    # ----- Definition -----

    @server.feature(lsp.TEXT_DOCUMENT_DEFINITION)
    def _definition(params: lsp.DefinitionParams) -> Optional[lsp.Location]:
        doc = server.workspace.get_text_document(params.text_document.uri)
        if _is_python_uri(params.text_document.uri):
            locations = kedi_docstrings.compute_kedi_docstring_definition(
                doc.source,
                params.text_document.uri,
                _uri_to_path(params.text_document.uri),
                params.position,
            )
            return cast(Optional[lsp.Location], locations[0] if locations else None)
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
        if _is_python_uri(params.text_document.uri):
            return cast(
                List[lsp.Location],
                kedi_docstrings.compute_kedi_docstring_references(
                    doc.source,
                    params.text_document.uri,
                    _uri_to_path(params.text_document.uri),
                    params.position,
                    include_declaration=params.context.include_declaration,
                ),
            )
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
        if _is_python_uri(params.text_document.uri):
            return []
        return features.compute_document_symbols(
            source=doc.source,
            source_path=_uri_to_path(params.text_document.uri),
        )

    # ----- Rename (prepare + apply) -----

    @server.feature(lsp.TEXT_DOCUMENT_PREPARE_RENAME)
    def _prepare_rename(params: lsp.PrepareRenameParams) -> Optional[lsp.Range]:
        doc = server.workspace.get_text_document(params.text_document.uri)
        if _is_python_uri(params.text_document.uri):
            return None
        return features.compute_prepare_rename(
            source=doc.source,
            source_path=_uri_to_path(params.text_document.uri),
            pos=params.position,
        )

    @server.feature(lsp.TEXT_DOCUMENT_RENAME)
    def _rename(params: lsp.RenameParams) -> Optional[lsp.WorkspaceEdit]:
        doc = server.workspace.get_text_document(params.text_document.uri)
        if _is_python_uri(params.text_document.uri):
            return None
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
        if _is_python_uri(params.text_document.uri):
            return []
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
        if _is_python_uri(params.text_document.uri):
            return None
        return features.compute_signature_help(
            source=doc.source,
            source_path=_uri_to_path(params.text_document.uri),
            pos=params.position,
        )

    # ----- Inlay hints -----

    @server.feature(lsp.TEXT_DOCUMENT_INLAY_HINT)
    def _inlay_hint(params: lsp.InlayHintParams) -> List[lsp.InlayHint]:
        doc = server.workspace.get_text_document(params.text_document.uri)
        if _is_python_uri(params.text_document.uri):
            return []
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
        if _is_python_uri(params.text_document.uri):
            return lsp.SemanticTokens(
                data=kedi_docstrings.compute_kedi_docstring_semantic_tokens(doc.source)["data"]
            )
        return features.compute_semantic_tokens_full(
            source=doc.source,
            source_path=_uri_to_path(params.text_document.uri),
        )

    # ----- Custom: embedded Python interop -----

    @server.feature("kedi/pythonRanges")
    def _python_ranges(params: Any) -> dict:
        """Return all Python regions in a Kedi document.

        ``params`` is a dict-like with ``textDocument.uri``. We return
        ``{"uri": ..., "ranges": [{kind, range, text}, ...]}``.
        """
        uri = _text_document_uri(params)
        if uri is None:
            return {"uri": "", "ranges": []}
        source = _request_source(server, uri, params)
        ranges = features.compute_python_ranges(source)
        return {"uri": uri, "ranges": ranges}

    @server.feature("kedi/pythonVirtualDocument")
    def _python_virtual_document(params: Any) -> dict:
        """Return a scope-aware virtual Python document for a Kedi file."""
        uri = _text_document_uri(params)
        if uri is None:
            return {
                "uri": "",
                "text": "",
                "ranges": [],
                "mappings": [],
                "symbols": [],
                "focusKey": None,
            }
        source = _request_source(server, uri, params)
        result = compute_python_virtual_document(
            source,
            source_path=_uri_to_path(uri),
            focus_line=_request_focus_line(params),
        )
        return {"uri": uri, **result}

    @server.feature("kedi/pythonCompletions")
    def _python_completions(params: Any) -> list[dict[str, Any]]:
        from kedi.lsp.python_virtual.completion import map_python_completions

        uri = _text_document_uri(params)
        if uri is None:
            return []
        get = (
            params.get
            if isinstance(params, dict)
            else lambda key, default: getattr(params, key, default)
        )
        return map_python_completions(
            _request_source(server, uri, params),
            _protocol_data(get("virtual", {})),
            _protocol_data(get("items", [])),
        )

    # ----- Custom: embedded Kedi-in-Python-docstring interop -----

    @server.feature("kedi/kediDocstringRanges")
    def _kedi_docstring_ranges(params: Any) -> dict:
        uri = _text_document_uri(params)
        if uri is None:
            return {"uri": "", "ranges": []}
        source = _request_source(server, uri, params)
        return {
            "uri": uri,
            "ranges": kedi_docstrings.compute_kedi_docstring_ranges(source),
        }

    @server.feature("kedi/kediDocstringDiagnostics")
    def _kedi_docstring_diagnostics(params: Any) -> dict:
        uri = _text_document_uri(params)
        if uri is None:
            return {"uri": "", "diagnostics": []}
        source = _request_source(server, uri, params)
        return {
            "uri": uri,
            "diagnostics": kedi_docstrings.compute_kedi_docstring_diagnostics(
                source,
                _uri_to_path(uri),
            ),
        }

    @server.feature("kedi/kediDocstringHover")
    def _kedi_docstring_hover(params: Any) -> dict | None:
        uri = _text_document_uri(params)
        position = _request_position(params)
        if uri is None or position is None:
            return None
        source = _request_source(server, uri, params)
        return kedi_docstrings.compute_kedi_docstring_hover(
            source,
            _uri_to_path(uri),
            position,
        )

    @server.feature("kedi/kediDocstringDefinition")
    def _kedi_docstring_definition(params: Any) -> dict:
        uri = _text_document_uri(params)
        position = _request_position(params)
        if uri is None or position is None:
            return {"uri": uri or "", "locations": []}
        source = _request_source(server, uri, params)
        return {
            "uri": uri,
            "locations": kedi_docstrings.compute_kedi_docstring_definition(
                source,
                uri,
                _uri_to_path(uri),
                position,
            ),
        }

    @server.feature("kedi/kediDocstringReferences")
    def _kedi_docstring_references(params: Any) -> dict:
        uri = _text_document_uri(params)
        position = _request_position(params)
        if uri is None or position is None:
            return {"uri": uri or "", "locations": []}
        source = _request_source(server, uri, params)
        return {
            "uri": uri,
            "locations": kedi_docstrings.compute_kedi_docstring_references(
                source,
                uri,
                _uri_to_path(uri),
                position,
                include_declaration=_request_include_declaration(params),
            ),
        }

    @server.feature("kedi/kediDocstringSemanticTokens")
    def _kedi_docstring_semantic_tokens(params: Any) -> dict:
        uri = _text_document_uri(params)
        if uri is None:
            return {"uri": "", "data": []}
        source = _request_source(server, uri, params)
        return {
            "uri": uri,
            **kedi_docstrings.compute_kedi_docstring_semantic_tokens(source),
        }

    return server


def _text_document_uri(params: Any) -> Optional[str]:
    # pygls/lsprotocol may hand us either a raw dict or a parsed
    # structure depending on the registration; accept both.
    if isinstance(params, dict):
        td = params.get("textDocument", {})
        return td.get("uri") if isinstance(td, dict) else None
    td = getattr(params, "text_document", None) or getattr(params, "textDocument", None)
    if td is not None:
        return getattr(td, "uri", None)
    return None


def _protocol_data(value: Any) -> Any:
    """Unwrap pygls' named-tuple Objects without losing protocol field names."""
    if hasattr(value, "_asdict"):
        value = value._asdict()
    if isinstance(value, dict):
        return {key: _protocol_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_protocol_data(item) for item in value]
    return value


def _request_source(server: LanguageServer, uri: str, params: Any) -> str:
    if isinstance(params, dict):
        text = params.get("text")
        if isinstance(text, str):
            return text
    text = getattr(params, "text", None)
    if isinstance(text, str):
        return text
    return server.workspace.get_text_document(uri).source


def _request_focus_line(params: Any) -> Optional[int]:
    if isinstance(params, dict):
        focus_line = params.get("focusLine")
        if isinstance(focus_line, int):
            return focus_line
        position = params.get("position")
        if isinstance(position, dict):
            line = position.get("line")
            if isinstance(line, int):
                return line
        return None
    focus_line = getattr(params, "focus_line", None)
    if focus_line is None:
        focus_line = getattr(params, "focusLine", None)
    if isinstance(focus_line, int):
        return focus_line
    position = getattr(params, "position", None)
    if position is not None:
        line = getattr(position, "line", None)
        if isinstance(line, int):
            return line
    return None


def _request_position(params: Any) -> Optional[lsp.Position]:
    if isinstance(params, dict):
        position = params.get("position")
        if isinstance(position, dict):
            line = position.get("line")
            character = position.get("character")
            if isinstance(line, int) and isinstance(character, int):
                return lsp.Position(line=line, character=character)
        return None
    position = getattr(params, "position", None)
    if position is None:
        return None
    line = getattr(position, "line", None)
    character = getattr(position, "character", None)
    if isinstance(line, int) and isinstance(character, int):
        return lsp.Position(line=line, character=character)
    return None


def _request_include_declaration(params: Any) -> bool:
    if isinstance(params, dict):
        value = params.get("includeDeclaration")
        if isinstance(value, bool):
            return value
        context = params.get("context")
        if isinstance(context, dict):
            value = context.get("includeDeclaration") or context.get("include_declaration")
            return bool(value)
        return False
    value = getattr(params, "include_declaration", None)
    if isinstance(value, bool):
        return value
    value = getattr(params, "includeDeclaration", None)
    if isinstance(value, bool):
        return value
    context = getattr(params, "context", None)
    if context is not None:
        value = getattr(context, "include_declaration", None)
        if isinstance(value, bool):
            return value
        value = getattr(context, "includeDeclaration", None)
        if isinstance(value, bool):
            return value
    return False


def _codemode_preload_warning_threshold(value: Any, *, default: int = 5) -> int:
    if not isinstance(value, Mapping):
        return default
    root = value.get("kedi", value)
    if not isinstance(root, Mapping):
        return default
    codemode = root.get("codemode")
    if not isinstance(codemode, Mapping):
        return default
    threshold = codemode.get("preloadWarningThreshold")
    if isinstance(threshold, int) and not isinstance(threshold, bool) and threshold >= 0:
        return threshold
    return default


def _publish_diagnostics(
    server: LanguageServer,
    uri: str,
    *,
    codemode_preload_warning_threshold: int = 5,
) -> None:
    doc = server.workspace.get_text_document(uri)
    if _is_python_uri(uri):
        diagnostics = cast(
            List[lsp.Diagnostic],
            kedi_docstrings.compute_kedi_docstring_diagnostics(
                doc.source,
                _uri_to_path(uri),
            ),
        )
    else:
        diagnostics = features.compute_diagnostics(
            source=doc.source,
            source_path=_uri_to_path(uri),
            codemode_preload_warning_threshold=codemode_preload_warning_threshold,
        )
    server.text_document_publish_diagnostics(
        lsp.PublishDiagnosticsParams(uri=uri, diagnostics=diagnostics)
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    server = make_server()
    server.start_io()


if __name__ == "__main__":  # pragma: no cover
    main()
