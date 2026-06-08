"""LSP feature tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from lsprotocol import types as lsp

from tests.lsp_helpers import position_of


def test_server_can_be_constructed():
    from kedi.lsp.server import make_server

    server = make_server()
    assert server is not None
    assert server.name == "kedi-lsp"


def test_server_custom_requests_prefer_request_text():
    from kedi.lsp.server import _request_source

    class Workspace:
        def get_text_document(self, uri):
            raise AssertionError("workspace cache should not be read")

    class Server:
        workspace = Workspace()

    fresh = "@p():\n  [x: int] = ```\n  return 1\n  ```\n  = <`x`>\n"
    assert (
        _request_source(
            cast(Any, Server()),
            "file:///tmp/x.kedi",
            {"textDocument": {"uri": "file:///tmp/x.kedi"}, "text": fresh},
        )
        == fresh
    )


def test_server_helpers_and_registered_handlers(monkeypatch):
    from kedi.lsp import server as server_module

    uri = "file:///tmp/example.kedi"
    source = "@hello(name: str) -> str:\n  = <name>\n"

    class Document:
        def __init__(self, text):
            self.source = text

    class Workspace:
        def get_text_document(self, requested_uri):
            assert requested_uri == uri
            return Document(source)

    server = server_module.make_server()
    cast(Any, server.protocol)._workspace = Workspace()
    handlers = server.protocol.fm.features
    published = []
    monkeypatch.setattr(
        server,
        "text_document_publish_diagnostics",
        lambda params: published.append(params),
    )

    calls = []

    def record(name, value):
        def inner(**kwargs):
            calls.append((name, kwargs))
            return value

        return inner

    monkeypatch.setattr(server_module.features, "compute_diagnostics", record("diagnostics", []))
    monkeypatch.setattr(server_module.features, "compute_hover", record("hover", "hovered"))
    monkeypatch.setattr(
        server_module.features, "compute_definition", record("definition", "definition")
    )
    monkeypatch.setattr(server_module.features, "compute_references", record("references", ["ref"]))
    monkeypatch.setattr(
        server_module.features, "compute_document_symbols", record("symbols", ["sym"])
    )
    monkeypatch.setattr(
        server_module.features, "compute_prepare_rename", record("prepare", "range")
    )
    monkeypatch.setattr(server_module.features, "compute_rename", record("rename", "edit"))
    monkeypatch.setattr(server_module.features, "compute_formatting", lambda src: ["format"])
    monkeypatch.setattr(
        server_module.features, "compute_signature_help", record("signature", "sig")
    )
    monkeypatch.setattr(server_module.features, "compute_inlay_hints", record("inlay", ["hint"]))
    monkeypatch.setattr(
        server_module.features,
        "compute_semantic_tokens_full",
        record("semantic", "tokens"),
    )
    monkeypatch.setattr(server_module.features, "compute_python_ranges", lambda src: ["range"])
    monkeypatch.setattr(
        server_module,
        "compute_python_virtual_document",
        lambda src, source_path=None, focus_line=None: {
            "text": "virtual",
            "ranges": [],
            "mappings": [],
            "symbols": [],
            "focusKey": focus_line,
        },
    )

    text_document = SimpleNamespace(uri=uri)
    position = position_of(source, "hello")

    handlers[lsp.TEXT_DOCUMENT_DID_OPEN](SimpleNamespace(text_document=text_document))
    handlers[lsp.TEXT_DOCUMENT_DID_CHANGE](SimpleNamespace(text_document=text_document))
    handlers[lsp.TEXT_DOCUMENT_DID_CLOSE](SimpleNamespace(text_document=text_document))
    assert handlers["workspace/didChangeWatchedFiles"](SimpleNamespace()) is None
    assert len(published) == 3
    assert published[-1].diagnostics == []

    assert (
        handlers[lsp.TEXT_DOCUMENT_HOVER](
            SimpleNamespace(text_document=text_document, position=position)
        )
        == "hovered"
    )
    assert (
        handlers[lsp.TEXT_DOCUMENT_DEFINITION](
            SimpleNamespace(text_document=text_document, position=position)
        )
        == "definition"
    )
    assert handlers[lsp.TEXT_DOCUMENT_REFERENCES](
        SimpleNamespace(
            text_document=text_document,
            position=position,
            context=SimpleNamespace(include_declaration=False),
        )
    ) == ["ref"]
    assert handlers[lsp.TEXT_DOCUMENT_DOCUMENT_SYMBOL](
        SimpleNamespace(text_document=text_document)
    ) == ["sym"]
    assert (
        handlers[lsp.TEXT_DOCUMENT_PREPARE_RENAME](
            SimpleNamespace(text_document=text_document, position=position)
        )
        == "range"
    )
    assert (
        handlers[lsp.TEXT_DOCUMENT_RENAME](
            SimpleNamespace(text_document=text_document, position=position, new_name="new")
        )
        == "edit"
    )
    assert handlers[lsp.TEXT_DOCUMENT_FORMATTING](SimpleNamespace(text_document=text_document)) == [
        "format"
    ]
    assert (
        handlers[lsp.TEXT_DOCUMENT_SIGNATURE_HELP](
            SimpleNamespace(text_document=text_document, position=position)
        )
        == "sig"
    )
    assert handlers[lsp.TEXT_DOCUMENT_INLAY_HINT](
        SimpleNamespace(text_document=text_document, range=lsp.Range(start=position, end=position))
    ) == ["hint"]
    assert (
        handlers[lsp.TEXT_DOCUMENT_SEMANTIC_TOKENS_FULL](
            SimpleNamespace(text_document=text_document)
        )
        == "tokens"
    )

    assert handlers["kedi/pythonRanges"]({}) == {"uri": "", "ranges": []}
    assert handlers["kedi/pythonRanges"]({"textDocument": {"uri": uri}, "text": source}) == {
        "uri": uri,
        "ranges": ["range"],
    }
    assert handlers["kedi/pythonVirtualDocument"]({}) == {
        "uri": "",
        "text": "",
        "ranges": [],
        "mappings": [],
        "symbols": [],
        "focusKey": None,
    }
    assert (
        handlers["kedi/pythonVirtualDocument"](
            {"textDocument": {"uri": uri}, "text": source, "position": {"line": 3}}
        )["focusKey"]
        == 3
    )

    assert server_module._uri_to_path("untitled:Untitled-1") is None
    assert server_module._uri_to_path("file:///tmp/a%20b.kedi") == "/tmp/a b.kedi"
    assert server_module._text_document_uri({"textDocument": "bad"}) is None
    assert (
        server_module._text_document_uri(SimpleNamespace(textDocument=SimpleNamespace(uri=uri)))
        == uri
    )
    assert server_module._text_document_uri(SimpleNamespace()) is None
    assert server_module._request_source(server, uri, {"text": 1}) == source
    assert server_module._request_source(server, uri, SimpleNamespace(text="fresh")) == "fresh"
    assert server_module._request_source(server, uri, SimpleNamespace()) == source
    assert server_module._request_focus_line({"focusLine": 4}) == 4
    assert server_module._request_focus_line({"position": {"line": "bad"}}) is None
    assert server_module._request_focus_line({"position": "bad"}) is None
    assert server_module._request_focus_line(SimpleNamespace(focus_line=5)) == 5
    assert server_module._request_focus_line(SimpleNamespace(focusLine=6)) == 6
    assert server_module._request_focus_line(SimpleNamespace(position=SimpleNamespace(line=7))) == 7
    assert (
        server_module._request_focus_line(SimpleNamespace(position=SimpleNamespace(line="bad")))
        is None
    )
    assert server_module._request_focus_line(SimpleNamespace()) is None


def test_server_main_starts_io(monkeypatch):
    from kedi.lsp import server as server_module

    started = []
    monkeypatch.setattr(
        server_module,
        "make_server",
        lambda: SimpleNamespace(start_io=lambda: started.append(True)),
    )

    server_module.main()

    assert started == [True]
