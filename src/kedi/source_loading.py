from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SourceDocument:
    path: str
    text: str
    start_line_in_merged: int

    @property
    def lines(self) -> list[str]:
        return self.text.splitlines()

    @property
    def line_count(self) -> int:
        return max(1, len(self.lines))


@dataclass(frozen=True)
class ResolvedLine:
    source_path: str
    source_line: int
    text: str
    source_id: str


@dataclass(frozen=True)
class CompositeSourceMap:
    documents: tuple[SourceDocument, ...]

    def resolve_line(self, merged_line: int) -> ResolvedLine | None:
        for document in reversed(self.documents):
            if merged_line >= document.start_line_in_merged:
                local_line = merged_line - document.start_line_in_merged + 1
                lines = document.lines
                if 1 <= local_line <= len(lines):
                    return ResolvedLine(
                        source_path=document.path,
                        source_line=local_line,
                        text=lines[local_line - 1],
                        source_id=document.path,
                    )
        return None

    def get_line(self, source_path: str, line_number: int) -> str | None:
        for document in self.documents:
            if document.path != source_path:
                continue
            lines = document.lines
            if 1 <= line_number <= len(lines):
                return lines[line_number - 1]
            return None
        return None

    def get_byte_offset(
        self,
        source_path: str,
        line_number: int,
        byte_column: int,
    ) -> int | None:
        """Return a document-local UTF-8 byte offset for a source point."""
        for document in self.documents:
            if document.path != source_path:
                continue
            raw = document.text.encode("utf-8")
            line_starts = [0]
            line_starts.extend(index + 1 for index, value in enumerate(raw) if value == 10)
            if not 1 <= line_number <= len(line_starts):
                return None
            line_start = line_starts[line_number - 1]
            line_end = line_starts[line_number] - 1 if line_number < len(line_starts) else len(raw)
            return line_start + max(0, min(byte_column, line_end - line_start))
        return None

    def has_path(self, source_path: str) -> bool:
        return any(document.path == source_path for document in self.documents)


@dataclass(frozen=True)
class LoadedKediSource:
    text: str
    source_map: CompositeSourceMap


def single_file_source_map(source_path: str | Path, text: str) -> CompositeSourceMap:
    path = str(source_path)
    return CompositeSourceMap((SourceDocument(path=path, text=text, start_line_in_merged=1),))


def load_kedi_source_bundle(
    source_path: str | Path,
    *,
    include_val: bool = True,
    include_cache: bool = True,
) -> LoadedKediSource:
    root = Path(source_path).resolve()
    documents: list[tuple[Path, str]] = [(root, root.read_text(encoding="utf-8"))]

    if include_val:
        val_path = root.with_suffix(".val.kedi")
        if val_path.exists():
            documents.append((val_path.resolve(), val_path.read_text(encoding="utf-8")))

    if include_cache:
        cache_path = root.with_suffix(".cache.kedi")
        if cache_path.exists():
            documents.append((cache_path.resolve(), cache_path.read_text(encoding="utf-8")))

    return merge_source_documents(documents)


def merge_source_documents(documents: Iterable[tuple[Path, str]]) -> LoadedKediSource:
    text_parts: list[str] = []
    mapped_docs: list[SourceDocument] = []
    merged_newlines = 0

    for index, (path, text) in enumerate(documents):
        if index > 0:
            separator = "\n\n"
            text_parts.append(separator)
            merged_newlines += separator.count("\n")
        text_parts.append(text)
        mapped_docs.append(
            SourceDocument(
                path=str(path),
                text=text,
                start_line_in_merged=merged_newlines + 1,
            )
        )
        merged_newlines += text.count("\n")

    return LoadedKediSource(
        text="".join(text_parts),
        source_map=CompositeSourceMap(tuple(mapped_docs)),
    )
