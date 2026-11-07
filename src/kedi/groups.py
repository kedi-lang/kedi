"""Named groups at run time: identity, the group a model call belongs to, and form assets.

A group is an ``> evolve:`` region, an ``> optimize:`` span, or an ``> auto:`` procedure. Its
identity, ``kedi:group:<module>:<qualified name>``, does not depend on source lines. Each model
call in the group's active form is ``kedi:template:<module>:<qualified name>#<ordinal>``, the
ordinal being the call's position in that form.
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from kedi.lang.ast import (
    ConditionalLoopStmt,
    FormProducer,
    IfStmt,
    LoopStmt,
    RawInvokeStmt,
    Stmt,
    TemplateBlock,
    TemplateLine,
)
from kedi.telemetry import TelemetryAsset, asset_fingerprint, capture_enabled, record_asset

GroupKind = Literal["evolve", "optimize", "auto"]


@dataclass(frozen=True, slots=True)
class GroupMember:
    """One model call of a group's active form."""

    group: str
    ordinal: int

    @property
    def template_locator(self) -> str:
        return f"kedi:template:{self.group.removeprefix('kedi:group:')}#{self.ordinal}"


@dataclass(frozen=True, slots=True)
class _ActiveGroup:
    locator: str
    ordinals: dict[int, int]


_GROUP: ContextVar[_ActiveGroup | None] = ContextVar("kedi_group", default=None)
_MEMBER: ContextVar[GroupMember | None] = ContextVar("kedi_group_member", default=None)


def group_locator(module: str, qualified_name: str) -> str:
    return f"kedi:group:{module}:{qualified_name}"


def module_name(path: Path | None, *, root: Path | None, spec: str | None) -> str:
    """A module's import name as the root program sees it, never an absolute path.

    Modules under the root program's directory are named by their path from it; packages and
    bundled modules keep the name they are imported by.
    """
    if path is None or path.name.startswith("<"):
        return spec or "main"
    if root is not None:
        try:
            return path.with_suffix("").relative_to(root.parent).as_posix()
        except ValueError:
            pass
    return spec or path.stem


def form_fingerprint(canonical: str) -> str:
    """A form's identity: its canonical text alone, whoever produced it."""
    return asset_fingerprint(canonical)


@contextmanager
def group_scope(
    *,
    module: str,
    qualified_name: str,
    kind: GroupKind,
    body: Sequence[Stmt],
    canonical: str,
    producer: FormProducer = "source",
    parent: str | None = None,
    note: str | None = None,
) -> Generator[None]:
    """Run a group's active form: record it, then attribute its model calls to it.

    ``parent`` is the fingerprint of the form this one derives from. Who produced the form and
    why are metadata; neither is part of its identity.
    """
    locator = group_locator(module, qualified_name)
    metadata = {"group": qualified_name, "group_kind": kind, "module": module, "producer": producer}
    if note is not None:
        metadata["note"] = note
    record_asset(
        TelemetryAsset(
            kind="form",
            local_id=qualified_name,
            name=qualified_name,
            source_locator=locator,
            content_fingerprint=form_fingerprint(canonical),
            content=canonical if capture_enabled("content") else None,
            metadata=metadata,
            sensitivity="sensitive",
            representation="definition",
            parent_fingerprint=parent,
        )
    )
    token = _GROUP.set(_ActiveGroup(locator, _model_call_ordinals(body)))
    try:
        yield
    finally:
        _GROUP.reset(token)


@contextmanager
def group_member(statement: Stmt) -> Generator[None]:
    """Attribute the model call a statement makes to the active group, if it belongs to it."""
    group = _GROUP.get()
    ordinal = None if group is None else group.ordinals.get(id(statement))
    if group is None or ordinal is None:
        yield
        return
    token = _MEMBER.set(GroupMember(group.locator, ordinal))
    try:
        yield
    finally:
        _MEMBER.reset(token)


def current_group_member() -> GroupMember | None:
    return _MEMBER.get()


def _model_call_ordinals(body: Sequence[Stmt]) -> dict[int, int]:
    """Number the statements of a form that call a model, in source order."""
    ordinals: dict[int, int] = {}

    def visit(statements: Sequence[Stmt]) -> None:
        for statement in statements:
            if isinstance(statement, (TemplateBlock, TemplateLine, RawInvokeStmt)) or (
                isinstance(statement, (IfStmt, ConditionalLoopStmt))
                and statement.condition_segments is not None
            ):
                ordinals[id(statement)] = len(ordinals)
            if isinstance(statement, IfStmt):
                visit(statement.body)
                visit(statement.else_body or [])
            elif isinstance(statement, LoopStmt):
                visit(statement.body)
                visit(statement.map_body or [])
            elif isinstance(statement, ConditionalLoopStmt):
                visit(statement.body)

    visit(body)
    return ordinals


__all__ = [
    "GroupMember",
    "current_group_member",
    "form_fingerprint",
    "group_locator",
    "group_member",
    "group_scope",
    "module_name",
]
