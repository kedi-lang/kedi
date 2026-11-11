from __future__ import annotations

from pathlib import Path

import pytest

import kedi
from kedi._interactive import state
from kedi._interactive.persistence import SessionLoadError, read_snapshot, write_snapshot


@pytest.mark.parametrize(
    "field, value",
    [
        ("cell_number", "x"),
        ("cell_number", -1),
        ("cell_number", 1.5),
        ("cell_number", True),
        ("cell_number", None),
        ("cwd", []),
        ("source_names", 5),
        ("source_names", "cell"),
        ("source_names", [1]),
        ("source_names", ["cell", "cell"]),
        ("fragments", {}),
        ("fragments", [None]),
        ("fragments", [{"source": [], "source_name": "cell", "digest": "digest"}]),
        ("fragments", [{"source": "", "source_name": [], "digest": "digest"}]),
        ("fragments", [{"source": "", "source_name": "cell", "digest": []}]),
        ("fragments", [{}]),
    ],
)
def test_invalid_snapshot_structure_is_rejected_before_allocating_session(
    tmp_path: Path, monkeypatch, field, value
) -> None:
    path = tmp_path / "session.kedi-state"
    with kedi.interactive() as session:
        session.execute("[value: int] = `1`")
        session.dump(path)
    payload = read_snapshot(path)
    payload[field] = value
    write_snapshot(path, payload)

    def unexpected_allocation(*args, **kwargs):
        pytest.fail("Malformed snapshot must be rejected before allocating conversation state")

    monkeypatch.setattr(state, "_load_conversation", unexpected_allocation)
    with pytest.raises(SessionLoadError, match=field):
        kedi.load_session(path)


def test_execution_lock_is_released_if_cell_number_increment_fails() -> None:
    with kedi.interactive() as session:
        session._cell_number = "invalid"
        with pytest.raises(TypeError):
            session.execute("[value: int] = `1`")
        assert not session._execute_lock.locked()
        session._cell_number = 0
        session.execute("[value: int] = `2`")
        assert session._env["value"] == 2
