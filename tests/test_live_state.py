"""Unit tests for live state persistence."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from midas.live_state import LiveState, StateFileError, load_state, save_atomic
from midas.models import PositionLot


def test_save_load_round_trip(tmp_path: Path) -> None:
    state = LiveState(
        available_cash=4823.50,
        cash_infusion_next_date=date(2026, 5, 15),
        high_water_marks={"PLTR": 24.18, "NVDA": 142.5},
        peak_equity=18420.0,
        lots={
            "PLTR": [
                PositionLot(shares=100.0, purchase_date=None, cost_basis=10.0),
                PositionLot(shares=50.0, purchase_date=date(2026, 4, 12), cost_basis=22.5),
            ],
            "NVDA": [PositionLot(shares=100.0, purchase_date=None, cost_basis=49.76)],
        },
    )
    path = tmp_path / "portfolio.state.yaml"
    save_atomic(state, path)
    loaded = load_state(path)
    assert loaded == state


def test_load_rejects_unknown_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "state.yaml"
    path.write_text("schema_version: 99\navailable_cash: 0\n")
    with pytest.raises(StateFileError, match="unsupported state version"):
        load_state(path)


def test_load_rejects_unparseable_yaml(tmp_path: Path) -> None:
    path = tmp_path / "state.yaml"
    path.write_text(":\n - this isn't\n  valid yaml: ::\n")
    with pytest.raises(StateFileError):
        load_state(path)


def test_save_atomic_leaves_canonical_intact_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = LiveState(available_cash=100.0, cash_infusion_next_date=None)
    path = tmp_path / "state.yaml"
    save_atomic(state, path)
    canonical_before = path.read_text()

    def boom(*args: object, **kwargs: object) -> None:
        msg = "simulated rename failure"
        raise OSError(msg)

    monkeypatch.setattr("os.replace", boom)
    with pytest.raises(OSError, match="simulated rename failure"):
        save_atomic(LiveState(available_cash=999.99, cash_infusion_next_date=None), path)
    assert path.read_text() == canonical_before
