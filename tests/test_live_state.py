"""Unit tests for live state persistence."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from midas.live_state import LiveState, StateFileError, load_or_seed, load_state, save_atomic
from midas.models import CashInfusion, Holding, PortfolioConfig, PositionLot


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


def test_load_rejects_non_dict_cash_infusion(tmp_path: Path) -> None:
    path = tmp_path / "state.yaml"
    path.write_text("schema_version: 1\navailable_cash: 0\ncash_infusion: today\n")
    with pytest.raises(StateFileError, match="cash_infusion"):
        load_state(path)


def test_load_or_seed_creates_state_from_portfolio(tmp_path: Path) -> None:
    portfolio = PortfolioConfig(
        holdings=[
            Holding(ticker="AAPL", shares=100.0, cost_basis=150.0),
            Holding(ticker="NVDA", shares=50.0, cost_basis=49.76),
        ],
        available_cash=5000.0,
        cash_infusion=CashInfusion(amount=1500.0, next_date=date(2026, 5, 15), frequency="biweekly"),
    )
    path = tmp_path / "portfolio.state.yaml"
    assert not path.exists()

    state = load_or_seed(portfolio, path)

    assert path.exists()  # seed wrote the file
    assert state.available_cash == 5000.0
    assert state.cash_infusion_next_date == date(2026, 5, 15)
    assert state.peak_equity is None
    assert state.high_water_marks == {}
    assert state.lots == {
        "AAPL": [PositionLot(shares=100.0, purchase_date=None, cost_basis=150.0)],
        "NVDA": [PositionLot(shares=50.0, purchase_date=None, cost_basis=49.76)],
    }


def test_load_or_seed_skips_holdings_with_zero_shares(tmp_path: Path) -> None:
    portfolio = PortfolioConfig(
        holdings=[
            Holding(ticker="AAPL", shares=100.0, cost_basis=150.0),
            Holding(ticker="MSFT", shares=0.0, cost_basis=None),
        ],
        available_cash=1000.0,
    )
    path = tmp_path / "state.yaml"
    state = load_or_seed(portfolio, path)
    assert "AAPL" in state.lots
    assert "MSFT" not in state.lots


def test_load_or_seed_warns_on_share_drift(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=100.0, cost_basis=150.0)],
        available_cash=1000.0,
    )
    path = tmp_path / "state.yaml"
    load_or_seed(portfolio, path)  # first call seeds

    portfolio_drifted = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=200.0, cost_basis=150.0)],  # state still has 100
        available_cash=1000.0,
    )
    with caplog.at_level("WARNING"):
        state = load_or_seed(portfolio_drifted, path)
    assert state.lots["AAPL"][0].shares == 100.0  # state file wins
    assert any("AAPL" in record.message and "200" in record.message for record in caplog.records)


def test_load_or_seed_warns_on_cash_drift(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=100.0, cost_basis=150.0)],
        available_cash=1000.0,
    )
    path = tmp_path / "state.yaml"
    load_or_seed(portfolio, path)

    portfolio_drifted = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=100.0, cost_basis=150.0)],
        available_cash=9999.99,
    )
    with caplog.at_level("WARNING"):
        state = load_or_seed(portfolio_drifted, path)
    assert state.available_cash == 1000.0  # state file wins
    assert any("available_cash" in record.message for record in caplog.records)


def test_load_or_seed_raises_when_held_ticker_removed_from_portfolio(tmp_path: Path) -> None:
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=100.0, cost_basis=150.0)],
        available_cash=1000.0,
    )
    path = tmp_path / "state.yaml"
    load_or_seed(portfolio, path)

    portfolio_without = PortfolioConfig(
        holdings=[],  # AAPL removed from portfolio while still held in state
        available_cash=1000.0,
    )
    with pytest.raises(StateFileError, match=r"AAPL.*100"):
        load_or_seed(portfolio_without, path)
