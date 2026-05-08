"""Unit tests for live state persistence."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from midas.live_state import (
    LiveState,
    StateFileError,
    aggregate_cost_basis,
    apply_buy,
    apply_sell,
    consume_lots_fifo,
    load_or_seed,
    load_state,
    save_atomic,
)
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
    # Seed peak_equity = cash + Σ shares * cost_basis
    # = 5000 + 100*150 + 50*49.76 = 5000 + 15000 + 2488 = 22488
    assert state.peak_equity == pytest.approx(22488.0)
    assert state.high_water_marks == {}
    assert state.lots == {
        "AAPL": [PositionLot(shares=100.0, purchase_date=None, cost_basis=150.0)],
        "NVDA": [PositionLot(shares=50.0, purchase_date=None, cost_basis=49.76)],
    }


def test_load_or_seed_initializes_peak_equity_to_starting_value(tmp_path: Path) -> None:
    """Backtest seeds state.peak_value = state.starting_value at _initialize_state;
    live's load_or_seed must do the same so CPPI/drawdown calcs don't diverge
    on the first tick when prices are below cost basis."""
    portfolio = PortfolioConfig(
        holdings=[
            Holding(ticker="AAPL", shares=10.0, cost_basis=150.0),
            Holding(ticker="NVDA", shares=20.0, cost_basis=50.0),
        ],
        available_cash=500.0,
    )
    state = load_or_seed(portfolio, tmp_path / "state.yaml")
    # 500 + 10*150 + 20*50 = 500 + 1500 + 1000 = 3000
    assert state.peak_equity == pytest.approx(3000.0)


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


def test_load_or_seed_warns_on_cost_basis_drift(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Hand-edits to portfolio.yaml's cost_basis are the field most likely to
    surprise the operator (it's exactly what TrailingStop/StopLoss clamp
    against). Surface it as a warning; state still wins per the documented
    "runtime fills are authoritative" policy.
    """
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=100.0, cost_basis=150.0)],
        available_cash=1000.0,
    )
    path = tmp_path / "state.yaml"
    load_or_seed(portfolio, path)

    portfolio_drifted = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=100.0, cost_basis=200.0)],  # 33% bump
        available_cash=1000.0,
    )
    with caplog.at_level("WARNING"):
        state = load_or_seed(portfolio_drifted, path)
    assert state.lots["AAPL"][0].cost_basis == 150.0  # state file wins
    assert any("cost_basis" in record.message for record in caplog.records)


def test_load_or_seed_does_not_warn_on_cost_basis_within_tolerance(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Sub-1% drift (e.g., 150.0 vs 150.5) is within tolerance — no warning."""
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=100.0, cost_basis=150.0)],
        available_cash=1000.0,
    )
    path = tmp_path / "state.yaml"
    load_or_seed(portfolio, path)

    portfolio_close = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=100.0, cost_basis=150.5)],  # ~0.33% drift
        available_cash=1000.0,
    )
    with caplog.at_level("WARNING"):
        load_or_seed(portfolio_close, path)
    assert not any("cost_basis" in record.message for record in caplog.records)


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


def test_aggregate_cost_basis_share_weighted() -> None:
    lots = [
        PositionLot(shares=100.0, purchase_date=None, cost_basis=10.0),
        PositionLot(shares=50.0, purchase_date=date(2026, 4, 12), cost_basis=22.0),
    ]
    # (100 * 10 + 50 * 22) / 150 = 14.0
    assert aggregate_cost_basis(lots) == pytest.approx(14.0)


def test_aggregate_cost_basis_empty_returns_zero() -> None:
    assert aggregate_cost_basis([]) == 0.0


def test_consume_lots_fifo_st_only() -> None:
    today = date(2026, 5, 7)
    lots = [PositionLot(shares=100.0, purchase_date=date(2026, 4, 1), cost_basis=10.0)]
    breakdown = consume_lots_fifo(lots, shares=40.0, day=today)
    assert breakdown.st_shares == 40.0
    assert breakdown.st_basis == pytest.approx(10.0)
    assert breakdown.st_weighted == pytest.approx(400.0)
    assert breakdown.lt_shares == 0.0
    assert breakdown.lt_weighted == 0.0
    assert lots[0].shares == 60.0  # mutated in place


def test_consume_lots_fifo_straddles_st_lt_boundary() -> None:
    today = date(2026, 5, 7)
    # First lot is >365 days old (LT); second is recent (ST).
    lots = [
        PositionLot(shares=30.0, purchase_date=date(2025, 4, 1), cost_basis=10.0),
        PositionLot(shares=20.0, purchase_date=date(2026, 4, 1), cost_basis=20.0),
    ]
    breakdown = consume_lots_fifo(lots, shares=40.0, day=today)
    assert breakdown.lt_shares == 30.0
    assert breakdown.lt_basis == pytest.approx(10.0)
    assert breakdown.st_shares == 10.0
    assert breakdown.st_basis == pytest.approx(20.0)
    assert lots == [PositionLot(shares=10.0, purchase_date=date(2026, 4, 1), cost_basis=20.0)]


def test_consume_lots_fifo_unknown_purchase_date_is_short_term() -> None:
    today = date(2026, 5, 7)
    lots = [PositionLot(shares=100.0, purchase_date=None, cost_basis=10.0)]
    breakdown = consume_lots_fifo(lots, shares=50.0, day=today)
    assert breakdown.st_shares == 50.0
    assert breakdown.lt_shares == 0.0


def test_consume_lots_fifo_oversell_consumes_all_available() -> None:
    """Selling more than the lot list holds consumes everything available
    and reports a partial breakdown. (Oversell-prevention is the caller's
    responsibility — backtest's _execute asserts new_position >= 0; live's
    apply_sell will be similarly bounded by the order sizer.)"""
    today = date(2026, 5, 7)
    lots = [PositionLot(shares=10.0, purchase_date=date(2026, 4, 1), cost_basis=20.0)]
    breakdown = consume_lots_fifo(lots, shares=999.0, day=today)
    assert breakdown.st_shares == 10.0  # only 10 available
    assert lots == []


def test_consume_lots_fifo_zero_shares_no_op() -> None:
    today = date(2026, 5, 7)
    lots = [PositionLot(shares=10.0, purchase_date=date(2026, 4, 1), cost_basis=20.0)]
    breakdown = consume_lots_fifo(lots, shares=0.0, day=today)
    assert breakdown.st_shares == 0.0
    assert breakdown.lt_shares == 0.0
    assert lots == [PositionLot(shares=10.0, purchase_date=date(2026, 4, 1), cost_basis=20.0)]


def test_consume_lots_fifo_exact_365_day_boundary_is_long_term() -> None:
    """Pin the exact-boundary classification: a lot purchased exactly 365
    days ago classifies as long-term (>= 365). This matches the existing
    backtest behavior; do not silently relax to > 365 in a future refactor."""
    today = date(2026, 5, 7)
    one_year_ago = today - timedelta(days=365)
    lots = [PositionLot(shares=10.0, purchase_date=one_year_ago, cost_basis=20.0)]
    breakdown = consume_lots_fifo(lots, shares=10.0, day=today)
    assert breakdown.lt_shares == 10.0
    assert breakdown.st_shares == 0.0


def test_consume_lots_fifo_full_consumption_leaves_empty_list() -> None:
    today = date(2026, 5, 7)
    lots = [PositionLot(shares=10.0, purchase_date=date(2026, 4, 1), cost_basis=20.0)]
    consume_lots_fifo(lots, shares=10.0, day=today)
    assert lots == []


def test_apply_buy_appends_lot_and_decrements_cash() -> None:
    state = LiveState(available_cash=1000.0, cash_infusion_next_date=None)
    apply_buy(state, "AAPL", shares=10.0, price=150.0, day=date(2026, 5, 7))
    assert state.available_cash == pytest.approx(1000.0 - 10.0 * 150.0)
    assert state.lots["AAPL"] == [PositionLot(shares=10.0, purchase_date=date(2026, 5, 7), cost_basis=150.0)]


def test_apply_buy_appends_to_existing_lots() -> None:
    state = LiveState(
        available_cash=1000.0,
        cash_infusion_next_date=None,
        lots={"AAPL": [PositionLot(shares=5.0, purchase_date=date(2026, 4, 1), cost_basis=140.0)]},
    )
    apply_buy(state, "AAPL", shares=10.0, price=150.0, day=date(2026, 5, 7))
    assert len(state.lots["AAPL"]) == 2
    assert state.available_cash == pytest.approx(1000.0 - 10.0 * 150.0)


def test_apply_sell_consumes_fifo_and_increments_cash() -> None:
    state = LiveState(
        available_cash=0.0,
        cash_infusion_next_date=None,
        lots={
            "AAPL": [
                PositionLot(shares=30.0, purchase_date=date(2025, 4, 1), cost_basis=10.0),  # LT
                PositionLot(shares=20.0, purchase_date=date(2026, 4, 1), cost_basis=20.0),  # ST
            ]
        },
    )
    st_pnl, lt_pnl = apply_sell(state, "AAPL", shares=40.0, price=25.0, day=date(2026, 5, 7))
    assert state.available_cash == pytest.approx(40.0 * 25.0)
    assert state.lots["AAPL"] == [PositionLot(shares=10.0, purchase_date=date(2026, 4, 1), cost_basis=20.0)]
    assert lt_pnl == pytest.approx(30.0 * (25.0 - 10.0))
    assert st_pnl == pytest.approx(10.0 * (25.0 - 20.0))


def test_apply_sell_drops_empty_ticker_entry() -> None:
    state = LiveState(
        available_cash=0.0,
        cash_infusion_next_date=None,
        lots={"AAPL": [PositionLot(shares=10.0, purchase_date=None, cost_basis=10.0)]},
    )
    apply_sell(state, "AAPL", shares=10.0, price=20.0, day=date(2026, 5, 7))
    assert "AAPL" not in state.lots  # cleared


def test_apply_sell_keeps_ticker_with_remaining_shares() -> None:
    state = LiveState(
        available_cash=0.0,
        cash_infusion_next_date=None,
        lots={"AAPL": [PositionLot(shares=10.0, purchase_date=None, cost_basis=10.0)]},
    )
    apply_sell(state, "AAPL", shares=4.0, price=20.0, day=date(2026, 5, 7))
    assert "AAPL" in state.lots
    assert state.lots["AAPL"][0].shares == 6.0


def test_apply_sell_raises_on_oversell() -> None:
    """Mirrors backtest's ``assert new_position >= 0`` invariant: ``apply_sell``
    must refuse to credit cash for shares that don't exist. ``OrderSizer.size_sells``
    clamps in production; this guards against a future regression silently
    fabricating cash."""
    state = LiveState(
        available_cash=0.0,
        cash_infusion_next_date=None,
        lots={"AAPL": [PositionLot(shares=10.0, purchase_date=None, cost_basis=10.0)]},
    )
    with pytest.raises(AssertionError, match="oversell"):
        apply_sell(state, "AAPL", shares=999.0, price=20.0, day=date(2026, 5, 7))
