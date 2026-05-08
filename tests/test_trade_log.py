"""Trade-log writer/reader unit tests."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from midas.models import Direction, HoldingPeriod, TradeRecord
from midas.trade_log import TradeLogError, append_trade, read_trades

TRADE_LOG_HEADER = (
    "date,ticker,direction,shares,price,strategy,holding_period,purchase_date,cost_basis,realized_pnl,return_pct\n"
)


def _buy(day: date) -> TradeRecord:
    return TradeRecord(
        date=day,
        ticker="AAPL",
        direction=Direction.BUY,
        shares=10.0,
        price=20.0,
        strategy_name="Momentum",
        purchase_date=day,
    )


def _sell(day: date, holding: HoldingPeriod, purchase: date | str | None) -> TradeRecord:
    return TradeRecord(
        date=day,
        ticker="AAPL",
        direction=Direction.SELL,
        shares=10.0,
        price=25.0,
        strategy_name="StopLoss",
        holding_period=holding,
        purchase_date=purchase,
    )


def test_first_append_writes_header(tmp_path: Path) -> None:
    path = tmp_path / "portfolio.state.yaml.trades.csv"
    append_trade(path, _buy(date(2026, 5, 8)), cost_basis=None, purchase_date=date(2026, 5, 8))
    contents = path.read_text()
    assert contents.startswith(TRADE_LOG_HEADER)
    # one data row after the header
    assert contents.count("\n") == 2


def test_subsequent_appends_do_not_duplicate_header(tmp_path: Path) -> None:
    path = tmp_path / "portfolio.state.yaml.trades.csv"
    append_trade(path, _buy(date(2026, 5, 8)), cost_basis=None, purchase_date=date(2026, 5, 8))
    append_trade(path, _buy(date(2026, 5, 9)), cost_basis=None, purchase_date=date(2026, 5, 9))
    text = path.read_text()
    assert text.count("date,ticker,direction") == 1
    assert text.count("\n") == 3  # header + 2 rows


def test_round_trip_buy_and_two_bucket_sell(tmp_path: Path) -> None:
    path = tmp_path / "portfolio.state.yaml.trades.csv"
    append_trade(path, _buy(date(2026, 5, 8)), cost_basis=None, purchase_date=date(2026, 5, 8))
    append_trade(
        path,
        _sell(date(2026, 6, 8), HoldingPeriod.SHORT_TERM, "various"),
        cost_basis=20.5,
        purchase_date="various",
    )
    append_trade(
        path,
        _sell(date(2026, 6, 8), HoldingPeriod.LONG_TERM, date(2024, 1, 1)),
        cost_basis=10.0,
        purchase_date=date(2024, 1, 1),
    )
    rows = read_trades(path)
    assert len(rows) == 3
    assert rows[0].direction == Direction.BUY
    assert rows[0].purchase_date == date(2026, 5, 8)
    assert rows[1].holding_period == HoldingPeriod.SHORT_TERM
    assert rows[1].purchase_date == "various"
    assert rows[1].cost_basis == 20.5
    assert rows[1].realized_pnl == pytest.approx((25.0 - 20.5) * 10.0)
    assert rows[2].purchase_date == date(2024, 1, 1)


def test_purchase_date_none_round_trips_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "portfolio.state.yaml.trades.csv"
    append_trade(
        path,
        _sell(date(2026, 6, 8), HoldingPeriod.SHORT_TERM, None),
        cost_basis=10.0,
        purchase_date=None,
    )
    rows = read_trades(path)
    assert rows[0].purchase_date is None


def test_header_drift_raises_with_named_divergence(tmp_path: Path) -> None:
    path = tmp_path / "portfolio.state.yaml.trades.csv"
    path.write_text("date,ticker,direction\n2026-05-08,AAPL,BUY\n")
    with pytest.raises(TradeLogError, match="header"):
        read_trades(path)


def test_partial_row_raises_with_line_number(tmp_path: Path) -> None:
    path = tmp_path / "portfolio.state.yaml.trades.csv"
    path.write_text(TRADE_LOG_HEADER + "2026-05-08,AAPL\n")  # 2 fields instead of 11
    with pytest.raises(TradeLogError, match="line 2"):
        read_trades(path)
