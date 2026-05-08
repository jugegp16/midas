"""Integration test: live engine writes a complete trade log across ticks."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from midas.allocator import Allocator
from midas.live import LiveEngine
from midas.live_state import LiveState, save_atomic
from midas.models import (
    AllocationConstraints,
    Direction,
    Holding,
    Order,
    OrderContext,
    PortfolioConfig,
    PositionLot,
)
from midas.order_sizer import OrderSizer
from midas.trade_log import read_trades

ProviderFactory = Callable[[dict[str, list[float]], list[date]], MagicMock]


def _fake_context(source: str = "test") -> OrderContext:
    return OrderContext(
        contributions={source: 1.0},
        blended_score=1.0,
        target_weight=1.0,
        current_weight=0.0,
        reason="test",
        source=source,
    )


def test_live_engine_writes_trade_log_for_full_exit_lt_sell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_provider: ProviderFactory
) -> None:
    """A full-exit SELL of an LT lot writes a single LT bucket row.

    The row's ``purchase_date`` is the original lot's purchase date and the
    ``cost_basis`` is the lot's basis. This exercises the live trade-log path
    end-to-end: state is persisted, breakdown is captured per-order, and the
    LT branch emits one row.
    """
    state_path = tmp_path / "portfolio.state.yaml"
    log_path = state_path.with_suffix(state_path.suffix + ".trades.csv")

    # Seed the state directly so we control purchase_date precisely.
    save_atomic(
        LiveState(
            available_cash=0.0,
            cash_infusion_next_date=None,
            high_water_marks={"AAPL": 100.0},
            peak_equity=10000.0,
            lots={
                "AAPL": [
                    PositionLot(shares=100.0, purchase_date=date(2024, 1, 1), cost_basis=50.0),
                ]
            },
        ),
        state_path,
    )

    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=100.0, cost_basis=50.0)],
        available_cash=0.0,
    )
    provider = make_provider({"AAPL": [80.0]}, [date(2026, 5, 7)])

    engine = LiveEngine(
        portfolio=portfolio,
        allocator=Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1),
        order_sizer=OrderSizer(),
        provider=provider,
        state_path=state_path,
    )

    # Inject a synthetic full-exit SELL via the order sizer's sell pass; this
    # bypasses allocator/exit-rule plumbing while exercising the real
    # apply-fills + trade-log paths.
    fake_sell = Order(
        ticker="AAPL",
        direction=Direction.SELL,
        shares=100.0,
        price=80.0,
        estimated_value=8000.0,
        context=_fake_context(source="StopLoss"),
    )
    monkeypatch.setattr(engine._order_sizer, "size_sells", lambda *a, **kw: [fake_sell])
    monkeypatch.setattr(engine._order_sizer, "size_buys", lambda *a, **kw: [])

    try:
        engine._tick(["AAPL"])
    finally:
        engine.close()

    assert log_path.exists()
    trades = read_trades(log_path)
    sells = [t for t in trades if t.direction == Direction.SELL]
    assert len(sells) == 1, f"expected exactly one SELL row, got {trades}"
    sell = sells[0]
    assert sell.holding_period is not None
    assert sell.holding_period.value == "long-term"
    assert sell.purchase_date == date(2024, 1, 1)
    assert sell.cost_basis == pytest.approx(50.0)
    assert sell.shares == pytest.approx(100.0)
    assert sell.strategy_name == "StopLoss"


def test_live_engine_writes_trade_log_across_three_ticks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_provider: ProviderFactory
) -> None:
    """Three ticks: BUY, BUY (second lot), partial SELL.

    The partial SELL consumes the older lot only (FIFO), producing one ST
    bucket row (because both lots were bought within 365 days). The log
    has three rows in order.

    Patches ``midas.live.date`` so each tick sees a distinct ``today``,
    pinning the per-row ``date`` and the BUY ``purchase_date`` columns.
    """
    state_path = tmp_path / "portfolio.state.yaml"
    log_path = state_path.with_suffix(state_path.suffix + ".trades.csv")

    portfolio = PortfolioConfig(
        holdings=[],
        available_cash=10000.0,
    )
    provider = make_provider(
        {"AAPL": [100.0, 110.0, 120.0]},
        [date(2026, 5, 6), date(2026, 5, 7), date(2026, 5, 8)],
    )

    engine = LiveEngine(
        portfolio=portfolio,
        allocator=Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1),
        order_sizer=OrderSizer(),
        provider=provider,
        state_path=state_path,
    )

    days = [date(2026, 5, 6), date(2026, 5, 7), date(2026, 5, 8)]
    day_idx = {"i": 0}

    class _FakeDate:
        @staticmethod
        def today() -> date:
            return days[day_idx["i"]]

    monkeypatch.setattr("midas.live.date", _FakeDate)

    # Tick 1: BUY 10 @ 100 on 2026-05-06.
    buy_a = Order(
        ticker="AAPL",
        direction=Direction.BUY,
        shares=10.0,
        price=100.0,
        estimated_value=1000.0,
        context=_fake_context("EntrySig"),
    )
    monkeypatch.setattr(engine._order_sizer, "size_sells", lambda *a, **kw: [])
    monkeypatch.setattr(engine._order_sizer, "size_buys", lambda *a, **kw: [buy_a])
    engine._tick(["AAPL"])

    # Tick 2: BUY 5 @ 110 on 2026-05-07 (second lot).
    day_idx["i"] = 1
    buy_b = Order(
        ticker="AAPL",
        direction=Direction.BUY,
        shares=5.0,
        price=110.0,
        estimated_value=550.0,
        context=_fake_context("EntrySig"),
    )
    monkeypatch.setattr(engine._order_sizer, "size_buys", lambda *a, **kw: [buy_b])
    engine._tick(["AAPL"])

    # Tick 3: partial SELL of 7 @ 120 on 2026-05-08 — consumes 7 of the first lot.
    day_idx["i"] = 2
    sell = Order(
        ticker="AAPL",
        direction=Direction.SELL,
        shares=7.0,
        price=120.0,
        estimated_value=840.0,
        context=_fake_context("StopLoss"),
    )
    monkeypatch.setattr(engine._order_sizer, "size_sells", lambda *a, **kw: [sell])
    monkeypatch.setattr(engine._order_sizer, "size_buys", lambda *a, **kw: [])
    try:
        engine._tick(["AAPL"])
    finally:
        engine.close()

    assert log_path.exists()
    trades = read_trades(log_path)
    assert len(trades) == 3, f"expected 3 rows (BUY, BUY, SELL), got {trades}"

    assert trades[0].direction == Direction.BUY
    assert trades[0].date == date(2026, 5, 6)
    assert trades[0].shares == pytest.approx(10.0)
    assert trades[0].price == pytest.approx(100.0)
    assert trades[0].cost_basis is None
    assert trades[0].purchase_date == date(2026, 5, 6)

    assert trades[1].direction == Direction.BUY
    assert trades[1].date == date(2026, 5, 7)
    assert trades[1].shares == pytest.approx(5.0)
    assert trades[1].price == pytest.approx(110.0)
    assert trades[1].purchase_date == date(2026, 5, 7)

    assert trades[2].direction == Direction.SELL
    assert trades[2].date == date(2026, 5, 8)
    assert trades[2].shares == pytest.approx(7.0)
    assert trades[2].price == pytest.approx(120.0)
    assert trades[2].holding_period is not None
    assert trades[2].holding_period.value == "short-term"
    # All consumed lots came from the first BUY on 2026-05-06.
    assert trades[2].purchase_date == date(2026, 5, 6)
    assert trades[2].cost_basis == pytest.approx(100.0)


def test_live_engine_writes_trade_log_for_mixed_lot_sell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_provider: ProviderFactory
) -> None:
    """A SELL spanning two lots with different purchase dates writes
    ``'various'`` in the purchase_date column and a share-weighted basis.
    Both lots are < 365 days old, so the entire sell is one ST bucket row.
    """
    state_path = tmp_path / "portfolio.state.yaml"
    log_path = state_path.with_suffix(state_path.suffix + ".trades.csv")

    save_atomic(
        LiveState(
            available_cash=0.0,
            cash_infusion_next_date=None,
            high_water_marks={"AAPL": 130.0},
            peak_equity=2000.0,
            lots={
                "AAPL": [
                    PositionLot(shares=10.0, purchase_date=date(2026, 1, 5), cost_basis=80.0),
                    PositionLot(shares=10.0, purchase_date=date(2026, 3, 5), cost_basis=120.0),
                ]
            },
        ),
        state_path,
    )

    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=20.0, cost_basis=100.0)],
        available_cash=0.0,
    )
    provider = make_provider({"AAPL": [130.0]}, [date(2026, 5, 7)])

    engine = LiveEngine(
        portfolio=portfolio,
        allocator=Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1),
        order_sizer=OrderSizer(),
        provider=provider,
        state_path=state_path,
    )

    # Sell 15 shares: consumes all 10 of lot1 (basis 80) + 5 of lot2 (basis 120).
    # Weighted avg basis = (10*80 + 5*120) / 15 = 1400 / 15 ~= 93.333.
    sell = Order(
        ticker="AAPL",
        direction=Direction.SELL,
        shares=15.0,
        price=130.0,
        estimated_value=1950.0,
        context=_fake_context("StopLoss"),
    )
    monkeypatch.setattr(engine._order_sizer, "size_sells", lambda *a, **kw: [sell])
    monkeypatch.setattr(engine._order_sizer, "size_buys", lambda *a, **kw: [])

    try:
        engine._tick(["AAPL"])
    finally:
        engine.close()

    trades = read_trades(log_path)
    sells = [t for t in trades if t.direction == Direction.SELL]
    assert len(sells) == 1
    sell_row = sells[0]
    assert sell_row.holding_period is not None
    assert sell_row.holding_period.value == "short-term"
    assert sell_row.purchase_date == "various"
    assert sell_row.cost_basis == pytest.approx(1400.0 / 15.0)
    assert sell_row.shares == pytest.approx(15.0)


def test_sell_spanning_st_and_lt_writes_two_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_provider: ProviderFactory
) -> None:
    """A single SELL consuming both old (LT) and recent (ST) lots writes
    one ST bucket row and one LT bucket row, each with the bucket's own
    purchase_date and cost_basis (not the order's blended basis)."""
    state_path = tmp_path / "portfolio.state.yaml"
    log_path = state_path.with_suffix(state_path.suffix + ".trades.csv")

    # Seed: 30 LT shares (purchased >365 days ago at $10) + 20 ST shares
    # (purchased <365 days ago at $20). Today (per the live engine) =
    # 2026-05-08 — set in the provider date below and via real ``date.today()``.
    save_atomic(
        LiveState(
            available_cash=0.0,
            cash_infusion_next_date=None,
            high_water_marks={"AAPL": 30.0},
            peak_equity=10000.0,
            lots={
                "AAPL": [
                    PositionLot(shares=30.0, purchase_date=date(2024, 1, 1), cost_basis=10.0),
                    PositionLot(shares=20.0, purchase_date=date(2026, 4, 1), cost_basis=20.0),
                ],
            },
        ),
        state_path,
    )

    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=50.0, cost_basis=14.0)],
        available_cash=0.0,
    )
    provider = make_provider({"AAPL": [30.0]}, [date(2026, 5, 8)])

    engine = LiveEngine(
        portfolio=portfolio,
        allocator=Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1),
        order_sizer=OrderSizer(),
        provider=provider,
        state_path=state_path,
    )

    # Pin today to 2026-05-08 so the ST/LT classification (>=365 days) is
    # deterministic regardless of when the test runs.
    class _FakeDate:
        @staticmethod
        def today() -> date:
            return date(2026, 5, 8)

    monkeypatch.setattr("midas.live.date", _FakeDate)

    # Sell all 50 shares at $30 — FIFO consumes 30 LT shares (lot 1) then
    # 20 ST shares (lot 2).
    sell = Order(
        ticker="AAPL",
        direction=Direction.SELL,
        shares=50.0,
        price=30.0,
        estimated_value=1500.0,
        context=_fake_context(source="StopLoss"),
    )
    monkeypatch.setattr(engine._order_sizer, "size_sells", lambda *a, **kw: [sell])
    monkeypatch.setattr(engine._order_sizer, "size_buys", lambda *a, **kw: [])

    try:
        engine._tick(["AAPL"])
    finally:
        engine.close()

    trades = read_trades(log_path)
    sells = [t for t in trades if t.direction == Direction.SELL]
    assert len(sells) == 2, f"expected ST + LT bucket rows, got {sells}"
    by_period = {s.holding_period.value: s for s in sells if s.holding_period is not None}
    assert "long-term" in by_period and "short-term" in by_period
    assert by_period["long-term"].purchase_date == date(2024, 1, 1)
    assert by_period["long-term"].cost_basis == pytest.approx(10.0)
    assert by_period["long-term"].shares == pytest.approx(30.0)
    assert by_period["short-term"].purchase_date == date(2026, 4, 1)
    assert by_period["short-term"].cost_basis == pytest.approx(20.0)
    assert by_period["short-term"].shares == pytest.approx(20.0)
