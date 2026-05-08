"""Tests for the backtest engine."""

import csv as csv_mod
import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import make_price_series, ph

from midas.allocator import Allocator
from midas.backtest import (
    BacktestEngine,
    ExecutionMode,
    _SimState,
)
from midas.metrics import (
    TRADING_DAYS_PER_YEAR,
    compute_annualized_return,
    compute_cagr,
    compute_max_drawdown,
    compute_sharpe,
    compute_sortino,
    compute_strategy_stats,
    compute_trade_stats,
)
from midas.models import (
    AllocationConstraints,
    CashInfusion,
    Direction,
    Holding,
    HoldingPeriod,
    Order,
    OrderContext,
    PortfolioConfig,
    PositionLot,
    TradeRecord,
    TradingRestrictions,
)
from midas.order_sizer import OrderSizer
from midas.restrictions import RestrictionTracker
from midas.results import BacktestResult, write_backtest_results
from midas.strategies.gap_down_recovery import GapDownRecovery
from midas.strategies.mean_reversion import MeanReversion
from midas.strategies.profit_taking import ProfitTaking
from midas.strategies.stop_loss import StopLoss
from midas.strategies.trailing_stop import TrailingStop


def _build_engine(
    entries=None,
    exit_rules=None,
    constraints=None,
    n_tickers=1,
    **kwargs,
):
    """Helper to build a BacktestEngine with the new allocator + order_sizer system."""
    entries = entries or []
    constraints = constraints or AllocationConstraints(
        min_buy_delta=0.01,
        max_position_pct=0.95,
    )
    allocator = Allocator(entries, constraints, n_tickers)
    order_sizer = OrderSizer()
    return BacktestEngine(
        allocator=allocator,
        order_sizer=order_sizer,
        exit_rules=exit_rules,
        constraints=constraints,
        **kwargs,
    )


def _make_backtest_data() -> tuple[PortfolioConfig, dict[str, pd.DataFrame]]:
    """Create a portfolio and price data that will generate trades."""
    portfolio = PortfolioConfig(
        holdings=[
            Holding(ticker="AAPL", shares=10, cost_basis=90.0),
        ],
        available_cash=2000.0,
    )

    # Price drops then recovers — triggers mean reversion buy
    returns = (
        [0.0] * 20  # flat at 100
        + [-0.008] * 20  # drop ~15%
        + [0.01] * 30  # recover
        + [0.0] * 30  # flat
    )
    prices = make_price_series(date(2024, 1, 2), 100, 100.0, returns, name="AAPL")
    return portfolio, {"AAPL": prices}


def test_backtest_produces_trades() -> None:
    portfolio, price_data = _make_backtest_data()
    mr = MeanReversion(window=20, threshold=0.05)
    engine = _build_engine(
        entries=[(mr, 1.0)],
        enable_split=False,
    )

    start = min(price_data["AAPL"].index)
    end = max(price_data["AAPL"].index)
    result = engine.run(portfolio, price_data, start, end)

    assert result.starting_value > 0
    assert result.final_value > 0
    assert len(result.trades) > 0
    for t in result.trades:
        assert t.ticker == "AAPL"


def test_backtest_runs_real_ohlcv_through_strategy() -> None:
    """End-to-end proof the OHLCV pipeline reaches the strategy via the engine.

    The conftest helper produces degenerate bars (O=H=L=C, flat volume), so a
    backtest using GapDownRecovery on that fixture can never fire — it needs a
    real open-vs-prior-close gap. Here we hand-build a frame whose CLOSE
    sequence shows no qualifying drop (close-to-close moves are <1%), but a
    single bar opens 5% below the prior close and recovers most of the way.
    If the engine ever silently degraded to close-only data, the buy below
    would not happen.
    """
    n = 30
    dates = [date(2024, 1, 2) + pd.Timedelta(days=i) for i in range(n)]
    dates = [d.date() for d in pd.to_datetime(dates)]
    closes = np.full(n, 100.0)
    opens = np.full(n, 100.0)
    highs = np.full(n, 100.0)
    lows = np.full(n, 100.0)
    volumes = np.full(n, 1_000_000.0)
    gap_day = 20
    # Real intraday gap-and-recover: open 5% below prior close, low $94,
    # close at $99 (close-to-close drop of just 1%, well under any
    # close-only "gap" proxy threshold).
    opens[gap_day] = 95.0
    lows[gap_day] = 94.0
    highs[gap_day] = 99.5
    closes[gap_day] = 99.0
    frame = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=dates,
    )
    frame.index.name = "date"

    # Start with a small seed position so the engine activates the ticker
    # in state.positions (zero-share holdings are skipped by _init_positions).
    # On flat days the score is 0 and the allocator holds the current weight,
    # so no trades fire. On the gap day the 0.8 score pushes the target to
    # the position cap, generating a buy delta.
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="GAPCO", shares=5, cost_basis=100.0)],
        available_cash=10_000.0,
    )
    engine = _build_engine(
        entries=[(GapDownRecovery(gap_threshold=0.03), 1.0)],
        enable_split=False,
    )

    result = engine.run(portfolio, {"GAPCO": frame}, dates[0], dates[-1])

    # Default engine mode is ``next_open``: the strategy scores the gap at
    # day ``gap_day`` close but the buy fills at day ``gap_day + 1``'s open.
    fill_day = dates[gap_day + 1]
    buys_on_fill_day = [t for t in result.trades if t.direction == Direction.BUY and t.date == fill_day]
    assert buys_on_fill_day, (
        "GapDownRecovery should fire on the real-OHLCV gap day (executing next "
        "open); if this fails, the engine is not threading open/high/low through "
        "to the strategy, or the execution-lag pipeline dropped the decision."
    )
    # And nothing fires on the flat days — proves the strategy is reading the
    # specific bar's OHLC, not e.g. a constant from the precompute fixture.
    other_buys = [t for t in result.trades if t.direction == Direction.BUY and t.date != fill_day]
    assert not other_buys, f"unexpected buys outside the gap-fill day: {other_buys}"


# --- execution lag (#46) ---
#
# Honest execution defers today's decision to tomorrow's open (or close)
# instead of filling on the bar the signal reads — eliminating the
# close-on-close lookahead that inflates realistic backtest returns.


def _gap_fill_frame(
    gap_day: int,
    fill_open: float,
    fill_close: float,
    *,
    n: int = 30,
) -> pd.DataFrame:
    """OHLCV frame with a real gap at ``gap_day`` and a pinned next bar.

    The fill bar's open and close are set independently so tests can
    distinguish ``next_open`` and ``next_close`` execution modes by the
    trade price they record.
    """
    dates = [d.date() for d in pd.bdate_range(start=date(2024, 1, 2), periods=n)]
    opens = np.full(n, 100.0)
    highs = np.full(n, 100.0)
    lows = np.full(n, 100.0)
    closes = np.full(n, 100.0)
    volumes = np.full(n, 1_000_000.0)
    # Real intraday gap: open 5% below prior close, recovers to $99.
    opens[gap_day] = 95.0
    lows[gap_day] = 94.0
    highs[gap_day] = 99.5
    closes[gap_day] = 99.0
    # Pin the fill bar so the trade price uniquely identifies which
    # bar (and which field) the engine filled against.
    opens[gap_day + 1] = fill_open
    highs[gap_day + 1] = max(fill_open, fill_close)
    lows[gap_day + 1] = min(fill_open, fill_close)
    closes[gap_day + 1] = fill_close
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=dates,
    )


def _lag_test_engine(execution_mode: ExecutionMode) -> BacktestEngine:
    constraints = AllocationConstraints(min_buy_delta=0.01, max_position_pct=0.95)
    allocator = Allocator([(GapDownRecovery(gap_threshold=0.03), 1.0)], constraints, n_tickers=1)
    return BacktestEngine(
        allocator=allocator,
        order_sizer=OrderSizer(default_slippage=0.0),  # clean numbers for price pinning
        exit_rules=[],
        constraints=constraints,
        enable_split=False,
        execution_mode=execution_mode,
    )


def _lag_test_portfolio() -> PortfolioConfig:
    return PortfolioConfig(
        holdings=[Holding(ticker="GAPCO", shares=5, cost_basis=100.0)],
        available_cash=10_000.0,
    )


def test_execution_lag_next_open_fills_at_next_bar_open() -> None:
    """Signal at day T close fires at day T+1's *open* under ``next_open``."""
    gap_day = 20
    frame = _gap_fill_frame(gap_day=gap_day, fill_open=105.0, fill_close=107.0)
    engine = _lag_test_engine("next_open")

    result = engine.run(_lag_test_portfolio(), {"GAPCO": frame}, frame.index[0], frame.index[-1])

    buys = [t for t in result.trades if t.direction == Direction.BUY]
    assert len(buys) == 1, f"expected a single buy from the gap signal, got {len(buys)}: {buys}"
    assert buys[0].date == frame.index[gap_day + 1]
    # Fill price is the next bar's open, not its close and not today's close.
    assert buys[0].price == 105.0, f"expected fill at next bar open ($105), got ${buys[0].price}"


def test_execution_lag_next_close_fills_at_next_bar_close() -> None:
    """Signal at day T close fires at day T+1's *close* under ``next_close``."""
    gap_day = 20
    frame = _gap_fill_frame(gap_day=gap_day, fill_open=105.0, fill_close=107.0)
    engine = _lag_test_engine("next_close")

    result = engine.run(_lag_test_portfolio(), {"GAPCO": frame}, frame.index[0], frame.index[-1])

    buys = [t for t in result.trades if t.direction == Direction.BUY]
    assert len(buys) == 1
    assert buys[0].date == frame.index[gap_day + 1]
    assert buys[0].price == 107.0, f"expected fill at next bar close ($107), got ${buys[0].price}"


def test_execution_lag_close_mode_preserves_legacy_same_day_fill() -> None:
    """Legacy ``close`` mode fills on the decision bar itself (optimistic)."""
    gap_day = 20
    frame = _gap_fill_frame(gap_day=gap_day, fill_open=105.0, fill_close=107.0)
    engine = _lag_test_engine("close")

    result = engine.run(_lag_test_portfolio(), {"GAPCO": frame}, frame.index[0], frame.index[-1])

    buys = [t for t in result.trades if t.direction == Direction.BUY]
    assert len(buys) == 1
    assert buys[0].date == frame.index[gap_day]
    # Legacy mode fills at the decision bar's close ($99, the gap recovery).
    assert buys[0].price == 99.0


def test_execution_lag_no_drift_warning_for_held_tickers() -> None:
    """Under lag, pure-hold tickers must not trip ``size_buys`` warnings.

    The allocator's Option-A rule marks a ticker with no positive
    entry-signal contributions as "hold at decision-day current weight".
    With execution lag, prices drift overnight and the ticker's *actual*
    T+1 weight diverges from the stored target, producing a fake delta.
    ``size_buys`` used to log a "no positive contributions" suppression
    warning for every held ticker every tick. Engine should now rewrite
    held-ticker targets to T+1 current weight so the delta collapses.
    """
    # Flat series — GapDownRecovery never fires, every tick is pure hold.
    n = 30
    dates = [d.date() for d in pd.bdate_range(start=date(2024, 1, 2), periods=n)]
    closes = 100.0 + np.arange(n, dtype=float)  # slow drift, $100 -> $129
    opens = closes - 0.5
    highs = closes + 0.5
    lows = closes - 1.0
    volumes = np.full(n, 1_000_000.0)
    frame = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=dates,
    )

    log_messages: list[str] = []
    engine = BacktestEngine(
        allocator=Allocator(
            [(GapDownRecovery(gap_threshold=0.03), 1.0)],
            AllocationConstraints(min_buy_delta=0.01, max_position_pct=0.95),
            n_tickers=1,
        ),
        order_sizer=OrderSizer(default_slippage=0.0),
        exit_rules=[],
        constraints=AllocationConstraints(min_buy_delta=0.01, max_position_pct=0.95),
        enable_split=False,
        execution_mode="next_open",
        log_fn=log_messages.append,
    )

    engine.run(_lag_test_portfolio(), {"GAPCO": frame}, dates[0], dates[-1])

    suppression_msgs = [m for m in log_messages if "Suppressing buy" in m or "no positive entry-signal" in m]
    assert suppression_msgs == [], f"held-ticker drift produced spurious warnings: {suppression_msgs}"


def test_execution_lag_last_day_decision_never_executes() -> None:
    """A signal on the final bar has no next bar to fill against — dropped.

    Under lag=1 semantics an order placed after the final session cannot
    fill inside the window. Otherwise the backtest would implicitly gain
    one free day of hindsight at its right edge.
    """
    # 22 bars, gap on bar 21 (the last). Under ``next_open`` the decision
    # is stored as pending but the loop ends — no trade should land.
    n = 22
    dates = [d.date() for d in pd.bdate_range(start=date(2024, 1, 2), periods=n)]
    opens = np.full(n, 100.0)
    highs = np.full(n, 100.0)
    lows = np.full(n, 100.0)
    closes = np.full(n, 100.0)
    volumes = np.full(n, 1_000_000.0)
    opens[n - 1] = 95.0
    lows[n - 1] = 94.0
    highs[n - 1] = 99.5
    closes[n - 1] = 99.0
    frame = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=dates,
    )

    engine = _lag_test_engine("next_open")
    result = engine.run(_lag_test_portfolio(), {"GAPCO": frame}, dates[0], dates[-1])

    buys = [t for t in result.trades if t.direction == Direction.BUY]
    assert buys == [], f"final-bar signal must not execute under lag=1, got: {buys}"


def test_execution_lag_with_exit_rule_stop_loss() -> None:
    """Exit rule (StopLoss) under lag fills at next bar's open, not decision close.

    The stop fires on the decision bar when price drops below cost basis by
    the threshold. Under ``next_open``, the sell should execute at the *next*
    bar's open price, and the clamp_attribution should survive drift
    neutralization (i.e. the sell is not suppressed).
    """
    n = 30
    dates = [d.date() for d in pd.bdate_range(start=date(2024, 1, 2), periods=n)]
    # Flat at $100 until day 20 where price drops to $90 (10% loss).
    closes = np.full(n, 100.0)
    opens = np.full(n, 100.0)
    highs = np.full(n, 100.0)
    lows = np.full(n, 100.0)
    volumes = np.full(n, 1_000_000.0)
    drop_day = 20
    opens[drop_day] = 92.0
    highs[drop_day] = 93.0
    lows[drop_day] = 88.0
    closes[drop_day] = 90.0
    # Next bar (fill bar) has distinct open/close for price verification.
    opens[drop_day + 1] = 89.0
    highs[drop_day + 1] = 91.0
    lows[drop_day + 1] = 88.0
    closes[drop_day + 1] = 91.0
    frame = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=dates,
    )

    constraints = AllocationConstraints(min_buy_delta=0.01, max_position_pct=0.95)
    allocator = Allocator(
        [(GapDownRecovery(gap_threshold=0.03), 1.0)],
        constraints,
        n_tickers=1,
    )
    engine = BacktestEngine(
        allocator=allocator,
        order_sizer=OrderSizer(default_slippage=0.0),
        exit_rules=[StopLoss(loss_threshold=0.05)],
        constraints=constraints,
        enable_split=False,
        execution_mode="next_open",
    )
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="GAPCO", shares=5, cost_basis=100.0)],
        available_cash=10_000.0,
    )

    result = engine.run(portfolio, {"GAPCO": frame}, dates[0], dates[-1])

    sells = [t for t in result.trades if t.direction == Direction.SELL]
    assert len(sells) >= 1, f"StopLoss should trigger a sell, got trades: {result.trades}"
    # Under next_open, the sell fills at the fill bar's open ($89), not the
    # decision bar's close ($90).
    assert sells[0].date == dates[drop_day + 1]
    assert sells[0].price == 89.0, f"expected fill at next open ($89), got ${sells[0].price}"


def test_backtest_with_split() -> None:
    portfolio, price_data = _make_backtest_data()
    mr = MeanReversion(window=20, threshold=0.05)
    engine = _build_engine(
        entries=[(mr, 1.0)],
        train_pct=0.7,
    )

    start = min(price_data["AAPL"].index)
    end = max(price_data["AAPL"].index)
    result = engine.run(portfolio, price_data, start, end)

    assert result.split_date is not None
    assert start < result.split_date < end


def test_backtest_results_output(tmp_path: Path) -> None:
    portfolio, price_data = _make_backtest_data()
    mr = MeanReversion(window=20, threshold=0.05)
    engine = _build_engine(
        entries=[(mr, 1.0)],
        enable_split=True,
    )

    start = min(price_data["AAPL"].index)
    end = max(price_data["AAPL"].index)
    result = engine.run(portfolio, price_data, start, end)

    out_dir = tmp_path / "results"
    write_backtest_results(result, out_dir)

    assert out_dir.is_dir()
    for name in ("trades.csv", "equity_curve.csv", "summary.json", "strategy_breakdown.csv"):
        assert (out_dir / name).exists()

    with open(out_dir / "trades.csv") as f:
        reader = csv_mod.DictReader(f)
        rows = list(reader)
    assert len(rows) > 0
    assert set(reader.fieldnames) == {  # type: ignore[arg-type]
        "date",
        "ticker",
        "direction",
        "shares",
        "price",
        "strategy",
        "holding_period",
        "purchase_date",
        "cost_basis",
        "realized_pnl",
        "return_pct",
    }
    for row in rows:
        assert "$" not in row["price"]
        assert "%" not in row.get("return_pct", "")
        assert row["ticker"] == "AAPL"

    with open(out_dir / "equity_curve.csv") as f:
        reader = csv_mod.DictReader(f)
        curve_rows = list(reader)
    assert len(curve_rows) > 0
    for row in curve_rows:
        assert float(row["nav"]) > 0
        assert float(row["drawdown"]) >= 0

    with open(out_dir / "summary.json") as f:
        summary = json.load(f)
    assert isinstance(summary["starting_value"], (int, float))
    assert "sharpe_ratio" in summary
    assert "split" in summary

    with open(out_dir / "strategy_breakdown.csv") as f:
        strat_rows = list(csv_mod.DictReader(f))
    assert len(strat_rows) > 0


def test_write_backtest_results_rejects_existing_file(tmp_path: Path) -> None:
    existing_file = tmp_path / "results"
    existing_file.write_text("oops")

    result = BacktestResult(
        trades=[],
        final_value=0,
        starting_value=0,
        buy_and_hold_value=0,
        train_trades=[],
        test_trades=[],
        train_return=0,
        test_return=0,
        train_bh_return=0,
        test_bh_return=0,
        split_date=None,
        twr=0,
        equity_curve=[],
        total_days=0,
        train_days=0,
        test_days=0,
        cagr=0,
        max_drawdown=0,
        sharpe_ratio=0,
        sortino_ratio=0,
        win_rate=0,
        profit_factor=0,
        avg_win=0,
        avg_loss=0,
        efficiency_ratio=0,
        strategy_stats=[],
        unrealized_pnl=0,
        unrealized_pnl_by_ticker={},
        basis_per_sell=[],
    )

    with pytest.raises(FileExistsError, match="existing file"):
        write_backtest_results(result, existing_file)


def test_backtest_cost_basis_uses_start_price() -> None:
    """Cost basis should be the start-date price, not the config value."""
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="TEST", shares=10, cost_basis=50.0)],
        available_cash=0.0,
    )
    # Flat at 100 for 50 days, then rise
    returns = [0.0] * 50 + [0.005] * 50
    prices = make_price_series(date(2024, 1, 2), 100, 100.0, returns, name="TEST")

    pt = ProfitTaking(gain_threshold=0.20)
    engine = _build_engine(
        exit_rules=[pt],
        enable_split=False,
    )

    start = min(prices.index)
    end = max(prices.index)
    result = engine.run(portfolio, {"TEST": prices}, start, end)

    sells = [t for t in result.trades if t.direction == Direction.SELL]
    if sells:
        assert sells[0].date > start


def test_backtest_deferred_ticker() -> None:
    """Tickers whose data starts after the backtest start date should
    begin with 0 shares and activate when data first appears."""
    portfolio = PortfolioConfig(
        holdings=[
            Holding(ticker="EARLY", shares=10),
            Holding(ticker="LATE", shares=5),
        ],
        available_cash=1000.0,
    )
    early = make_price_series(date(2024, 1, 2), 100, 100.0, name="EARLY")
    late = make_price_series(date(2024, 3, 20), 50, 80.0, name="LATE")

    mr = MeanReversion(window=10, threshold=0.05)
    log_messages: list[str] = []
    engine = _build_engine(
        entries=[(mr, 1.0)],
        n_tickers=2,
        enable_split=False,
        log_fn=log_messages.append,
    )

    start = date(2024, 1, 2)
    end = date(2024, 6, 30)
    result = engine.run(portfolio, {"EARLY": early, "LATE": late}, start, end)

    deferred_msgs = [m for m in log_messages if "deferred" in m]
    activated_msgs = [m for m in log_messages if "activated" in m]
    assert len(deferred_msgs) == 1
    assert "LATE" in deferred_msgs[0]
    assert len(activated_msgs) == 1
    assert "LATE" in activated_msgs[0]

    assert result.starting_value == 2000.0 + 5 * 80.0


def test_backtest_consumes_warmup_prefix() -> None:
    """Bars before ``start`` should prime strategy signals from day one.

    Without a warmup prefix, a `window=20` strategy spends the first 20
    days of the simulation in warmup and emits no signals. With a warmup
    prefix fetched ahead of ``start``, the allocator can produce valid
    conviction scores on the very first simulation day.
    """
    # 60 bars total: bars 0-19 are warmup, bars 20-59 are the sim window.
    # Flat at 100 through bar 30, then drop — so the drop lands inside
    # the sim window but needs the warmup bars to compute its 20-day MA.
    returns = [0.0] * 30 + [-0.02] * 10 + [0.0] * 20
    prices = make_price_series(date(2024, 1, 2), 60, 100.0, returns, name="AAPL")
    trading_days = list(prices.index)
    sim_start = trading_days[20]
    sim_end = trading_days[-1]

    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=10, cost_basis=100.0)],
        available_cash=5000.0,
    )

    mr = MeanReversion(window=20, threshold=0.05)
    engine = _build_engine(
        entries=[(mr, 1.0)],
        enable_split=False,
    )
    result = engine.run(portfolio, {"AAPL": prices}, sim_start, sim_end)

    # With warmup consumed, the drop triggers a mean-reversion buy inside
    # the sim window. Without warmup, the strategy would still be in its
    # 20-day cold start when the drop started and miss it entirely.
    buys = [t for t in result.trades if t.direction == Direction.BUY]
    assert buys, "Expected at least one buy — warmup prefix was not consumed"
    assert all(sim_start <= t.date <= sim_end for t in buys)


def test_backtest_logs_insufficient_warmup() -> None:
    """A ticker with less history than a strategy needs should log a warning."""
    # Only 25 bars total, starting exactly at the sim start — no prefix.
    prices = make_price_series(date(2024, 1, 2), 25, 100.0, name="AAPL")
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=10, cost_basis=100.0)],
        available_cash=1000.0,
    )

    # window=50 → warmup_period=50, but only ~25 bars are available.
    mr = MeanReversion(window=50, threshold=0.05)
    log_messages: list[str] = []
    engine = _build_engine(
        entries=[(mr, 1.0)],
        enable_split=False,
        log_fn=log_messages.append,
    )
    engine.run(portfolio, {"AAPL": prices}, min(prices.index), max(prices.index))

    warmup_msgs = [m for m in log_messages if "warmup" in m.lower()]
    assert warmup_msgs, f"Expected warmup warning, got: {log_messages}"
    assert "AAPL" in warmup_msgs[0]


def test_backtest_excluded_ticker() -> None:
    """Tickers with no data in the range should be excluded and logged."""
    portfolio = PortfolioConfig(
        holdings=[
            Holding(ticker="REAL", shares=10),
            Holding(ticker="GHOST", shares=5),
        ],
        available_cash=1000.0,
    )
    real = make_price_series(date(2024, 1, 2), 100, 100.0, name="REAL")

    mr = MeanReversion(window=10, threshold=0.05)
    log_messages: list[str] = []
    engine = _build_engine(
        entries=[(mr, 1.0)],
        enable_split=False,
        log_fn=log_messages.append,
    )

    start = date(2024, 1, 2)
    end = date(2024, 6, 30)
    result = engine.run(portfolio, {"REAL": real}, start, end)

    excluded_msgs = [m for m in log_messages if "excluded" in m]
    assert len(excluded_msgs) == 1
    assert "GHOST" in excluded_msgs[0]

    assert result.starting_value == 1000.0 + 10 * 100.0


def test_backtest_cash_infusion_credits_cash() -> None:
    """Cash infusions should be credited on their next_date during backtest."""
    prices = make_price_series(date(2024, 1, 2), 100, 100.0, name="AAPL")
    trading_days = list(prices.index)
    # Pick an infusion date that falls on a trading day in the middle
    infusion_date = trading_days[50]

    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=10)],
        available_cash=1000.0,
        cash_infusion=CashInfusion(
            amount=2000.0,
            next_date=infusion_date,
        ),
    )

    mr = MeanReversion(window=20, threshold=0.05)
    engine = _build_engine(
        entries=[(mr, 1.0)],
        enable_split=False,
    )

    start = trading_days[0]
    end = trading_days[-1]
    result = engine.run(portfolio, {"AAPL": prices}, start, end)

    # Final value should reflect the 2000 infusion (starting cash 1000 + infusion 2000 = 3000 base)
    # Even with no trades, cash portion should include the infusion
    assert result.final_value >= 3000.0


def test_backtest_recurring_cash_infusion() -> None:
    """Recurring cash infusions should credit multiple times."""
    prices = make_price_series(date(2024, 1, 2), 100, 100.0, name="AAPL")
    trading_days = list(prices.index)
    # Start infusion early so multiple biweekly infusions land in the window
    infusion_date = trading_days[5]

    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=10)],
        available_cash=500.0,
        cash_infusion=CashInfusion(
            amount=1000.0,
            next_date=infusion_date,
            frequency="biweekly",
        ),
    )

    mr = MeanReversion(window=20, threshold=0.05)
    engine = _build_engine(
        entries=[(mr, 1.0)],
        enable_split=False,
    )

    start = trading_days[0]
    end = trading_days[-1]
    result = engine.run(portfolio, {"AAPL": prices}, start, end)

    # With ~100 trading days (~140 calendar days), biweekly = ~10 infusions of $1000
    # Final value must exceed starting holdings + multiple infusions
    assert result.final_value > 500.0 + 10 * 100.0 + 5000.0


# ---------------------------------------------------------------------------
# Unit tests for the metric helpers
# ---------------------------------------------------------------------------


def _curve(values: list[float], start: date = date(2024, 1, 1)) -> list[tuple[date, float]]:
    """Build an equity curve from a list of daily values."""
    return [(date.fromordinal(start.toordinal() + i), v) for i, v in enumerate(values)]


def _sell(d: date, ticker: str, shares: float, price: float, strategy: str = "S1") -> TradeRecord:
    return TradeRecord(
        date=d, ticker=ticker, direction=Direction.SELL, shares=shares, price=price, strategy_name=strategy
    )


def _buy(d: date, ticker: str, shares: float, price: float, strategy: str = "S1") -> TradeRecord:
    return TradeRecord(
        date=d, ticker=ticker, direction=Direction.BUY, shares=shares, price=price, strategy_name=strategy
    )


# --- compute_cagr ---


def test_cagr_one_year_double() -> None:
    # 100 → 200 over ~one year ≈ 100% CAGR.
    # `days` is an int and the function divides by 365.25, so use a slightly
    # larger window to clear the rounding gap.
    cagr = compute_cagr(100.0, 200.0, 366)
    assert math.isclose(cagr, 1.0, abs_tol=1e-2)


def test_cagr_two_year_quadruple() -> None:
    # 100 → 400 over 2 years = 100% CAGR
    cagr = compute_cagr(100.0, 400.0, 731)
    assert math.isclose(cagr, 1.0, abs_tol=1e-2)


def test_cagr_zero_days_returns_zero() -> None:
    assert compute_cagr(100.0, 200.0, 0) == 0.0


def test_cagr_invalid_starting_returns_zero() -> None:
    assert compute_cagr(0.0, 200.0, 365) == 0.0
    assert compute_cagr(-10.0, 200.0, 365) == 0.0


def test_cagr_loss() -> None:
    # 100 → 50 over ~1 year = -50% CAGR
    cagr = compute_cagr(100.0, 50.0, 366)
    assert math.isclose(cagr, -0.5, abs_tol=1e-2)


# --- compute_annualized_return ---


def test_annualized_return_one_year_pass_through() -> None:
    # A ~21% cumulative return over ~1 year annualizes to ~21%.
    # Anchors the identity: annualized(r, 365.25 days) == r.
    ann = compute_annualized_return(0.21, 366)
    assert math.isclose(ann, 0.21, abs_tol=1e-2)


def test_annualized_return_compounds_short_window() -> None:
    # 10% over ~half a year compounds to ~21% annualized (1.10^2 - 1).
    # This is the key guard against the PR's most likely regression —
    # forgetting to exponentiate and just dividing.
    ann = compute_annualized_return(0.10, 183)
    assert math.isclose(ann, 0.21, abs_tol=2e-2)


def test_annualized_return_multi_year_takes_root() -> None:
    # 100% cumulative over 2 years ≈ 41.4% annualized (sqrt(2) - 1).
    # Guards the inverse direction — the exponent is (1/years), not years.
    ann = compute_annualized_return(1.0, 731)
    assert math.isclose(ann, 2**0.5 - 1.0, abs_tol=1e-2)


def test_annualized_return_zero_days_returns_zero() -> None:
    # Matches compute_cagr's sentinel so aggregates mixing the two don't
    # get poisoned by inf/NaN.
    assert compute_annualized_return(0.25, 0) == 0.0
    assert compute_annualized_return(0.25, -5) == 0.0


def test_annualized_return_total_loss_returns_minus_one() -> None:
    # Growth factor ≤ 0 (cumulative loss wipes out starting value). No
    # sensible annualization exists, so the function returns -1.0 rather
    # than raising or returning NaN.
    assert compute_annualized_return(-1.0, 365) == -1.0
    assert compute_annualized_return(-1.5, 365) == -1.0


def test_annualized_return_survivable_loss() -> None:
    # 100 → 50 over ~1 year annualizes to ~-50%.
    ann = compute_annualized_return(-0.5, 366)
    assert math.isclose(ann, -0.5, abs_tol=1e-2)


def test_annualized_return_extrapolates_short_windows_aggressively() -> None:
    # 30-day 10% → ~219% annualized. Documents the known amplification;
    # if someone "fixes" this by clamping, this test will fail and prompt
    # a discussion.
    ann = compute_annualized_return(0.10, 30)
    assert ann > 2.0


# --- compute_max_drawdown ---


def test_max_drawdown_monotone_up_is_zero() -> None:
    assert compute_max_drawdown(_curve([100, 110, 120, 130])) == 0.0


def test_max_drawdown_monotone_down() -> None:
    # 100 → 50 = 50% drawdown
    assert compute_max_drawdown(_curve([100, 90, 75, 50])) == 0.5


def test_max_drawdown_peak_then_recover() -> None:
    # peak 120 → trough 60 = 50%
    dd = compute_max_drawdown(_curve([100, 120, 90, 60, 80, 110]))
    assert math.isclose(dd, 0.5, abs_tol=1e-9)


def test_max_drawdown_single_point() -> None:
    assert compute_max_drawdown(_curve([100])) == 0.0


def test_max_drawdown_empty() -> None:
    assert compute_max_drawdown([]) == 0.0


# --- compute_sharpe ---


def test_sharpe_flat_curve_is_zero() -> None:
    assert compute_sharpe(_curve([100, 100, 100, 100])) == 0.0


def test_sharpe_too_few_points() -> None:
    assert compute_sharpe(_curve([100, 101])) == 0.0
    assert compute_sharpe([]) == 0.0


def test_sharpe_positive_when_mean_positive() -> None:
    # Mix of up and down days with positive bias
    curve = _curve([100, 102, 101, 103, 102, 104, 103, 105])
    s = compute_sharpe(curve)
    assert s > 0


def test_sharpe_negative_when_mean_negative() -> None:
    curve = _curve([100, 98, 99, 97, 98, 96, 97, 95])
    assert compute_sharpe(curve) < 0


def test_sharpe_annualization_factor() -> None:
    # Returns with positive mean & nonzero std should scale by sqrt(252).
    curve = _curve([100, 101, 102, 103])  # roughly +1%/day with shrinking returns
    s = compute_sharpe(curve)
    # mean ≈ 0.0099, std small but nonzero — annualized factor present
    assert s > math.sqrt(TRADING_DAYS_PER_YEAR) * 0.5


# --- compute_sortino ---


def test_sortino_no_downside_returns_zero_not_inf() -> None:
    # All-up curve → no negative returns → undefined → 0.0 (NOT inf)
    curve = _curve([100, 101, 102, 103, 104])
    result = compute_sortino(curve)
    assert result == 0.0
    assert not math.isinf(result)


def test_sortino_flat_returns_zero() -> None:
    assert compute_sortino(_curve([100, 100, 100, 100])) == 0.0


def test_sortino_too_few_points() -> None:
    assert compute_sortino(_curve([100, 101])) == 0.0
    assert compute_sortino([]) == 0.0


def test_sortino_negative_when_mean_negative() -> None:
    curve = _curve([100, 98, 99, 97, 98, 96, 97, 95])
    assert compute_sortino(curve) < 0


def test_sortino_positive_when_mean_positive_with_some_downside() -> None:
    curve = _curve([100, 102, 101, 103, 102, 104, 103, 105])
    assert compute_sortino(curve) > 0


# --- compute_trade_stats ---


def test_trade_stats_no_sells() -> None:
    trades = [_buy(date(2024, 1, 1), "AAPL", 10, 100.0)]
    win_rate, pf, avg_win, avg_loss = compute_trade_stats(trades, [])
    assert (win_rate, pf, avg_win, avg_loss) == (0.0, 0.0, 0.0, 0.0)


def test_trade_stats_all_wins() -> None:
    trades = [
        _sell(date(2024, 1, 2), "AAPL", 10, 110.0),
        _sell(date(2024, 1, 3), "AAPL", 5, 120.0),
    ]
    basis = [100.0, 100.0]  # gains: +100, +100
    win_rate, pf, avg_win, avg_loss = compute_trade_stats(trades, basis)
    assert win_rate == 1.0
    assert math.isinf(pf)
    assert avg_win == 100.0
    assert avg_loss == 0.0


def test_trade_stats_all_losses() -> None:
    trades = [_sell(date(2024, 1, 2), "AAPL", 10, 90.0)]
    basis = [100.0]  # loss: -100
    win_rate, pf, avg_win, avg_loss = compute_trade_stats(trades, basis)
    assert win_rate == 0.0
    assert pf == 0.0
    assert avg_win == 0.0
    assert avg_loss == -100.0


def test_trade_stats_mixed() -> None:
    trades = [
        _sell(date(2024, 1, 2), "AAPL", 10, 110.0),  # +100
        _sell(date(2024, 1, 3), "AAPL", 10, 90.0),  # -100
        _sell(date(2024, 1, 4), "AAPL", 10, 120.0),  # +200
    ]
    basis = [100.0, 100.0, 100.0]
    win_rate, pf, avg_win, avg_loss = compute_trade_stats(trades, basis)
    assert math.isclose(win_rate, 2 / 3, abs_tol=1e-9)
    assert math.isclose(pf, 300.0 / 100.0, abs_tol=1e-9)
    assert math.isclose(avg_win, 150.0, abs_tol=1e-9)
    assert math.isclose(avg_loss, -100.0, abs_tol=1e-9)


def test_trade_stats_same_day_same_ticker_no_collision() -> None:
    """Two sells of the same ticker on the same day must each see their own basis."""
    day = date(2024, 1, 2)
    trades = [
        _sell(day, "AAPL", 5, 110.0, strategy="A"),  # basis 100 → +50
        _sell(day, "AAPL", 5, 110.0, strategy="B"),  # basis 80 → +150
    ]
    basis = [100.0, 80.0]
    win_rate, pf, avg_win, _avg_loss = compute_trade_stats(trades, basis)
    assert win_rate == 1.0
    # Both wins: 50 + 150 = 200; avg = 100
    assert math.isclose(avg_win, 100.0, abs_tol=1e-9)
    assert math.isinf(pf)


def test_trade_stats_breakeven_counts_as_win() -> None:
    trades = [_sell(date(2024, 1, 2), "AAPL", 10, 100.0)]
    basis = [100.0]
    win_rate, _pf, _avg_win, _avg_loss = compute_trade_stats(trades, basis)
    assert win_rate == 1.0


# --- compute_strategy_stats ---


def test_strategy_stats_groups_by_strategy() -> None:
    trades = [
        _buy(date(2024, 1, 1), "AAPL", 10, 100.0, strategy="A"),
        _sell(date(2024, 1, 2), "AAPL", 10, 110.0, strategy="A"),  # +100
        _buy(date(2024, 1, 1), "MSFT", 5, 200.0, strategy="B"),
        _sell(date(2024, 1, 3), "MSFT", 5, 190.0, strategy="B"),  # -50
    ]
    basis = [100.0, 200.0]  # parallel to sells in trade order
    stats = compute_strategy_stats(trades, basis)
    by_name = {s.name: s for s in stats}
    assert by_name["A"].buys == 1
    assert by_name["A"].sells == 1
    assert by_name["A"].pnl == 100.0
    assert by_name["A"].win_rate == 1.0
    assert by_name["B"].pnl == -50.0
    assert by_name["B"].win_rate == 0.0


def test_strategy_stats_same_day_collision() -> None:
    """Two strategies sell the same ticker on the same day with different bases."""
    day = date(2024, 1, 2)
    trades = [
        _sell(day, "AAPL", 5, 110.0, strategy="A"),  # basis 100 → +50
        _sell(day, "AAPL", 5, 110.0, strategy="B"),  # basis 80  → +150
    ]
    basis = [100.0, 80.0]
    stats = compute_strategy_stats(trades, basis)
    by_name = {s.name: s for s in stats}
    assert by_name["A"].pnl == 50.0
    assert by_name["B"].pnl == 150.0


def test_strategy_stats_empty() -> None:
    assert compute_strategy_stats([], []) == []


# --- end-to-end check that BacktestResult populates new metrics ---


def test_backtest_populates_new_metrics() -> None:
    portfolio, price_data = _make_backtest_data()
    mr = MeanReversion(window=10, threshold=-0.05)
    engine = _build_engine(entries=[(mr, 1.0)], enable_split=False)
    idx = list(price_data["AAPL"].index)
    start, end = idx[0], idx[-1]
    result = engine.run(portfolio, price_data, start, end)
    assert len(result.equity_curve) > 0
    assert result.equity_curve[-1][0] == end
    assert result.max_drawdown >= 0.0
    # Sortino must never be inf — even on a perfect run we cap to 0.
    assert not math.isinf(result.sortino_ratio)


def test_efficiency_ratio_uses_annualized_returns() -> None:
    """Efficiency ratio must divide annualized test return by annualized train.

    Train and test windows have different lengths, so dividing cumulative
    returns would mix periods of different duration. Annualizing both sides
    makes the ratio dimensionally consistent (and matches the walk-forward
    efficiency_ratio convention).
    """
    portfolio, price_data = _make_backtest_data()
    mr = MeanReversion(window=10, threshold=-0.05)
    engine = _build_engine(entries=[(mr, 1.0)], train_pct=0.4)
    start = min(price_data["AAPL"].index)
    end = max(price_data["AAPL"].index)
    result = engine.run(portfolio, price_data, start, end)

    assert result.split_date is not None
    assert result.train_days > 0
    assert result.test_days > 0
    assert result.train_days != result.test_days  # 40/60 split on 100 bars

    train_ann = compute_annualized_return(result.train_return, result.train_days)
    test_ann = compute_annualized_return(result.test_return, result.test_days)
    if train_ann == 0:
        pytest.skip("Degenerate fixture: no train-side return to compare against")
    expected = test_ann / train_ann
    # Engine computes efficiency from unrounded returns then rounds the final
    # ratio; recomputing from the rounded stored returns can drift slightly,
    # so compare with a small tolerance rather than exact equality.
    assert math.isclose(result.efficiency_ratio, expected, rel_tol=1e-2)

    # Regression guard: the cumulative ratio (test_return / train_return) is
    # wrong because train and test windows span different lengths here.
    # Should differ meaningfully from the annualized ratio.
    if result.train_return and result.test_return:
        cumulative_ratio = result.test_return / result.train_return
        assert not math.isclose(cumulative_ratio, expected, rel_tol=1e-2), (
            "Efficiency ratio matches both cumulative and annualized ratios — "
            "test fixture doesn't discriminate the fix. Use distinct train/test lengths."
        )


def test_summary_json_includes_annualized_keys(tmp_path: Path) -> None:
    """summary.json must expose *_annualized alongside cumulative keys.

    Downstream consumers key off these names; renaming or dropping them is
    a breaking change. The annualized values must also match
    compute_annualized_return applied to the matching cumulative value —
    otherwise we've silently desynced the two.
    """
    portfolio, price_data = _make_backtest_data()
    mr = MeanReversion(window=20, threshold=0.05)
    engine = _build_engine(entries=[(mr, 1.0)], train_pct=0.7)
    start = min(price_data["AAPL"].index)
    end = max(price_data["AAPL"].index)
    result = engine.run(portfolio, price_data, start, end)

    out_dir = tmp_path / "results"
    write_backtest_results(result, out_dir)
    with open(out_dir / "summary.json") as f:
        summary = json.load(f)

    # Top-level annualized keys exist and reconcile against cumulative + days.
    # Annualized values are rounded to match their cumulative sibling's precision
    # (6 decimals for total_return / buy_and_hold_return, 4 for twr and splits),
    # so tolerances track that rounding.
    assert "total_return_annualized" in summary
    assert "twr_annualized" in summary
    assert "buy_and_hold_return_annualized" in summary
    total_days = result.total_days
    assert math.isclose(
        summary["total_return_annualized"],
        compute_annualized_return(summary["total_return"], total_days),
        abs_tol=1e-5,
    )
    assert math.isclose(
        summary["twr_annualized"],
        compute_annualized_return(result.twr, total_days),
        abs_tol=5e-5,
    )

    # Split block likewise carries annualized mirrors.
    assert "split" in summary
    split = summary["split"]
    for key in (
        "train_return_annualized",
        "test_return_annualized",
        "train_bh_return_annualized",
        "test_bh_return_annualized",
    ):
        assert key in split
    assert math.isclose(
        split["train_return_annualized"],
        compute_annualized_return(result.train_return, result.train_days),
        abs_tol=5e-5,
    )
    assert math.isclose(
        split["test_return_annualized"],
        compute_annualized_return(result.test_return, result.test_days),
        abs_tol=5e-5,
    )


# --- _fifo_consumed_basis unit tests ---
#
# FIFO basis underpins realized-P&L attribution. These pin the contract
# directly so regressions surface at the unit level instead of bleeding
# through end-to-end backtests.


def _pl(shares: float, basis: float) -> PositionLot:
    return PositionLot(
        shares=shares,
        purchase_date=date(2024, 1, 1),
        cost_basis=basis,
    )


def test_fifo_basis_empty_lots() -> None:
    assert BacktestEngine._fifo_consumed_basis([], 5) == 0.0


def test_fifo_basis_zero_shares() -> None:
    assert BacktestEngine._fifo_consumed_basis([_pl(10, 100.0)], 0) == 0.0


def test_fifo_basis_single_lot_partial() -> None:
    basis = BacktestEngine._fifo_consumed_basis([_pl(10, 100.0)], 4)
    assert basis == 100.0


def test_fifo_basis_crosses_lot_boundary() -> None:
    """Consume 8 shares across two lots: 5 @ $100 + 3 @ $80 → $92.50."""
    lots = [_pl(5, 100.0), _pl(10, 80.0)]
    basis = BacktestEngine._fifo_consumed_basis(lots, 8)
    assert basis == (5 * 100.0 + 3 * 80.0) / 8


def test_fifo_basis_respects_lot_order() -> None:
    """Reversing the lot list must change the answer — this is FIFO, not LIFO."""
    lot_a = _pl(5, 100.0)
    lot_b = _pl(10, 80.0)
    fifo = BacktestEngine._fifo_consumed_basis([lot_a, lot_b], 8)
    lifo = BacktestEngine._fifo_consumed_basis([lot_b, lot_a], 8)
    assert fifo != lifo
    assert lifo == 80.0  # first 8 all come from the $80 lot


def test_fifo_basis_does_not_mutate_lots() -> None:
    lots = [_pl(5, 100.0), _pl(10, 80.0)]
    BacktestEngine._fifo_consumed_basis(lots, 8)
    assert lots[0].shares == 5
    assert lots[1].shares == 10


# --- restriction-before-sizing regression ---


def test_blocked_sell_does_not_leak_into_buy_sizing() -> None:
    """Restriction-before-sizing invariant: blocked sells must not inflate the
    ``cash`` the buy pass sees.

    The backtest orders Phase 3 as:

        1. size sells from clamped targets
        2. filter restriction-blocked sells
        3. compute ``post_sell_cash = state.cash + sum(filtered)``
        4. call ``size_buys(..., cash=post_sell_cash, ...)``

    If step 2 is skipped (or moved after step 3), blocked sell proceeds
    leak into ``post_sell_cash`` and ``size_buys`` can authorize buys
    against cash that will never arrive in phase 5.

    This test pins the order by spying on ``size_buys`` and asserting the
    ``cash`` argument equals ``state.cash + proceeds_of_unblocked_sells``,
    not ``state.cash + proceeds_of_all_sells``.
    """
    constraints = AllocationConstraints(min_buy_delta=0.01, max_position_pct=0.95)
    mr = MeanReversion(window=10, threshold=0.05)
    allocator = Allocator([(mr, 1.0)], constraints, n_tickers=1)
    sizer = OrderSizer(default_slippage=0.0)

    # Spy on size_buys to capture the cash it was called with.
    captured: dict[str, float] = {}
    real_size_buys = sizer.size_buys

    def spy_size_buys(*args, **kwargs):
        # cash is the 4th positional arg in size_buys
        captured["cash"] = args[3] if len(args) > 3 else kwargs["cash"]
        return real_size_buys(*args, **kwargs)

    sizer.size_buys = spy_size_buys  # type: ignore[method-assign]

    engine = BacktestEngine(
        allocator=allocator,
        order_sizer=sizer,
        exit_rules=[ProfitTaking(gain_threshold=0.10)],
        constraints=constraints,
        execution_mode="close",
    )

    # A single held ticker in profit → ProfitTaking fires. Round-trip
    # restriction blocks the sell.
    a_prices = ph(np.array([100.0] * 15))

    state = _SimState(cash=0.0)
    state.positions = {"A": 5.0}
    state.lots = {
        "A": [
            PositionLot(
                shares=5.0,
                purchase_date=date(2024, 1, 1),
                cost_basis=80.0,
            )
        ]
    }
    state.high_water_marks = {"A": 100.0}
    state.restriction_tracker = RestrictionTracker(TradingRestrictions(round_trip_days=30))
    state.restriction_tracker.record_trade("A", Direction.BUY, date(2024, 1, 15))

    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="A", shares=5, cost_basis=80.0)],
        available_cash=0.0,
        trading_restrictions=TradingRestrictions(round_trip_days=30),
    )

    engine._run_day(state, portfolio, {"A": a_prices}, date(2024, 1, 16))

    # post_sell_cash should equal state.cash ($0) because the single sell
    # was blocked. If blocked proceeds leaked in, cash would be ~$500.
    assert "cash" in captured, "size_buys was not called"
    assert captured["cash"] == 0.0, f"blocked sell proceeds leaked into size_buys cash: {captured['cash']}"


# --- competing exit rules: realized + unrealized reconciliation ---


def test_competing_exit_rules_collapse_to_one_sell() -> None:
    """Regression: ProfitTaking + TrailingStop firing the same tick over-sold.

    When a held lot is in deep profit *and* its HWM has already drifted
    below the trail threshold, both rules independently want to liquidate
    the entire position. Pre-fix, ``size_sells`` produced two sell orders
    each sized against the full position — the second sell drained shares
    that no longer existed, fabricated cash, and broke the realized-P&L
    reconciliation against ``final - start``.

    Drive ``_run_day`` directly with a hand-crafted state so we can pin
    exactly the lot/HWM/price configuration that triggers the double-fire,
    then assert: (a) only one sell fires, (b) it's credited to the more
    aggressive rule, and (c) ``final - start == realized + unrealized``.
    """
    constraints = AllocationConstraints(min_buy_delta=0.01, max_position_pct=0.95)
    allocator = Allocator([], constraints, n_tickers=1)
    sizer = OrderSizer(default_slippage=0.0)
    engine = BacktestEngine(
        allocator=allocator,
        order_sizer=sizer,
        exit_rules=[ProfitTaking(gain_threshold=0.10), TrailingStop(trail_pct=0.05)],
        constraints=constraints,
        execution_mode="close",
    )

    # Lot @ $80 basis with HWM=$130 (the peak the position has seen).
    # Today's price is $115:
    #   PT: gain = (115-80)/80 = 43.75% > 10% → wants full liquidation
    #   TS: drawdown = (130-115)/130 = 11.5% > 5% → wants full liquidation
    # Both rules fire on the entire 10-share position the same tick.
    state = _SimState(cash=0.0)
    state.cash = 0.0
    state.positions = {"A": 10.0}
    state.lots = {
        "A": [
            PositionLot(
                shares=10.0,
                purchase_date=date(2024, 1, 1),
                cost_basis=80.0,
            )
        ]
    }
    state.high_water_marks = {"A": 130.0}
    state.starting_value = 800.0  # 10 shares at $80 basis
    state.twr_base_value = 800.0

    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="A", shares=10, cost_basis=80.0)],
        available_cash=0.0,
    )

    # Price array with current=$115; backstop history gives the rules
    # something to read but the only price they act on is the last bar.
    prices_raw = np.array([100.0, 110.0, 120.0, 130.0, 125.0, 120.0, 115.0])
    prices = ph(prices_raw)
    engine._run_day(state, portfolio, {"A": prices}, date(2024, 2, 1))

    sells = [t for t in state.trades if t.direction == Direction.SELL]
    assert len(sells) == 1, f"expected 1 sell after collapse, got {len(sells)}: {sells}"
    assert sells[0].shares == 10  # the full position, not 20
    # TrailingStop's intent (full liquidation at current price) ties with
    # ProfitTaking's. Either is acceptable; what matters is one wins, not
    # both. We assert it's one of the two configured sources.
    assert sells[0].strategy_name in {"ProfitTaking", "TrailingStop"}

    # Reconciliation: with one sell, FIFO consumed basis = real basis,
    # state.cash exactly reflects sell proceeds, and the position is empty.
    final_price = float(prices_raw[-1])
    final_value = state.cash + sum(sum(lot.shares for lot in lots) * final_price for lots in state.lots.values())
    realized = sum(
        (t.price - state.basis_per_sell[i]) * t.shares
        for i, t in enumerate(t for t in state.trades if t.direction == Direction.SELL)
    )
    unrealized = sum(sum(lot.shares * (final_price - lot.cost_basis) for lot in lots) for lots in state.lots.values())
    delta = final_value - state.starting_value
    assert math.isclose(delta, realized + unrealized, abs_tol=1.0), (
        f"reconciliation broken: final-start=${delta:,.2f} but realized+unrealized=${realized + unrealized:,.2f}"
    )


# --- HWM lifecycle: stale peaks must not survive a full exit ---


def _sell_order(ticker: str, shares: float, price: float, source: str = "TestRule") -> Order:
    return Order(
        ticker=ticker,
        direction=Direction.SELL,
        shares=shares,
        price=price,
        estimated_value=shares * price,
        context=OrderContext(
            contributions={},
            blended_score=0.0,
            target_weight=0.0,
            current_weight=0.0,
            reason="test",
            source=source,
        ),
    )


def test_full_exit_clears_high_water_mark() -> None:
    """Regression: ``state.high_water_marks[ticker]`` must be cleared when a
    position is fully exited. Otherwise a later re-entry inherits the stale
    peak and TrailingStop misfires on day 1 of the new lot against a price
    the new position has never reached.
    """
    engine = _build_engine()
    state = _SimState(cash=0.0)
    state.positions = {"A": 10.0}
    state.lots = {"A": [PositionLot(shares=10.0, purchase_date=date(2024, 1, 1), cost_basis=100.0)]}
    state.high_water_marks = {"A": 200.0}  # AAPL rallied from $100 to $200

    order = _sell_order("A", 10.0, 160.0, source="TrailingStop")
    engine._execute(order, date(2024, 2, 1), state)

    assert state.positions["A"] == 0
    assert "A" not in state.high_water_marks, "stale HWM survived a full exit — TrailingStop would misfire on re-entry"


def test_partial_exit_preserves_high_water_mark() -> None:
    """Partial exits leave HWM intact — the position still exists."""
    engine = _build_engine()
    state = _SimState(cash=0.0)
    state.positions = {"A": 10.0}
    state.lots = {"A": [PositionLot(shares=10.0, purchase_date=date(2024, 1, 1), cost_basis=100.0)]}
    state.high_water_marks = {"A": 200.0}

    order = _sell_order("A", 4.0, 180.0)
    engine._execute(order, date(2024, 2, 1), state)

    assert state.positions["A"] == 6.0
    assert state.high_water_marks["A"] == 200.0


# --- Mixed holding-period sells split into per-period records ---


def test_sell_crossing_st_lt_boundary_splits_into_two_records() -> None:
    """A FIFO sell that consumes lots straddling the 365-day boundary must
    emit two TradeRecords — one short-term, one long-term — each with the
    correct per-group share count, weighted cost basis, and classification.

    Pre-fix, ``lots[0].purchase_date`` classified the entire sell (whichever
    bucket the oldest lot fell into), misreporting the other portion.
    """
    engine = _build_engine()
    state = _SimState(cash=0.0)
    # Lot 0 is ancient (1000 days old — LT), lot 1 is fresh (30 days old — ST).
    # FIFO consumes lot 0 first then lot 1.
    state.positions = {"A": 15.0}
    state.lots = {
        "A": [
            PositionLot(shares=5.0, purchase_date=date(2022, 1, 1), cost_basis=80.0),
            PositionLot(shares=10.0, purchase_date=date(2024, 1, 1), cost_basis=120.0),
        ]
    }
    state.high_water_marks = {"A": 150.0}

    # Sell 12 shares — consumes all 5 LT shares + 7 ST shares from lot 1.
    order = _sell_order("A", 12.0, 130.0)
    day = date(2024, 2, 1)  # lot 0: 762 days (LT), lot 1: 31 days (ST)
    records = engine._execute(order, day, state)

    assert len(records) == 2, f"expected 2 records (ST + LT), got {len(records)}"

    # Records are returned ST-first, then LT.
    st_trade, st_basis = records[0]
    lt_trade, lt_basis = records[1]

    assert st_trade.holding_period == HoldingPeriod.SHORT_TERM
    assert st_trade.shares == 7.0
    assert st_basis == 120.0  # all 7 ST shares came from lot 1 @ $120

    assert lt_trade.holding_period == HoldingPeriod.LONG_TERM
    assert lt_trade.shares == 5.0
    assert lt_basis == 80.0  # all 5 LT shares came from lot 0 @ $80

    # Total shares reconcile.
    assert st_trade.shares + lt_trade.shares == order.shares
    # Position correctly decremented and 3 shares of lot 1 remain.
    assert state.positions["A"] == 3.0
    assert len(state.lots["A"]) == 1
    assert state.lots["A"][0].shares == 3.0


def test_sell_single_period_emits_one_record() -> None:
    """A sell consuming only ST lots emits a single ST record."""
    engine = _build_engine()
    state = _SimState(cash=0.0)
    state.positions = {"A": 10.0}
    state.lots = {"A": [PositionLot(shares=10.0, purchase_date=date(2024, 1, 1), cost_basis=100.0)]}

    order = _sell_order("A", 6.0, 110.0)
    records = engine._execute(order, date(2024, 2, 1), state)

    assert len(records) == 1
    trade, basis = records[0]
    assert trade.holding_period == HoldingPeriod.SHORT_TERM
    assert trade.shares == 6.0
    assert basis == 100.0


def test_sell_pure_long_term_emits_one_lt_record() -> None:
    """A sell consuming only LT lots emits a single LT record."""
    engine = _build_engine()
    state = _SimState(cash=0.0)
    state.positions = {"A": 10.0}
    state.lots = {"A": [PositionLot(shares=10.0, purchase_date=date(2022, 1, 1), cost_basis=100.0)]}

    order = _sell_order("A", 6.0, 110.0)
    records = engine._execute(order, date(2024, 2, 1), state)

    assert len(records) == 1
    trade, basis = records[0]
    assert trade.holding_period == HoldingPeriod.LONG_TERM
    assert trade.shares == 6.0
    assert basis == 100.0


def test_sell_mixed_basis_within_single_period() -> None:
    """Two ST lots at different bases aggregate into a single ST record
    with a weighted cost basis.
    """
    engine = _build_engine()
    state = _SimState(cash=0.0)
    state.positions = {"A": 10.0}
    state.lots = {
        "A": [
            PositionLot(shares=4.0, purchase_date=date(2024, 1, 1), cost_basis=80.0),
            PositionLot(shares=6.0, purchase_date=date(2024, 1, 15), cost_basis=100.0),
        ]
    }

    # Sell all 10: weighted basis = (4*80 + 6*100) / 10 = 92.0
    order = _sell_order("A", 10.0, 110.0)
    records = engine._execute(order, date(2024, 2, 1), state)

    assert len(records) == 1
    trade, basis = records[0]
    assert trade.holding_period == HoldingPeriod.SHORT_TERM
    assert trade.shares == 10.0
    assert basis == 92.0


# --- TradeRecord.purchase_date population in _execute ---


def _buy_order(ticker: str, shares: float, price: float, source: str = "Momentum") -> Order:
    return Order(
        ticker=ticker,
        direction=Direction.BUY,
        shares=shares,
        price=price,
        estimated_value=shares * price,
        context=OrderContext(
            contributions={source: 1.0},
            blended_score=1.0,
            target_weight=0.5,
            current_weight=0.0,
            reason="entry",
            source=source,
        ),
    )


def test_execute_buy_records_purchase_date_as_day() -> None:
    """BUY records carry the fill date as their purchase_date."""
    engine = _build_engine()
    state = _SimState(cash=10000.0, starting_value=10000.0)

    order = _buy_order("AAPL", 10.0, 20.0)
    records = engine._execute(order, date(2026, 5, 8), state)
    trade, _basis = records[0]
    assert trade.purchase_date == date(2026, 5, 8)


def test_execute_sell_single_lot_records_lot_purchase_date() -> None:
    """SELL bucket consuming one lot records that lot's purchase date."""
    engine = _build_engine()
    state = _SimState(cash=0.0, starting_value=0.0)
    state.lots["AAPL"] = [PositionLot(shares=10.0, purchase_date=date(2026, 1, 1), cost_basis=10.0)]
    state.positions["AAPL"] = 10.0

    order = _sell_order("AAPL", 10.0, 15.0, source="StopLoss")
    records = engine._execute(order, date(2026, 5, 8), state)
    trade, _basis = records[0]
    assert trade.purchase_date == date(2026, 1, 1)


def test_execute_sell_mixed_lot_records_various() -> None:
    """SELL bucket spanning multiple lots with different dates records 'various'."""
    engine = _build_engine()
    state = _SimState(cash=0.0, starting_value=0.0)
    state.lots["AAPL"] = [
        PositionLot(shares=5.0, purchase_date=date(2026, 1, 1), cost_basis=10.0),
        PositionLot(shares=5.0, purchase_date=date(2026, 2, 1), cost_basis=11.0),
    ]
    state.positions["AAPL"] = 10.0

    order = _sell_order("AAPL", 10.0, 15.0, source="StopLoss")
    records = engine._execute(order, date(2026, 5, 8), state)
    # Only one ST bucket since both lots are <365 days from sell day → both ST → mixed.
    trade, _basis = records[0]
    assert trade.purchase_date == "various"


def test_execute_sell_single_unseeded_lot_records_none() -> None:
    """SELL bucket consuming one lot with purchase_date=None records None,
    not 'various'. Mirrors live-mode unseeded lots from a fresh state file."""
    engine = _build_engine()
    state = _SimState(cash=0.0, starting_value=0.0)
    state.lots["AAPL"] = [PositionLot(shares=10.0, purchase_date=None, cost_basis=10.0)]
    state.positions["AAPL"] = 10.0

    order = _sell_order("AAPL", 10.0, 15.0, source="StopLoss")
    records = engine._execute(order, date(2026, 5, 8), state)
    trade, _basis = records[0]
    assert trade.purchase_date is None


def _make_minimal_result(
    trades: list[TradeRecord],
    basis_per_sell: list[float],
) -> BacktestResult:
    """Build a minimal ``BacktestResult`` for output-writer tests."""
    return BacktestResult(
        trades=trades,
        final_value=0,
        starting_value=0,
        buy_and_hold_value=0,
        train_trades=[],
        test_trades=[],
        train_return=0,
        test_return=0,
        train_bh_return=0,
        test_bh_return=0,
        split_date=None,
        twr=0,
        equity_curve=[],
        total_days=0,
        train_days=0,
        test_days=0,
        cagr=0,
        max_drawdown=0,
        sharpe_ratio=0,
        sortino_ratio=0,
        win_rate=0,
        profit_factor=0,
        avg_win=0,
        avg_loss=0,
        efficiency_ratio=0,
        strategy_stats=[],
        unrealized_pnl=0,
        unrealized_pnl_by_ticker={},
        basis_per_sell=basis_per_sell,
    )


def test_trades_csv_includes_purchase_date_column(tmp_path: Path) -> None:
    """Backtest output's trades.csv has a purchase_date column populated for BUYs and SELLs."""
    import csv

    from midas.results import _write_trades_csv

    trades = [
        TradeRecord(
            date=date(2026, 1, 5),
            ticker="AAPL",
            direction=Direction.BUY,
            shares=10.0,
            price=20.0,
            strategy_name="Momentum",
            purchase_date=date(2026, 1, 5),
        ),
        TradeRecord(
            date=date(2026, 4, 1),
            ticker="AAPL",
            direction=Direction.SELL,
            shares=10.0,
            price=25.0,
            strategy_name="StopLoss",
            holding_period=HoldingPeriod.SHORT_TERM,
            purchase_date=date(2026, 1, 5),
        ),
    ]
    result = _make_minimal_result(trades=trades, basis_per_sell=[20.0])
    out = tmp_path / "trades.csv"
    _write_trades_csv(result, out)
    rows = list(csv.DictReader(out.open()))
    assert "purchase_date" in rows[0]
    assert rows[0]["purchase_date"] == "2026-01-05"
    assert rows[1]["purchase_date"] == "2026-01-05"


def test_backtest_trades_csv_round_trips_through_trade_log_reader(tmp_path: Path) -> None:
    """Backtest's trades.csv shape must match TRADE_LOG_COLUMNS exactly so
    the live and backtest paths share one reader. read_trades raises
    TradeLogError on any header drift, so a successful round-trip pins
    the column shape."""
    from midas.results import _write_trades_csv
    from midas.trade_log import read_trades

    trades = [
        TradeRecord(
            date=date(2026, 1, 5),
            ticker="AAPL",
            direction=Direction.BUY,
            shares=10.0,
            price=20.0,
            strategy_name="Momentum",
            purchase_date=date(2026, 1, 5),
        ),
        TradeRecord(
            date=date(2026, 4, 1),
            ticker="AAPL",
            direction=Direction.SELL,
            shares=10.0,
            price=25.0,
            strategy_name="StopLoss",
            holding_period=HoldingPeriod.SHORT_TERM,
            purchase_date=date(2026, 1, 5),
        ),
    ]
    result = _make_minimal_result(trades=trades, basis_per_sell=[20.0])
    out = tmp_path / "trades.csv"
    _write_trades_csv(result, out)
    rows = read_trades(out)  # raises on shape drift
    assert len(rows) == 2
    assert rows[0].direction == Direction.BUY
    assert rows[0].purchase_date == date(2026, 1, 5)
    assert rows[1].direction == Direction.SELL
    assert rows[1].holding_period == HoldingPeriod.SHORT_TERM
    assert rows[1].purchase_date == date(2026, 1, 5)
    assert rows[1].cost_basis == 20.0


def _stop_loss_frame() -> tuple[PortfolioConfig, dict[str, pd.DataFrame]]:
    """Frame and portfolio that triggers a StopLoss exit (one realized SELL)."""
    n = 30
    dates = [d.date() for d in pd.bdate_range(start=date(2024, 1, 2), periods=n)]
    closes = np.full(n, 100.0)
    opens = np.full(n, 100.0)
    highs = np.full(n, 100.0)
    lows = np.full(n, 100.0)
    volumes = np.full(n, 1_000_000.0)
    drop_day = 20
    opens[drop_day] = 92.0
    highs[drop_day] = 93.0
    lows[drop_day] = 88.0
    closes[drop_day] = 90.0
    opens[drop_day + 1] = 89.0
    highs[drop_day + 1] = 91.0
    lows[drop_day + 1] = 88.0
    closes[drop_day + 1] = 91.0
    frame = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=dates,
    )
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="GAPCO", shares=5, cost_basis=100.0)],
        available_cash=10_000.0,
    )
    return portfolio, {"GAPCO": frame}


def test_backtest_result_after_tax_fields_populated_with_tax_config() -> None:
    """End-to-end: a backtest with TaxConfig set populates after_tax_* fields and tax_summary."""
    from midas.models import TaxConfig

    portfolio, price_data = _stop_loss_frame()
    constraints = AllocationConstraints(min_buy_delta=0.01, max_position_pct=0.95)
    allocator = Allocator(
        [(GapDownRecovery(gap_threshold=0.03), 1.0)],
        constraints,
        n_tickers=1,
    )
    tax = TaxConfig(short_term_rate=0.30, long_term_rate=0.15)
    engine = BacktestEngine(
        allocator=allocator,
        order_sizer=OrderSizer(default_slippage=0.0),
        exit_rules=[StopLoss(loss_threshold=0.05)],
        constraints=constraints,
        enable_split=False,
        execution_mode="next_open",
        tax_config=tax,
    )
    frame = price_data["GAPCO"]
    start, end = frame.index[0], frame.index[-1]
    result = engine.run(portfolio, price_data, start, end)

    # The fixture produces at least one realized SELL, so the tax pipeline
    # has data to operate on.
    sells = [t for t in result.trades if t.direction == Direction.SELL]
    assert sells, "fixture must produce at least one SELL for the after-tax pipeline to engage"

    assert result.after_tax_final_value is not None
    assert result.after_tax_total_return is not None
    assert result.after_tax_cagr is not None
    assert result.after_tax_twr is not None
    assert result.tax_summary, "tax_summary should be non-empty when sells happened"
    assert len(result.after_tax_equity_curve) == len(result.equity_curve)

    # tax_cost_ratio is only meaningful when gross CAGR is positive; otherwise
    # the field is None (a losing strategy still has tax drag, but expressing
    # it as a ratio of negative CAGR is not interpretable).
    if result.cagr > 0:
        assert result.tax_cost_ratio is not None
    else:
        assert result.tax_cost_ratio is None

    # The StopLoss fixture realizes a loss, so tax_owed is negative (a refund
    # via the deductible-loss credit) and the after-tax curve sits above the
    # gross curve. In a winning scenario, tax_owed > 0 and the after-tax curve
    # would sit below. Either way: the sign of (after_tax - gross) must match
    # the sign of (-tax_owed), and the after-tax curve must reflect that drag
    # (or refund).
    total_tax = sum(s.tax_owed for s in result.tax_summary)
    if total_tax > 0:
        assert result.after_tax_final_value < result.final_value
        assert result.after_tax_cagr is not None and result.after_tax_cagr < result.cagr
    elif total_tax < 0:
        assert result.after_tax_final_value > result.final_value
    else:
        assert result.after_tax_final_value == result.final_value

    # Structural shape of multi-year summaries: years and payment_dates monotonic.
    if len(result.tax_summary) > 1:
        years = [s.year for s in result.tax_summary]
        payment_dates = [s.payment_date for s in result.tax_summary]
        assert years == sorted(years)
        assert payment_dates == sorted(payment_dates)


def test_backtest_result_after_tax_fields_none_without_tax_config() -> None:
    """Backwards-compatible: without TaxConfig, all after-tax fields are None/empty."""
    portfolio, price_data = _stop_loss_frame()
    constraints = AllocationConstraints(min_buy_delta=0.01, max_position_pct=0.95)
    allocator = Allocator(
        [(GapDownRecovery(gap_threshold=0.03), 1.0)],
        constraints,
        n_tickers=1,
    )
    engine = BacktestEngine(
        allocator=allocator,
        order_sizer=OrderSizer(default_slippage=0.0),
        exit_rules=[StopLoss(loss_threshold=0.05)],
        constraints=constraints,
        enable_split=False,
        execution_mode="next_open",
    )
    frame = price_data["GAPCO"]
    start, end = frame.index[0], frame.index[-1]
    result = engine.run(portfolio, price_data, start, end)

    assert result.after_tax_final_value is None
    assert result.after_tax_total_return is None
    assert result.after_tax_cagr is None
    assert result.after_tax_twr is None
    assert result.after_tax_equity_curve == []
    assert result.tax_summary == []
    assert result.tax_cost_ratio is None


def test_equity_curve_csv_includes_nav_after_tax_when_set(tmp_path: Path) -> None:
    """equity_curve.csv gains a parallel ``nav_after_tax`` column when after-tax curve is populated."""
    from midas.results import _write_equity_curve_csv

    result = _make_minimal_result(trades=[], basis_per_sell=[])
    result.equity_curve = [(date(2026, 1, 5), 1000.0), (date(2026, 1, 6), 1010.0)]
    result.after_tax_equity_curve = [(date(2026, 1, 5), 1000.0), (date(2026, 1, 6), 990.0)]

    out = tmp_path / "equity_curve.csv"
    _write_equity_curve_csv(result, out)
    rows = list(csv_mod.DictReader(out.open()))
    assert "nav_after_tax" in rows[0]
    assert rows[0]["nav_after_tax"] == "1000.0"
    assert rows[1]["nav_after_tax"] == "990.0"


def test_equity_curve_csv_omits_nav_after_tax_when_not_set(tmp_path: Path) -> None:
    """equity_curve.csv has no nav_after_tax column when after-tax curve is empty."""
    from midas.results import _write_equity_curve_csv

    result = _make_minimal_result(trades=[], basis_per_sell=[])
    result.equity_curve = [(date(2026, 1, 5), 1000.0), (date(2026, 1, 6), 1010.0)]

    out = tmp_path / "equity_curve.csv"
    _write_equity_curve_csv(result, out)
    rows = list(csv_mod.DictReader(out.open()))
    assert "nav_after_tax" not in rows[0]


def test_summary_json_includes_after_tax_block_when_set(tmp_path: Path) -> None:
    """summary.json includes after-tax fields and tax_summary array when populated."""
    from midas.results import _write_summary_json
    from midas.tax import AnnualTaxSummary

    result = _make_minimal_result(trades=[], basis_per_sell=[])
    result.starting_value = 1000.0
    result.final_value = 1100.0
    result.after_tax_final_value = 1080.0
    result.after_tax_total_return = 0.08
    result.after_tax_cagr = 0.075
    result.after_tax_twr = 0.078
    result.tax_cost_ratio = 0.05
    result.tax_summary = [
        AnnualTaxSummary(
            year=2026,
            st_realized=100.0,
            lt_realized=0.0,
            net_after_cross=100.0,
            deductible_loss=0.0,
            carry_forward=0.0,
            tax_owed=20.0,
            payment_date=date(2027, 4, 15),
        )
    ]

    out = tmp_path / "summary.json"
    _write_summary_json(result, out)
    summary = json.loads(out.read_text())
    assert summary["after_tax_final_value"] == 1080.0
    assert summary["after_tax_total_return"] == 0.08
    assert summary["after_tax_cagr"] == 0.075
    assert summary["after_tax_twr"] == 0.078
    assert summary["tax_cost_ratio"] == 0.05
    assert len(summary["tax_summary"]) == 1
    assert summary["tax_summary"][0]["year"] == 2026
    assert summary["tax_summary"][0]["tax_owed"] == 20.0
    assert summary["tax_summary"][0]["payment_date"] == "2027-04-15"


def test_summary_json_omits_after_tax_block_when_not_set(tmp_path: Path) -> None:
    """summary.json has no after-tax fields when fields are None and tax_summary empty."""
    from midas.results import _write_summary_json

    result = _make_minimal_result(trades=[], basis_per_sell=[])
    result.starting_value = 1000.0
    result.final_value = 1100.0

    out = tmp_path / "summary.json"
    _write_summary_json(result, out)
    summary = json.loads(out.read_text())
    assert "after_tax_final_value" not in summary
    assert "after_tax_total_return" not in summary
    assert "after_tax_cagr" not in summary
    assert "after_tax_twr" not in summary
    assert "tax_cost_ratio" not in summary
    assert "tax_summary" not in summary
