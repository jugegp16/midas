"""Bar-for-bar parity between the backtest and live engines.

Drives a deterministic synthetic price series through both the backtest and
the live engine (with a fake DataProvider). Asserts that lot lists, HWMs,
peak equity, and available cash agree at the end of the run.

The simplest version uses no strategies and no exit rules — both engines
just track time, prices, and the seed position. This isolates the state-
evolution mechanics from any strategy-driven order flow. Note that with
no order flow, lot-consumption parity (FIFO/LIFO accounting under sells)
is not exercised end-to-end here — that requires the strategy-driven
follow-up.

Notes on alignment:

- The backtest's ``_init_positions`` reseeds each holding's cost basis to
  the day-0 close (so exit rules don't fire on pre-backtest gains). The
  live engine's ``load_or_seed`` uses the YAML cost basis. We pre-seed
  the live state YAML on disk to match the backtest's seed exactly,
  isolating state-evolution mechanics from seeding policy.
- The backtest uses ``execution_mode="next_open"`` (lag=1) by default,
  but with no strategies and no exit rules there are no decisions and
  no fills, so the lag is a no-op and both engines reduce to the same
  "track HWM/peak across closes" loop.
- ``date.today()`` is monkeypatched per tick so the live engine sees the
  same calendar day as the backtest's current bar.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from midas.allocator import Allocator
from midas.backtest import BacktestEngine, _SimState
from midas.live import LiveEngine
from midas.live_state import LiveState, load_state, save_atomic
from midas.models import (
    AllocationConstraints,
    Holding,
    PortfolioConfig,
    PositionLot,
)
from midas.order_sizer import OrderSizer
from midas.results import BacktestResult
from midas.strategies.stop_loss import StopLoss


def _build_price_frame(start: date, n_bars: int, seed: int = 42) -> pd.DataFrame:
    """Deterministic synthetic OHLCV frame indexed by business days."""
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n_bars)))
    dates: list[date] = []
    current = start
    while len(dates) < n_bars:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": np.full(n_bars, 1_000_000.0),
        },
        index=dates,
    )


class _CapturingBacktestEngine(BacktestEngine):
    """Subclass that stashes the final ``_SimState`` for test inspection.

    ``BacktestEngine.run`` builds state internally and only returns a
    ``BacktestResult`` summary. We need the raw lot list, HWMs, peak,
    and cash to compare against ``LiveState`` — capture them via
    ``_build_result``, the last hook the engine calls before returning.
    """

    captured_state: _SimState | None = None

    def _build_result(
        self,
        state: _SimState,
        portfolio: PortfolioConfig,
        price_data: dict[str, pd.DataFrame],
        trading_days: list[date],
        split_date: date | None,
    ) -> BacktestResult:
        self.captured_state = state
        return super()._build_result(state, portfolio, price_data, trading_days, split_date)


def _make_provider_for_window(
    prices: pd.DataFrame,
    ticker: str,
    upto: date,
) -> MagicMock:
    """Build a fake provider returning *prices* sliced to ``index <= upto``."""
    provider = MagicMock()

    def get_history(ticker_arg: str, start: date, end: date) -> pd.DataFrame:
        if ticker_arg != ticker:
            msg = f"unexpected ticker {ticker_arg!r}"
            raise KeyError(msg)
        return prices[prices.index <= upto]

    provider.get_history.side_effect = get_history
    return provider


def _seed_state_to_match_backtest(
    state_path: Path,
    ticker: str,
    shares: float,
    seed_close: float,
    seed_day: date,
    cash: float,
) -> None:
    """Pre-seed the live state YAML to mirror backtest's ``_init_positions``.

    Backtest reseeds the seed lot's cost basis to the day-0 close and uses
    ``trading_days[0]`` as ``purchase_date``; live's default ``load_or_seed``
    uses the YAML basis with ``purchase_date=None``. We write the matching
    seed straight to disk so the engine loads it on construction.
    """
    seeded = LiveState(
        available_cash=cash,
        cash_infusion_next_date=None,
        high_water_marks={ticker: seed_close},
        peak_equity=cash + shares * seed_close,
        lots={
            ticker: [
                PositionLot(shares=shares, purchase_date=seed_day, cost_basis=seed_close),
            ]
        },
    )
    save_atomic(seeded, state_path)


def test_no_action_state_evolution_matches_backtest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no strategies and no exit rules, both engines should converge on
    the same final state for a held position over a fixed price series.

    The backtest's ``_init_positions`` rewrites the seed lot's cost basis to
    the day-0 close and stamps ``purchase_date=trading_days[0]``. We pre-seed
    the live state file with that same lot so any divergence is purely from
    state evolution (HWM, peak, cash), not from differing seeding policy.
    """
    ticker = "AAPL"
    shares = 10.0
    cash = 0.0
    n_bars = 30
    start = date(2026, 1, 5)
    prices = _build_price_frame(start, n_bars)
    seed_day = prices.index[0]
    seed_close = float(prices["close"].iloc[0])

    portfolio = PortfolioConfig(
        holdings=[Holding(ticker=ticker, shares=shares, cost_basis=seed_close)],
        available_cash=cash,
    )

    # ---- Backtest --------------------------------------------------------
    bt_allocator = Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1)
    bt_engine = _CapturingBacktestEngine(
        allocator=bt_allocator,
        order_sizer=OrderSizer(),
        exit_rules=[],
        constraints=AllocationConstraints(),
        enable_split=False,
    )
    bt_engine.run(portfolio, {ticker: prices}, prices.index[0], prices.index[-1])
    bt_state = bt_engine.captured_state
    assert bt_state is not None

    # ---- Live ------------------------------------------------------------
    state_path = tmp_path / "state.yaml"
    _seed_state_to_match_backtest(
        state_path=state_path,
        ticker=ticker,
        shares=shares,
        seed_close=seed_close,
        seed_day=seed_day,
        cash=cash,
    )

    # The live engine constructs its provider once at __init__, but the
    # provider's slice depends on each tick's date — we rebuild the engine
    # per tick with a provider whose ``get_history`` returns prices through
    # the current day. (Construction is cheap and idempotent because the
    # state file persists across invocations.)
    for current_day in prices.index:
        provider = _make_provider_for_window(prices, ticker, current_day)

        # Stub ``date.today()`` to return the current loop bar so live's
        # ``_tick`` sees the same "today" the backtest's day-T decision is
        # comparing against. We use a class attribute (not a closure over
        # ``current_day``) to satisfy ruff's B023 check on loop-variable
        # capture. The stub is intentionally narrow: it overrides only
        # ``today()``, not the ``date`` constructor — ``_tick`` doesn't
        # call other ``date`` factories.
        class FakeDate:
            value: date = current_day

            @staticmethod
            def today() -> date:
                return FakeDate.value

        monkeypatch.setattr("midas.live.date", FakeDate)

        live_engine = LiveEngine(
            portfolio=portfolio,
            allocator=Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1),
            order_sizer=OrderSizer(),
            provider=provider,
            state_path=state_path,
            history_days=n_bars * 2,
        )
        try:
            live_engine._tick([ticker])
        finally:
            # Release the state lockfile so the next loop iteration's engine
            # can reacquire it. In production a single engine lives for the
            # whole process; this teardown is test-only.
            live_engine.close()

    final_live = load_state(state_path)

    # ---- Compare ---------------------------------------------------------
    # Build comparable summaries from each engine's final state so any
    # divergence is reported with the field, the ticker, and both values.
    bt_summary = {
        "lots": {
            tk: [(lot.shares, lot.purchase_date, lot.cost_basis) for lot in lots] for tk, lots in bt_state.lots.items()
        },
        "hwm": dict(bt_state.high_water_marks),
        "peak": bt_state.peak_value,
        "cash": bt_state.cash,
    }
    live_summary = {
        "lots": {
            tk: [(lot.shares, lot.purchase_date, lot.cost_basis) for lot in lots]
            for tk, lots in final_live.lots.items()
        },
        "hwm": dict(final_live.high_water_marks),
        "peak": final_live.peak_equity,
        "cash": final_live.available_cash,
    }

    assert bt_summary["lots"].keys() == live_summary["lots"].keys(), (
        f"lot tickers diverge: backtest={set(bt_summary['lots'])}, live={set(live_summary['lots'])}"
    )
    for tk in bt_summary["lots"]:
        bt_lots = bt_summary["lots"][tk]
        live_lots = live_summary["lots"][tk]
        assert len(bt_lots) == len(live_lots), (
            f"{tk}: lot count diverges: backtest={len(bt_lots)}, live={len(live_lots)}"
        )
        for idx, (bt_lot, live_lot) in enumerate(zip(bt_lots, live_lots, strict=True)):
            assert bt_lot[0] == pytest.approx(live_lot[0]), (
                f"{tk} lot {idx}: shares diverge: backtest={bt_lot[0]}, live={live_lot[0]}"
            )
            assert bt_lot[1] == live_lot[1], (
                f"{tk} lot {idx}: purchase_date diverges: backtest={bt_lot[1]}, live={live_lot[1]}"
            )
            assert bt_lot[2] == pytest.approx(live_lot[2]), (
                f"{tk} lot {idx}: cost_basis diverges: backtest={bt_lot[2]}, live={live_lot[2]}"
            )

    assert bt_summary["hwm"].keys() == live_summary["hwm"].keys(), (
        f"HWM tickers diverge: backtest={set(bt_summary['hwm'])}, live={set(live_summary['hwm'])}"
    )
    for tk, bt_hwm in bt_summary["hwm"].items():
        assert bt_hwm == pytest.approx(live_summary["hwm"][tk]), (
            f"{tk}: HWM diverges: backtest={bt_hwm}, live={live_summary['hwm'][tk]}"
        )

    assert bt_summary["peak"] == pytest.approx(live_summary["peak"]), (
        f"peak diverges: backtest.peak_value={bt_summary['peak']}, live.peak_equity={live_summary['peak']}"
    )
    assert bt_summary["cash"] == pytest.approx(live_summary["cash"]), (
        f"cash diverges: backtest.cash={bt_summary['cash']}, live.available_cash={live_summary['cash']}"
    )


def _build_drop_price_frame(start: date, n_bars: int, drop_at: int, drop_pct: float) -> pd.DataFrame:
    """Flat-then-drop OHLCV frame: ``drop_pct`` decline at bar ``drop_at``.

    Deterministic — no randomness — so the test asserts exact equivalence
    rather than tolerance windows.
    """
    closes = np.full(n_bars, 100.0)
    closes[drop_at:] = 100.0 * (1.0 - drop_pct)
    dates: list[date] = []
    current = start
    while len(dates) < n_bars:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": np.full(n_bars, 1_000_000.0),
        },
        index=dates,
    )


def test_stoploss_sell_state_evolution_matches_backtest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both engines execute the same StopLoss sell and end with matching state.

    Drives a price drop large enough to trigger ``StopLoss(loss_threshold=0.10)``
    on a held position. Asserts the resulting lot list (empty after full exit),
    cash (credited with sale proceeds), HWM (cleared on full exit), and peak
    equity all match between backtest and live. The earlier no-action parity
    test only exercises HWM/peak ratchets; this one exercises the order
    sizer, FIFO consumption, and the on-exit cleanup invariants — the actual
    plumbing that makes the two engines comparable in real use.
    """
    ticker = "AAPL"
    shares = 10.0
    cash = 0.0
    n_bars = 20
    drop_at = 5
    start = date(2026, 1, 5)
    prices = _build_drop_price_frame(start, n_bars, drop_at, drop_pct=0.20)
    seed_day = prices.index[0]
    seed_close = float(prices["close"].iloc[0])

    portfolio = PortfolioConfig(
        holdings=[Holding(ticker=ticker, shares=shares, cost_basis=seed_close)],
        available_cash=cash,
    )
    constraints = AllocationConstraints()

    # ---- Backtest --------------------------------------------------------
    bt_engine = _CapturingBacktestEngine(
        allocator=Allocator(entries=[], constraints=constraints, n_tickers=1),
        order_sizer=OrderSizer(),
        exit_rules=[StopLoss(loss_threshold=0.10)],
        constraints=constraints,
        enable_split=False,
        execution_mode="close",  # Avoid next-day fill lag for cleaner parity.
    )
    bt_engine.run(portfolio, {ticker: prices}, prices.index[0], prices.index[-1])
    bt_state = bt_engine.captured_state
    assert bt_state is not None
    assert bt_state.positions.get(ticker, 0) == 0, "backtest should fully exit on StopLoss"

    # ---- Live ------------------------------------------------------------
    state_path = tmp_path / "state.yaml"
    _seed_state_to_match_backtest(
        state_path=state_path,
        ticker=ticker,
        shares=shares,
        seed_close=seed_close,
        seed_day=seed_day,
        cash=cash,
    )

    for current_day in prices.index:
        provider = _make_provider_for_window(prices, ticker, current_day)

        class FakeDate:
            value: date = current_day

            @staticmethod
            def today() -> date:
                return FakeDate.value

        monkeypatch.setattr("midas.live.date", FakeDate)

        live_engine = LiveEngine(
            portfolio=portfolio,
            allocator=Allocator(entries=[], constraints=constraints, n_tickers=1),
            order_sizer=OrderSizer(),
            provider=provider,
            state_path=state_path,
            exit_rules=[StopLoss(loss_threshold=0.10)],
            constraints=constraints,
            history_days=n_bars * 2,
        )
        try:
            live_engine._tick([ticker])
        finally:
            live_engine.close()

    final_live = load_state(state_path)

    # ---- Compare ---------------------------------------------------------
    assert ticker not in final_live.lots, "live should empty lots after full exit"
    # HWM clear-on-exit invariant: backtest pops state.high_water_marks on
    # full exit at backtest.py:_execute. Live now does the same in
    # apply_sell — this asserts the symmetry.
    assert ticker not in final_live.high_water_marks, (
        "live should clear HWM on full exit (regression for the C2 leak the reviewer flagged)"
    )
    assert bt_state.cash == pytest.approx(final_live.available_cash), (
        f"cash diverges after StopLoss sell: backtest={bt_state.cash}, live={final_live.available_cash}"
    )
    # Peak must match — both engines saw the same equity trajectory.
    assert bt_state.peak_value == pytest.approx(final_live.peak_equity), (
        f"peak diverges: backtest.peak_value={bt_state.peak_value}, live.peak_equity={final_live.peak_equity}"
    )
