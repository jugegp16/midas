"""Live analysis engine — polls prices and emits alerts in real time."""

from __future__ import annotations

import logging
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

from midas.allocator import AllocationResult, Allocator
from midas.data.price_history import PriceHistory
from midas.data.provider import DataProvider
from midas.live_state import LiveState, aggregate_cost_basis, apply_buy, apply_sell, load_or_seed, save_atomic
from midas.models import (
    AllocationConstraints,
    Direction,
    PortfolioConfig,
)
from midas.order_sizer import OrderSizer
from midas.output import print_alert, print_status
from midas.restrictions import RestrictionTracker
from midas.strategies.base import ExitRule, max_warmup, warmup_bars_to_calendar_days

logger = logging.getLogger(__name__)


class LiveEngine:
    def __init__(
        self,
        portfolio: PortfolioConfig,
        allocator: Allocator,
        order_sizer: OrderSizer,
        provider: DataProvider,
        state_path: Path,
        exit_rules: list[ExitRule] | None = None,
        constraints: AllocationConstraints | None = None,
        poll_interval: int = 60,
        dry_run: bool = False,
        history_days: int | None = None,
    ) -> None:
        self._state_path = state_path
        self._state: LiveState = load_or_seed(portfolio, state_path)
        self._portfolio = portfolio
        self._allocator = allocator
        self._order_sizer = order_sizer
        self._exit_rules = exit_rules or []
        self._constraints = constraints or AllocationConstraints()
        self._provider = provider
        self._poll_interval = poll_interval
        self._dry_run = dry_run
        # Derive the history window from the largest warmup required across
        # configured strategies (plus slack for weekends/holidays). An explicit
        # ``history_days`` override is still honored for tests.
        if history_days is not None:
            self._history_days = history_days
        else:
            warmup_bars = max_warmup([*allocator.strategies, *self._exit_rules])
            self._history_days = warmup_bars_to_calendar_days(warmup_bars)
        # Track (ticker, direction, shares) from last tick to suppress duplicate alerts
        self._last_order_keys: set[tuple[str, Direction, float]] = set()
        self._restriction_tracker: RestrictionTracker | None = None
        if portfolio.trading_restrictions:
            self._restriction_tracker = RestrictionTracker(
                portfolio.trading_restrictions,
            )

    def run(self) -> None:
        tickers = [holding.ticker for holding in self._portfolio.holdings]
        print_status(
            f"Starting {'dry run' if self._dry_run else 'live'} analysis "
            f"for {len(tickers)} tickers, polling every {self._poll_interval}s"
        )

        try:
            while True:
                self._tick(tickers)
                time.sleep(self._poll_interval)
        except KeyboardInterrupt:
            print_status("Stopped.")

    def _tick(self, tickers: list[str]) -> None:
        today = date.today()

        # Fetch recent history for all tickers and convert to PriceHistory at
        # the boundary. Both calls share a try/except so a misshapen frame
        # (missing OHLCV columns, bad index) skips that ticker for the tick
        # instead of crashing the entire poll loop.
        end = today
        start = end - timedelta(days=self._history_days)
        price_data: dict[str, pd.DataFrame] = {}
        price_history: dict[str, PriceHistory] = {}
        for ticker in tickers:
            try:
                df = self._provider.get_history(ticker, start, end)
                price_history[ticker] = PriceHistory.from_dataframe(df)
                price_data[ticker] = df
            except Exception as exc:
                print_status(f"Warning: failed to fetch {ticker}: {exc}")

        if not price_data:
            return

        # If any held position is missing from price_data, we can't compute an
        # accurate portfolio denominator for current_weights — skip the tick
        # rather than let Option A hold inflated weights based on partial info.
        missing_held = [
            holding.ticker
            for holding in self._portfolio.holdings
            if holding.shares > 0 and holding.ticker not in price_data
        ]
        if missing_held:
            print_status(f"Skipping tick: missing price data for held positions {missing_held}. Will retry next poll.")
            return

        current_prices: dict[str, float] = {}
        for ticker in tickers:
            if ticker in price_data and len(price_data[ticker]) > 0:
                current_prices[ticker] = float(price_data[ticker]["close"].iloc[-1])

        active_tickers = [ticker for ticker in tickers if ticker in price_data]

        # Advance per-ticker HWM in state to the latest close (never regress).
        for ticker in active_tickers:
            close = current_prices[ticker]
            self._state.high_water_marks[ticker] = max(self._state.high_water_marks.get(ticker, close), close)

        # Current positions + weights (weights feed Option A: neutral=hold).
        # Positions are derived from state lots, not the YAML.
        positions = {ticker: sum(lot.shares for lot in self._state.lots.get(ticker, [])) for ticker in active_tickers}

        # Pass None (not {}) when the denominator is zero so the allocator
        # falls back to its equal-weight baseline.
        total_value = self._state.available_cash + sum(
            positions[ticker] * current_prices[ticker] for ticker in active_tickers
        )
        current_weights: dict[str, float] | None = None
        if total_value > 0:
            current_weights = {
                ticker: (positions[ticker] * current_prices[ticker]) / total_value for ticker in active_tickers
            }

        # Phase 1: Allocator scores entry signals and blends to target weights.
        allocation = self._allocator.allocate(
            active_tickers,
            price_history,
            current_weights=current_weights,
        )

        # Phase 2: Exit rules clamp proposed targets downward (LEAN pattern).
        clamped_targets = dict(allocation.targets)
        clamp_attribution: dict[str, tuple[str, str]] = {}
        for rule in self._exit_rules:
            for ticker in active_tickers:
                if positions.get(ticker, 0.0) <= 0:
                    continue
                proposed = clamped_targets.get(ticker, 0.0)
                if proposed <= 0:
                    continue
                cost_basis = aggregate_cost_basis(self._state.lots.get(ticker, []))
                hwm = self._state.high_water_marks.get(ticker, current_prices[ticker])
                clamped = rule.clamp_target(ticker, proposed, price_history[ticker], cost_basis, hwm)
                if clamped < proposed:
                    clamped_targets[ticker] = clamped
                    if ticker not in clamp_attribution:
                        reason = rule.clamp_reason(ticker, price_history[ticker], cost_basis, hwm)
                        clamp_attribution[ticker] = (rule.name, reason)

        # Size sells and filter restriction-blocked sells *before* computing
        # post-sell cash. Otherwise a blocked sell would leak phantom proceeds
        # into the buy pass, sizing buys against cash that will never arrive.
        exit_orders = self._order_sizer.size_sells(
            clamped_targets,
            positions,
            current_prices,
            total_value,
            clamp_attribution,
        )
        if self._restriction_tracker:
            exit_orders = [
                order
                for order in exit_orders
                if not self._restriction_tracker.is_blocked(order.ticker, order.direction, today)
            ]
        sell_proceeds = sum(order.estimated_value for order in exit_orders)
        post_sell_cash = self._state.available_cash + sell_proceeds

        clamped_allocation = AllocationResult(
            targets=clamped_targets,
            contributions=allocation.contributions,
            blended_scores=allocation.blended_scores,
        )

        buy_orders = self._order_sizer.size_buys(
            clamped_allocation,
            positions,
            current_prices,
            post_sell_cash,
            self._constraints,
            total_value=total_value,
        )
        if self._restriction_tracker:
            buy_orders = [
                order
                for order in buy_orders
                if not self._restriction_tracker.is_blocked(order.ticker, order.direction, today)
            ]

        filtered = exit_orders + buy_orders

        # Apply assumed fills to the in-memory state.
        for order in filtered:
            if order.shares <= 0:
                continue
            if order.direction == Direction.BUY:
                apply_buy(self._state, order.ticker, order.shares, order.price, today)
            else:
                apply_sell(self._state, order.ticker, order.shares, order.price, today)

        # Update peak equity from the current portfolio value (post-fills).
        positions_after = {
            ticker: sum(lot.shares for lot in self._state.lots.get(ticker, [])) for ticker in active_tickers
        }
        current_equity = self._state.available_cash + sum(
            positions_after[ticker] * current_prices[ticker] for ticker in active_tickers
        )
        self._state.peak_equity = max(self._state.peak_equity or 0.0, current_equity)

        # Advance cash infusion if due.
        infusion = self._portfolio.cash_infusion
        if (
            infusion is not None
            and self._state.cash_infusion_next_date is not None
            and today >= self._state.cash_infusion_next_date
        ):
            self._state.available_cash += infusion.amount
            # CashInfusion.advance() mutates next_date in place; align it with state, advance, copy back.
            infusion.next_date = self._state.cash_infusion_next_date
            infusion.advance()
            self._state.cash_infusion_next_date = infusion.next_date

        # Persist state at the end of the tick (HWM/peak/infusion always advance,
        # even on no-change ticks where alert printing is suppressed below).
        save_atomic(self._state, self._state_path)

        # Emit alerts only when the order set changes
        current_keys = {(order.ticker, order.direction, order.shares) for order in filtered if order.shares > 0}
        if current_keys == self._last_order_keys:
            return
        self._last_order_keys = current_keys

        now = datetime.now(tz=UTC)
        remaining_cash = self._state.available_cash
        for order in filtered:
            if order.shares <= 0:
                continue
            if order.direction == Direction.BUY:
                remaining_cash -= order.estimated_value
            else:
                remaining_cash += order.estimated_value
            print_alert(order, remaining_cash, now, dry_run=self._dry_run)
