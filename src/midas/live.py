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
from midas.live_state import LiveState, load_or_seed
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

        # Current positions + weights (weights feed Option A: neutral=hold).
        positions = {}
        for ticker in active_tickers:
            held = self._portfolio.get_holding(ticker)
            positions[ticker] = held.shares if held else 0.0

        # Pass None (not {}) when the denominator is zero so the allocator
        # falls back to its equal-weight baseline.
        total_value = self._portfolio.available_cash + sum(
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
                holding = self._portfolio.get_holding(ticker)
                if holding is None:
                    continue
                if holding.cost_basis is None:
                    logger.warning(
                        "%s: no cost_basis in portfolio config — using current "
                        "price as fallback. Stop-loss and profit-taking exits "
                        "are effectively disabled for this ticker until a real "
                        "basis is recorded.",
                        ticker,
                    )
                    cost_basis = current_prices[ticker]
                else:
                    cost_basis = holding.cost_basis
                hwm = max(cost_basis, current_prices[ticker])
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
        post_sell_cash = self._portfolio.available_cash + sell_proceeds

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

        # Emit alerts only when the order set changes
        current_keys = {(order.ticker, order.direction, order.shares) for order in filtered if order.shares > 0}
        if current_keys == self._last_order_keys:
            return
        self._last_order_keys = current_keys

        now = datetime.now(tz=UTC)
        remaining_cash = self._portfolio.available_cash
        for order in filtered:
            if order.shares <= 0:
                continue
            if order.direction == Direction.BUY:
                remaining_cash -= order.estimated_value
            else:
                remaining_cash += order.estimated_value
            print_alert(order, remaining_cash, now, dry_run=self._dry_run)
