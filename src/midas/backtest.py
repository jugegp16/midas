"""Backtest engine — replays historical data through strategies."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd

from midas.allocator import AllocationResult, Allocator, AllocatorRiskTelemetry
from midas.data.price_history import PriceHistory
from midas.live_state import aggregate_cost_basis, consume_lots_fifo
from midas.metrics import (
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
    Direction,
    HoldingPeriod,
    Order,
    PortfolioConfig,
    PositionLot,
    TradeRecord,
)
from midas.order_sizer import OrderSizer
from midas.restrictions import RestrictionTracker
from midas.results import BacktestResult
from midas.risk import per_ticker_vol_contribution
from midas.risk_metrics import RiskHistory, compute_risk_metrics
from midas.strategies.base import ExitRule, max_warmup

# ---------------------------------------------------------------------------
# Constants & type aliases
# ---------------------------------------------------------------------------

ExecutionMode = Literal["close", "next_open", "next_close"]
"""When orders computed on day T actually execute.

- ``"close"`` — day T close (legacy, optimistic; signals see the bar they
  trade on). Useful for pinning regression tests but not a realistic
  simulation of live trading.
- ``"next_open"`` — day T+1 open (honest default; orders submitted at T's
  close fill at the next session's open).
- ``"next_close"`` — day T+1 close (market-on-close at the next session).

Under lagged modes the last simulated day's decision never executes —
there is no T+1 bar in the window. That matches reality: an order placed
after the final session can't fill inside the backtest.
"""

DEFAULT_TRAIN_PCT = 0.70


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class _TickerIndex:
    """Pre-computed PriceHistory and cursor for a single ticker's price data."""

    dates: list[date]
    history: PriceHistory
    ptr: int = 0


@dataclass
class _Decision:
    """Allocator+exit-rule output, deferred for later execution under lag.

    ``allocation.targets`` is the *clamped* target weight dict (after
    exit rules reduced the raw allocator proposal). ``clamp_attribution``
    records which exit rule fired per ticker for sell-side attribution.
    Under ``execution_mode="close"`` the decision is sized and executed
    the same tick; under lagged modes it is stored on ``_SimState.pending``
    and re-sized against the next bar's prices when it fills.
    """

    allocation: AllocationResult
    clamp_attribution: dict[str, tuple[str, str]]
    active_tickers: list[str]
    decision_day: date

    def filtered(self, available: set[str]) -> _Decision | None:
        """Return a copy keeping only tickers in *available*, or None if empty."""
        active = [ticker for ticker in self.active_tickers if ticker in available]
        if not active:
            return None
        keep = set(active)
        # ``risk_telemetry`` is intentionally not threaded through this copy —
        # the engine consumes telemetry once, in ``_run_day``, from the
        # original ``_decide`` output (before any pending-fill or
        # rebalance copy is made). A filtered/pending decision feeds only
        # the order sizer, which doesn't read ``risk_telemetry``.
        return _Decision(
            allocation=AllocationResult(
                targets={ticker: weight for ticker, weight in self.allocation.targets.items() if ticker in keep},
                contributions={ticker: val for ticker, val in self.allocation.contributions.items() if ticker in keep},
                blended_scores={
                    ticker: val for ticker, val in self.allocation.blended_scores.items() if ticker in keep
                },
            ),
            clamp_attribution={key: val for key, val in self.clamp_attribution.items() if key in keep},
            active_tickers=active,
            decision_day=self.decision_day,
        )


@dataclass
class _SimState:
    """Mutable simulation state carried across trading days."""

    positions: dict[str, float] = field(default_factory=dict)
    # Cost basis recorded at the moment each SELL trade was executed, in the
    # same order as SELL entries in `trades`. Each entry is the share-weighted
    # average of the FIFO lots actually consumed by that sell — not the
    # weighted-average of all lots — so partial sells of mixed-basis positions
    # attribute P&L to the correct lots. A parallel list (rather than a
    # (date, ticker)-keyed dict) is required so that multiple sells of the
    # same ticker on the same day — e.g. from different strategies — each get
    # their own basis snapshot instead of overwriting one another.
    basis_per_sell: list[float] = field(default_factory=list)
    bh_positions: dict[str, float] = field(default_factory=dict)
    # Per-ticker FIFO list of open lots. Used for cost-basis accounting
    # and FIFO holding-period classification on sells.
    lots: dict[str, list[PositionLot]] = field(default_factory=dict)
    # Aggregate per-ticker high-water mark for exit rule evaluation.
    # Updated each tick with the day's closing price.
    high_water_marks: dict[str, float] = field(default_factory=dict)
    cash: float = 0.0
    starting_value: float = 0.0
    trades: list[TradeRecord] = field(default_factory=list)
    last_day: date | None = None
    split_value: float | None = None
    split_bh_value: float | None = None
    restriction_tracker: RestrictionTracker | None = None
    twr_base_value: float = 0.0  # portfolio value after last cash infusion
    twr_periods: list[float] = field(default_factory=list)  # sub-period returns
    twr_split_idx: int | None = None  # index into twr_periods at train/test split
    equity_curve: list[tuple[date, float]] = field(default_factory=list)
    # Buy-and-hold equity curve: portfolio.available_cash + sum(bh_positions *
    # close_prices) per bar. Same semantics as the final B&H value in
    # _build_result — ignores cash infusions, contributes 0 for tickers
    # whose data hasn't yet started — so the curve aligns with the strategy
    # equity curve at t=0 and at the activation step of deferred tickers.
    bh_equity_curve: list[tuple[date, float]] = field(default_factory=list)
    # Running peak portfolio value, updated each bar. Read by the allocator's
    # CPPI overlay to compute the bar's current_drawdown.
    peak_value: float = 0.0
    # Per-ticker cost-basis-weighted attribution: {ticker: {strategy: share}}.
    # Shares sum to 1.0 per ticker. Updated on buy as a basis-weighted blend
    # of the prior dict and the order's contributions; consumed on sell to
    # split realized P&L into per-strategy buckets. Reset on full exit.
    attribution: dict[str, dict[str, float]] = field(default_factory=dict)
    # Cumulative attributed realised P&L by strategy. Surfaced via RiskMetrics.
    cumulative_strategy_pnl: dict[str, float] = field(default_factory=dict)
    # Decision made at the previous decision day, waiting to execute on
    # the current bar under ``execution_mode="next_open"|"next_close"``.
    # Always ``None`` under legacy ``execution_mode="close"``.
    pending: _Decision | None = None
    # Per-bar risk-engine telemetry from the latest allocator call. Refreshed
    # each tick by ``_run_day``; consumed by ``_simulate`` when appending to
    # ``risk_history`` so per-bar arrays stay aligned with ``equity_curve``.
    last_risk_telemetry: AllocatorRiskTelemetry = field(default_factory=AllocatorRiskTelemetry)
    risk_history: RiskHistory = field(default_factory=RiskHistory)
    vol_target_skip_count: int = 0

    def portfolio_value(self, close_prices: dict[str, float]) -> float:
        """Cash + mark-to-market of held positions at *close_prices*."""
        return self.cash + sum(
            shares * close_prices[ticker]
            for ticker, shares in self.positions.items()
            if shares > 0 and ticker in close_prices
        )

    def close_twr_period(self, current_value: float) -> None:
        """Close a TWR sub-period and reset the base for the next one."""
        if self.twr_base_value > 0:
            self.twr_periods.append(current_value / self.twr_base_value)
        self.twr_base_value = current_value


def _close_prices(current_data: dict[str, PriceHistory]) -> dict[str, float]:
    """Latest close price for every ticker in *current_data*."""
    return {ticker: float(current_data[ticker].close[-1]) for ticker in current_data}


def _ticker_basis(state: _SimState, ticker: str) -> float:
    """Aggregate cost basis for *ticker* — sum of (shares * cost_basis) across open lots."""
    return sum(lot.shares * lot.cost_basis for lot in state.lots.get(ticker, []))


def _update_buy_attribution(state: _SimState, ticker: str, order: Order) -> None:
    """Blend the order's contributions into the ticker's running attribution dict.

    The blend is cost-basis-weighted: prior attribution gets weight
    ``prior_basis / new_basis``; the new order's contributions get
    ``buy_size / new_basis``. The resulting dict is renormalised to sum to 1.

    The typical case (a single strategy entering, optionally adding) is exact;
    accuracy degrades gracefully when many strategies contribute over many bars
    (spec §"Risk Telemetry").
    """
    contributions = order.context.contributions
    if not contributions:
        return
    buy_size = order.shares * order.price
    if buy_size <= 0:
        return
    # Prior basis = sum of all open lots EXCLUDING the buy we just appended.
    prior_basis = max(_ticker_basis(state, ticker) - buy_size, 0.0)
    new_basis = prior_basis + buy_size
    if new_basis <= 0:
        return
    prior = state.attribution.get(ticker, {})
    blended: dict[str, float] = {strat: share * (prior_basis / new_basis) for strat, share in prior.items()}
    for strat, contrib in contributions.items():
        blended[strat] = blended.get(strat, 0.0) + contrib * (buy_size / new_basis)
    total = sum(blended.values())
    if total > 0:
        state.attribution[ticker] = {strat: share / total for strat, share in blended.items()}


def _split_pnl_by_attribution(state: _SimState, ticker: str, realized_pnl: float) -> None:
    """Add *realized_pnl* into ``state.cumulative_strategy_pnl`` split by current attribution."""
    attr = state.attribution.get(ticker)
    if not attr:
        return
    for strat, share in attr.items():
        state.cumulative_strategy_pnl[strat] = state.cumulative_strategy_pnl.get(strat, 0.0) + realized_pnl * share


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------


class BacktestEngine:
    """Replays historical price data through strategies to evaluate performance.

    Orchestrates allocation, exit rules, order sizing, and trade execution
    across a date range, producing a BacktestResult with performance metrics.
    """

    def __init__(
        self,
        allocator: Allocator,
        order_sizer: OrderSizer,
        exit_rules: list[ExitRule] | None = None,
        constraints: AllocationConstraints | None = None,
        train_pct: float = DEFAULT_TRAIN_PCT,
        enable_split: bool = True,
        log_fn: Callable[[str], None] | None = None,
        execution_mode: ExecutionMode = "next_open",
    ) -> None:
        self._allocator = allocator
        self._order_sizer = order_sizer
        self._exit_rules = exit_rules or []
        self._constraints = constraints or AllocationConstraints()
        self._train_pct = train_pct
        self._enable_split = enable_split
        self._log = log_fn or (lambda _msg: None)
        self._execution_mode: ExecutionMode = execution_mode

    def run(
        self,
        portfolio: PortfolioConfig,
        price_data: dict[str, pd.DataFrame],
        start: date,
        end: date,
    ) -> BacktestResult:
        """Execute a full backtest over the given date range.

        Args:
            portfolio: Holdings, cash, and configuration.
            price_data: Per-ticker OHLCV DataFrames keyed by
                ticker symbol.
            start: First calendar date of the simulation window.
            end: Last calendar date of the simulation window.

        Returns:
            A BacktestResult containing trades, metrics, and the
            equity curve.
        """
        trading_days = self._collect_trading_days(price_data, start, end)
        _, split_date = self._compute_split(trading_days)
        ticker_idx = self._build_ticker_index(price_data, start, end)
        state = self._init_positions(portfolio, price_data, trading_days, start, end)

        # Precompute strategy signals over the full PriceHistory (one-time cost).
        full_history = {ticker: idx.history for ticker, idx in ticker_idx.items()}
        self._allocator.precompute_signals(full_history)

        deferred = self._find_deferred(portfolio, price_data, trading_days, start, end)

        self._simulate(
            state,
            portfolio,
            trading_days,
            ticker_idx,
            deferred,
            split_date,
        )

        return self._build_result(
            state,
            portfolio,
            price_data,
            trading_days,
            split_date,
        )

    # -- Phase helpers --------------------------------------------------------

    def _collect_trading_days(
        self,
        price_data: dict[str, pd.DataFrame],
        start: date,
        end: date,
    ) -> list[date]:
        all_dates: set[date] = set()
        for df in price_data.values():
            all_dates.update(dt for dt in df.index if start <= dt <= end)
        days = sorted(all_dates)
        if not days:
            msg = "No trading days found in date range"
            raise ValueError(msg)
        return days

    def _compute_split(
        self,
        trading_days: list[date],
    ) -> tuple[int, date | None]:
        if self._enable_split:
            split_idx = int(len(trading_days) * self._train_pct)
            if split_idx < len(trading_days):
                return split_idx, trading_days[split_idx]
        return len(trading_days), None

    def _build_ticker_index(
        self,
        price_data: dict[str, pd.DataFrame],
        start: date,
        end: date,
    ) -> dict[str, _TickerIndex]:
        """Build per-ticker PriceHistory keeping the warmup prefix intact.

        The returned history spans ``[max(first_available, start - warmup_bars), end]``
        so ``precompute_signals`` and the per-day pointer advance see a prefix
        of history before the user's ``start``. The simulation loop iterates
        over ``trading_days`` (which are already filtered to ``start..end``),
        so the pointer walks through warmup dates on day one of the sim.
        """
        warmup_bars = self._warmup_bars()
        index: dict[str, _TickerIndex] = {}
        for ticker, df in price_data.items():
            bounded = df[df.index <= end]
            if len(bounded) == 0:
                continue
            # Identify the first trading day inside the sim window.
            sim_mask = np.asarray(bounded.index >= start)
            if not sim_mask.any():
                continue
            sim_first_idx = int(np.argmax(sim_mask))
            # Keep up to ``warmup_bars`` prior bars for precompute.
            warmup_first_idx = max(0, sim_first_idx - warmup_bars)
            available_warmup = sim_first_idx - warmup_first_idx
            if warmup_bars > 0 and available_warmup < warmup_bars:
                self._log(
                    f"{ticker}: only {available_warmup} warmup bars available "
                    f"(requested {warmup_bars}) — strategies will score on "
                    f"partial history until enough bars accumulate"
                )
            sliced = bounded.iloc[warmup_first_idx:]
            index[ticker] = _TickerIndex(
                dates=list(sliced.index),
                history=PriceHistory.from_dataframe(sliced),
            )
        return index

    def _warmup_bars(self) -> int:
        """Max warmup required across entry signals and exit rules."""
        return max_warmup([*self._allocator.strategies, *self._exit_rules])

    def _first_data_dates(
        self,
        price_data: dict[str, pd.DataFrame],
        start: date,
    ) -> dict[str, date]:
        result: dict[str, date] = {}
        for ticker, df in price_data.items():
            in_range = df[df.index >= start]
            if len(in_range) > 0:
                result[ticker] = in_range.index[0]
        return result

    def _init_positions(
        self,
        portfolio: PortfolioConfig,
        price_data: dict[str, pd.DataFrame],
        trading_days: list[date],
        start: date,
        end: date,
    ) -> _SimState:
        state = _SimState(cash=portfolio.available_cash)
        if portfolio.trading_restrictions:
            state.restriction_tracker = RestrictionTracker(
                portfolio.trading_restrictions,
            )
        first_dates = self._first_data_dates(price_data, start)

        for holding in portfolio.holdings:
            if holding.shares <= 0:
                continue

            if holding.ticker not in first_dates:
                self._log(f"{holding.ticker}: no price data in backtest range ({start} to {end}) — excluded")
                continue

            ticker_start = first_dates[holding.ticker]
            if ticker_start <= trading_days[0]:
                # Backtest seeds the cost basis from the start-day market price, not the YAML
                # ``cost_basis``. The YAML value is the user's real purchase basis (used by the
                # live engine and for display) — using it here would let exit rules fire on
                # pre-backtest gains, distorting strategy performance.
                entry_df = price_data[holding.ticker][price_data[holding.ticker].index >= start]
                entry_price = float(entry_df["close"].iloc[0])
                state.positions[holding.ticker] = holding.shares
                state.bh_positions[holding.ticker] = holding.shares
                state.lots[holding.ticker] = [
                    PositionLot(
                        shares=holding.shares,
                        purchase_date=trading_days[0],
                        cost_basis=entry_price,
                    )
                ]
                state.high_water_marks[holding.ticker] = entry_price

        state.starting_value = state.cash + sum(
            lot.shares * lot.cost_basis for lots in state.lots.values() for lot in lots
        )
        state.twr_base_value = state.starting_value
        state.peak_value = state.starting_value
        return state

    def _find_deferred(
        self,
        portfolio: PortfolioConfig,
        price_data: dict[str, pd.DataFrame],
        trading_days: list[date],
        start: date,
        end: date,
    ) -> dict[str, float]:
        first_dates = self._first_data_dates(price_data, start)
        deferred: dict[str, float] = {}

        for holding in portfolio.holdings:
            if holding.shares <= 0 or holding.ticker not in first_dates:
                continue
            ticker_start = first_dates[holding.ticker]
            if ticker_start > trading_days[0]:
                deferred[holding.ticker] = holding.shares
                self._log(
                    f"{holding.ticker}: data starts {ticker_start} (after backtest start {start})"
                    f" — position deferred until first available date"
                )
        return deferred

    def _simulate(
        self,
        state: _SimState,
        portfolio: PortfolioConfig,
        trading_days: list[date],
        ticker_idx: dict[str, _TickerIndex],
        deferred: dict[str, float],
        split_date: date | None,
    ) -> None:
        for ticker, shares in deferred.items():
            state.bh_positions[ticker] = shares
            state.positions[ticker] = 0.0

        deferred_activated: set[str] = set()

        for day in trading_days:
            state.last_day = day

            # Advance pointers and build current slices
            current_data: dict[str, PriceHistory] = {}
            for ticker, idx in ticker_idx.items():
                while idx.ptr < len(idx.dates) and idx.dates[idx.ptr] <= day:
                    idx.ptr += 1
                if idx.ptr > 0:
                    current_data[ticker] = idx.history[: idx.ptr]

            # Activate deferred holdings
            self._activate_deferred(
                state,
                deferred,
                deferred_activated,
                current_data,
                day,
            )

            # Capture split snapshot
            if split_date and day == split_date and state.split_value is None:
                closes = _close_prices(current_data)
                state.split_value = state.portfolio_value(closes)
                state.split_bh_value = portfolio.available_cash + sum(
                    state.bh_positions.get(ticker, 0) * closes.get(ticker, 0) for ticker in state.bh_positions
                )
                state.close_twr_period(state.split_value)
                state.twr_split_idx = len(state.twr_periods)

            self._run_day(state, portfolio, current_data, day)

            close_prices = _close_prices(current_data)
            value = state.portfolio_value(close_prices)
            state.equity_curve.append((day, value))
            if value > state.peak_value:
                state.peak_value = value

            bh_value = portfolio.available_cash + sum(
                shares * close_prices.get(ticker, 0.0) for ticker, shares in state.bh_positions.items()
            )
            state.bh_equity_curve.append((day, bh_value))

            telemetry = state.last_risk_telemetry
            drawdown = (state.peak_value - value) / state.peak_value if state.peak_value > 0 else 0.0
            # Record *actual* gross exposure (market value of positions / total
            # value), not the allocator's target output. The allocator
            # constructs-to-budget so its targets are always near
            # ``(1 - min_cash_pct)`` by design — a flat line that hides what
            # the strategy actually deployed. ``positions_value / total_value``
            # captures real cash drag from min_buy_delta, exit clamps, and
            # waiting-for-fills lag.
            positions_value = sum(
                shares * close_prices.get(ticker, 0.0) for ticker, shares in state.positions.items() if shares > 0
            )
            actual_gross = positions_value / value if value > 0 else 0.0
            state.risk_history.dates.append(day)
            state.risk_history.gross_exposure.append(actual_gross)
            state.risk_history.cppi_scale.append(telemetry.cppi_scale)
            state.risk_history.vol_target_scale.append(telemetry.vol_target_scale)
            state.risk_history.vol_target_predicted_vol.append(telemetry.vol_target_predicted_vol)
            state.risk_history.drawdown.append(drawdown)

    def _activate_deferred(
        self,
        state: _SimState,
        deferred: dict[str, float],
        activated: set[str],
        current_data: dict[str, PriceHistory],
        day: date,
    ) -> None:
        for ticker, shares in deferred.items():
            if ticker in activated:
                continue
            if ticker in current_data:
                entry_price = float(current_data[ticker].close[-1])
                added_value = shares * entry_price

                # Treat deferred activation like a capital infusion for TWR: close the current
                # sub-period on existing positions (excluding the newly activated ticker), then
                # reset the base to include the new position. Otherwise the new capital would be
                # counted as pure return by the closing final_value / twr_base ratio.
                pre_activation_value = state.portfolio_value(_close_prices(current_data))
                state.close_twr_period(pre_activation_value)
                state.twr_base_value = pre_activation_value + added_value

                state.positions[ticker] = shares
                state.lots[ticker] = [
                    PositionLot(
                        shares=shares,
                        purchase_date=day,
                        cost_basis=entry_price,
                    )
                ]
                state.high_water_marks[ticker] = entry_price
                state.starting_value += added_value
                activated.add(ticker)
                self._log(f"{ticker}: activated on {day} at ${entry_price:.2f} ({shares} shares)")

    def _run_day(
        self,
        state: _SimState,
        portfolio: PortfolioConfig,
        current_data: dict[str, PriceHistory],
        day: date,
    ) -> None:
        """Single-day tick: credit cash, execute any pending, decide, maybe execute.

        Under ``close`` mode, today's decision executes immediately at
        today's close.  Under ``next_open``/``next_close``, today's
        decision is stashed as ``pending`` and fills on the next bar.
        The last simulated day's decision under lag modes intentionally
        never executes — there is no T+1 bar in the window.
        """
        self._credit_cash_infusion(state, portfolio, current_data, day)

        # Execute yesterday's pending decision at today's prices.
        if state.pending is not None:
            pending = state.pending.filtered(set(current_data))
            if pending is not None:
                price_field = "open" if self._execution_mode == "next_open" else "close"
                exec_prices = {
                    ticker: float(getattr(current_data[ticker], price_field)[-1]) for ticker in pending.active_tickers
                }
                self._size_and_execute(state, pending, exec_prices, day)
            state.pending = None

        self._update_high_water_marks(state, current_data)
        decision = self._decide(state, current_data, day)
        if decision is None:
            state.last_risk_telemetry = AllocatorRiskTelemetry()
            return

        state.last_risk_telemetry = decision.allocation.risk_telemetry
        if state.last_risk_telemetry.vol_target_skipped:
            state.vol_target_skip_count += 1

        if self._execution_mode == "close":
            self._size_and_execute(state, decision, _close_prices(current_data), day)
        else:
            state.pending = decision

    def _credit_cash_infusion(
        self,
        state: _SimState,
        portfolio: PortfolioConfig,
        current_data: dict[str, PriceHistory],
        day: date,
    ) -> None:
        """Credit a due cash infusion, closing the current TWR sub-period.

        A fresh deposit is indistinguishable from return unless we
        snapshot portfolio value *before* crediting and reset the TWR
        base *after*. Otherwise the final_value / twr_base ratio rolls
        the infusion into return, inflating the sim's reported TWR.
        """
        infusion = portfolio.cash_infusion
        if not infusion or infusion.next_date > day:
            return
        pre_infusion_value = state.portfolio_value(_close_prices(current_data))
        state.close_twr_period(pre_infusion_value)
        state.cash += infusion.amount
        state.twr_base_value = pre_infusion_value + infusion.amount
        infusion.advance()

    def _update_high_water_marks(
        self,
        state: _SimState,
        current_data: dict[str, PriceHistory],
    ) -> None:
        """Bump per-ticker HWM to today's close on still-held positions.

        Read by ``TrailingStop`` and any other HWM-driven exit rule when
        it clamps. Must run *before* ``_decide`` so today's decision sees
        a fresh peak.
        """
        for ticker, pos in state.positions.items():
            if pos > 0 and ticker in current_data:
                price = float(current_data[ticker].close[-1])
                state.high_water_marks[ticker] = max(price, state.high_water_marks.get(ticker, 0.0))

    def _decide(
        self,
        state: _SimState,
        current_data: dict[str, PriceHistory],
        day: date,
    ) -> _Decision | None:
        """Run allocator (phase 1) and exit clamps (phase 2).

        Returns ``None`` when there are no active tickers — nothing to
        decide and nothing to execute. Otherwise returns a ``_Decision``
        whose ``allocation.targets`` is the clamped weight dict and whose
        ``clamp_attribution`` identifies which exit rule drove each
        clamp for sell-side attribution.
        """
        active_tickers = [
            ticker for ticker in state.positions if state.positions.get(ticker, 0) > 0 or ticker in current_data
        ]
        active_tickers = [ticker for ticker in active_tickers if ticker in current_data]

        if not active_tickers:
            return None

        current_prices = _close_prices(current_data)

        # Current portfolio weights so the allocator can hold (not
        # drift-correct) tickers whose entry signals don't score today.
        # Pass ``None`` (not ``{}``) when the denominator is zero so the
        # allocator falls back to its equal-weight baseline rather than
        # anchoring held tickers at 0.
        total_value = state.cash + sum(
            state.positions.get(ticker, 0.0) * current_prices[ticker] for ticker in active_tickers
        )
        current_weights: dict[str, float] | None = (
            {
                ticker: (state.positions.get(ticker, 0.0) * current_prices[ticker]) / total_value
                for ticker in active_tickers
            }
            if total_value > 0
            else None
        )

        # Phase 1: allocator scores entry signals and blends to target weights.
        # current_drawdown feeds the optional CPPI overlay (Phase 4a); when the
        # overlay is disabled the allocator ignores this argument.
        current_drawdown = (
            (state.peak_value - total_value) / state.peak_value
            if state.peak_value > 0 and total_value < state.peak_value
            else 0.0
        )
        allocation = self._allocator.allocate(
            active_tickers,
            current_data,
            current_weights=current_weights,
            current_drawdown=current_drawdown,
        )

        # Phase 2: exit rules clamp targets downward (LEAN pattern). Each
        # rule can only reduce a target, never increase. First clamper
        # wins attribution for that ticker.
        clamped_targets = dict(allocation.targets)
        clamp_attribution: dict[str, tuple[str, str]] = {}
        for rule in self._exit_rules:
            for ticker in active_tickers:
                if state.positions.get(ticker, 0.0) <= 0:
                    continue
                proposed = clamped_targets.get(ticker, 0.0)
                if proposed <= 0:
                    continue
                cost_basis = aggregate_cost_basis(state.lots.get(ticker, []))
                hwm = state.high_water_marks.get(ticker, 0.0)
                clamped = rule.clamp_target(
                    ticker,
                    proposed,
                    current_data[ticker],
                    cost_basis,
                    hwm,
                )
                if clamped < proposed:
                    clamped_targets[ticker] = clamped
                    if ticker not in clamp_attribution:
                        reason = rule.clamp_reason(
                            ticker,
                            current_data[ticker],
                            cost_basis,
                            hwm,
                        )
                        clamp_attribution[ticker] = (rule.name, reason)

        clamped_allocation = AllocationResult(
            targets=clamped_targets,
            contributions=allocation.contributions,
            blended_scores=allocation.blended_scores,
            risk_telemetry=allocation.risk_telemetry,
        )
        return _Decision(
            allocation=clamped_allocation,
            clamp_attribution=clamp_attribution,
            active_tickers=active_tickers,
            decision_day=day,
        )

    def _size_and_execute(
        self,
        state: _SimState,
        decision: _Decision,
        exec_prices: dict[str, float],
        day: date,
    ) -> None:
        """Run sell/buy sizing (phases 3-4) and execute (phase 5).

        ``exec_prices`` is the per-ticker fill price for this execution —
        today's close under ``execution_mode="close"``, today's open under
        ``next_open``, or today's close under ``next_close``. The sizer
        re-derives ``total_value`` / ``current_weights`` from these prices
        so the delta math is self-consistent with what will actually fill.
        """
        active_tickers = decision.active_tickers
        positions = {ticker: state.positions.get(ticker, 0.0) for ticker in active_tickers}
        total_value = state.cash + sum(positions[ticker] * exec_prices[ticker] for ticker in active_tickers)

        # Preserve "hold" semantics across price drift between decision and
        # fill.  A pure-hold ticker (no positive entry-signal contributions,
        # no exit clamp) gets its target rewritten to the current weight at
        # exec_prices so the delta collapses to zero.  Under ``close`` mode
        # exec_prices == decision prices so this is a no-op.
        rebalanced_targets = dict(decision.allocation.targets)
        contribs_map = decision.allocation.contributions
        for ticker in active_tickers:
            if ticker in decision.clamp_attribution:
                continue
            contribs = contribs_map.get(ticker, {})
            if any(val > 0 for val in contribs.values()):
                continue
            if total_value <= 0:
                rebalanced_targets[ticker] = 0.0
                continue
            rebalanced_targets[ticker] = (positions[ticker] * exec_prices[ticker]) / total_value

        # ``risk_telemetry`` is intentionally omitted: this rebalance feeds
        # only the order sizer, and ``state.last_risk_telemetry`` was already
        # captured from the pre-rebalance ``decision`` in ``_run_day``.
        rebalanced_allocation = AllocationResult(
            targets=rebalanced_targets,
            contributions=contribs_map,
            blended_scores=decision.allocation.blended_scores,
        )

        # Phase 3: size sells and filter restriction-blocked ones *before*
        # computing post-sell cash. Leaking blocked proceeds would let
        # ``size_buys`` authorize buys against cash that never arrives.
        def unrestricted(orders: list[Order]) -> list[Order]:
            if not state.restriction_tracker:
                return orders
            return [
                order
                for order in orders
                if not state.restriction_tracker.is_blocked(order.ticker, order.direction, day)
            ]

        exit_orders = unrestricted(
            self._order_sizer.size_sells(
                rebalanced_targets,
                positions,
                exec_prices,
                total_value,
                decision.clamp_attribution,
            )
        )
        post_sell_cash = state.cash + sum(order.estimated_value for order in exit_orders)

        buy_orders = unrestricted(
            self._order_sizer.size_buys(
                rebalanced_allocation,
                positions,
                exec_prices,
                post_sell_cash,
                self._constraints,
                total_value=total_value,
            )
        )

        # Phase 5: sells first (proceeds land before buys), then buys.
        # ``_execute`` emits one TradeRecord per holding-period group on
        # sells, so mixed-lot sells crossing the 365-day boundary split
        # into separate ST and LT records with per-group cost basis.
        for order in exit_orders:
            if order.shares <= 0:
                continue
            records = self._execute(order, day, state)
            if not records:
                continue
            for trade, basis in records:
                state.trades.append(trade)
                state.basis_per_sell.append(basis)
            if state.restriction_tracker:
                state.restriction_tracker.record_trade(order.ticker, order.direction, day)
            state.cash += order.estimated_value

        for order in buy_orders:
            if order.shares <= 0:
                continue
            records = self._execute(order, day, state)
            if not records:
                continue
            state.trades.append(records[0][0])
            if state.restriction_tracker:
                state.restriction_tracker.record_trade(order.ticker, order.direction, day)
            state.cash -= order.estimated_value

    @staticmethod
    def _resolve_purchase_date(dates: tuple[date | None, ...]) -> date | str | None:
        """Map a tuple of consumed lots' purchase dates to a single CSV value.

        Empty tuple → None. Single unique non-None date → that date. Anything
        else (multiple dates, any None present) → the literal string 'various'.

        Args:
            dates: Purchase dates of the lots consumed by a single SELL bucket,
                in FIFO order. May contain ``None`` for unseeded lots.

        Returns:
            The single shared date, the literal string ``'various'``, or
            ``None`` when there are no consumed lots.
        """
        if not dates:
            return None
        unique = set(dates)
        if len(unique) == 1:
            only = next(iter(unique))
            return only  # may be a date or None
        return "various"

    @staticmethod
    def _fifo_consumed_basis(lots: list[PositionLot], shares: float) -> float:
        """Share-weighted cost basis of the first *shares* shares in FIFO order.

        Mirrors what ``_execute`` will pop off the lot list, without mutating
        it. Used to record the actual basis for the lots leaving the position
        on a sell, rather than the weighted average of all open lots.
        """
        if shares <= 0 or not lots:
            return 0.0
        remaining = shares
        weighted = 0.0
        consumed = 0.0
        for lot in lots:
            if remaining <= 0:
                break
            take = min(lot.shares, remaining)
            weighted += take * lot.cost_basis
            consumed += take
            remaining -= take
        return weighted / consumed if consumed > 0 else 0.0

    def _execute(
        self,
        order: Order,
        day: date,
        state: _SimState,
    ) -> list[tuple[TradeRecord, float]]:
        """Execute an order and return ``(TradeRecord, cost_basis)`` pairs.

        Buys return a single pair with ``cost_basis=0.0`` (unused). Sells
        FIFO-consume the lot list and emit one pair per holding-period
        bucket, so a sell straddling the 365-day ST/LT boundary produces
        two records — one short-term, one long-term — each with its own
        share count and weighted cost basis.
        """
        ticker = order.ticker
        strategy_name = order.context.source

        if order.direction == Direction.BUY:
            state.positions[ticker] = state.positions.get(ticker, 0) + order.shares
            state.lots.setdefault(ticker, []).append(
                PositionLot(shares=order.shares, purchase_date=day, cost_basis=order.price)
            )
            _update_buy_attribution(state, ticker, order)
            return [
                (
                    TradeRecord(
                        date=day,
                        ticker=ticker,
                        direction=Direction.BUY,
                        shares=order.shares,
                        price=order.price,
                        strategy_name=strategy_name,
                        purchase_date=day,
                    ),
                    0.0,
                )
            ]

        # SELL — FIFO consume lots, bucketing each consumed slice into the
        # short-term or long-term group based on its own purchase date.
        lots = state.lots.get(ticker, [])
        if not lots:
            return []

        breakdown = consume_lots_fifo(lots, order.shares, day)
        st_shares = breakdown.st_shares
        st_weighted_basis = breakdown.st_weighted
        lt_shares = breakdown.lt_shares
        lt_weighted_basis = breakdown.lt_weighted

        new_position = state.positions.get(ticker, 0) - order.shares
        # ``size_sells`` caps at held shares, so reaching _execute with a
        # sell larger than the position is a logic bug — not a float rounding
        # blip — and would fabricate cash if silently clamped to 0. Fail loud.
        assert new_position >= 0, (
            f"sell exceeds position on {ticker}: {order.shares} shares against {state.positions.get(ticker, 0)} held"
        )
        state.positions[ticker] = new_position

        # Split realised P&L into per-strategy buckets via the running
        # cost-basis-weighted attribution dict for this ticker.
        total_basis = st_weighted_basis + lt_weighted_basis
        realized_pnl = order.price * order.shares - total_basis
        _split_pnl_by_attribution(state, ticker, realized_pnl)

        # Full exit resets the high-water mark and the attribution dict.
        # Otherwise a subsequent re-entry would inherit stale attribution
        # shares from the closed position.
        if new_position == 0:
            state.high_water_marks.pop(ticker, None)
            state.attribution.pop(ticker, None)

        records: list[tuple[TradeRecord, float]] = []
        if st_shares > 0:
            records.append(
                (
                    TradeRecord(
                        date=day,
                        ticker=ticker,
                        direction=Direction.SELL,
                        shares=st_shares,
                        price=order.price,
                        strategy_name=strategy_name,
                        holding_period=HoldingPeriod.SHORT_TERM,
                        purchase_date=self._resolve_purchase_date(breakdown.st_purchase_dates),
                    ),
                    st_weighted_basis / st_shares,
                )
            )
        if lt_shares > 0:
            records.append(
                (
                    TradeRecord(
                        date=day,
                        ticker=ticker,
                        direction=Direction.SELL,
                        shares=lt_shares,
                        price=order.price,
                        strategy_name=strategy_name,
                        holding_period=HoldingPeriod.LONG_TERM,
                        purchase_date=self._resolve_purchase_date(breakdown.lt_purchase_dates),
                    ),
                    lt_weighted_basis / lt_shares,
                )
            )
        return records

    def _build_result(
        self,
        state: _SimState,
        portfolio: PortfolioConfig,
        price_data: dict[str, pd.DataFrame],
        trading_days: list[date],
        split_date: date | None,
    ) -> BacktestResult:
        # Use prices at the last trading day within the backtest range, not the
        # last row of the raw frame (which may extend beyond `end` when the
        # caller reuses one price_data dict across multiple sub-windows — e.g.
        # walk-forward fold evaluation).
        end_day = trading_days[-1]
        final_prices: dict[str, float] = {}
        for ticker, df in price_data.items():
            if len(df) == 0:
                continue
            in_range = df[df.index <= end_day]
            if len(in_range) > 0:
                final_prices[ticker] = float(in_range["close"].iloc[-1])
        final_value = state.portfolio_value(final_prices)
        bh_value = portfolio.available_cash + sum(
            state.bh_positions.get(ticker, 0) * final_prices.get(ticker, 0) for ticker in state.bh_positions
        )

        # Close the final TWR sub-period and compound all periods.
        state.close_twr_period(final_value)
        twr = math.prod(state.twr_periods) - 1.0

        starting_val = state.starting_value
        split_bh_val = state.split_bh_value

        if split_date:
            train_trades = [trade for trade in state.trades if trade.date < split_date]
            test_trades = [trade for trade in state.trades if trade.date >= split_date]
        else:
            train_trades = list(state.trades)
            test_trades = []

        # Compute train/test TWR from sub-period boundaries.
        split_idx = state.twr_split_idx
        if split_idx is not None:
            train_twr = 1.0
            for period in state.twr_periods[:split_idx]:
                train_twr *= period
            train_twr -= 1.0
            test_twr = 1.0
            for period in state.twr_periods[split_idx:]:
                test_twr *= period
            test_twr -= 1.0
        else:
            train_twr = twr
            test_twr = twr

        train_return = train_twr
        test_return = test_twr
        if split_bh_val is not None and starting_val > 0:
            train_bh_return = (split_bh_val - starting_val) / starting_val
        elif starting_val > 0:
            train_bh_return = (bh_value - starting_val) / starting_val
        else:
            train_bh_return = 0.0
        if split_bh_val is not None and split_bh_val > 0:
            test_bh_return = (bh_value - split_bh_val) / split_bh_val
        elif starting_val > 0:
            test_bh_return = (bh_value - starting_val) / starting_val
        else:
            test_bh_return = 0.0

        # New metrics
        equity_curve = state.equity_curve
        total_days = (trading_days[-1] - trading_days[0]).days if len(trading_days) > 1 else 0
        if split_date and len(trading_days) > 1:
            train_days = (split_date - trading_days[0]).days
            test_days = (trading_days[-1] - split_date).days
        else:
            train_days = 0
            test_days = 0
        cagr = compute_cagr(starting_val, final_value, total_days)
        max_drawdown = compute_max_drawdown(equity_curve)
        sharpe = compute_sharpe(equity_curve)
        sortino = compute_sortino(equity_curve)
        win_rate, profit_factor, avg_win, avg_loss = compute_trade_stats(state.trades, state.basis_per_sell)
        # Annualize both sides of the efficiency ratio so train and test windows
        # of different lengths compare apples-to-apples (matches the walk-forward
        # convention). Without this, a shorter test window's smaller cumulative
        # return would understate parameter decay, and vice versa.
        if split_date:
            train_ann = compute_annualized_return(train_return, train_days)
            test_ann = compute_annualized_return(test_return, test_days)
            efficiency = test_ann / train_ann if train_ann != 0 else 0.0
        else:
            efficiency = 0.0
        strategy_stats = compute_strategy_stats(state.trades, state.basis_per_sell)

        # Unrealized P&L: mark-to-market gain on positions still held at end.
        unrealized_pnl_by_ticker: dict[str, float] = {}
        for ticker, lots in state.lots.items():
            ticker_pnl = sum(lot.shares * (final_prices.get(ticker, 0.0) - lot.cost_basis) for lot in lots)
            if lots:
                unrealized_pnl_by_ticker[ticker] = round(ticker_pnl, 2)
        unrealized_pnl = sum(unrealized_pnl_by_ticker.values())

        # Per-strategy attribution covers realised P&L from sells. At end-of-run,
        # split each ticker's unrealised P&L into the same per-strategy buckets
        # by current attribution shares so the final RiskMetrics.per_strategy_pnl
        # reflects total attributed P&L (realised + unrealised), per spec line 183.
        attributed_strategy_pnl = dict(state.cumulative_strategy_pnl)
        for ticker, ticker_unrealized in unrealized_pnl_by_ticker.items():
            attr = state.attribution.get(ticker)
            if not attr:
                continue
            for strat, share in attr.items():
                attributed_strategy_pnl[strat] = attributed_strategy_pnl.get(strat, 0.0) + ticker_unrealized * share

        return BacktestResult(
            trades=state.trades,
            final_value=round(final_value, 2),
            starting_value=round(starting_val, 2),
            buy_and_hold_value=round(bh_value, 2),
            train_trades=train_trades,
            test_trades=test_trades,
            train_return=round(train_return, 4),
            test_return=round(test_return, 4),
            train_bh_return=round(train_bh_return, 4),
            test_bh_return=round(test_bh_return, 4),
            split_date=split_date,
            twr=round(twr, 4),
            equity_curve=equity_curve,
            total_days=total_days,
            train_days=train_days,
            test_days=test_days,
            cagr=round(cagr, 4),
            max_drawdown=round(max_drawdown, 4),
            sharpe_ratio=round(sharpe, 4),
            sortino_ratio=round(sortino, 4),
            win_rate=round(win_rate, 4),
            profit_factor=(round(profit_factor, 4) if not math.isinf(profit_factor) else float("inf")),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            efficiency_ratio=round(efficiency, 4),
            strategy_stats=strategy_stats,
            unrealized_pnl=round(unrealized_pnl, 2),
            unrealized_pnl_by_ticker=unrealized_pnl_by_ticker,
            basis_per_sell=state.basis_per_sell,
            risk_metrics=compute_risk_metrics(
                equity_curve=equity_curve,
                vol_target=(self._allocator.risk_config.vol_target if self._allocator.risk_config else None),
                per_strategy_pnl=attributed_strategy_pnl,
                per_ticker_vol_contribution=self._end_state_vol_contribution(state, price_data, end_day),
                risk_history=state.risk_history,
                vol_target_skip_count=state.vol_target_skip_count,
            ),
            risk_history=state.risk_history,
            bh_equity_curve=state.bh_equity_curve,
        )

    def _end_state_vol_contribution(
        self,
        state: _SimState,
        price_data: dict[str, pd.DataFrame],
        end_day: date,
    ) -> dict[str, float]:
        """Per-ticker share of portfolio vol from end-of-run positions.

        Uses the same lookback window as the configured Phase 4b (default 60
        bars when no ``risk_config`` is set). Returns ``{}`` when no risk
        config is configured, or when any held ticker has insufficient
        history / non-positive prices / zero stdev in the window — same
        skip semantics as ``_apply_vol_target``.
        """
        risk_config = self._allocator.risk_config
        if risk_config is None:
            return {}
        held = [ticker for ticker, shares in state.positions.items() if shares > 0]
        if not held:
            return {}
        lookback = risk_config.vol_lookback_days
        end_prices: dict[str, float] = {}
        log_return_columns: list[np.ndarray] = []
        for ticker in held:
            df = price_data.get(ticker)
            if df is None:
                return {}
            in_range = df[df.index <= end_day]
            if len(in_range) < lookback + 1:
                return {}
            window = np.asarray(in_range["close"].iloc[-(lookback + 1) :], dtype=float)
            if np.any(window <= 0):
                return {}
            series = np.diff(np.log(window))
            if np.std(series, ddof=1) == 0.0:
                return {}
            log_return_columns.append(series)
            end_prices[ticker] = float(window[-1])

        position_values = np.array([state.positions[ticker] * end_prices[ticker] for ticker in held], dtype=float)
        total_value = state.cash + float(position_values.sum())
        if total_value <= 0:
            return {}
        weights = position_values / total_value
        log_returns = np.column_stack(log_return_columns)
        contributions = per_ticker_vol_contribution(weights, log_returns)
        return {ticker: float(contrib) for ticker, contrib in zip(held, contributions, strict=True)}
