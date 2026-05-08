"""Performance metrics and strategy statistics.

Pure functions that compute risk/return metrics from equity curves and
trade lists. No dependency on the backtest engine — these operate on
the output data structures only.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date

from midas.models import Direction, TradeRecord

TRADING_DAYS_PER_YEAR = 252
DAYS_PER_YEAR = 365.25  # average calendar days/year (accounts for leap years)
SHORT_WINDOW_THRESHOLD_DAYS = 365  # windows below this are "sub-one-year" for UX warnings


@dataclass
class StrategyStats:
    """Per-strategy performance breakdown, optionally scoped to a ticker."""

    name: str
    ticker: str | None
    trades: int
    buys: int
    sells: int
    win_rate: float  # fraction of profitable sells
    pnl: float  # total realized P&L from sells


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _drawdown_series(equity_curve: list[tuple[date, float]]) -> list[float]:
    """Per-point drawdown (fraction from running peak) for each equity curve entry."""
    if not equity_curve:
        return []
    peak = equity_curve[0][1]
    result: list[float] = []
    for _, value in equity_curve:
        if value > peak:
            peak = value
        result.append((peak - value) / peak if peak > 0 else 0.0)
    return result


# ---------------------------------------------------------------------------
# Public compute functions
# ---------------------------------------------------------------------------


def pair_sells_with_basis(
    trades: list[TradeRecord],
    basis_per_sell: list[float],
) -> list[tuple[TradeRecord, float]]:
    """Zip SELL trades with their recorded cost basis (parallel-list order).

    Falls back to the trade price (zero P&L) for any sell beyond the recorded
    basis list — defensive only; the lists should always be the same length.
    """
    sells = [trade for trade in trades if trade.direction == Direction.SELL]
    paired: list[tuple[TradeRecord, float]] = []
    for idx, trade in enumerate(sells):
        basis = basis_per_sell[idx] if idx < len(basis_per_sell) else trade.price
        paired.append((trade, basis))
    return paired


def compute_cagr(starting: float, final: float, days: int) -> float:
    """Compound annual growth rate from total return over *days* calendar days."""
    if starting <= 0 or final <= 0 or days <= 0:
        return 0.0
    years = days / DAYS_PER_YEAR
    return float((final / starting) ** (1.0 / years) - 1.0)


def compute_annualized_return(cumulative_return: float, days: int) -> float:
    """Annualize a cumulative return over *days* calendar days.

    ``cumulative_return`` is expressed as a fraction (0.25 == +25%). Returns
    0.0 for non-positive ``days`` and -1.0 when the cumulative loss wipes out
    the starting value (i.e. growth factor ≤ 0), since compounding past total
    loss is undefined. The 0.0 sentinel for ``days <= 0`` matches
    :func:`compute_cagr` so aggregates mixing the two don't get poisoned.

    Caveat: annualizing windows shorter than one year extrapolates aggressively
    — a 30-day +10% return projects to roughly +219% annualized. The number is
    mathematically correct but statistically noisy; callers that display short
    windows should surface the window length alongside so users can discount
    accordingly.
    """
    if days <= 0:
        return 0.0
    growth = 1.0 + cumulative_return
    if growth <= 0:
        return -1.0
    years = days / DAYS_PER_YEAR
    return float(growth ** (1.0 / years) - 1.0)


def compute_max_drawdown(equity_curve: list[tuple[date, float]]) -> float:
    """Maximum peak-to-trough percentage decline."""
    dd = _drawdown_series(equity_curve)
    return max(dd) if len(dd) >= 2 else 0.0


def compute_sharpe(equity_curve: list[tuple[date, float]]) -> float:
    """Annualized Sharpe ratio (risk-free = 0) from daily returns."""
    if len(equity_curve) < 3:
        return 0.0
    values = [val for _, val in equity_curve]
    daily_returns = [(values[i] - values[i - 1]) / values[i - 1] for i in range(1, len(values)) if values[i - 1] > 0]
    if len(daily_returns) < 2:
        return 0.0
    mean_ret = sum(daily_returns) / len(daily_returns)
    variance = sum((ret - mean_ret) ** 2 for ret in daily_returns) / (len(daily_returns) - 1)
    std_ret = math.sqrt(variance) if variance > 0 else 0.0
    if std_ret == 0:
        return 0.0
    return (mean_ret / std_ret) * math.sqrt(TRADING_DAYS_PER_YEAR)


def compute_sortino(equity_curve: list[tuple[date, float]]) -> float:
    """Annualized Sortino ratio (risk-free = 0, downside deviation only).

    When there is no downside (no negative daily returns) the metric is
    undefined. We return 0.0 as a sentinel rather than `inf` so that callers
    averaging across multiple windows (e.g. walk-forward folds) aren't
    poisoned by a single zero-downside fold.
    """
    if len(equity_curve) < 3:
        return 0.0
    values = [val for _, val in equity_curve]
    daily_returns = [(values[i] - values[i - 1]) / values[i - 1] for i in range(1, len(values)) if values[i - 1] > 0]
    if len(daily_returns) < 2:
        return 0.0
    mean_ret = sum(daily_returns) / len(daily_returns)
    downside = [ret for ret in daily_returns if ret < 0]
    if not downside:
        return 0.0
    downside_var = sum(ret**2 for ret in downside) / len(daily_returns)
    downside_dev = math.sqrt(downside_var) if downside_var > 0 else 0.0
    if downside_dev == 0:
        return 0.0
    return (mean_ret / downside_dev) * math.sqrt(TRADING_DAYS_PER_YEAR)


def compute_trade_stats(
    trades: list[TradeRecord],
    basis_per_sell: list[float],
) -> tuple[float, float, float, float]:
    """Return (win_rate, profit_factor, avg_win, avg_loss) from sell trades.

    Breakeven sells (`pnl == 0`) are counted as wins by convention.
    """
    paired = pair_sells_with_basis(trades, basis_per_sell)
    if not paired:
        return 0.0, 0.0, 0.0, 0.0

    wins: list[float] = []
    losses: list[float] = []
    for trade, basis in paired:
        pnl = (trade.price - basis) * trade.shares
        if pnl >= 0:
            wins.append(pnl)
        else:
            losses.append(pnl)

    total = len(wins) + len(losses)
    win_rate = len(wins) / total if total > 0 else 0.0
    gross_wins = sum(wins) if wins else 0.0
    gross_losses = abs(sum(losses)) if losses else 0.0
    if gross_losses > 0:
        profit_factor = gross_wins / gross_losses
    elif gross_wins > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0
    avg_win = gross_wins / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0  # negative number
    return win_rate, profit_factor, avg_win, avg_loss


def compute_strategy_stats(
    trades: list[TradeRecord],
    basis_per_sell: list[float],
) -> list[StrategyStats]:
    """Compute per-(strategy, ticker) trade breakdown."""
    sell_basis: dict[int, float] = {id(trade): basis for trade, basis in pair_sells_with_basis(trades, basis_per_sell)}

    by_key: dict[tuple[str, str], list[TradeRecord]] = defaultdict(list)
    for trade in trades:
        by_key[(trade.strategy_name, trade.ticker)].append(trade)

    stats: list[StrategyStats] = []
    for (name, ticker), strategy_trades in sorted(by_key.items()):
        buys = [trade for trade in strategy_trades if trade.direction == Direction.BUY]
        sells = [trade for trade in strategy_trades if trade.direction == Direction.SELL]
        winning_sells = 0
        total_pnl = 0.0
        for trade in sells:
            basis = sell_basis.get(id(trade), trade.price)
            pnl = (trade.price - basis) * trade.shares
            total_pnl += pnl
            if pnl >= 0:
                winning_sells += 1
        win_rate = winning_sells / len(sells) if sells else 0.0
        stats.append(
            StrategyStats(
                name=name,
                ticker=ticker,
                trades=len(strategy_trades),
                buys=len(buys),
                sells=len(sells),
                win_rate=round(win_rate, 4),
                pnl=round(total_pnl, 2),
            )
        )
    return stats


def aggregate_strategy_stats(stats: list[StrategyStats]) -> list[StrategyStats]:
    """Aggregate per-(strategy, ticker) stats into per-strategy totals."""
    by_strategy: dict[str, list[StrategyStats]] = defaultdict(list)
    for stat in stats:
        by_strategy[stat.name].append(stat)
    result: list[StrategyStats] = []
    for name, group in sorted(by_strategy.items()):
        total_sells = sum(stat.sells for stat in group)
        total_pnl = sum(stat.pnl for stat in group)
        winning = sum(round(stat.win_rate * stat.sells) for stat in group)
        agg_win_rate = winning / total_sells if total_sells > 0 else 0.0
        result.append(
            StrategyStats(
                name=name,
                ticker=None,
                trades=sum(stat.trades for stat in group),
                buys=sum(stat.buys for stat in group),
                sells=total_sells,
                win_rate=round(agg_win_rate, 4),
                pnl=round(total_pnl, 2),
            )
        )
    return result
