"""BacktestResult container and serialization helpers.

The result dataclass holds all output from a backtest run. The ``write_*``
functions persist it to disk as CSV/JSON for downstream analysis.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from midas.metrics import (
    StrategyStats,
    _drawdown_series,
    _pair_sells_with_basis,
    aggregate_strategy_stats,
    compute_annualized_return,
)
from midas.models import Direction, TradeRecord
from midas.risk_metrics import RiskHistory, RiskMetrics
from midas.trade_log import format_holding_period, format_purchase_date


@dataclass
class BacktestResult:
    """Complete output of a single backtest run."""

    trades: list[TradeRecord]
    final_value: float
    starting_value: float
    buy_and_hold_value: float
    train_trades: list[TradeRecord]
    test_trades: list[TradeRecord]
    train_return: float
    test_return: float
    train_bh_return: float
    test_bh_return: float
    split_date: date | None
    twr: float  # time-weighted return (accounts for cash infusions)
    equity_curve: list[tuple[date, float]]  # daily (date, portfolio_value)
    total_days: int  # calendar days spanned by the full backtest
    train_days: int  # calendar days spanned by the train window (0 if no split)
    test_days: int  # calendar days spanned by the test window (0 if no split)
    cagr: float  # compound annual growth rate
    max_drawdown: float  # peak-to-trough percentage decline
    sharpe_ratio: float  # annualized, risk-free=0
    sortino_ratio: float  # annualized, downside deviation only
    win_rate: float  # fraction of round-trip sells that were profitable
    profit_factor: float  # gross wins / gross losses (inf if no losses)
    avg_win: float  # average P&L of winning sells
    avg_loss: float  # average P&L of losing sells
    efficiency_ratio: float  # test_return / train_return (0 if no split)
    strategy_stats: list[StrategyStats]
    unrealized_pnl: float  # mark-to-market gain on positions still held at end
    unrealized_pnl_by_ticker: dict[str, float]  # per-ticker unrealized P&L
    basis_per_sell: list[float]  # cost basis for each SELL trade (parallel list)
    risk_metrics: RiskMetrics | None = None  # populated when the engine wires it in
    risk_history: RiskHistory | None = None  # per-bar risk telemetry across the run
    bh_equity_curve: list[tuple[date, float]] = field(default_factory=list)
    """Per-bar buy-and-hold equity, parallel to ``equity_curve``."""


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def write_backtest_results(result: BacktestResult, output_dir: Path) -> None:
    """Write backtest results to a directory of machine-readable files."""
    if output_dir.is_file():
        msg = f"Output path '{output_dir}' is an existing file, not a directory"
        raise FileExistsError(msg)
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_trades_csv(result, output_dir / "trades.csv")
    _write_equity_curve_csv(result, output_dir / "equity_curve.csv")
    _write_summary_json(result, output_dir / "summary.json")
    _write_strategy_breakdown_csv(result, output_dir / "strategy_breakdown.csv")


def _write_trades_csv(result: BacktestResult, path: Path) -> None:
    sell_basis = {id(trade): basis for trade, basis in _pair_sells_with_basis(result.trades, result.basis_per_sell)}
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
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
            ]
        )
        for trade in result.trades:
            common = [
                trade.date.isoformat(),
                trade.ticker,
                trade.direction.value,
                trade.shares,
                trade.price,
                trade.strategy_name,
                format_holding_period(trade.holding_period),
                format_purchase_date(trade.purchase_date),
            ]
            if trade.direction == Direction.SELL:
                basis = sell_basis.get(id(trade), trade.price)
                pnl = round((trade.price - basis) * trade.shares, 4)
                ret = round((trade.price - basis) / basis, 6) if basis != 0 else 0.0
                writer.writerow([*common, round(basis, 4), pnl, ret])
            else:
                writer.writerow([*common, "", "", ""])


def _write_equity_curve_csv(result: BacktestResult, path: Path) -> None:
    drawdowns = _drawdown_series(result.equity_curve)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "nav", "drawdown"])
        for (dt, nav), drawdown in zip(result.equity_curve, drawdowns, strict=True):
            writer.writerow([dt.isoformat(), round(nav, 2), round(drawdown, 6)])


def _write_summary_json(result: BacktestResult, path: Path) -> None:
    starting_val = result.starting_value
    total_return = (result.final_value - starting_val) / starting_val if starting_val > 0 else 0.0
    bh_return = (result.buy_and_hold_value - starting_val) / starting_val if starting_val > 0 else 0.0
    total_days = result.total_days

    summary: dict[str, object] = {
        "starting_value": starting_val,
        "final_value": result.final_value,
        "total_return": round(total_return, 6),
        "total_return_annualized": round(compute_annualized_return(total_return, total_days), 6),
        "cagr": result.cagr,
        "twr": result.twr,
        "twr_annualized": round(compute_annualized_return(result.twr, total_days), 4),
        "buy_and_hold_value": result.buy_and_hold_value,
        "buy_and_hold_return": round(bh_return, 6),
        "buy_and_hold_return_annualized": round(compute_annualized_return(bh_return, total_days), 6),
        "total_trades": len(result.trades),
        "max_drawdown": result.max_drawdown,
        "sharpe_ratio": result.sharpe_ratio,
        "sortino_ratio": result.sortino_ratio,
        "win_rate": result.win_rate,
        "profit_factor": result.profit_factor if not math.isinf(result.profit_factor) else "inf",
        "avg_win": result.avg_win,
        "avg_loss": result.avg_loss,
        "unrealized_pnl": result.unrealized_pnl,
        "efficiency_ratio": result.efficiency_ratio,
    }

    if result.split_date:
        summary["split"] = {
            "date": result.split_date.isoformat(),
            "train_return": result.train_return,
            "train_return_annualized": round(compute_annualized_return(result.train_return, result.train_days), 4),
            "test_return": result.test_return,
            "test_return_annualized": round(compute_annualized_return(result.test_return, result.test_days), 4),
            "train_bh_return": result.train_bh_return,
            "train_bh_return_annualized": round(
                compute_annualized_return(result.train_bh_return, result.train_days), 4
            ),
            "test_bh_return": result.test_bh_return,
            "test_bh_return_annualized": round(compute_annualized_return(result.test_bh_return, result.test_days), 4),
            "train_trades": len(result.train_trades),
            "test_trades": len(result.test_trades),
        }

    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")


def _write_strategy_breakdown_csv(result: BacktestResult, path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "ticker", "trades", "buys", "sells", "win_rate", "pnl"])

        # Per-(strategy, ticker) rows
        for stat in result.strategy_stats:
            writer.writerow(
                [
                    stat.name,
                    stat.ticker,
                    stat.trades,
                    stat.buys,
                    stat.sells,
                    round(stat.win_rate, 4) if stat.sells > 0 else "",
                    round(stat.pnl, 2) if stat.sells > 0 else "",
                ]
            )

        # Aggregate per-strategy rows
        for agg in aggregate_strategy_stats(result.strategy_stats):
            writer.writerow(
                [
                    agg.name,
                    "*",
                    agg.trades,
                    agg.buys,
                    agg.sells,
                    agg.win_rate if agg.sells > 0 else "",
                    agg.pnl if agg.sells > 0 else "",
                ]
            )

        # Per-ticker open positions
        for ticker, pnl in sorted(result.unrealized_pnl_by_ticker.items()):
            writer.writerow(["Open Positions (Unrealized)", ticker, "", "", "", "", round(pnl, 2)])
        writer.writerow(["Open Positions (Unrealized)", "*", "", "", "", "", round(result.unrealized_pnl, 2)])
