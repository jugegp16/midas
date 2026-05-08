"""Rich terminal output for alerts and status messages."""

from __future__ import annotations

import math
from datetime import datetime
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from midas.metrics import (
    DAYS_PER_YEAR,
    SHORT_WINDOW_THRESHOLD_DAYS,
    aggregate_strategy_stats,
    compute_annualized_return,
)
from midas.models import Direction, Order
from midas.strategies.base import Strategy

if TYPE_CHECKING:
    from midas.results import BacktestResult

console = Console()

BACKTEST_TABLE_WIDTH = 100
# Split table in half so the column divider is centered.
# Account for 4 chars of box borders/separator (outer borders + center separator).
METRIC_COL_WIDTH = (BACKTEST_TABLE_WIDTH - 4) // 2
VALUE_COL_WIDTH = BACKTEST_TABLE_WIDTH - 4 - METRIC_COL_WIDTH


def print_alert(
    order: Order,
    remaining_cash: float,
    timestamp: datetime,
    *,
    dry_run: bool = False,
) -> None:
    color = "green" if order.direction == Direction.BUY else "red"
    prefix = "[DRY RUN] " if dry_run else ""

    ctx = order.context
    dominant = ctx.source

    lines = [
        f"[bold]{order.ticker}[/bold] — ${order.price:,.2f}",
        ctx.reason,
        f"Target weight: {ctx.target_weight:.1%} | Current: {ctx.current_weight:.1%}",
        f"Blended score: {ctx.blended_score:+.3f}",
        f"Primary strategy: {dominant}",
        f"Suggested order: {order.shares} share{'s' if order.shares != 1 else ''} "
        f"@ ${order.price:,.2f} = ${order.estimated_value:,.2f}",
    ]

    lines.append(f"Available cash after: ${remaining_cash:,.2f}")

    console.print(
        Panel(
            "\n".join(lines),
            title=f"{prefix}[{color}][{order.direction.value}][/{color}]",
            border_style=color,
            subtitle=timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
        )
    )


def print_status(message: str) -> None:
    console.print(f"[dim]{message}[/dim]")


def print_strategy_table(strategies: list[Strategy]) -> None:
    table = Table(title="Available Strategies")
    table.add_column("Name", style="bold")
    table.add_column("Tier", style="magenta")
    table.add_column("Description")
    table.add_column("Suitability", style="cyan")

    for strat in strategies:
        tags = ", ".join(tag.value for tag in strat.suitability)
        table.add_row(strat.name, strat.tier_label, strat.description, tags)

    console.print(table, justify="center")


def color_signed(value: float, fmt: str = ".2%") -> str:
    """Color-code a numeric value green/red based on sign."""
    style = "green" if value >= 0 else "red"
    return f"[{style}]{value:{fmt}}[/{style}]"


def make_metric_table(title: str) -> Table:
    """2-column metric/value table — the canonical layout for summary outputs."""
    table = Table(title=title, show_lines=True, width=BACKTEST_TABLE_WIDTH)
    table.add_column("Metric", style="bold", width=METRIC_COL_WIDTH)
    table.add_column("Value", justify="right", width=VALUE_COL_WIDTH)
    return table


def make_wide_table(title: str, width: int = BACKTEST_TABLE_WIDTH) -> Table:
    """Multi-column table at the standard width (caller adds columns)."""
    return Table(title=title, title_style="bold", show_lines=True, width=width)


def print_centered(table: Table) -> None:
    """Centered render — used by all summary tables for visual consistency."""
    console.print(table, justify="center")


def print_run_info(rows: list[tuple[str, str]], title: str = "Run Info") -> None:
    """Render a small key/value table for run metadata (trials, output path, etc)."""
    table = make_metric_table(title)
    for label, value in rows:
        table.add_row(label, value)
    print_centered(table)


def print_params_table(
    title: str,
    params: dict[str, dict[str, float]],
    global_key: str | None = None,
) -> None:
    """Render an optimizer's per-strategy parameter table.

    `global_key`, if supplied, is the synthetic strategy name used to hold
    portfolio-wide allocation knobs; it's relabeled as "Global" for display.
    """
    table = make_wide_table(title)
    table.add_column("Strategy", style="bold")
    table.add_column("Parameters")
    for name, param_dict in params.items():
        display = "Global" if global_key is not None and name == global_key else name
        table.add_row(display, ", ".join(f"{key}={val}" for key, val in param_dict.items()))
    print_centered(table)


def _return_row(cum: float, days: int) -> str:
    """Format a return as 'cumulative (annualized)' for display.

    Args:
        cum: Cumulative return as a fraction (0.25 == +25%).
        days: Calendar days spanned by the return window. Non-positive values
            have no meaningful annualization, so the annualized portion is
            rendered as ``"—"``.

    Returns:
        Rich-formatted string of the form ``"+25.00% (+21.33% annualized)"``
        with each number sign-colored via :func:`color_signed`.
    """
    if days <= 0:
        return f"{color_signed(cum)} (— annualized)"
    annualized = compute_annualized_return(cum, days)
    return f"{color_signed(cum)} ({color_signed(annualized)} annualized)"


def print_backtest_summary(result: BacktestResult, *, show_charts: bool = False) -> None:
    starting_val = result.starting_value
    final_val = result.final_value
    bh_val = result.buy_and_hold_value
    total_return = (final_val - starting_val) / starting_val if starting_val > 0 else 0
    bh_return = (bh_val - starting_val) / starting_val if starting_val > 0 else 0
    total_days = result.total_days

    # --- Performance ---
    perf = make_metric_table("Performance")
    perf.add_row("Starting Value", f"${starting_val:,.2f}")
    perf.add_row("Final Value", f"${final_val:,.2f}")
    perf.add_row("Total Return", _return_row(total_return, total_days))
    perf.add_row("CAGR (Annualized)", color_signed(result.cagr))
    perf.add_row("Time-Weighted Return", _return_row(result.twr, total_days))
    perf.add_row("Buy & Hold Value", f"${bh_val:,.2f}")
    perf.add_row("Buy & Hold Return", _return_row(bh_return, total_days))
    perf.add_row("Total Trades", str(len(result.trades)))
    print_centered(perf)
    if 0 < total_days < SHORT_WINDOW_THRESHOLD_DAYS:
        years = total_days / DAYS_PER_YEAR
        console.print(
            f"[yellow]Note: backtest window is {total_days} days (~{years:.2f} years). "
            f"Annualized figures extrapolate from a sub-one-year sample and can be "
            f"noisy — interpret alongside the cumulative number.[/yellow]",
            justify="center",
        )

    # --- After-Tax Performance ---
    if result.after_tax_final_value is not None:
        after_tax = make_metric_table("After-Tax Performance")
        after_tax.add_row("After-Tax Final Value", f"${result.after_tax_final_value:,.2f}")
        after_tax.add_row(
            "After-Tax Total Return",
            _return_row(result.after_tax_total_return or 0.0, total_days),
        )
        after_tax.add_row("After-Tax CAGR", color_signed(result.after_tax_cagr or 0.0))
        if result.tax_cost_ratio is not None:
            after_tax.add_row("Tax Cost Ratio", f"{result.tax_cost_ratio:.2%}")
        print_centered(after_tax)

        if result.tax_summary:
            tax_table = make_wide_table("Realized Tax (per year)")
            tax_table.add_column("Year", style="bold")
            tax_table.add_column("ST Realized", justify="right")
            tax_table.add_column("LT Realized", justify="right")
            tax_table.add_column("Net (after netting)", justify="right")
            tax_table.add_column("Tax Owed", justify="right")
            tax_table.add_column("Carry Forward", justify="right")
            for s in result.tax_summary:
                tax_table.add_row(
                    str(s.year),
                    f"${s.st_realized:+,.2f}",
                    f"${s.lt_realized:+,.2f}",
                    f"${s.net_after_cross:+,.2f}",
                    f"${s.tax_owed:+,.2f}",
                    f"${s.carry_forward:,.2f}",
                )
            print_centered(tax_table)

    # --- Train / Test Split ---
    if result.split_date:
        split = make_metric_table("Train / Test Split")
        split.add_row("Split Date", result.split_date.isoformat())
        split.add_row("Train Return", _return_row(result.train_return, result.train_days))
        split.add_row("Train B&H Return", _return_row(result.train_bh_return, result.train_days))
        split.add_row("Train Trades", str(len(result.train_trades)))
        split.add_row("Test Return", _return_row(result.test_return, result.test_days))
        split.add_row("Test B&H Return", _return_row(result.test_bh_return, result.test_days))
        split.add_row("Test Trades", str(len(result.test_trades)))
        split.add_row("Efficiency Ratio", f"{result.efficiency_ratio:.0%}")
        print_centered(split)

    # --- Risk Metrics ---
    risk_table = make_metric_table("Risk Metrics")
    risk_table.add_row("Max Drawdown", f"[red]{result.max_drawdown:.2%}[/red]")
    risk_table.add_row("Sharpe Ratio", color_signed(result.sharpe_ratio, fmt=".2f"))
    risk_table.add_row("Sortino Ratio", color_signed(result.sortino_ratio, fmt=".2f"))
    if result.risk_metrics is not None:
        risk_table.add_row("Realized Vol (60d)", f"{result.risk_metrics.realized_vol_60d:.2%}")
        if result.risk_metrics.vol_target is not None:
            risk_table.add_row("Vol Target", f"{result.risk_metrics.vol_target:.2%}")
        risk_table.add_row("Drawdown From Peak", f"[red]{result.risk_metrics.drawdown_from_peak:.2%}[/red]")
        risk_table.add_row("Rolling Sharpe (252d)", color_signed(result.risk_metrics.rolling_sharpe_252d, fmt=".2f"))
        risk_table.add_row("Avg Gross Exposure", f"{result.risk_metrics.avg_gross_exposure:.2%}")
        risk_table.add_row("Min Gross Exposure", f"{result.risk_metrics.min_gross_exposure:.2%}")
    print_centered(risk_table)

    # --- Risk Engine Activity ---
    # Show this section when CPPI fired on any bar OR when vol target is
    # configured (regardless of whether it bound). The vol-target clause is
    # configuration-gated so a configured-but-non-binding target shows
    # "Avg Scale: 100.00%" as confirmation it ran, rather than an empty
    # section indistinguishable from "feature was disabled". CPPI does not
    # have an analogous configuration field on RiskMetrics, so its rows
    # surface only when activity registers.
    if result.risk_metrics is not None and (
        result.risk_metrics.cppi_active_pct > 0
        or result.risk_metrics.cppi_min_scale < 1.0
        or result.risk_metrics.vol_target is not None
    ):
        phase_table = make_metric_table("Risk Engine Activity")
        if result.risk_metrics.cppi_active_pct > 0 or result.risk_metrics.cppi_min_scale < 1.0:
            phase_table.add_row("CPPI Active (% of bars)", f"{result.risk_metrics.cppi_active_pct:.1%}")
            phase_table.add_row("CPPI Avg Scale", f"{result.risk_metrics.cppi_avg_scale:.2%}")
            phase_table.add_row("CPPI Min Scale", f"[red]{result.risk_metrics.cppi_min_scale:.2%}[/red]")
        if result.risk_metrics.vol_target is not None:
            phase_table.add_row("Vol Target Bound (% of bars)", f"{result.risk_metrics.vol_target_bind_pct:.1%}")
            phase_table.add_row("Vol Target Avg Scale", f"{result.risk_metrics.vol_target_avg_scale:.2%}")
        if result.risk_metrics.vol_target_skip_count > 0:
            phase_table.add_row(
                "Vol Target Skipped (bars)",
                f"[yellow]{result.risk_metrics.vol_target_skip_count}[/yellow]",
            )
        print_centered(phase_table)

    # --- Per-Ticker Vol Contribution ---
    if result.risk_metrics is not None and result.risk_metrics.per_ticker_vol_contribution:
        contrib_table = make_metric_table("Per-Ticker Vol Contribution")
        items = sorted(
            result.risk_metrics.per_ticker_vol_contribution.items(),
            key=lambda kv: -abs(kv[1]),
        )
        max_share = max((abs(share) for _, share in items), default=0.0)
        bar_width = 24
        for ticker, share in items:
            # Inline bar glyphs scale the largest contribution to ``bar_width``
            # full blocks so concentration patterns are readable at a glance,
            # in addition to the precise percentage. Using a single color keeps
            # the visual quiet — adjacent rows are differentiable by length
            # alone.
            blocks = round((abs(share) / max_share) * bar_width) if max_share > 0 else 0
            bar = "█" * blocks
            contrib_table.add_row(ticker, f"[cyan]{bar}[/cyan] {share:.1%}")
        print_centered(contrib_table)

    # --- Per-Strategy P&L Attribution ---
    if result.risk_metrics is not None and result.risk_metrics.per_strategy_pnl:
        attr_table = make_metric_table("Per-Strategy P&L Attribution")
        for strat, pnl in sorted(
            result.risk_metrics.per_strategy_pnl.items(),
            key=lambda kv: -kv[1],
        ):
            attr_table.add_row(strat, f"${pnl:+,.2f}")
        print_centered(attr_table)

    # --- Trade Quality ---
    if any(trade.direction == Direction.SELL for trade in result.trades):
        trade_table = make_metric_table("Trade Quality")
        trade_table.add_row("Win Rate", color_signed(result.win_rate))
        pf_str = f"{result.profit_factor:.2f}" if not math.isinf(result.profit_factor) else "∞"
        trade_table.add_row("Profit Factor", pf_str)
        trade_table.add_row("Avg Win", f"[green]${result.avg_win:,.2f}[/green]")
        trade_table.add_row("Avg Loss", f"[red]${result.avg_loss:,.2f}[/red]")
        print_centered(trade_table)

    # --- Per-Strategy Breakdown (aggregate) ---
    if result.strategy_stats:
        agg_table = make_wide_table("Strategy Breakdown")
        agg_table.add_column("Strategy", style="bold")
        agg_table.add_column("Trades", justify="right")
        agg_table.add_column("Buys", justify="right")
        agg_table.add_column("Sells", justify="right")
        agg_table.add_column("Win Rate", justify="right")
        agg_table.add_column("P&L", justify="right")

        for agg in aggregate_strategy_stats(result.strategy_stats):
            pnl_style = "green" if agg.pnl >= 0 else "red"
            agg_table.add_row(
                agg.name,
                str(agg.trades),
                str(agg.buys),
                str(agg.sells),
                f"{agg.win_rate:.0%}" if agg.sells > 0 else "—",
                f"[{pnl_style}]${agg.pnl:,.2f}[/{pnl_style}]" if agg.sells > 0 else "—",
            )

        unr = result.unrealized_pnl
        unr_style = "green" if unr >= 0 else "red"
        agg_table.add_row(
            "[dim]Open Positions (Unrealized)[/dim]",
            "—",
            "—",
            "—",
            "—",
            f"[{unr_style}]${unr:,.2f}[/{unr_style}]",
        )

        print_centered(agg_table)

        # --- Strategy x Ticker Breakdown ---
        detail_table = make_wide_table("Strategy x Ticker Breakdown")
        detail_table.add_column("Strategy", style="bold")
        detail_table.add_column("Ticker", style="bold")
        detail_table.add_column("Trades", justify="right")
        detail_table.add_column("Buys", justify="right")
        detail_table.add_column("Sells", justify="right")
        detail_table.add_column("Win Rate", justify="right")
        detail_table.add_column("P&L", justify="right")

        for stat in result.strategy_stats:
            pnl_style = "green" if stat.pnl >= 0 else "red"
            detail_table.add_row(
                stat.name,
                stat.ticker,
                str(stat.trades),
                str(stat.buys),
                str(stat.sells),
                f"{stat.win_rate:.0%}" if stat.sells > 0 else "—",
                f"[{pnl_style}]${stat.pnl:,.2f}[/{pnl_style}]" if stat.sells > 0 else "—",
            )

        detail_table.add_section()
        for ticker, pnl in sorted(result.unrealized_pnl_by_ticker.items()):
            pnl_style = "green" if pnl >= 0 else "red"
            detail_table.add_row(
                "[dim]Open Positions (Unrealized)[/dim]",
                ticker,
                "—",
                "—",
                "—",
                "—",
                f"[{pnl_style}]${pnl:,.2f}[/{pnl_style}]",
            )

        print_centered(detail_table)
        console.print(
            "[dim italic]Note: P&L is credited to the exit strategy. "
            "Open Positions shows unrealized gain/loss on shares still "
            "held at backtest end.[/dim italic]",
            justify="center",
        )

    if show_charts:
        from midas.charts import render_charts

        render_charts(result)
