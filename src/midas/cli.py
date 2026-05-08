"""CLI interface — the sole user-facing interface for MVP."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import click
import pandas as pd

from midas.allocator import Allocator
from midas.backtest import DEFAULT_TRAIN_PCT, BacktestEngine, ExecutionMode
from midas.config import load_portfolio, load_strategies
from midas.data import CachedYFinanceProvider
from midas.models import (
    AllocationConstraints,
    Direction,
    PortfolioConfig,
    RiskConfig,
    StrategyConfig,
    TaxConfig,
    TradeRecord,
)
from midas.order_sizer import OrderSizer
from midas.output import print_backtest_summary, print_status, print_strategy_table
from midas.results import write_backtest_results
from midas.strategies import STRATEGY_REGISTRY, EntrySignal, ExitRule, Strategy
from midas.strategies.base import max_warmup, warmup_bars_to_calendar_days
from midas.tax import AnnualTaxSummary
from midas.trade_log import LoggedTrade


def _build_strategy(cfg: StrategyConfig) -> Strategy:
    cls = STRATEGY_REGISTRY.get(cfg.name)
    if cls is None:
        msg = f"Unknown strategy '{cfg.name}'. Available: {', '.join(STRATEGY_REGISTRY)}"
        raise click.ClickException(msg)
    return cls(**cfg.params)


def _build_components(
    strategy_configs: list[StrategyConfig] | None,
    constraints: AllocationConstraints,
    n_tickers: int,
    risk_config: RiskConfig | None = None,
) -> tuple[Allocator, OrderSizer, list[ExitRule]]:
    """Build allocator, order sizer, and exit rules from config."""
    configs = strategy_configs or [StrategyConfig(name=name) for name in STRATEGY_REGISTRY]

    entries: list[tuple[EntrySignal, float]] = []
    exits: list[ExitRule] = []

    for cfg in configs:
        strategy = _build_strategy(cfg)

        if isinstance(strategy, ExitRule):
            exits.append(strategy)
        elif isinstance(strategy, EntrySignal):
            entries.append((strategy, cfg.weight))
        else:
            msg = f"Strategy {cfg.name!r} is neither EntrySignal nor ExitRule"
            raise click.ClickException(msg)

    allocator = Allocator(entries, constraints, n_tickers, risk_config=risk_config)
    order_sizer = OrderSizer()

    return allocator, order_sizer, exits


def _to_date(dt: date | datetime) -> date:
    """Coerce click.DateTime output to a plain date."""
    return dt.date() if isinstance(dt, datetime) else dt


def _fetch_prices(
    portfolio: PortfolioConfig,
    start: date,
    end: date,
    warmup_bars: int = 0,
) -> dict[str, pd.DataFrame]:
    """Fetch price history from ``start - warmup_buffer`` through ``end``.

    ``warmup_bars`` is the maximum number of trading days any configured
    strategy needs before it can emit valid scores. The fetch start is
    shifted backward by an equivalent calendar-day buffer so strategies
    have history available on day one of the simulation.
    """
    provider = CachedYFinanceProvider()
    price_data: dict[str, pd.DataFrame] = {}
    tickers = [holding.ticker for holding in portfolio.holdings]
    buffer_days = warmup_bars_to_calendar_days(warmup_bars)
    fetch_start = start - timedelta(days=buffer_days)
    print_status(f"Fetching data for {', '.join(tickers)} (with {buffer_days}-day warmup buffer from {fetch_start})...")
    for ticker in tickers:
        try:
            price_data[ticker] = provider.get_history(ticker, fetch_start, end)
        except Exception as exc:
            print_status(f"Skipping {ticker}: {exc}")
    return price_data


@click.group()
def cli() -> None:
    """Midas — Portfolio Signal Engine."""


@cli.command()
@click.option(
    "--portfolio",
    "-p",
    required=True,
    type=click.Path(exists=True),
    help="Path to portfolio YAML config.",
)
@click.option(
    "--strategies",
    "-s",
    default=None,
    type=click.Path(exists=True),
    help="Path to strategies YAML config. Defaults to all strategies.",
)
@click.option("--start", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--end", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option(
    "--output",
    "-o",
    default="backtest_results",
    help="Output directory path.",
)
@click.option(
    "--train-pct",
    default=DEFAULT_TRAIN_PCT,
    help="Train/test split ratio (0-1).",
)
@click.option("--no-split", is_flag=True, help="Disable train/test split.")
@click.option(
    "--execution-mode",
    type=click.Choice(["close", "next_open", "next_close"]),
    default="next_open",
    show_default=True,
    help=(
        "When orders computed on day T actually fill. "
        "'close' = same day (legacy, optimistic); "
        "'next_open' = next session's open (honest default); "
        "'next_close' = next session's close."
    ),
)
@click.option(
    "--charts/--no-charts",
    default=True,
    show_default=True,
    help="Render terminal ASCII charts (equity, drawdown, exposure) after the summary.",
)
def backtest(
    portfolio: str,
    strategies: str | None,
    start: date,
    end: date,
    output: str,
    train_pct: float,
    no_split: bool,
    execution_mode: ExecutionMode,
    charts: bool,
) -> None:
    """Run a backtest over historical data."""
    port = load_portfolio(Path(portfolio))
    strat_configs, constraints, risk_config, tax_config = (
        load_strategies(Path(strategies)) if strategies else (None, AllocationConstraints(), RiskConfig(), None)
    )

    start_d, end_d = _to_date(start), _to_date(end)

    n_tickers = sum(1 for holding in port.holdings if holding.shares > 0)
    allocator, order_sizer, exit_rules = _build_components(
        strat_configs,
        constraints,
        n_tickers,
        risk_config=risk_config,
    )

    warmup_bars = max_warmup([*allocator.strategies, *exit_rules])
    price_data = _fetch_prices(port, start_d, end_d, warmup_bars=warmup_bars)

    engine = BacktestEngine(
        allocator=allocator,
        order_sizer=order_sizer,
        exit_rules=exit_rules,
        constraints=constraints,
        train_pct=train_pct,
        enable_split=not no_split,
        log_fn=print_status,
        execution_mode=execution_mode,
        tax_config=tax_config,
    )

    print_status("Running backtest...")
    result = engine.run(port, price_data, start_d, end_d)

    out_path = Path(output)
    write_backtest_results(result, out_path)
    print_status(f"Results written to {out_path}/")
    print_backtest_summary(result, show_charts=charts)


@cli.command()
@click.option(
    "--portfolio",
    "-p",
    required=True,
    type=click.Path(exists=True),
    help="Path to portfolio YAML config.",
)
@click.option(
    "--strategies",
    "-s",
    default=None,
    type=click.Path(exists=True),
    help="Path to strategies YAML config. Defaults to all strategies.",
)
@click.option("--interval", default=60, help="Poll interval in seconds.")
@click.option("--dry-run", is_flag=True, help="Log signals without alerts.")
def live(
    portfolio: str,
    strategies: str | None,
    interval: int,
    dry_run: bool,
) -> None:
    """Run live analysis with real-time price polling."""
    from midas.live import LiveEngine

    portfolio_path = Path(portfolio)
    port = load_portfolio(portfolio_path)
    state_path = port.state_file if port.state_file is not None else portfolio_path.with_suffix(".state.yaml")
    strat_configs, constraints, risk_config, _ = (
        load_strategies(Path(strategies)) if strategies else (None, AllocationConstraints(), RiskConfig(), None)
    )
    provider = CachedYFinanceProvider()

    n_tickers = sum(1 for holding in port.holdings if holding.shares > 0)
    allocator, order_sizer, exit_rules = _build_components(
        strat_configs,
        constraints,
        n_tickers,
        risk_config=risk_config,
    )

    with LiveEngine(
        portfolio=port,
        allocator=allocator,
        order_sizer=order_sizer,
        provider=provider,
        state_path=state_path,
        exit_rules=exit_rules,
        constraints=constraints,
        poll_interval=interval,
        dry_run=dry_run,
    ) as engine:
        engine.run()


@cli.command(name="tax-report")
@click.option(
    "--portfolio",
    "-p",
    default=None,
    type=click.Path(exists=True),
    help="Portfolio YAML; resolves the trade log next to its state file unless --from-trades is given.",
)
@click.option(
    "--strategies",
    "-s",
    required=True,
    type=click.Path(exists=True),
    help="Strategies YAML containing the tax: block. Required — rates have no defaults at the CLI.",
)
@click.option(
    "--from-trades",
    "from_trades",
    default=None,
    type=click.Path(exists=True),
    help="Explicit path to a trades.csv. Overrides --portfolio resolution.",
)
@click.option("--year", type=int, default=None, help="Calendar year to report (e.g. 2026).")
@click.option("--start", type=click.DateTime(formats=["%Y-%m-%d"]), default=None)
@click.option("--end", type=click.DateTime(formats=["%Y-%m-%d"]), default=None)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output CSV path. Defaults to schedule_d_<year>.csv (or schedule_d_<start>_<end>.csv).",
)
def tax_report(
    portfolio: str | None,
    strategies: str,
    from_trades: str | None,
    year: int | None,
    start: datetime | None,
    end: datetime | None,
    output: str | None,
) -> None:
    """Year-end realized-P&L report (Schedule D-shaped) from a trade log."""
    from midas.tax import compute_tax_summary
    from midas.trade_log import read_trades

    if year is None and (start is None or end is None):
        msg = "either --year or both --start and --end must be provided"
        raise click.UsageError(msg)

    _strats, _constraints, _risk, tax_config = load_strategies(Path(strategies))
    if tax_config is None:
        msg = (
            "strategies file has no `tax:` block; tax-report requires configured rates "
            "(short_term_rate, long_term_rate). See docs/tax-reporting.md."
        )
        raise click.UsageError(msg)

    if from_trades is not None:
        trades_path = Path(from_trades)
    else:
        if portfolio is None:
            raise click.UsageError("either --portfolio or --from-trades is required")
        port = load_portfolio(Path(portfolio))
        portfolio_path = Path(portfolio)
        state_path = port.state_file if port.state_file is not None else portfolio_path.with_suffix(".state.yaml")
        trades_path = state_path.with_suffix(state_path.suffix + ".trades.csv")
        if not trades_path.exists():
            msg = f"trade log not found at {trades_path}"
            raise click.UsageError(msg)

    if year is not None:
        start_d = date(year, 1, 1)
        end_d = date(year, 12, 31)
        period_label = str(year)
    else:
        assert start is not None and end is not None
        start_d = _to_date(start)
        end_d = _to_date(end)
        period_label = f"{start_d.isoformat()}_{end_d.isoformat()}"

    rows: list[LoggedTrade] = [
        row for row in read_trades(trades_path) if row.direction == Direction.SELL and start_d <= row.date <= end_d
    ]

    if not rows:
        click.echo(f"No realized sales in {period_label}.")
        if output is not None:
            Path(output).write_text(",".join(_TAX_REPORT_COLUMNS) + "\n")
        return

    out_path = Path(output) if output is not None else Path(f"schedule_d_{period_label}.csv")

    trade_records: list[TradeRecord] = []
    basis_per_sell: list[float] = []
    for row in rows:
        trade_records.append(
            TradeRecord(
                date=row.date,
                ticker=row.ticker,
                direction=row.direction,
                shares=row.shares,
                price=row.price,
                strategy_name=row.strategy_name,
                holding_period=row.holding_period,
                purchase_date=row.purchase_date,
            )
        )
        basis_per_sell.append(row.cost_basis if row.cost_basis is not None else row.price)

    summary = compute_tax_summary(trade_records, basis_per_sell, tax_config, end_date=end_d)
    _print_tax_report(rows, basis_per_sell, summary, period_label)
    _write_tax_report_csv(rows, basis_per_sell, summary, out_path)
    click.echo(f"\nWrote {out_path}")


@cli.command()
@click.option(
    "--portfolio",
    "-p",
    required=True,
    type=click.Path(exists=True),
    help="Path to portfolio YAML config.",
)
@click.option(
    "--strategies",
    "-s",
    default=None,
    type=click.Path(exists=True),
    help="Strategies to optimize. Defaults to all.",
)
@click.option("--start", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--end", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option(
    "--output",
    "-o",
    default="optimized_strategies.yaml",
    help="Output YAML path.",
)
@click.option(
    "--n-trials",
    "-n",
    default=200,
    show_default=True,
    help="Number of Optuna optimisation trials.",
)
@click.option(
    "--train-pct",
    default=DEFAULT_TRAIN_PCT,
    show_default=True,
    help="Train/test split ratio (0-1).",
)
@click.option(
    "--walk-forward",
    is_flag=True,
    default=False,
    help="Use walk-forward optimization (auto-determines folds from date range).",
)
@click.option(
    "--wf-min-train-pct",
    default=None,
    type=float,
    help="Walk-forward: minimum initial training window as fraction of data (0-1). Default 0.60.",
)
@click.option(
    "--wf-min-test-days",
    default=None,
    type=int,
    help="Walk-forward: minimum trading days per test fold. Default 63 (~3 months).",
)
def optimize(
    portfolio: str,
    strategies: str | None,
    start: date,
    end: date,
    output: str,
    n_trials: int,
    train_pct: float,
    walk_forward: bool,
    wf_min_train_pct: float | None,
    wf_min_test_days: int | None,
) -> None:
    """Find optimal strategy parameters via Bayesian optimisation (Optuna TPE)."""
    from midas.optimizer import (
        ALLOCATION_KEY,
        WF_MIN_TEST_DAYS,
        WF_MIN_TRAIN_PCT,
        max_warmup_for_search,
        walk_forward_optimize,
        write_strategies_yaml,
    )
    from midas.optimizer import (
        optimize as run_optimize,
    )

    # Validation
    if walk_forward and train_pct != DEFAULT_TRAIN_PCT:
        raise click.UsageError("--train-pct cannot be used with --walk-forward (walk-forward has its own split logic).")
    if not walk_forward and (wf_min_train_pct is not None or wf_min_test_days is not None):
        raise click.UsageError("--wf-min-train-pct and --wf-min-test-days require --walk-forward.")
    if train_pct <= 0 or train_pct > 1:
        raise click.UsageError("--train-pct must be in (0, 1].")
    if wf_min_train_pct is not None and (wf_min_train_pct <= 0 or wf_min_train_pct >= 1):
        raise click.UsageError("--wf-min-train-pct must be in (0, 1).")
    if wf_min_test_days is not None and wf_min_test_days < 1:
        raise click.UsageError("--wf-min-test-days must be >= 1.")

    port = load_portfolio(Path(portfolio))

    strategy_names: list[str] | None = None
    min_cash_pct = AllocationConstraints().min_cash_pct
    risk_config: RiskConfig = RiskConfig()
    tax_config: TaxConfig | None = None
    if strategies:
        strat_configs, strat_constraints, risk_config, tax_config = load_strategies(Path(strategies))
        strategy_names = [cfg.name for cfg in strat_configs]
        min_cash_pct = strat_constraints.min_cash_pct

    start_d, end_d = _to_date(start), _to_date(end)
    n_tickers = sum(1 for holding in port.holdings if holding.shares > 0)
    warmup_bars = max_warmup_for_search(strategy_names, min_cash_pct, n_tickers)
    price_data = _fetch_prices(port, start_d, end_d, warmup_bars=warmup_bars)

    from midas.metrics import SHORT_WINDOW_THRESHOLD_DAYS
    from midas.output import (
        color_signed,
        console,
        make_metric_table,
        make_wide_table,
        print_backtest_summary,
        print_centered,
        print_params_table,
        print_run_info,
    )

    if walk_forward:
        wf_result = walk_forward_optimize(
            portfolio=port,
            price_data=price_data,
            start=start_d,
            end=end_d,
            strategy_names=strategy_names,
            n_trials=n_trials,
            min_cash_pct=min_cash_pct,
            min_train_pct=wf_min_train_pct or WF_MIN_TRAIN_PCT,
            min_test_days=wf_min_test_days or WF_MIN_TEST_DAYS,
            log_fn=print_status,
            risk_config=risk_config,
        )

        write_strategies_yaml(
            wf_result.best_params,
            output,
            min_cash_pct=min_cash_pct,
            risk_config=risk_config,
            tax_config=tax_config,
        )

        console.print()

        # Per-fold results — wider since this table has 9 columns.
        fold_table = make_wide_table("Walk-Forward Analysis", width=140)
        fold_table.add_column("Fold", justify="center", style="bold")
        fold_table.add_column("IS Period")
        fold_table.add_column("OOS Period")
        fold_table.add_column("IS Return (Annualized)", justify="right")
        fold_table.add_column("OOS Return (Annualized)", justify="right")
        fold_table.add_column("Max DD", justify="right")
        fold_table.add_column("Sharpe", justify="right")
        fold_table.add_column("Sortino", justify="right")
        fold_table.add_column("Win Rate", justify="right")
        for fold in wf_result.folds:
            fold_table.add_row(
                str(fold.fold),
                f"{fold.train_start} → {fold.train_end}",
                f"{fold.test_start} → {fold.test_end}",
                color_signed(fold.train_return),
                color_signed(fold.test_return),
                f"[red]{fold.max_drawdown:.2%}[/red]",
                color_signed(fold.sharpe_ratio, fmt=".2f"),
                color_signed(fold.sortino_ratio, fmt=".2f"),
                f"{fold.win_rate:.0%}" if fold.win_rate > 0 else "—",
            )
        print_centered(fold_table)

        short_folds = [
            fold for fold in wf_result.folds if (fold.test_end - fold.test_start).days < SHORT_WINDOW_THRESHOLD_DAYS
        ]
        if short_folds:
            shortest = min((fold.test_end - fold.test_start).days for fold in short_folds)
            console.print(
                f"[yellow]Note: {len(short_folds)}/{len(wf_result.folds)} OOS windows are "
                f"under one year (shortest: {shortest} days). Per-fold annualized returns "
                f"extrapolate from short samples and can be noisy.[/yellow]",
                justify="center",
            )

        # Aggregate metrics — same layout as the backtest summary tables.
        n_folds = len(wf_result.folds)
        agg = make_metric_table("Walk-Forward Aggregate")
        agg.add_row("Annualized OOS Return (CAGR)", color_signed(wf_result.annualized_return))
        agg.add_row(
            "Per-Fold OOS Mean ± Std (Annualized)",
            f"{wf_result.mean_test_return:.2%} ± {wf_result.std_test_return:.2%}",
        )
        agg.add_row("Winning Folds", f"{wf_result.winning_folds}/{n_folds}")
        agg.add_row(
            "Best / Worst Fold",
            f"{color_signed(wf_result.best_fold_return)} / {color_signed(wf_result.worst_fold_return)}",
        )
        agg.add_row("Efficiency Ratio", f"{wf_result.efficiency_ratio:.0%}")
        agg.add_row("Mean Max Drawdown", f"[red]{wf_result.mean_max_drawdown:.2%}[/red]")
        agg.add_row("Mean Sharpe Ratio", color_signed(wf_result.mean_sharpe, fmt=".2f"))
        agg.add_row("Mean Sortino Ratio", color_signed(wf_result.mean_sortino, fmt=".2f"))
        agg.add_row(
            "Mean Win Rate",
            f"{wf_result.mean_win_rate:.0%}" if wf_result.mean_win_rate > 0 else "—",
        )
        print_centered(agg)

        print_run_info([("Total Trials", str(wf_result.total_trials)), ("Output", output)])
        print_params_table(
            "Deployed Parameters (from latest fold)",
            wf_result.best_params,
            global_key=ALLOCATION_KEY,
        )

    else:
        result = run_optimize(
            portfolio=port,
            price_data=price_data,
            start=start_d,
            end=end_d,
            strategy_names=strategy_names,
            n_trials=n_trials,
            min_cash_pct=min_cash_pct,
            train_pct=train_pct,
            log_fn=print_status,
            risk_config=risk_config,
        )

        write_strategies_yaml(
            result.best_params,
            output,
            min_cash_pct=min_cash_pct,
            risk_config=risk_config,
            tax_config=tax_config,
        )

        console.print()

        # Reuse the backtest summary tables for the optimized strategy.
        assert result.best_result is not None
        print_backtest_summary(result.best_result)

        print_run_info(
            [
                ("Trials", str(result.trials_run)),
                ("Train/Test Split", f"{train_pct:.0%} / {1 - train_pct:.0%}" if train_pct < 1.0 else "100% / 0%"),
                ("Output", output),
            ]
        )
        print_params_table("Optimized Parameters", result.best_params, global_key=ALLOCATION_KEY)


@cli.command(name="strategies")
def list_strategies() -> None:
    """List all available strategies."""
    print_strategy_table([cls() for cls in STRATEGY_REGISTRY.values()])


_TAX_REPORT_COLUMNS = (
    "ticker",
    "shares",
    "purchase_date",
    "sale_date",
    "cost_basis",
    "proceeds",
    "realized_pnl",
    "holding_period_days",
    "classification",
)


def _print_tax_report(
    rows: list[LoggedTrade],
    basis_per_sell: list[float],
    summary: list[AnnualTaxSummary],
    period_label: str,
) -> None:
    from rich.console import Console

    from midas.output import make_wide_table

    table_width = 140
    table = make_wide_table(f"Schedule D — {period_label}", width=table_width)
    table.add_column("Ticker", style="bold")
    table.add_column("Shares", justify="right")
    table.add_column("Purchase Date")
    table.add_column("Sale Date")
    table.add_column("Cost Basis", justify="right")
    table.add_column("Proceeds", justify="right")
    table.add_column("Realized P&L", justify="right")
    table.add_column("Days Held", justify="right")
    table.add_column("Classification")

    for row, basis in zip(rows, basis_per_sell, strict=True):
        proceeds = row.shares * row.price
        pnl = proceeds - basis * row.shares
        if isinstance(row.purchase_date, date):
            days_held = (row.date - row.purchase_date).days
            purchase_disp = row.purchase_date.isoformat()
            days_disp = str(days_held)
        else:
            purchase_disp = row.purchase_date or ""
            days_disp = ""
        classification = row.holding_period.value if row.holding_period else ""
        table.add_row(
            row.ticker,
            f"{row.shares:.4f}",
            purchase_disp,
            row.date.isoformat(),
            f"${basis:,.2f}",
            f"${proceeds:,.2f}",
            f"${pnl:+,.2f}",
            days_disp,
            classification,
        )
    # Use a width-pinned console so the rendered table isn't truncated when
    # stdout isn't a TTY (e.g. piped output, CliRunner in tests).
    Console(width=table_width).print(table, justify="center")

    for s in summary:
        click.echo(
            f"\nYear {s.year}: ST {s.st_realized:+,.2f}  LT {s.lt_realized:+,.2f}  "
            f"Net {s.net_after_cross:+,.2f}  Deductible {s.deductible_loss:,.2f}  "
            f"Tax {s.tax_owed:+,.2f}  Carry-Forward {s.carry_forward:,.2f}"
        )


def _write_tax_report_csv(
    rows: list[LoggedTrade],
    basis_per_sell: list[float],
    summary: list[AnnualTaxSummary],
    path: Path,
) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(_TAX_REPORT_COLUMNS)
        for row, basis in zip(rows, basis_per_sell, strict=True):
            proceeds = row.shares * row.price
            pnl = proceeds - basis * row.shares
            if isinstance(row.purchase_date, date):
                days_held: object = (row.date - row.purchase_date).days
                purchase_cell: str = row.purchase_date.isoformat()
            else:
                days_held = ""
                purchase_cell = row.purchase_date or ""
            writer.writerow(
                [
                    row.ticker,
                    row.shares,
                    purchase_cell,
                    row.date.isoformat(),
                    round(basis, 4),
                    round(proceeds, 4),
                    round(pnl, 4),
                    days_held,
                    row.holding_period.value if row.holding_period else "",
                ]
            )
        for s in summary:
            writer.writerow([])
            writer.writerow([f"Year {s.year}", "", "", "", "", "", "", "", ""])
            writer.writerow(["ST realized", "", "", "", "", "", round(s.st_realized, 4), "", ""])
            writer.writerow(["LT realized", "", "", "", "", "", round(s.lt_realized, 4), "", ""])
            writer.writerow(["Net (after netting)", "", "", "", "", "", round(s.net_after_cross, 4), "", ""])
            writer.writerow(["Deductible loss", "", "", "", "", "", round(s.deductible_loss, 4), "", ""])
            writer.writerow(["Tax owed", "", "", "", "", "", round(s.tax_owed, 4), "", ""])
            writer.writerow(["Carry forward", "", "", "", "", "", round(s.carry_forward, 4), "", ""])
