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
    PortfolioConfig,
    RiskConfig,
    StrategyConfig,
    TaxConfig,
)
from midas.order_sizer import OrderSizer
from midas.output import print_backtest_summary, print_status, print_strategy_table
from midas.results import write_backtest_results
from midas.strategies import STRATEGY_REGISTRY, EntrySignal, ExitRule, Strategy
from midas.strategies.base import max_warmup, warmup_bars_to_calendar_days


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
    strat_configs, constraints, risk_config, _ = (
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
