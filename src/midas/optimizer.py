"""Bayesian optimizer for strategy parameters using Optuna TPE."""

from __future__ import annotations

import decimal
import logging
import os
import threading
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date
from typing import Any

import optuna
import pandas as pd
import yaml

from midas.allocator import Allocator
from midas.backtest import DEFAULT_TRAIN_PCT, BacktestEngine
from midas.metrics import DAYS_PER_YEAR, compute_annualized_return
from midas.models import (
    DEFAULT_MIN_CASH_PCT,
    AllocationConstraints,
    PortfolioConfig,
    RiskConfig,
    TaxConfig,
)
from midas.order_sizer import OrderSizer
from midas.results import BacktestResult
from midas.strategies import STRATEGY_REGISTRY
from midas.strategies.base import EntrySignal, ExitRule

# Parameters that should be cast to int when building strategy instances.
INT_PARAMS = {
    "window",
    "short_window",
    "long_window",
    "fast_period",
    "slow_period",
    "signal_period",
    "frequency_days",
}

# Meta-params prefixed with _ are not passed to the strategy constructor.
# _weight: blending weight for entry signals.
META_PARAMS = {"_weight"}

# Synthetic key for global allocation knobs (softmax_temperature, min_buy_delta).
ALLOCATION_KEY = "_global"

# Default parameter ranges per strategy.
# Each entry: (min, max, step) — step used for Optuna discretisation.
PARAM_RANGES: dict[str, dict[str, tuple[float, float, float]]] = {
    # --- Entry signals (use _weight for blending) ---
    "MeanReversion": {
        "window": (10, 100, 5),
        "threshold": (0.03, 0.25, 0.01),
        "_weight": (0.5, 3.0, 0.25),
    },
    "Momentum": {
        "window": (5, 50, 2),
        "momentum_scale": (0.02, 0.10, 0.01),
        "_weight": (0.5, 3.0, 0.25),
    },
    "RSIOversold": {
        "window": (7, 28, 2),
        "oversold_threshold": (15.0, 40.0, 2.0),
        "_weight": (0.5, 3.0, 0.25),
    },
    "BollingerBand": {
        "window": (10, 50, 5),
        "num_std": (1.5, 3.0, 0.25),
        "_weight": (0.5, 3.0, 0.25),
    },
    "DonchianBreakout": {
        "window": (10, 60, 5),
        "breakout_scale": (0.01, 0.06, 0.005),
        "_weight": (0.5, 3.0, 0.25),
    },
    "MACDCrossover": {
        "fast_period": (8, 16, 2),
        "slow_period": (20, 40, 2),
        "signal_period": (5, 13, 2),
        "_weight": (0.5, 3.0, 0.25),
    },
    "GapDownRecovery": {
        "gap_threshold": (0.02, 0.08, 0.005),
        "_weight": (0.5, 3.0, 0.25),
    },
    "KeltnerChannel": {
        "window": (10, 50, 5),
        "multiplier": (1.0, 4.0, 0.25),
        "_weight": (0.5, 3.0, 0.25),
    },
    "VWAPReversion": {
        "window": (10, 50, 5),
        "threshold": (0.01, 0.05, 0.005),
        "_weight": (0.5, 3.0, 0.25),
    },
    "MovingAverageCrossover": {
        "short_window": (10, 30, 2),
        "long_window": (40, 100, 5),
        "spread_scale": (0.02, 0.10, 0.01),
        "_weight": (0.5, 3.0, 0.25),
    },
    # --- Exit rules (no _weight — exits don't participate in blending) ---
    "ProfitTaking": {
        "gain_threshold": (0.10, 0.80, 0.03),
    },
    "TrailingStop": {
        "trail_pct": (0.05, 0.25, 0.02),
    },
    "StopLoss": {
        "loss_threshold": (0.05, 0.25, 0.02),
    },
    "ChandelierStop": {
        "window": (10, 40, 2),
        "multiplier": (1.5, 5.0, 0.25),
    },
    "MACDExit": {
        "fast_period": (8, 16, 2),
        "slow_period": (20, 40, 2),
        "signal_period": (5, 13, 2),
    },
    "MovingAverageCrossoverExit": {
        "short_window": (10, 30, 2),
        "long_window": (40, 100, 5),
    },
    "ParabolicSARExit": {
        "af_start": (0.01, 0.04, 0.005),
        "af_step": (0.01, 0.04, 0.005),
        "af_max": (0.10, 0.30, 0.02),
    },
    ALLOCATION_KEY: {
        # softmax_temperature: low = concentrated, high = uniform split.
        "softmax_temperature": (0.2, 1.0, 0.1),
        "min_buy_delta": (0.01, 0.05, 0.005),
        # min_cash_pct is a user risk preference, not optimized
        # max_position_pct is computed dynamically in optimize() from n_tickers
    },
}

DEFAULT_N_TRIALS = 200

# Walk-forward defaults: reserve 60% for the first training window,
# then carve the remaining 40% into test windows of ~63 trading days
# (~3 calendar months) each.
WF_MIN_TRAIN_PCT = 0.60
WF_MIN_TEST_DAYS = 63


@dataclass
class OptimizeResult:
    best_params: dict[str, dict[str, float]]
    best_return: float
    best_bh_return: float
    best_train_return: float
    best_test_return: float
    trials_run: int
    best_result: BacktestResult | None = None


@dataclass
class FoldResult:
    """Result of a single walk-forward fold.

    ``train_return`` and ``test_return`` are annualized (so folds of different
    lengths compare apples-to-apples). ``train_return_raw`` and
    ``test_return_raw`` are the raw (un-annualized) TWR over the fold's own
    window — used by the overall-CAGR compounding loop, which must not
    double-annualize.
    """

    fold: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    best_params: dict[str, dict[str, float]]
    train_return: float
    test_return: float
    train_return_raw: float
    test_return_raw: float
    trials_run: int
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    win_rate: float = 0.0


@dataclass
class WalkForwardResult:
    """Aggregated result of walk-forward optimisation across all folds."""

    folds: list[FoldResult]
    annualized_return: float  # CAGR over the OOS period
    mean_test_return: float
    std_test_return: float
    winning_folds: int  # number of folds with positive OOS return
    best_fold_return: float  # best single-fold OOS return
    worst_fold_return: float  # worst single-fold OOS return
    efficiency_ratio: float  # mean OOS return / mean train return
    best_params: dict[str, dict[str, float]]  # from last fold (most recent data)
    total_trials: int
    mean_max_drawdown: float = 0.0
    mean_sharpe: float = 0.0
    mean_sortino: float = 0.0
    mean_win_rate: float = 0.0


def _suggest_params(
    trial: optuna.Trial,
    strategy_name: str,
    ranges: dict[str, tuple[float, float, float]],
) -> dict[str, float]:
    """Use Optuna trial to suggest parameter values for one strategy."""
    params: dict[str, float] = {}
    for param, (lo, hi, step) in ranges.items():
        key = f"{strategy_name}__{param}"
        # Snap hi down so (hi - lo) is divisible by step, avoiding Optuna warnings.
        # Use Decimal to match Optuna's own check and avoid float noise.
        d_lo, d_hi, d_step = decimal.Decimal(str(lo)), decimal.Decimal(str(hi)), decimal.Decimal(str(step))
        hi = float((d_hi - d_lo) // d_step * d_step + d_lo)
        if param in INT_PARAMS:
            params[param] = float(trial.suggest_int(key, int(lo), int(hi), step=int(step)))
        else:
            params[param] = trial.suggest_float(key, lo, hi, step=step)
    return params


def _run_trial(
    strategy_params: dict[str, dict[str, float]],
    portfolio: PortfolioConfig,
    price_data: dict[str, pd.DataFrame],
    start: date,
    end: date,
    min_cash_pct: float = DEFAULT_MIN_CASH_PCT,
    train_pct: float = DEFAULT_TRAIN_PCT,
    enable_split: bool = True,
    risk_config: RiskConfig | None = None,
) -> tuple[float, float, float, float, float, BacktestResult]:
    """Run a single backtest trial with the allocator + order_sizer + exit_rules system.

    Returns (total_return, bh_return, train_return, test_return, twr, result).
    """
    # Extract global allocation knobs
    global_params = strategy_params.get(ALLOCATION_KEY, {})
    entries: list[tuple[EntrySignal, float]] = []
    exits: list[ExitRule] = []

    for name, params in strategy_params.items():
        if name == ALLOCATION_KEY:
            continue
        cls = STRATEGY_REGISTRY[name]
        weight = params.get("_weight", 1.0)
        clean_params = {
            key: int(val) if key in INT_PARAMS else val for key, val in params.items() if key not in META_PARAMS
        }
        strategy = cls(**clean_params)

        if isinstance(strategy, ExitRule):
            exits.append(strategy)
        elif isinstance(strategy, EntrySignal):
            entries.append((strategy, weight))
        else:
            msg = f"Strategy {name!r} is neither EntrySignal nor ExitRule"
            raise TypeError(msg)

    # Count tickers in portfolio
    n_tickers = sum(1 for holding in portfolio.holdings if holding.shares > 0)
    constraints = AllocationConstraints(
        max_position_pct=global_params.get("max_position_pct"),
        min_cash_pct=min_cash_pct,
        softmax_temperature=global_params.get("softmax_temperature", 0.5),
        min_buy_delta=global_params.get("min_buy_delta", 0.02),
    )

    allocator = Allocator(entries, constraints, n_tickers, risk_config=risk_config)
    order_sizer = OrderSizer()

    engine = BacktestEngine(
        allocator=allocator,
        order_sizer=order_sizer,
        exit_rules=exits,
        constraints=constraints,
        train_pct=train_pct,
        enable_split=enable_split,
    )
    result = engine.run(portfolio, price_data, start, end)

    if result.starting_value <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, result

    total_return = (result.final_value - result.starting_value) / result.starting_value
    bh_return = (result.buy_and_hold_value - result.starting_value) / result.starting_value
    return total_return, bh_return, result.train_return, result.test_return, result.twr, result


worker_state: dict[str, Any] = {}


def _init_worker(
    portfolio: PortfolioConfig,
    price_data: dict[str, pd.DataFrame],
    start: date,
    end: date,
    min_cash_pct: float,
    train_pct: float,
    enable_split: bool = True,
    risk_config: RiskConfig | None = None,
) -> None:
    # Suppress allocator warnings during trial evaluation — the optimizer
    # explores boundary values that trigger heuristic warnings but are fine
    # to evaluate. Scoped to the worker process so the main-process final
    # re-run at the end of optimize() still surfaces warnings for the chosen
    # config, and subsequent backtest/live calls in the same session aren't
    # silently muted.
    logging.getLogger("midas.allocator").setLevel(logging.ERROR)
    worker_state.update(
        portfolio=portfolio,
        price_data=price_data,
        start=start,
        end=end,
        min_cash_pct=min_cash_pct,
        train_pct=train_pct,
        enable_split=enable_split,
        risk_config=risk_config,
    )


def _trial_worker(strategy_params: dict[str, dict[str, float]]) -> tuple[float, ...]:
    total_ret, bh_ret, train_ret, test_ret, twr, _result = _run_trial(strategy_params, **worker_state)
    return total_ret, bh_ret, train_ret, test_ret, twr


def _wf_init_worker(
    portfolio: PortfolioConfig,
    price_data: dict[str, pd.DataFrame],
    min_cash_pct: float,
    risk_config: RiskConfig | None = None,
) -> None:
    """Initialise walk-forward workers with static state only (dates vary per call)."""
    logging.getLogger("midas.allocator").setLevel(logging.ERROR)
    worker_state.update(
        portfolio=portfolio,
        price_data=price_data,
        min_cash_pct=min_cash_pct,
        risk_config=risk_config,
    )


def _wf_trial_worker(
    strategy_params: dict[str, dict[str, float]],
    start: date,
    end: date,
) -> tuple[float, ...]:
    total_ret, bh_ret, train_ret, test_ret, twr, _result = _run_trial(
        strategy_params,
        worker_state["portfolio"],
        worker_state["price_data"],
        start,
        end,
        worker_state["min_cash_pct"],
        enable_split=False,
        risk_config=worker_state.get("risk_config"),
    )
    return total_ret, bh_ret, train_ret, test_ret, twr


def max_warmup_for_search(
    strategy_names: list[str] | None,
    min_cash_pct: float = DEFAULT_MIN_CASH_PCT,
    n_tickers: int = 1,
) -> int:
    """Upper bound on warmup bars across the optimizer's search space.

    Walk-forward and standard optimizer trials sample different parameter
    values, so the prefetched warmup buffer must cover the worst case.
    For each optimizable strategy, instantiate it with every integer
    parameter pinned to its search-range upper bound and take the max
    ``warmup_period``.
    """
    names, ranges = _prepare_names_and_ranges(strategy_names, min_cash_pct, n_tickers)
    max_warmup_val = 0
    for name in names:
        if name == ALLOCATION_KEY:
            continue
        cls = STRATEGY_REGISTRY[name]
        params: dict[str, Any] = {}
        for pname, (lo, hi, step) in ranges.get(name, {}).items():
            if pname in META_PARAMS:
                continue
            # Snap hi down the same way _suggest_params does so the upper
            # bound here matches values the optimizer actually tries.
            d_lo, d_hi, d_step = decimal.Decimal(str(lo)), decimal.Decimal(str(hi)), decimal.Decimal(str(step))
            snapped_hi = float((d_hi - d_lo) // d_step * d_step + d_lo)
            params[pname] = int(snapped_hi) if pname in INT_PARAMS else snapped_hi
        # Let TypeError propagate — if a search-range key doesn't match a
        # constructor param, that's a real configuration bug we want loud.
        instance = cls(**params)
        max_warmup_val = max(max_warmup_val, instance.warmup_period)
    return max_warmup_val


def _prepare_names_and_ranges(
    strategy_names: list[str] | None,
    min_cash_pct: float,
    n_tickers: int,
) -> tuple[list[str], dict[str, dict[str, tuple[float, float, float]]]]:
    """Resolve strategy names and build parameter ranges (shared by optimize/walk-forward)."""
    names = strategy_names or [key for key in PARAM_RANGES if key != ALLOCATION_KEY]
    names = [name for name in names if name in PARAM_RANGES]

    if not names:
        msg = "No optimizable strategies found"
        raise ValueError(msg)

    names.append(ALLOCATION_KEY)

    equal_weight = (1.0 - min_cash_pct) / max(n_tickers, 1)
    lo = max(round(1.5 * equal_weight, 2), 0.10)
    hi = min(round(5.0 * equal_weight, 2), 0.80)
    if lo >= hi:
        lo, hi = 0.10, 0.80
    step = round((hi - lo) / 8, 2) or 0.01
    ranges = {name: dict(PARAM_RANGES[name]) for name in names if name in PARAM_RANGES}
    ranges.setdefault(ALLOCATION_KEY, {})
    ranges[ALLOCATION_KEY]["max_position_pct"] = (lo, hi, step)

    return names, ranges


def optimize(
    portfolio: PortfolioConfig,
    price_data: dict[str, pd.DataFrame],
    start: date,
    end: date,
    strategy_names: list[str] | None = None,
    n_trials: int = DEFAULT_N_TRIALS,
    min_cash_pct: float = DEFAULT_MIN_CASH_PCT,
    train_pct: float = DEFAULT_TRAIN_PCT,
    log_fn: Callable[[str], None] | None = None,
    risk_config: RiskConfig | None = None,
) -> OptimizeResult:
    """Bayesian optimization over strategy parameters using Optuna TPE.

    Runs *n_trials* Optuna trials (default 200).  Each trial samples a
    parameter combination via the Tree-structured Parzen Estimator and
    evaluates it with a full backtest.  Backtests are executed in a worker
    pool to utilise multiple CPU cores.
    """
    log = log_fn or (lambda _: None)

    n_tickers = sum(1 for holding in portfolio.holdings if holding.shares > 0)
    names, ranges = _prepare_names_and_ranges(strategy_names, min_cash_pct, n_tickers)

    max_workers = min((os.cpu_count() or 4) // 2, n_trials) or 1

    strat_names = [name for name in names if name != ALLOCATION_KEY]
    log(f"Optimizing {len(strat_names)} strategies over {start} to {end}")
    log(f"  {n_trials} trials across {max_workers} workers")

    # Suppress Optuna's default logging (we provide our own via log_fn).
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # -- Objective that runs in the main process but farms backtest to pool --
    pool = ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(portfolio, price_data, start, end, min_cash_pct, train_pct, True, risk_config),
    )

    trials_done = 0
    progress_lock = threading.Lock()

    def objective(trial: optuna.Trial) -> float:
        nonlocal trials_done

        strategy_params: dict[str, dict[str, float]] = {}
        for name in names:
            strategy_params[name] = _suggest_params(
                trial,
                name,
                ranges[name],
            )

        _total_ret, bh_ret, train_ret, test_ret, _twr = pool.submit(
            _trial_worker,
            strategy_params,
        ).result()

        # Store auxiliary metrics as user attributes for later retrieval.
        trial.set_user_attr("bh_return", bh_ret)
        trial.set_user_attr("train_return", train_ret)
        trial.set_user_attr("test_return", test_ret)
        trial.set_user_attr("params", strategy_params)

        with progress_lock:
            trials_done += 1
            if trials_done % 25 == 0 or trials_done == n_trials:
                pct = trials_done * 100 // n_trials
                log(f"  [{pct:3d}%] {trials_done}/{n_trials} trials — best return: {study.best_value:.2%}")

        return train_ret

    try:
        study.optimize(objective, n_trials=n_trials, n_jobs=max_workers)
    finally:
        pool.shutdown(wait=True)

    best = study.best_trial
    best_params: dict[str, dict[str, float]] = best.user_attrs["params"]

    log(f"Optimization complete — best train return: {best.value:.2%}")

    # Re-run best params in the main process to capture the full BacktestResult
    # (risk/trade metrics aren't serialised through the worker pool).
    _total, _bh, _train, _test, _twr, best_result = _run_trial(
        best_params,
        portfolio,
        price_data,
        start,
        end,
        min_cash_pct=min_cash_pct,
        train_pct=train_pct,
        risk_config=risk_config,
    )

    log(
        f"  Max drawdown: {best_result.max_drawdown:.2%} | "
        f"Sharpe: {best_result.sharpe_ratio:.2f} | "
        f"Win rate: {best_result.win_rate:.2%}"
    )

    return OptimizeResult(
        best_params=best_params,
        best_return=round(best.value or 0.0, 4),
        best_bh_return=round(best.user_attrs["bh_return"], 4),
        best_train_return=round(best.user_attrs["train_return"], 4),
        best_test_return=round(best.user_attrs["test_return"], 4),
        trials_run=len(study.trials),
        best_result=best_result,
    )


def walk_forward_optimize(
    portfolio: PortfolioConfig,
    price_data: dict[str, pd.DataFrame],
    start: date,
    end: date,
    strategy_names: list[str] | None = None,
    n_trials: int = DEFAULT_N_TRIALS,
    min_cash_pct: float = DEFAULT_MIN_CASH_PCT,
    min_train_pct: float = WF_MIN_TRAIN_PCT,
    min_test_days: int = WF_MIN_TEST_DAYS,
    log_fn: Callable[[str], None] | None = None,
    risk_config: RiskConfig | None = None,
) -> WalkForwardResult:
    """Walk-forward optimisation with anchored training windows.

    Reserves *min_train_pct* of trading days as the minimum training window,
    then carves the remainder into test windows of *min_test_days* each.
    Each fold grows the training window while the test window slides forward.
    Parameters are re-optimised per fold so every test period is genuinely
    out-of-sample.
    """
    log = log_fn or (lambda _: None)

    n_tickers = sum(1 for holding in portfolio.holdings if holding.shares > 0)
    names, ranges = _prepare_names_and_ranges(strategy_names, min_cash_pct, n_tickers)

    # Collect trading days across all tickers.
    all_dates: set[date] = set()
    for df in price_data.values():
        all_dates.update(dt for dt in df.index if start <= dt <= end)
    trading_days = sorted(all_dates)

    n_days = len(trading_days)
    train_cutoff = int(n_days * min_train_pct)
    remaining = n_days - train_cutoff
    if remaining < min_test_days * 2:
        min_needed = int(min_test_days * 2 / (1 - min_train_pct))
        msg = f"Not enough data for walk-forward ({n_days} days, need ≥{min_needed})"
        raise ValueError(msg)

    n_folds = max(remaining // min_test_days, 2)
    test_size = remaining // n_folds

    # Build fold boundaries: [train_cutoff, train_cutoff + test_size, ...]
    fold_boundaries = [train_cutoff + i * test_size for i in range(n_folds + 1)]
    fold_boundaries[-1] = n_days  # last fold absorbs remainder

    trials_per_fold = max(n_trials // n_folds, 10)
    max_workers = min((os.cpu_count() or 4) // 2, trials_per_fold) or 1

    strat_names = [name for name in names if name != ALLOCATION_KEY]
    log(f"Walk-forward optimization — {len(strat_names)} strategies, {start} to {end}")
    log(f"  {n_folds} folds, ~{test_size} trading days per test window, {trials_per_fold} trials/fold")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    fold_results: list[FoldResult] = []
    prev_best_flat: dict[str, float] | None = None

    # Single pool reused across all folds — dates are passed per call.
    pool = ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_wf_init_worker,
        initargs=(portfolio, price_data, min_cash_pct, risk_config),
    )

    try:
        for fold_idx in range(n_folds):
            # Anchored training: always starts at day 0, grows each fold.
            fold_train_start = trading_days[0]
            fold_train_end = trading_days[fold_boundaries[fold_idx] - 1]
            fold_test_start = trading_days[fold_boundaries[fold_idx]]
            fold_test_end = trading_days[fold_boundaries[fold_idx + 1] - 1]

            log(f"Fold {fold_idx + 1}/{n_folds}")
            log(f"  Train: {fold_train_start} → {fold_train_end}")
            log(f"  Test:  {fold_test_start} → {fold_test_end}")

            # --- Optimise on training window (no internal split) ---
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42 + fold_idx),
            )

            # Warm-start: seed with previous fold's best params so Optuna
            # doesn't explore blindly when adjacent folds share most data.
            if prev_best_flat is not None:
                study.enqueue_trial(prev_best_flat)

            def _make_objective(
                pool_ref: ProcessPoolExecutor,
                names_ref: list[str],
                ranges_ref: dict[str, dict[str, tuple[float, float, float]]],
                study_ref: optuna.Study,
                fold_trials: int,
                train_start: date,
                train_end: date,
            ) -> Callable[[optuna.Trial], float]:
                counter = [0]
                lock = threading.Lock()

                def objective(trial: optuna.Trial) -> float:
                    strategy_params: dict[str, dict[str, float]] = {}
                    for name in names_ref:
                        strategy_params[name] = _suggest_params(trial, name, ranges_ref[name])
                    _total_ret, _bh, _train, _test, twr = pool_ref.submit(
                        _wf_trial_worker,
                        strategy_params,
                        train_start,
                        train_end,
                    ).result()
                    trial.set_user_attr("params", strategy_params)
                    with lock:
                        counter[0] += 1
                        if counter[0] % 25 == 0 or counter[0] == fold_trials:
                            pct = counter[0] * 100 // fold_trials
                            log(f"  [{pct:3d}%] {counter[0]}/{fold_trials} trials — best: {study_ref.best_value:.2%}")
                    return twr

                return objective

            study.optimize(
                _make_objective(pool, names, ranges, study, trials_per_fold, fold_train_start, fold_train_end),
                n_trials=trials_per_fold,
                n_jobs=max_workers,
            )

            best_params: dict[str, dict[str, float]] = study.best_trial.user_attrs["params"]
            train_return_raw = study.best_trial.value or 0.0
            prev_best_flat = dict(study.best_trial.params)

            # --- Evaluate best params on test window ---
            _test_total, _bh, _train, _test, test_twr, test_result = _run_trial(
                best_params,
                portfolio,
                price_data,
                fold_test_start,
                fold_test_end,
                min_cash_pct,
                enable_split=False,
                risk_config=risk_config,
            )

            train_days = (fold_train_end - fold_train_start).days
            test_days = (fold_test_end - fold_test_start).days
            train_ann = compute_annualized_return(train_return_raw, train_days)
            test_ann = compute_annualized_return(test_twr, test_days)

            log(f"  Result — train: {train_ann:.2%} annualized | out-of-sample: {test_ann:.2%} annualized")

            fold_results.append(
                FoldResult(
                    fold=fold_idx + 1,
                    train_start=fold_train_start,
                    train_end=fold_train_end,
                    test_start=fold_test_start,
                    test_end=fold_test_end,
                    best_params=best_params,
                    train_return=round(train_ann, 4),
                    test_return=round(test_ann, 4),
                    train_return_raw=round(train_return_raw, 4),
                    test_return_raw=round(test_twr, 4),
                    trials_run=len(study.trials),
                    max_drawdown=round(test_result.max_drawdown, 4),
                    sharpe_ratio=round(test_result.sharpe_ratio, 4),
                    sortino_ratio=round(test_result.sortino_ratio, 4),
                    win_rate=round(test_result.win_rate, 4),
                )
            )
    finally:
        pool.shutdown(wait=True)

    # Per-fold test returns are already annualized, so aggregates (mean, std,
    # best/worst) compare folds of different lengths apples-to-apples.
    test_returns = [fold.test_return for fold in fold_results]
    mean_test = sum(test_returns) / len(test_returns)
    variance = sum((ret - mean_test) ** 2 for ret in test_returns) / max(len(test_returns) - 1, 1)
    std_test = variance**0.5

    # Overall OOS CAGR: compound the *raw* per-fold returns (what actually
    # happened to the equity chain) and then annualize over the full OOS span.
    # Doing this on raw values avoids the double-annualization artefact that
    # would come from chaining already-annualized fold returns.
    compounded = 1.0
    for fold in fold_results:
        compounded *= 1.0 + fold.test_return_raw
    first_test_start = fold_results[0].test_start
    last_test_end = fold_results[-1].test_end
    years = (last_test_end - first_test_start).days / DAYS_PER_YEAR
    annualized = compounded ** (1.0 / years) - 1.0 if years > 0 and compounded > 0 else 0.0

    winning_folds = sum(1 for ret in test_returns if ret > 0)
    best_fold = max(test_returns)
    worst_fold = min(test_returns)

    # Efficiency ratio: how much in-sample performance survives out-of-sample.
    # Both sides are annualized so the ratio is dimensionally consistent even
    # though IS windows are anchored (longer) and OOS windows are shorter.
    train_returns = [fold.train_return for fold in fold_results]
    mean_train = sum(train_returns) / len(train_returns)
    efficiency = mean_test / mean_train if mean_train != 0 else 0.0

    num_folds = len(fold_results)
    mean_dd = sum(fold.max_drawdown for fold in fold_results) / num_folds
    mean_sharpe = sum(fold.sharpe_ratio for fold in fold_results) / num_folds
    mean_sortino = sum(fold.sortino_ratio for fold in fold_results) / num_folds
    mean_wr = sum(fold.win_rate for fold in fold_results) / num_folds

    log("")
    log("Walk-forward complete")
    log(f"  Annualized OOS return (CAGR): {annualized:.2%}")
    log(f"  Per-fold mean: {mean_test:.2%} ± {std_test:.2%}")
    log(f"  Winning folds: {winning_folds}/{num_folds} | Best: {best_fold:.2%} | Worst: {worst_fold:.2%}")
    log(f"  Efficiency ratio: {efficiency:.0%}")
    log(f"  Mean max drawdown: {mean_dd:.2%} | Mean Sharpe: {mean_sharpe:.2f}")

    return WalkForwardResult(
        folds=fold_results,
        annualized_return=round(annualized, 4),
        mean_test_return=round(mean_test, 4),
        std_test_return=round(std_test, 4),
        winning_folds=winning_folds,
        best_fold_return=round(best_fold, 4),
        worst_fold_return=round(worst_fold, 4),
        efficiency_ratio=round(efficiency, 4),
        best_params=fold_results[-1].best_params,
        total_trials=sum(fold.trials_run for fold in fold_results),
        mean_max_drawdown=round(mean_dd, 4),
        mean_sharpe=round(mean_sharpe, 4),
        mean_sortino=round(mean_sortino, 4),
        mean_win_rate=round(mean_wr, 4),
    )


def write_strategies_yaml(
    params: dict[str, dict[str, float]],
    path: str,
    min_cash_pct: float = DEFAULT_MIN_CASH_PCT,
    risk_config: RiskConfig | None = None,
    tax_config: TaxConfig | None = None,
) -> None:
    """Write optimized parameters to a strategies YAML file.

    The optimizer does not search risk or tax knobs (both are policy, not
    tunables). When the user supplied a ``risk:`` or ``tax:`` block in the
    input, it must round-trip to the optimized output unchanged so the next
    run honors the same policy; otherwise the optimized YAML silently drops
    the user's config.

    Args:
        params: Per-strategy parameter dict from the optimizer (also includes
            ``ALLOCATION_KEY`` for portfolio-wide knobs).
        path: Output path.
        min_cash_pct: Preserved from the user's input config.
        risk_config: Preserved from the user's input config. When present and
            differing from defaults, emitted as a ``risk:`` block. ``None`` or
            an all-default ``RiskConfig`` is omitted.
        tax_config: Preserved from the user's input config. When present,
            emitted as a ``tax:`` block. ``None`` is omitted.
    """
    output: dict[str, object] = {}

    # Emit global allocation knobs as top-level keys
    if ALLOCATION_KEY in params:
        for key, val in params[ALLOCATION_KEY].items():
            output[key] = round(val, 4)

    # min_cash_pct is not optimized — preserve the user's configured value
    output["min_cash_pct"] = round(min_cash_pct, 4)

    risk_block = _risk_block_for_yaml(risk_config)
    if risk_block:
        output["risk"] = risk_block

    if tax_config is not None:
        output["tax"] = {
            "short_term_rate": tax_config.short_term_rate,
            "long_term_rate": tax_config.long_term_rate,
            "deductible_loss_cap": tax_config.deductible_loss_cap,
            "payment_lag_days": tax_config.payment_lag_days,
        }

    strategies = []
    for name, param_dict in params.items():
        if name == ALLOCATION_KEY:
            continue
        entry: dict[str, object] = {"name": name}
        clean_params: dict[str, object] = {}
        for key, val in param_dict.items():
            if key == "_weight":
                entry["weight"] = round(val, 4)
            elif key in INT_PARAMS:
                clean_params[key] = int(val)
            else:
                clean_params[key] = round(val, 4)
        if clean_params:
            entry["params"] = clean_params
        strategies.append(entry)

    output["strategies"] = strategies

    with open(path, "w") as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)


def _risk_block_for_yaml(risk_config: RiskConfig | None) -> dict[str, object]:
    """Render a ``risk:`` block dict, omitting fields that match the dataclass default.

    Returns ``{}`` when ``risk_config`` is ``None`` or every field is at its
    default — there is no behavioral signal to preserve. Otherwise emits only
    the fields the user (implicitly or explicitly) set to a non-default value,
    matching the spec's "omit to disable" YAML conventions.
    """
    if risk_config is None:
        return {}
    default = RiskConfig()
    block: dict[str, object] = {}
    if risk_config.weighting != default.weighting:
        block["weighting"] = risk_config.weighting
    if risk_config.vol_lookback_days != default.vol_lookback_days:
        block["vol_lookback_days"] = int(risk_config.vol_lookback_days)
    if risk_config.vol_target is not None:
        block["vol_target"] = round(float(risk_config.vol_target), 4)
    if risk_config.drawdown_penalty is not None:
        block["drawdown_penalty"] = round(float(risk_config.drawdown_penalty), 4)
    if risk_config.drawdown_floor is not None:
        block["drawdown_floor"] = round(float(risk_config.drawdown_floor), 4)
    return block
