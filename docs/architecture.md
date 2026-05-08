# Architecture

Midas is a target-weight portfolio allocation engine. Entry signals score buy candidates, an allocator blends those scores into target weights via softmax, exit rules clamp those targets downward (LEAN-style override layer), and the order sizer turns the deltas against current holdings into buy and sell orders.

## The Two-Tier Model

Strategies fall into exactly two disjoint tiers:

- **EntrySignal** — scores ticker bullishness in `[0, 1]`. Pure buy-side. Returning 0 means "no opinion" (not "sell"). Returning `None` means "abstain" (insufficient data, missing context). Multiple entry signals are blended into a single conviction per ticker, then softmaxed into target weights.

- **ExitRule** — a downstream override layer that clamps the allocator's proposed target weights downward. Each rule's `clamp_target(ticker, proposed_target, price_history, cost_basis, high_water_mark)` returns an adjusted target weight that must be ≤ the proposed target. Returning 0.0 means "full liquidation." Exit rules never participate in target-weight construction or softmax; they sit downstream as a veto/reduce layer. The order sizer computes sell orders from negative deltas (clamped target < current weight).

The two tiers are enforced at the type level (`EntrySignal` and `ExitRule` are separate base classes in `strategies/base.py`) and at runtime by an `isinstance` partition in the optimizer, config loader, and backtest engine. The two share a thin `Strategy` base for shared bookkeeping (`name`, `warmup_period`, `suitability`, `description`), but the scoring interfaces (`EntrySignal.score` and `ExitRule.clamp_target`) are completely disjoint. There is no third tier; entry-signal logic cannot accidentally produce a sell, and exit-rule logic cannot accidentally inflate a buy.

### How this compares to LEAN

QuantConnect LEAN splits the same problem across **AlphaModel** (signals), **PortfolioConstructionModel** (weights), and **RiskManagementModel** (overrides like stop loss). Midas's `EntrySignal` is the alpha + portfolio-construction half, and `ExitRule` is the risk-management half. Like LEAN's `RiskManagementModel`, midas's exit rules act as a downstream clamp — they can reduce proposed target weights but never increase them. Both systems evaluate exits at the aggregate position level (not per-lot), and sells arise from negative deltas between clamped targets and current weights.

## Core Engine

The engine follows a linear pipeline on every tick (one simulated day in backtesting, one poll interval in live mode):

1. **Entry signals** score every ticker
2. **Allocator** blends those scores into target weights (softmax construct-to-budget)
3. **Exit rules** clamp proposed targets downward (can reduce, never increase; first clamper wins attribution)
4. **Order sizer** sizes sells from negative deltas (clamped target < current weight), then buys from positive deltas

Sells are sized first, freeing up cash for the buy pass to use in the same tick.

Every component is stateless within a single tick. State management (positions, cash, lot list, trade log) is the responsibility of whichever execution mode is driving the engine.

### Entry Signals

Entry signals are stateless, ticker-agnostic scorers that return a number in `[0, 1]` (or `None` to abstain). All entry signals inherit from `EntrySignal` and are registered by name in `strategies/__init__.py`. The CLI, optimizer, and config loader all use this registry to instantiate entry signals by name from YAML.

#### Precomputation

For backtest performance, entry signals can optionally compute scores for every day of the price series in a single vectorized pass. The allocator caches these results and looks them up by day index during simulation, avoiding per-day function calls.

#### Warmup

Each strategy declares a `warmup_period` — the bars of price history it needs before it produces a valid score. The CLI fetches a lookback buffer equal to the maximum warmup across configured entry signals *and* exit rules (plus slack for weekends/holidays) so signals are valid from day one of the simulation rather than spending the first N days in cold start. For recursive indicators (RSI, MACD), the nominal period is multiplied by a TA-Lib-style unstable-period factor so the indicator has room to converge. Live mode derives the same history window, and the walk-forward optimizer prefetches a single buffer sized for the upper bound of its parameter search space.

See [Strategies](strategies.md) for the full reference.

### Allocator

The allocator turns entry-signal scores into target portfolio weights. It runs in three phases.

#### Phase 1: Score and Blend

For each ticker, the allocator collects scores from all entry signals and computes a weighted average. Each entry signal has a configurable `weight` (default 1.0) that controls its influence. A signal returning `None` is excluded entirely — it doesn't pull the average toward zero, it simply doesn't participate.

Tickers fall into two buckets:

- **Active** — at least one entry signal scored > 0. The blended score is positive.
- **Held** — every entry signal returned 0 or `None`. The allocator treats this as "no opinion" and holds the ticker at its current weight (or the equal-weight base on the very first allocation).

#### Phase 2: Softmax Budget Allocation

The active tickers' blended scores need to become target portfolio weights that sum to the *active* budget — that is, the investable budget (`1 − min_cash_pct`) minus whatever the held tickers consume. The allocator uses softmax — the same construct-to-budget operator used by QuantConnect LEAN's `InsightWeightingPortfolioConstructionModel`, mean-variance optimizers, and risk-parity libraries.

```
target_i = active_budget * exp(blended_i / T) / sum_j(exp(blended_j / T))
```

By construction, `sum(active targets) == active_budget` exactly, always. There is no separate normalize step — oversubscription is mathematically impossible. The `softmax_temperature` parameter `T` follows the standard ML softmax convention: low `T` concentrates budget on the highest-conviction ticker (winner-take-most, `T → 0` is argmax), `T = 1` is the unscaled softmax over raw scores, and high `T` approaches a uniform split. Midas defaults to `T = 0.5`, a mild concentration bias.

**Neutral = hold.** When all entry signals abstain or score 0 for a ticker, the allocator holds that ticker's current weight rather than dragging it back to equal-weight. This avoids churn from drift-correction trades on days when no signal is firing on a held position.

#### Phase 3: Soft Position Cap

Any active ticker whose softmax target exceeds `max_position_pct` is pinned at the cap, and the freed budget is redistributed to the uncapped survivors by re-running softmax over them with the reduced budget. The loop runs until no survivor exceeds the cap.

The cap is **soft**: it can refuse to allocate *more* budget to an over-target ticker, but it never forces a sell. If a ticker drifts above the cap because of price appreciation, the allocator simply stops buying more — it does not generate a corrective sell. Sells are exclusively the domain of `ExitRule` strategies. This is the cleanest way to keep the buy-only allocator from accidentally producing exits.

If `max_position_pct` is not configured, it's auto-computed as 2.5x the equal-weight baseline (capped at 25%), which allows meaningful overweighting without extreme concentration.

#### Phase 4a: CPPI Drawdown Overlay (optional)

When `risk.drawdown_penalty` and `risk.drawdown_floor` are both configured, the allocator multiplies the gross investable budget by `max(1 - penalty * current_drawdown, floor)` *before* the softmax runs. The freed budget becomes deliberate cash reserve and recovers automatically as the portfolio heals. The driver (`backtest.py`) tracks the running peak portfolio value; live mode warns at startup and treats the overlay as inert pending peak persistence (v2).

#### Phase 4b: Portfolio Vol Target (optional)

After the cap converges, if `risk.vol_target` is configured, the allocator computes predicted annualized portfolio vol from a Ledoit-Wolf-shrunk covariance matrix and scales the entire weight vector down by `vol_target / predicted` if the predicted exceeds the target. Slack flows to cash. The cap is *not* re-applied after scaling — scaling only shrinks weights, so an upper cap remains satisfied.

Phases 4a and 4b are both reduce-only risk overlays. Phase 4a runs before Phase 1 and shrinks the budget that Phase 1 has to allocate; Phase 4b runs after Phase 3 and shrinks the resulting weight vector. Together they cap risk from both ends of the pipeline. While Phase 4b is non-binding, the two stack multiplicatively (e.g. a 20% drawdown with `drawdown_penalty: 1.5` shrinks gross to 70% and Phase 4b leaves it alone). When Phase 4b binds it normalizes predicted vol to target, mathematically erasing prior gross-scaling. With aggressive settings during deep drawdowns the resulting gross can drop well below `drawdown_floor`. This is intentional.

#### Inverse-Vol Weighting

When `risk.weighting: inverse_vol`, the softmax exponent gains a per-ticker offset of `-log(max(vol_i, vol_floor))` *added outside the `/T` divider*:

```
weight_i ∝ exp(blended_i / T + offset_i)
```

The form keeps the offset's contribution invariant to softmax temperature — a 10× vol gap is always a 10× weight gap from vol alone, regardless of how concentrated conviction is. (PR #63 used `(1/vol)^(1/T)` and was rejected for coupling inverse-vol intensity to `T`.) Tickers with insufficient history or zero realized vol fall back to Option A (held at current weight, excluded from the softmax).

#### Risk Telemetry

`RiskMetrics` (rolling 60-day vol, drawdown from peak, rolling 252-day Sharpe, per-strategy P&L attribution) is computed from the equity curve and surfaced through `output.py`. Per-strategy attribution uses a cost-basis-weighted running blend per position, consistent with the existing aggregate-position philosophy. Strictly observational — never feeds back into construction.

### Exit Rules

Exit rules run *downstream* of the allocator as a LEAN-style override/veto layer. Each rule's `clamp_target(ticker, proposed_target, price_history, cost_basis, high_water_mark)` receives the ticker's proposed target weight, aggregate cost basis, aggregate high-water mark, and price history; it returns an adjusted target ≤ the proposed target. Returning 0.0 means full liquidation.

Exit rules evaluate at the **aggregate position level** — they see a single cost basis (share-weighted average of all lots) and a single high-water mark (the ticker's all-time peak since entry). This matches how LEAN, Zipline, and Backtrader handle exits. Per-lot logic belongs at execution time (FIFO consumption), not at exit-rule evaluation time.

Exit rules are applied sequentially. Each rule can only reduce a target, never increase it. The first rule to clamp a ticker wins attribution for that sell order. If multiple rules would fire on the same ticker, only the first clamper's reason and source appear on the order.

### Order Sizer

`OrderSizer` is a stateless converter from target-weight deltas to concrete orders. It exposes two methods:

- **`size_buys(allocation, positions, prices, cash, constraints)`** — diffs the clamped target weights against the current weights and emits buy orders for any underweight ticker whose delta exceeds `min_buy_delta`. **Buy-only.** A ticker that has drifted *above* its target never produces a sell — that's the soft-cap principle from the allocator carried through to the order sizer.

- **`size_sells(clamped_targets, positions, prices, total_value, clamp_attribution)`** — computes sell orders from negative deltas (clamped target < current weight). Only tickers present in `clamp_attribution` (i.e., where an exit rule fired) produce sells. Shares are capped at actual held shares. Attribution comes from the exit rule that first clamped the target.

Sells run before buys in every tick so that exit proceeds become available cash for the buy pass.

#### Slippage and Circuit Breaker

All orders include a slippage estimate (default 0.05%) to model realistic execution costs. Buy prices are adjusted slightly upward, sell prices slightly downward. Share counts are floored to whole numbers since fractional shares aren't supported.

A daily deployment cap limits total buy value to 25% of portfolio value per day, preventing the engine from going all-in during a single volatile session.

### Trading Restrictions

The restriction tracker enforces round-trip rules. When `round_trip_days` is configured (e.g., 30 days), you cannot buy then sell (or sell then buy) the same ticker within that window. This prevents wash sales and models real-world brokerage restrictions. Orders violating this constraint are filtered out before execution.

## Execution Modes

The core engine doesn't run itself -- it needs a driver that feeds it price data, manages portfolio state, and decides what to do with the resulting orders. Midas provides three execution modes, and the typical workflow follows them in order: optimize to find good parameters, backtest to validate performance, and live to act on real-time signals.

### Optimizer

The optimizer is usually the starting point. Rather than hand-tuning parameters, you let the optimizer search for a combination that performs well on historical data. It outputs a strategies YAML that you can then feed into backtest or live mode.

The optimizer uses Bayesian optimization (Optuna's TPE sampler) to search jointly over:

| Layer | What it controls | Search range |
|-------|-----------------|--------------|
| Entry signal parameters | When a signal fires and how strong | `window`, `threshold`, etc. |
| Entry signal weights | How much influence each signal has in the blend | 0.5 to 3.0 |
| Exit rule parameters | When an exit triggers | `loss_threshold`, `trail_pct`, `gain_threshold`, etc. |
| Softmax temperature | How aggressively the allocator concentrates budget | 0.2 to 1.0 |
| Min buy delta | Minimum weight diff to trigger a buy | 0.01 to 0.05 |
| Max position % | Maximum weight for any single position | 0.15 to 0.50 |

Default search ranges are defined in `PARAM_RANGES` in `optimizer.py`. Exit rules don't get a `weight` field — they fire on their own conditions, not as a contributor to a blended score.

**Standard Mode** -- Runs a configurable number of trials (default 200). Each trial suggests a parameter combination, runs a full backtest with train/test split, and returns the training return as the optimization objective. Trials are distributed across CPU cores via multiprocessing for parallel evaluation.

#### Walk-Forward Optimization

[Walk-forward optimization](https://en.wikipedia.org/wiki/Walk_forward_optimization) is considered the gold standard for validating trading strategies. It determines optimal parameters while testing their robustness against overfitting.

Standard optimization can overfit -- parameters that look great on historical data may not work going forward. Walk-forward fixes this by repeatedly optimizing on in-sample data, then testing on out-of-sample data that was never used during optimization. The time window rolls forward and the process repeats until all available data is used.

```
Fold 1: train [2020───2023.01]  test [2023.01───2023.04]  → 4.3%
Fold 2: train [2020───2023.04]  test [2023.04───2023.07]  → 2.1%
Fold 3: train [2020───2023.07]  test [2023.07───2023.10]  → 3.8%
```

The optimizer only sees training data when picking parameters -- it has no access to the test window. So when you evaluate those parameters on the test window, the results tell you how the strategy would have performed on data it wasn't tuned for. A robust strategy shows consistent positive out-of-sample results across multiple folds. The summary reports annualized CAGR, per-fold OOS mean/std, best/worst fold, and an efficiency ratio (how much of the training performance holds up out-of-sample).

Parameters written to the output YAML come from the last fold (trained on the most data). Each fold is warm-started with the previous fold's best parameters to exploit correlation between adjacent time periods.

See [CLI Reference](cli.md#optimize) for all optimizer options.

### Backtest

Once you have a set of parameters (from the optimizer or hand-tuned), backtesting lets you see exactly how they would have performed over a historical period. It produces a detailed trade log, return metrics, and a comparison against a buy-and-hold baseline.

The backtest engine simulates the full pipeline over historical data, stepping through one trading day at a time. On each trading day, the engine runs the complete pipeline: cash infusion (if scheduled), entry-signal scoring, allocation, exit-rule target clamping, sell pass, buy pass, restriction filtering, and order execution. After execution, it updates positions, the lot list, cash balance, and the trade log.

**Execution Lag** -- By default the backtest fills at the **next bar's open** (`--execution-mode=next_open`). Signals read at day T's close can only be acted on after the bell rings again, so orders computed from T's history fill at T+1's open price. This matches what a real operator can actually do: see the close, compute a decision overnight, send the order, let it fill at the open. Two other modes are available:

- `next_close` — fill at T+1's close (market-on-close at the next session). Useful when you model end-of-day rebalancing.
- `close` — fill at T's close, same bar the signal was computed from. **Optimistic.** This is the old default, preserved for regression pinning and for comparing against the lookahead bias in prior reports — not a realistic live simulation.

Under lagged modes the decision computed on the *final* simulated bar never executes — there is no T+1 bar inside the window. That matches reality: an order placed after the last session cannot fill inside the backtest. Switching from `close` to `next_open` typically trims 50–200 bps/yr off reported returns depending on how quickly strategies react to closes (mean reversion and RSI-family signals lose the most).

**Lot Tracking and Cost Basis** -- The engine tracks individual purchase lots as a `list[PositionLot]` per ticker. Every buy fill appends a new lot at the execution price; every sell consumes lots FIFO (first-in, first-out). Exit rules evaluate at the aggregate level — they see a share-weighted average cost basis and a per-ticker high-water mark, not individual lots. FIFO execution is the standard US broker default and the method used by every major backtesting framework (LEAN, Zipline, Backtrader). See [Lot Tracking](#lot-tracking) for details. Trades are classified as short-term (held less than 365 days) or long-term for tax awareness.

> **Note on initial cost basis.** The backtest seeds each starting position's cost basis from the *start-day market price*, not the YAML `cost_basis`. The YAML value is the user's real purchase basis (used by the live engine and for display), but using it inside a backtest would let exit rules fire on pre-window gains, distorting strategy performance.

**Train/Test Split** -- By default, the backtest splits the date range 70/30 into training and test periods. Returns are reported separately for each. The optimizer uses train return as its objective and test return to measure how well the parameters generalize to unseen data.

**Time-Weighted Return** -- TWR accounts for external cash infusions (e.g., biweekly contributions) by breaking the simulation into sub-periods at each infusion point and compounding the sub-period returns. This gives an accurate measure of strategy performance independent of when cash enters the portfolio.

**Deferred Holdings** -- If a portfolio ticker has no price data at the backtest start date (e.g., the company IPO'd mid-backtest), the position is deferred and automatically activated when data becomes available.

See [CLI Reference](cli.md#backtest) for all backtest options.

### Live

After optimizing and backtesting, live mode puts the strategy to work on real-time market data. It polls current prices and tells you what trades to make right now based on your portfolio's actual holdings.

The live engine polls real-time prices on a configurable interval (default 60 seconds) and emits order alerts for manual execution. On each tick, it fetches the last 120 days of price history, runs the full allocation and exit-rule pipeline, and compares the resulting order set to the previous tick. If nothing changed, it stays quiet. If new orders appear or existing ones change, it emits an alert with the ticker, price, reason, source strategy, and suggested share count.

The live engine carries persistent runtime state across ticks and runs via a YAML sidecar (see [Live State Persistence](#live-state-persistence) below). Exit rules see a share-weighted cost basis from the live lot list and a per-ticker high-water mark tracked as the running peak of price observed since seed. This matches backtest semantics: `TrailingStop` fires on real drawdown from a tracked peak, the CPPI overlay scales budget against tracked peak equity, and FIFO sell consumption preserves short-term/long-term holding-period classification. The live engine does not execute trades; it's designed for operators who execute manually through their broker, and it assumes every emitted alert is filled at the alert price when it updates state.

See [CLI Reference](cli.md#live) for all live options.

### Live State Persistence

The live engine persists runtime state to a YAML sidecar (`<portfolio>.state.yaml` by default, alongside `portfolio.yaml`, or the path under the optional top-level `state_file:` field in `portfolio.yaml`). The schema is owned by `src/midas/live_state.py` and documented in [the design spec](specs/2026-05-07-live-per-lot-tracking-design.md).

After first seed, `portfolio.yaml` is read-only seed config. The state file owns positions (per-lot), available cash, per-ticker high-water marks, peak equity, and the cash-infusion `next_date`. Edits to `portfolio.yaml`'s aggregate `shares`, `cost_basis`, or `available_cash` after seed have no effect; the engine warns on drift but trusts the state file.

The engine writes the state file atomically (tempfile + `os.replace`) at the end of every tick, on the assumption that emitted alerts are filled at the alert price. Operators who need to reflect slippage or manual overrides can hand-edit the state file — it is plain YAML.

## Lot Tracking

Midas tracks individual purchase lots per ticker for FIFO execution and cost-basis accounting. This matches the approach used by LEAN, Zipline, and Backtrader.

Exit rules evaluate at the **aggregate position level** — they see a share-weighted average cost basis and a single high-water mark per ticker, not individual lots. This is the industry standard: LEAN's `RiskManagementModel` operates on aggregate positions, and per-lot evaluation adds complexity without real-world value (the same exit signal that fires on lot B would fire on the aggregate position anyway).

**Execution is FIFO.** When the backtest engine executes a sell order, it consumes lots first-in-first-out. This is the US broker default for equities (IRS default tax-lot identification method) and the approach used universally across backtesting frameworks. Trades are classified as short-term (held less than 365 days) or long-term based on the FIFO lot's purchase date.

A future extension could add configurable lot-consumption methods (LIFO, highest-cost, specific-ID) at the execution layer, but FIFO is the correct default.
