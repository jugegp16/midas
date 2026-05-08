# Strategies

Every midas strategy is one of two disjoint types:

- **EntrySignal** — receives a price history array and returns a bullish score in `[0, 1]` (or `None` to abstain). Pure buy-side: a score of 0 means "no opinion", not "sell". Multiple entry signals are blended into a single conviction per ticker, then turned into a target weight by the allocator's softmax.

- **ExitRule** — a downstream override layer that clamps the allocator's proposed target weights downward. Each rule's `clamp_target(ticker, proposed_target, price_history, cost_basis, high_water_mark)` returns an adjusted target weight ≤ the proposed target. Returning 0.0 means "full liquidation." Exit rules run *outside* the allocator and never participate in target-weight construction or softmax. Sells arise from negative deltas between clamped targets and current weights.

The two tiers exist as separate base classes (`midas.strategies.base.EntrySignal` and `midas.strategies.base.ExitRule`). There is no shared parent and no third tier. Entry-signal logic cannot accidentally produce a sell, and exit-rule logic cannot accidentally inflate a buy. See [Architecture](architecture.md#the-two-tier-model) for the design rationale and the comparison to LEAN's AlphaModel/PortfolioConstructionModel/RiskManagementModel split.

To add a new strategy: implement `EntrySignal` or `ExitRule` in a new file under `strategies/`, register it in `strategies/__init__.py`, and optionally add a search range in `PARAM_RANGES` in `optimizer.py` to make it optimizable. Entry-signal entries get a `weight` field; exit rules don't (they fire on their own conditions, not as a contributor to a blend).

## Summary

| Strategy | Type | What it does |
|----------|------|--------------|
| BollingerBand | Entry | Bullish at the lower volatility band of the moving average |
| DonchianBreakout | Entry | Bullish when price breaks above the rolling N-bar high (Turtle-style) |
| GapDownRecovery | Entry | Bullish after a gap-down event starts recovering |
| KeltnerChannel | Entry | Bullish on a breakout above the SMA + k x ATR upper band |
| MACDCrossover | Entry | Bullish when the MACD line is above its signal line |
| MeanReversion | Entry | Bullish when price drops below its moving average |
| Momentum | Entry | Bullish when price is above its moving average |
| MovingAverageCrossover | Entry | Bullish on the golden cross (short MA above long MA) |
| RSIOversold | Entry | Bullish when RSI dips below 50 (oversold conditions) |
| VWAPReversion | Entry | Bullish when price is below the average price (VWAP proxy) |
| ChandelierStop | Exit | Clamps target to 0 when price falls k x ATR below the rolling N-bar high |
| MACDExit | Exit | Clamps target to 0 on a bearish MACD crossover |
| MovingAverageCrossoverExit | Exit | Clamps target to 0 on the death cross |
| ParabolicSARExit | Exit | Clamps target to 0 when Wilder's Parabolic SAR flips above price |
| ProfitTaking | Exit | Clamps target to 0 when unrealized gain exceeds the threshold |
| StopLoss | Exit | Clamps target to 0 when unrealized loss exceeds the threshold |
| TrailingStop | Exit | Clamps target to 0 after a drawdown from the high-water mark |

## Composing Strategy Files

A strategy file needs at least one entry signal and at least one exit rule. Without entries, no buys are ever generated. Without exits, the engine accumulates positions indefinitely — there is no fallback "automatic" exit and no veto-style escape hatch.

**Principles for picking strategies:**

- **Pair entries with exits.** Every entry signal needs at least one exit rule to close the position it opens. Pick exits that match the entry's thesis: `MovingAverageCrossover` pairs naturally with `MovingAverageCrossoverExit` (same indicator, symmetric trigger), and dip-buying entries (`BollingerBand`, `RSIOversold`, `MeanReversion`) pair naturally with `StopLoss` to floor the damage when "cheap" gets cheaper.

- **Mix signal types.** Strategies that measure the same thing (e.g. `MeanReversion` and `VWAPReversion` both compare price to a moving average) provide redundant signals. Strategies that measure different things (e.g. `BollingerBand` measures distance from the MA in *volatility units*; `RSIOversold` measures the up-day/down-day ratio) provide independent confirmation. When two independent signals agree, the conviction is more trustworthy.

- **Match your market thesis.** Trend-following entries (`Momentum`, `MovingAverageCrossover`, `MACDCrossover`) and mean-reversion entries (`MeanReversion`, `BollingerBand`, `VWAPReversion`) have opposing views of how markets work. Using both isn't wrong — their contributions partially cancel, producing more moderate positions — but be intentional about it.

- **Stack exits if the entry is risky.** A `StopLoss` floor and a `TrailingStop` ratchet are not redundant: stop loss caps absolute downside from cost basis, trailing stop protects accumulated gains from a drawdown. Adding `ProfitTaking` on top gives you a third independent exit at a fixed gain target.

**Pre-built examples** in `example-strategies/` demonstrate these principles:

- **[Trend-Following](../example-strategies/trend-following.yaml)** — Two entry signals (`MovingAverageCrossover` + `MACDCrossover`) confirmed across timescales, paired with their symmetric exits (`MovingAverageCrossoverExit` + `MACDExit`) and a `ProfitTaking` target.
- **[Dip-Buying](../example-strategies/dip-buying.yaml)** — `BollingerBand` and `RSIOversold` for independent confirmation that an asset is oversold, protected by a `StopLoss` floor.
- **[Balanced Growth](../example-strategies/balanced-growth.yaml)** — `Momentum` and `MeanReversion` give entries in both trending and recovering markets; `ProfitTaking` harvests gains at a fixed target and `ChandelierStop` provides a volatility-adjusted trailing stop that works whether the position is in profit or underwater.

## Risk Discipline

Optional risk policy lives under a top-level `risk:` block in the strategies YAML. Omit it for current behavior bit-for-bit. All vol quantities are **annualized**.

```yaml
risk:
  weighting: inverse_vol      # equal | inverse_vol; default equal
  vol_lookback_days: 60       # rolling window for vol and covariance estimates

  vol_target: 0.20            # annualized; null/omit disables Phase 4b

  drawdown_penalty: 1.5       # exposure = max(1 - penalty * dd, floor)
  drawdown_floor: 0.5         # both required, both must be set or both omitted
```

The optimizer **does not** search risk knobs — risk is policy, easy to overfit, and changing it is a deliberate user act. To experiment, edit the YAML and rerun. For A/B comparisons keep two strategies files in version control.

CPPI (`drawdown_penalty` / `drawdown_floor`) is supported in `live` mode — peak equity persists across runs through the live state sidecar (see [Architecture: Live State Persistence](architecture.md#live-state-persistence)).

See [Architecture: Risk-aware allocator phases](architecture.md#phase-4a-cppi-drawdown-overlay-optional) for the phase-by-phase behavior.

## Entry Signals

Entry signals score a ticker's bullishness in `[0, 1]`. They contribute to the allocator's per-ticker blend via a configurable `weight` (default 1.0). A signal returning `None` is excluded from the blend entirely — it doesn't pull the average toward zero, it simply doesn't participate. A signal returning 0 means "no opinion" — the ticker is treated as held at its current weight rather than as an active buy candidate.

---

### MeanReversion

**Type**: Entry signal

Bullish when price drops below its moving average, expecting it to revert back up. The score ramps linearly with distance below the MA, reaching full conviction at the `threshold` percentage. This is a contrarian entry — it bets against recent price movement.

| Param | Default | Description |
|-------|---------|-------------|
| `window` | 30 | Moving average lookback period (trading days) |
| `threshold` | 0.10 | Distance below the MA at which the score reaches 1.0 |

**Suited for**: Broad market ETFs, large caps

**Interactions**: Natural counterweight to `Momentum` — `MeanReversion` buys dips while `Momentum` buys strength, so using both produces more moderate positions. Overlaps significantly with `VWAPReversion` and `BollingerBand` since all three compare price to a moving average; using `MeanReversion` alongside either provides redundant rather than independent signals. Pairs cleanly with `RSIOversold`, which measures a different dimension (up-day/down-day ratio vs. price distance from average). Always pair with `StopLoss` — dip-buying without a floor catches falling knives.

---

### Momentum

**Type**: Entry signal

The opposite of mean reversion — bullish when price is above its moving average, riding the trend. The score scales with how far above the MA the price has moved.

| Param | Default | Description |
|-------|---------|-------------|
| `window` | 20 | Moving average lookback period (trading days) |
| `momentum_scale` | 0.05 | Distance above the MA at which the score reaches 1.0 |

**Suited for**: All asset classes

**Interactions**: Aligns with `MovingAverageCrossover` and `MACDCrossover`, which are also trend-following but measure trend on different timescales. Conflicts with `MeanReversion` since they have opposing theses. Pairs naturally with `MovingAverageCrossoverExit` or `MACDExit` as the symmetric exit when the trend reverses.

---

### RSIOversold

**Type**: Entry signal

Uses the Relative Strength Index to detect oversold conditions. The score increases as RSI drops toward the oversold threshold. Neutral when RSI is at or above 50.

| Param | Default | Description |
|-------|---------|-------------|
| `window` | 14 | RSI calculation period (trading days) |
| `oversold_threshold` | 30.0 | RSI level at which the score reaches 1.0 |

**Suited for**: All asset classes

**Interactions**: Provides independent confirmation alongside `BollingerBand` since they measure different things (up-day/down-day ratio vs. volatility-adjusted distance from MA). Good complement to `Momentum` since they buy in different scenarios (trending vs. oversold). Always pair with `StopLoss`.

---

### BollingerBand

**Type**: Entry signal

A volatility-aware mean reversion entry. Computes how many standard deviations the current price is below its moving average and maps that to a conviction score. When price touches or pierces the lower band, the strategy is bullish — it expects a bounce. The `num_std` parameter controls the band width; at `-num_std` standard deviations, the score reaches 1.0.

Unlike plain `MeanReversion`, `BollingerBand` adapts to the stock's recent volatility. In calm markets the bands are narrow and small moves trigger conviction; in volatile markets the bands widen and it takes a larger move to reach the same score.

| Param | Default | Description |
|-------|---------|-------------|
| `window` | 20 | Moving average and standard deviation lookback (trading days) |
| `num_std` | 2.0 | Number of standard deviations defining the band width |

**Suited for**: Broad market ETFs, large caps

**Interactions**: Overlaps with `MeanReversion` and `VWAPReversion` (all are MA-based mean reversion) — prefer one of the three rather than stacking them. Provides strong independent confirmation when paired with `RSIOversold`. Conflicts with trend-following entries.

---

### KeltnerChannel

**Type**: Entry signal

A volatility-adjusted breakout entry. Computes an SMA centerline and an ATR-based band half-width, then fires when the current close breaks above the upper band (centerline + `multiplier` x ATR). The score is measured in ATR units of excess above the band: 1 ATR above the band reaches full conviction. ATR is approximated from close-to-close absolute differences since the engine's price history is close-only.

Unlike `BollingerBand` — which is a *mean-reversion* entry that fires at the *lower* band — `KeltnerChannel` is trend-following and fires at the *upper* band. Their signals are symmetric but opposite: `BollingerBand` buys the dip to the bottom band, `KeltnerChannel` buys the breakout above the top band. Keltner also uses ATR rather than standard deviation, which makes the bands smoother and less sensitive to single outliers.

| Param | Default | Description |
|-------|---------|-------------|
| `window` | 20 | SMA and ATR lookback (trading days) |
| `multiplier` | 2.0 | ATR multiple defining the upper band distance above the centerline |

**Suited for**: All asset classes, especially trending markets

**Interactions**: Aligns with `Momentum`, `MovingAverageCrossover`, `MACDCrossover`, and `DonchianBreakout` (all trend-following). Conflicts with `BollingerBand` and other mean-reversion entries — they'd fight each other. Pairs naturally with `ChandelierStop`, which uses the same ATR-based volatility framing for its exit distance.

---

### MACDCrossover

**Type**: Entry signal

A trend-following entry based on the convergence and divergence of two exponential moving averages. The MACD line (fast EMA minus slow EMA) measures the trend's strength and direction; the signal line (an EMA of the MACD line itself) smooths out noise. Bullish when MACD is above the signal line.

The raw difference between MACD and signal is normalized by the current price so the score is comparable across tickers at different price levels.

| Param | Default | Description |
|-------|---------|-------------|
| `fast_period` | 12 | Fast EMA period |
| `slow_period` | 26 | Slow EMA period |
| `signal_period` | 9 | Signal line EMA period |

**Suited for**: All asset classes

**Interactions**: Confirms `MovingAverageCrossover` on a different timescale. Pairs naturally with its symmetric exit `MACDExit` so the position closes on the same indicator that opened it.

---

### VWAPReversion

**Type**: Entry signal

Mean reversion around the average price over a lookback window. Bullish when price is below the average. Functionally similar to `MeanReversion` but uses a different threshold scale. Currently uses a simple moving average as a VWAP proxy since volume data is not available from the data provider.

| Param | Default | Description |
|-------|---------|-------------|
| `window` | 20 | Average price lookback (trading days) |
| `threshold` | 0.02 | Distance below the average at which the score reaches 1.0 |

**Suited for**: Large caps, broad market ETFs

**Interactions**: Highly redundant with `MeanReversion` and `BollingerBand` — choose one rather than stacking. The tighter default threshold (2% vs. 10% for `MeanReversion`) makes it more sensitive.

---

### MovingAverageCrossover

**Type**: Entry signal

The classic golden cross strategy. Tracks two moving averages of different lengths. When the short-term MA crosses above the long-term MA (golden cross), the score goes bullish and scales continuously with the spread between the two averages.

| Param | Default | Description |
|-------|---------|-------------|
| `short_window` | 20 | Short-term moving average period (trading days) |
| `long_window` | 50 | Long-term moving average period (trading days) |
| `spread_scale` | 0.05 | Spread between MAs at which the score reaches 1.0 |

**Suited for**: All asset classes

**Interactions**: Confirms `MACDCrossover`. Aligns with `Momentum`. Pairs naturally with `MovingAverageCrossoverExit` so entries and exits use the same indicator on the same timescale.

---

### DonchianBreakout

**Type**: Entry signal

The classic Turtle Trading entry. Bullish when the current close exceeds the highest close over the prior `window` bars — a strict breakout. Once the breakout fires, the score ramps linearly with the excess over the prior high, reaching full conviction at `breakout_scale` (default 2%). Before the breakout the score is 0; no partial credit for approaching the level.

Unlike MA-based trend entries (`Momentum`, `MovingAverageCrossover`) which score continuously along a gradient, Donchian is binary-then-linear: either you're making a new high or you're not. This makes it late to enter trends but immune to whipsaws inside the lookback range.

| Param | Default | Description |
|-------|---------|-------------|
| `window` | 20 | Lookback for the prior high (trading days). The classical Turtle values are 20 (short system) and 55 (long system) |
| `breakout_scale` | 0.02 | Excess over the prior high at which the score reaches 1.0 |

**Suited for**: All asset classes, especially trending equities and commodities-like assets

**Interactions**: Aligns with `Momentum` and `MovingAverageCrossover` (all trend-following) but measures a different thing — price vs. prior high rather than price vs. moving average — so it adds independent confirmation rather than redundancy. Conflicts with mean-reversion entries. Pairs naturally with `ChandelierStop`, which also uses a rolling-window high as its reference point.

---

### GapDownRecovery

**Type**: Entry signal

A short-term opportunistic entry that looks for gap-down events followed by recovery. When a stock opens significantly below the previous close (a gap-down) and then starts recovering, the score goes bullish. The score reflects how much of the gap has been recovered — partial recovery produces a partial score.

| Param | Default | Description |
|-------|---------|-------------|
| `gap_threshold` | 0.03 | Minimum gap-down size (as fraction of previous close) to trigger |

**Suited for**: Individual equities, high-volatility stocks

**Interactions**: Fires rarely and on specific events, so it doesn't conflict with anything. Always pair with `StopLoss` since gap-down buying is inherently risky.

---

## Exit Rules

Exit rules act as a downstream override/veto layer (following the LEAN `RiskManagementModel` pattern). Each rule's `clamp_target(ticker, proposed_target, price_history, cost_basis, high_water_mark)` receives the allocator's proposed target weight, the position's aggregate cost basis, aggregate high-water mark, and price history. It returns an adjusted target ≤ the proposed target. Returning 0.0 triggers full liquidation.

Exit rules evaluate at the **aggregate position level** — they see a share-weighted average cost basis and a single high-water mark per ticker, not individual lots. This matches how LEAN, Zipline, and Backtrader handle exits. Per-lot logic belongs at execution time (FIFO consumption). Exit rules are applied sequentially; the first rule to clamp a ticker wins attribution for that sell.

> **Note on initial cost basis in backtests.** The backtest seeds each starting position with the *start-day market price* as cost basis, not the YAML `cost_basis` value. The YAML value is the user's real purchase basis (used by the live engine and for display) — using it inside a backtest would let exit rules fire on pre-window gains and distort strategy performance. The live engine uses the YAML basis directly.

---

### StopLoss

**Type**: Exit rule

Clamps the target weight to 0 when the position's unrealized loss (from aggregate cost basis) exceeds `loss_threshold`.

| Param | Default | Description |
|-------|---------|-------------|
| `loss_threshold` | 0.10 | Loss percentage at which a lot is liquidated |

**Suited for**: All asset classes

**Interactions**: Complements `TrailingStop` — `StopLoss` caps absolute downside from cost basis, `TrailingStop` protects accumulated gains. They don't overlap: `StopLoss` fires on losing lots, `TrailingStop` fires on profitable ones. Essential alongside any dip-buying entry (`BollingerBand`, `RSIOversold`, `MeanReversion`, `GapDownRecovery`).

---

### TrailingStop

**Type**: Exit rule

Clamps the target weight to 0 when the position's drawdown from its aggregate high-water mark exceeds `trail_pct`. Only fires when the position is in profit (current price > cost basis) — this prevents `TrailingStop` from compounding with `StopLoss` on losing positions.

| Param | Default | Description |
|-------|---------|-------------|
| `trail_pct` | 0.10 | Drawdown from the ticker's high-water mark that triggers liquidation |

**Suited for**: All asset classes

**Interactions**: Complements `StopLoss` and `ProfitTaking`. `ProfitTaking` exits at a fixed gain threshold; `TrailingStop` dynamically protects whatever gains have accumulated, even beyond that threshold. Together they create layered exits: `ProfitTaking` harvests at the target, `TrailingStop` catches sharp reversals after gains have run further.

---

### ChandelierStop

**Type**: Exit rule

A volatility-adjusted trailing stop based on Chuck LeBeau's Chandelier Exit. Clamps the target weight to 0 when the current price falls more than `multiplier` × ATR below the highest close over the most recent `window` bars. ATR is approximated from close-to-close absolute differences since the engine's price history is close-only.

Unlike `TrailingStop`, `ChandelierStop` uses a rolling window high rather than an all-time high-water-mark since entry, so its reference point breathes with the recent price range rather than being pinned to a months-old peak. It also has no `in_profit` gate — it fires on both profitable and losing positions, which makes it a drop-in replacement for `StopLoss + TrailingStop` rather than a layered partner. The stop distance scales with realized volatility, so the same `multiplier` produces a tighter stop in calm markets and a wider stop in choppy ones.

| Param | Default | Description |
|-------|---------|-------------|
| `window` | 22 | Rolling lookback for both the highest close and the ATR (trading days) |
| `multiplier` | 3.0 | ATR multiple that sets the stop distance below the rolling high |

**Suited for**: All asset classes

**Interactions**: Generally substitutes for the `StopLoss` + `TrailingStop` pair — stacking all three produces overlapping protection with unclear attribution. Still pairs cleanly with `ProfitTaking` as the gain-harvest mechanism. Useful when fixed-percent stops feel regime-ignorant, since a single `multiplier` adapts across tickers with different volatility profiles.

---

### ParabolicSARExit

**Type**: Exit rule

J. Welles Wilder's Parabolic SAR (Stop And Reverse) as a trailing-stop exit. Computes a self-accelerating trailing stop that starts slow and ratchets tighter as the trend extends: each time the price makes a new extreme in the direction of the trend, the acceleration factor (AF) is bumped up by `af_step` (capped at `af_max`), which pulls the SAR closer to price. When the SAR finally flips above price, the uptrend is considered broken and the rule clamps the target to 0.

The defining property of SAR is that it converts elapsed trend time into stop tightness. A fresh breakout gets a generous stop (small AF, SAR trails far below); a long-running uptrend gets a tight one (AF near cap, SAR hugging price). This makes it well-suited for riding long trends without giving back too much at the end — the stop tightens on its own as the move matures.

| Param | Default | Description |
|-------|---------|-------------|
| `af_start` | 0.02 | Initial acceleration factor — the fraction of the SAR-to-extreme gap closed per bar at trend start |
| `af_step` | 0.02 | Increment added to AF on each new trend extreme |
| `af_max` | 0.20 | Upper cap on AF — Wilder's original value |

**Suited for**: All asset classes, especially trending markets

**Interactions**: Overlaps functionally with `TrailingStop` and `ChandelierStop` — all three are trailing-stop exits — but uses a fundamentally different tightening rule. `TrailingStop` is a fixed percentage from the HWM; `ChandelierStop` scales with ATR; `ParabolicSARExit` scales with trend duration. Don't stack all three or they'll fight over attribution. Pairs naturally with strong trend-following entries like `DonchianBreakout`, `KeltnerChannel`, or `Momentum`, where the late-stage tightening matches the "ride the move, cut when it breaks" thesis.

---

### ProfitTaking

**Type**: Exit rule

Clamps the target weight to 0 when the position's unrealized gain (from aggregate cost basis) exceeds `gain_threshold`.

| Param | Default | Description |
|-------|---------|-------------|
| `gain_threshold` | 0.20 | Gain percentage at which a lot is liquidated |

**Suited for**: All asset classes

**Interactions**: Complements `TrailingStop` — `ProfitTaking` provides a fixed exit target while `TrailingStop` provides a dynamic one. Pairs naturally with any entry signal as the gain-harvest mechanism.

---

### MACDExit

**Type**: Exit rule

Symmetric counterpart to `MACDCrossover`. Clamps the target weight to 0 when the MACD line crosses below the signal line — the bearish crossover that mirrors the bullish entry trigger. Does not use cost basis or high-water mark.

| Param | Default | Description |
|-------|---------|-------------|
| `fast_period` | 12 | Fast EMA period |
| `slow_period` | 26 | Slow EMA period |
| `signal_period` | 9 | Signal line EMA period |

**Suited for**: All asset classes

**Interactions**: Pairs with `MACDCrossover`. Use matching parameters on both so the entry and exit react to the same crossover symmetry.

---

### MovingAverageCrossoverExit

**Type**: Exit rule

Symmetric counterpart to `MovingAverageCrossover`. Clamps the target weight to 0 on a death cross — the short-term MA crossing below the long-term MA. Does not use cost basis or high-water mark.

| Param | Default | Description |
|-------|---------|-------------|
| `short_window` | 20 | Short-term moving average period (trading days) |
| `long_window` | 50 | Long-term moving average period (trading days) |

**Suited for**: All asset classes

**Interactions**: Pairs with `MovingAverageCrossover`. Use matching `short_window`/`long_window` on both so the entry and exit react to the same crossover symmetry.
