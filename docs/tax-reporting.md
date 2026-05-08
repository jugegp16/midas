# Tax reporting

Midas can report realized capital gains in a Schedule D-shaped format and (in
backtest mode only) compute after-tax return metrics. Tax accounting is opt-in:
without a `tax:` block in the strategies YAML, all output is unchanged from
pre-#66 behavior.

## Configuring rates

Add a `tax:` block to the strategies YAML alongside the existing top-level
keys (`min_cash_pct`, `softmax_temperature`, optional `risk:`):

```yaml
tax:
  short_term_rate: 0.37   # decimal fraction; default top federal bracket
  long_term_rate: 0.20    # decimal fraction; default top federal LTCG
  deductible_loss_cap: 3000.0
  payment_lag_days: 105   # Dec 31 → ~Apr 15 of following year
```

All four fields have defaults; the block is parsed leniently. The validator
also enforces `long_term_rate <= short_term_rate` to catch transposed values.

## Backtest: after-tax metrics

Running `midas backtest` with a `tax:` block in the strategies YAML adds:

- `after_tax_final_value`, `after_tax_total_return`, `after_tax_cagr`,
  `after_tax_twr` to `summary.json`.
- A `nav_after_tax` column in `equity_curve.csv` parallel to `nav`.
- A `tax_summary` array in `summary.json` with one entry per calendar year
  containing realized sales (`st_realized`, `lt_realized`, `net_after_cross`,
  `deductible_loss`, `carry_forward`, `tax_owed`, `payment_date`).
- A `tax_cost_ratio` field equal to `(cagr - after_tax_cagr) / cagr` (or
  `null` when CAGR is non-positive — taxes can't reduce a losing run's
  return measurably and the ratio is undefined).

The summary terminal output gains an "After-Tax Performance" block and a
per-year "Realized Tax" table. The equity-curve chart overlays gross and
after-tax curves in cyan and magenta.

Tax owed for year Y is deducted from the after-tax curve at
`Dec 31 of Y + payment_lag_days` (default Apr 15 of Y+1). If the payment
date falls past the end of the backtest, it's clamped to the final bar
— a known overstatement of tax drag in the run-end year.

## Live: trade log

Each tick that produces fills appends rows to `<state>.trades.csv` next to
the state file. Format matches `<output>/trades.csv` from `midas backtest`,
plus a `purchase_date` column added in both modes:

| Column | Notes |
|---|---|
| `date` | Fill date. |
| `ticker` | Symbol. |
| `direction` | `BUY` or `SELL`. |
| `shares`, `price` | Numeric. |
| `strategy` | Source signal name. |
| `holding_period` | `short-term` / `long-term` / empty for buys. |
| `purchase_date` | Buy date for BUYs. SELL bucket: a single date when all consumed lots share one, the literal string `various` for mixed-lot buckets, or empty when the lot's date was unknown. |
| `cost_basis` | Share-weighted bucket basis on SELL bucket rows. |
| `realized_pnl` | `(price - cost_basis) * shares` on SELL rows. |
| `return_pct` | `(price - cost_basis) / cost_basis` on SELL rows. |

The log is append-only. Hand-edits are an explicit escape valve (correcting
a recorded price after broker confirms a different fill). The reader is
strict on schema (header drift or partial rows raise `TradeLogError` with a
line number) and lenient on content.

## `midas tax-report`

```
midas tax-report --strategies strategies.yaml \
                 --portfolio portfolio.yaml --year 2026 \
                 [--output schedule_d_2026.csv]
```

Or against an explicit log path (e.g. backtest output):

```
midas tax-report --strategies strategies.yaml \
                 --from-trades output/trades.csv --year 2026
```

Prints a per-row table (ticker, shares, purchase/sale dates, basis,
proceeds, P&L, days held, classification) and writes the same data plus a
per-year aggregate footer to `--output`.

`--start` / `--end` accept arbitrary date ranges instead of `--year`.

## Caveats

- **No wash-sale detection.** Broker 1099-B handles it; we don't replicate
  broker-specific adjustments.
- **No tax-aware allocation.** The allocator is and stays tax-blind. See
  issue #69 for the in-progress follow-up that adds an after-tax objective
  to the optimizer.
- **FIFO only.** Specific-lot or LIFO accounting is out of scope.
- **Carryforward loses ST/LT character on rollover.** IRS preserves
  character; we use a single signed scalar. For most operators the
  difference is small (carryforward rate is the higher of ST/LT in the
  year it's used).
- **Pre-existing live deployments upgrading mid-year:** the trade log
  starts fresh on the first post-upgrade tick. Year-end report for the
  upgrade year is partial; merge with broker statements.
