# Tax-shaped trade log + realized-P&L report (issue #66)

**Status:** Design
**Date:** 2026-05-08
**Issue:** [#66](https://github.com/jugegp16/Midas/issues/66)
**Predecessor:** PR #67 (live per-lot tracking, #36)

## Problem

Live mode (after #67) keeps per-lot positions in `<portfolio>.state.yaml` but
discards trade history: `apply_sell` returns `(st_realized_pnl, lt_realized_pnl)`
per fill and the engine throws it away. After a year of `midas live`, the
operator has current state but no record of fills — nothing for Schedule D.

Backtest already has the data: `BacktestResult.trades` is the authoritative
list, and `_write_trades_csv` already emits a CSV with `holding_period`,
`cost_basis`, and `realized_pnl` columns. What's missing is a `purchase_date`
column and a year-end aggregator that produces a Schedule D-shaped report.

The operator also wants to see how capital-gains tax drag affects backtested
returns: an after-tax equity curve, after-tax CAGR/TWR, and a tax-cost ratio.

## Scope

In:

1. Live mode writes an append-only `<state_path>.trades.csv` with the same shape
   as backtest's existing `trades.csv` (plus a new `purchase_date` column added
   to both modes).
2. New `midas tax-report --year YYYY -p portfolio.yaml` subcommand consumes the
   trade log and emits a Schedule D-style printed table + CSV.
3. Backtest gains optional after-tax accounting controlled by an opt-in
   `TaxConfig` block in the strategies YAML: after-tax fields on
   `BacktestResult`, after-tax equity curve column in `equity_curve.csv`, chart
   overlay, and a tax-cost-ratio metric in `summary.json`.

Out (deferred to follow-ups):

- Tax-aware optimizer objective (`--objective gross|after_tax`) — separate issue.
- Wash-sale detection — broker 1099-B handles it, replication is broker-specific.
- Tax-aware allocation at runtime — explicit guardrail in #66; allocator stays
  tax-blind.
- LIFO / specific-lot accounting — engine is FIFO-only.
- Per-strategy after-tax breakdown in `strategy_breakdown.csv` — tax netting
  doesn't decompose cleanly across strategies; defer.

## Decisions

### Trade log shape

- One row per BUY fill.
- One row per non-empty ST/LT *bucket* on each SELL fill (mixed-lot sells emit
  up to two rows). This matches backtest's existing `TradeRecord` semantics; we
  do not split into per-lot-slice rows.
- `purchase_date` column is new. For BUYs it is the fill date. For SELL bucket
  rows, it is the consumed lots' single shared date when all lots in the bucket
  share one date, otherwise the literal string `various` (Schedule D convention).
  Empty when an unseeded live lot has `purchase_date=None`.

### Trade log location

- Sibling of state file: `state_path.with_suffix(state_path.suffix + ".trades.csv")`,
  e.g. `portfolio.state.yaml` → `portfolio.state.yaml.trades.csv`. Inherits the
  engine's existing `flock` on the state file; no new locking primitive.
- Backtest continues to write `<output_dir>/trades.csv` via `write_backtest_results`.
  Same column shape as live; `tax-report --from-trades <path>` can consume either.

### Tax configuration

`TaxConfig` is opt-in via the strategies YAML, alongside `RiskConfig`:

```yaml
tax:
  short_term_rate: 0.37
  long_term_rate: 0.20
  deductible_loss_cap: 3000.0
  payment_lag_days: 105   # Dec 31 → ~Apr 15 of following year
```

Defaults match top federal brackets. Omitting the block disables tax accounting
entirely (after-tax fields stay `None`, no extra CSV column, no chart overlay).

### Tax timing & netting

- Tax owed for calendar year Y is deducted from the after-tax equity curve at
  `Dec 31 of Y + payment_lag_days`. If that date falls past `end_d`, deduct on
  the final bar instead — known overstatement of tax drag, documented.
- Annual netting follows IRS Schedule D math:
  1. Net within ST bucket; net within LT bucket.
  2. If signs differ, cross-net buckets.
  3. If net is a loss, deduct up to `deductible_loss_cap` (default $3,000)
     against ordinary income — credited at `short_term_rate`.
  4. Remainder carries forward.
- Carryforward is a single signed scalar per portfolio threaded chronologically;
  it loses ST/LT character on rollover (simplification — IRS preserves character,
  but this is a reporting-layer approximation, documented).

### `apply_sell` signature

Currently returns `(st_pnl: float, lt_pnl: float)`. Changes to return the
already-existing `SellBreakdown` dataclass, with two new fields added:

```python
@dataclass(frozen=True)
class SellBreakdown:
    st_shares: float
    st_basis: float
    st_weighted: float
    lt_shares: float
    lt_basis: float
    lt_weighted: float
    st_purchase_dates: tuple[date | None, ...]   # new
    lt_purchase_dates: tuple[date | None, ...]   # new
```

Strict superset of today's return; one internal caller in `live.py` and the
test suite update.

## Architecture

### New modules

**`src/midas/trade_log.py`** — append-only writer/reader.

```python
def append_trade(
    path: Path,
    record: TradeRecord,
    cost_basis: float | None,
    purchase_date: date | str | None,
) -> None: ...

def read_trades(path: Path) -> list[LoggedTrade]: ...

class TradeLogError(ValueError): ...
```

`LoggedTrade` is a small dataclass mirroring the CSV columns (date, ticker,
direction, shares, price, strategy, holding_period, purchase_date, cost_basis,
realized_pnl, return_pct).

`append_trade` creates the file with header on first write, appends rows only
afterwards. Header drift on read raises `TradeLogError` naming the divergence.

**`src/midas/tax.py`** — pure tax math.

```python
@dataclass(frozen=True)
class TaxConfig:
    short_term_rate: float = 0.37
    long_term_rate: float = 0.20
    deductible_loss_cap: float = 3000.0
    payment_lag_days: int = 105

@dataclass(frozen=True)
class AnnualTaxSummary:
    year: int
    st_realized: float
    lt_realized: float
    net_after_cross: float
    deductible_loss: float
    carry_forward: float
    tax_owed: float
    payment_date: date

def compute_tax_summary(
    trades: Sequence[TradeRecord],
    basis_per_sell: Sequence[float],
    config: TaxConfig,
    end_date: date,
) -> list[AnnualTaxSummary]: ...

def compute_after_tax_curve(
    equity_curve: Sequence[tuple[date, float]],
    summaries: Sequence[AnnualTaxSummary],
    end_date: date,
) -> list[tuple[date, float]]: ...
```

All math here. Reused by `BacktestResult` construction and by the `tax-report`
subcommand so backtest after-tax numbers and tax-report numbers are bit-identical
when fed identical trade rows.

### Extended types

**`models.TaxConfig`** — frozen dataclass added alongside `RiskConfig`. Loaded
from the optional `tax:` block in the strategies YAML.

**`models.TradeRecord.purchase_date`** — new field, type `date | str | None`.
`'various'` is permitted as a string sentinel for mixed-lot bucket sells.

**`live_state.SellBreakdown`** — gains `st_purchase_dates` and
`lt_purchase_dates` tuples. `consume_lots_fifo` populates them as it iterates
the lot list.

**`live_state.apply_sell`** — return type changes from `tuple[float, float]` to
`SellBreakdown`. Existing pnl is recoverable as
`breakdown.st_shares * price - breakdown.st_weighted` (and analogous for LT),
matching today's math exactly.

**`results.BacktestResult`** — new fields, all `None`/empty when
`TaxConfig` is not set:

```python
after_tax_final_value: float | None = None
after_tax_total_return: float | None = None
after_tax_cagr: float | None = None
after_tax_twr: float | None = None
after_tax_equity_curve: list[tuple[date, float]] = field(default_factory=list)
tax_cost_ratio: float | None = None
tax_summary: list[AnnualTaxSummary] = field(default_factory=list)
```

### CLI

```
midas tax-report --year YYYY -p <portfolio.yaml> [--output <csv-path>] [--from-trades <path>]
```

- Resolves trade-log path the same way `midas live` resolves the state path,
  then suffixes `.trades.csv`. `--from-trades` overrides for backtest output
  paths or arbitrary log files.
- Prints a Schedule D-style table to stdout: per-row (ticker, shares,
  purchase_date, sale_date, basis, proceeds, pnl, holding_period_days,
  classification) + footer (ST total, LT total, net, deductible, tax_owed at
  configured rates).
- Writes a CSV at `--output` (default `schedule_d_<year>.csv`) with the same rows.
- `--start` / `--end` flags accepted as an alternative to `--year` for
  arbitrary date ranges.

## Data flow

### Live tick

1. `_tick` runs as today through fill emission.
2. For each fill: `apply_buy` mutates state OR `apply_sell` returns `SellBreakdown`.
3. **Order: append to log, then `save_atomic` state, then print alerts.**
   If the log append fails, state is at its pre-fill snapshot and the operator
   sees the error; re-running re-applies fills and re-attempts the append. State
   and log can never silently disagree about whether a fill happened.
4. Append uses the existing engine-wide `flock`; no new lock.

### Backtest result construction

1. `state.trades` and `state.basis_per_sell` accumulate as today, plus
   `purchase_date` populated per record (single date, `'various'`, or `None`).
2. `_write_trades_csv` emits the new `purchase_date` column.
3. If `TaxConfig` is set:
   - `compute_tax_summary(state.trades, state.basis_per_sell, tax_config, end_d)`
     groups SELL records by year, runs IRS netting + carryforward.
   - `compute_after_tax_curve(equity_curve, summaries, end_d)` deducts each
     year's `tax_owed` at its `payment_date` (clamped to `end_d`).
   - `after_tax_*` metrics derived via the same metrics functions used for
     gross.
   - `tax_cost_ratio = (cagr - after_tax_cagr) / cagr` if `cagr > 0` else `0.0`.
4. `equity_curve.csv` gains a parallel `nav_after_tax` column.
5. `summary.json` gains `after_tax_*`, `tax_cost_ratio`, and a `tax_summary`
   array (one entry per year).
6. `print_backtest_summary` prints an after-tax block under the gross block;
   chart overlays both curves.

### `tax-report`

1. Resolve trade-log path; read via `trade_log.read_trades`.
2. Filter SELL rows to the requested period.
3. Run `compute_tax_summary` (same call backtest uses).
4. Render console table + write CSV.

## Edge cases

| Case | Behavior |
|---|---|
| `TaxConfig=None` | All after-tax fields stay `None`; `equity_curve.csv` omits the new column; chart skips overlay; no behavior change vs. today. |
| Backtest ends mid-year | Year's `tax_owed` deducted at `end_d`. `tax_summary[-1].payment_date` records actual Apr 15 for honest reporting. |
| Empty trades | `compute_tax_summary` returns `[]`. After-tax curve == gross curve. `tax_cost_ratio=0.0`. |
| `tax-report` with no SELLs in range | Print "No realized sales in `<year>`"; emit header-only CSV; exit 0. |
| Pre-existing live deployments upgrading mid-year | First tick after upgrade creates fresh trade log. Pre-upgrade fills are gone (live didn't record them). Year-end report for upgrade year is partial; operator merges with broker statement. |
| Hand-edited log | Strict CSV parser. Schema mismatch → `TradeLogError` naming the row. Content (negative shares, future dates, etc.) flows through. |
| Partial-write log corruption | Strict reader raises with offending line number. No auto-recovery; operator restores from git or hand-fixes. |
| Header drift on existing log | Refuses to start at engine init with the divergence printed. Operator renames or hand-fixes; we don't auto-migrate. |
| Mixed-lot SELL bucket | `purchase_date='various'`; `cost_basis` is the share-weighted bucket basis (already true today). |
| `purchase_date=None` (unseeded lot) | Empty string in CSV. Counts as ST in classification per existing `consume_lots_fifo` semantics. |

## Testing

**`tests/test_trade_log.py` (new)**
- Round-trip BUYs + SELLs (mixed bucket cases).
- Header creation on first write, append on subsequent.
- Header drift raises with row 0 named.
- `'various'` and empty `purchase_date` round-trip correctly.
- Partial-row corruption → strict reader raises with line number.

**`tests/test_tax.py` (new)**
- Pure netting:
  - All ST gain.
  - ST gain + LT loss → cross-bucket netting.
  - Net loss `< $3K` → full deductible at ST rate.
  - Net loss `> $3K` → cap deductible, carry remainder.
  - Multi-year carryforward absorbing future gains.
- `compute_after_tax_curve` with payment-date scenarios (lag inside range, lag past `end_d`).
- `tax_cost_ratio` math: `cagr > 0` and `cagr == 0` paths.
- Empty trades → empty summary, after-tax == gross.

**`tests/test_live_trade_log.py` (new) — integration**
- 3-tick run with planned fills (BUY, mixed-lot SELL, full-exit SELL).
- Assert row count, `purchase_date` per row, header presence.
- Kill mid-tick → state at last good snapshot; log never half-written.

**Additions**
- `tests/test_backtest.py`: extend trades.csv parity test for the new column;
  2-year backtest asserts `after_tax_*` populated with `TaxConfig` and `None` without.
- `tests/test_live_state.py`: update `apply_sell` tests for the new return shape;
  cover `st_purchase_dates`/`lt_purchase_dates` aggregation.
- `tests/test_live_engine.py`: assert `<state>.trades.csv` is created and grows.
- `tests/test_live_backtest_parity.py`: assert trade-log CSVs from a backtest
  run and equivalent live replay are byte-identical (strong guarantee tax math
  cannot diverge between modes).
- `tests/test_cli_live.py`: smoke test `tax-report` against a fixture log;
  assert footer numbers and exit code.
- `tests/test_charts.py`: snapshot test the after-tax overlay.

`uv run mypy src` and `uv run ruff check .` clean across new modules.

## Implementation order

A natural build order falls out of the dependency graph:

1. `models.TaxConfig` + strategies-loader wiring (no behavior change yet).
2. `models.TradeRecord.purchase_date` + `live_state.SellBreakdown` field
   additions + `consume_lots_fifo` purchase-date tracking.
3. `live_state.apply_sell` return-type change + caller updates.
4. `src/midas/trade_log.py` + tests.
5. Backtest `trades.csv` `purchase_date` column + parity test extension.
6. Live engine wiring: append fills to `<state>.trades.csv`.
7. `src/midas/tax.py` + tests.
8. `BacktestResult` after-tax fields, `equity_curve.csv` column, `summary.json`
   additions, summary print block, chart overlay.
9. `midas tax-report` CLI subcommand.
10. Documentation: brief mention in `docs/architecture.md`; a new
    `docs/tax-reporting.md` covering the YAML block, CLI, and Schedule D
    interpretation caveats.
