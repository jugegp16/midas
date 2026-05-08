# Tax-Shaped Trade Log + Realized-P&L Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add (1) an append-only `<state>.trades.csv` written by `midas live`, (2) a `midas tax-report --year YYYY` subcommand that emits a Schedule D-shaped report from the trade log, and (3) opt-in after-tax accounting for backtests (after-tax CAGR/TWR, after-tax equity curve overlay, tax-cost-ratio metric) controlled by a new `tax:` block in the strategies YAML.

**Architecture:** Two new modules — `src/midas/trade_log.py` (CSV writer/reader) and `src/midas/tax.py` (pure netting + after-tax-curve math). `consume_lots_fifo` is extended to track per-lot purchase dates so trade-log SELL rows can populate a new `purchase_date` column (single date, `'various'` for mixed-lot buckets, or empty when unknown). `apply_sell` returns the full `SellBreakdown` instead of `(st_pnl, lt_pnl)` so the live engine has the inputs it needs to write bucket rows. Backtest's existing `trades.csv` gains the same `purchase_date` column so a single tax-report reader serves both modes. After-tax pipeline is opt-in via a `TaxConfig` (defaults None → all after-tax fields stay None, no behavior change).

**Tech Stack:** Python 3.14, dataclasses, PyYAML, click, plotext, pytest. No new deps.

**Spec:** `docs/specs/2026-05-08-tax-trade-log-design.md`

---

## File Structure

**Created:**
- `src/midas/trade_log.py` — append-only writer, strict reader, `TradeLogError`, `LoggedTrade` dataclass.
- `src/midas/tax.py` — `TaxConfig`, `AnnualTaxSummary`, `compute_tax_summary`, `compute_after_tax_curve`.
- `tests/test_trade_log.py` — round-trip, header creation, drift detection, `'various'` and empty `purchase_date`, partial-row corruption.
- `tests/test_tax.py` — pure netting cases, multi-year carryforward, `compute_after_tax_curve`, `tax_cost_ratio`, empty trades.
- `tests/test_live_trade_log.py` — integration: 3-tick run with planned fills writes the expected log.
- `tests/test_cli_tax_report.py` — smoke-test the new `tax-report` subcommand.
- `docs/tax-reporting.md` — operator-facing doc on the YAML block, CLI, and Schedule D caveats.

**Modified:**
- `src/midas/models.py` — add `TaxConfig`; widen `TradeRecord.purchase_date` to `date | str | None`.
- `src/midas/config.py` — parse optional `tax:` block; `load_strategies` returns `tax_config` as a 4th tuple element.
- `src/midas/live_state.py` — `SellBreakdown` gains `st_purchase_dates` / `lt_purchase_dates`; `consume_lots_fifo` populates them; `apply_sell` returns `SellBreakdown` instead of `(float, float)`.
- `src/midas/live.py` — caller of `apply_sell` updated; new fill-append step writes to `<state_path>.trades.csv` after `save_atomic`.
- `src/midas/backtest.py` — `_execute` resolves and stores `purchase_date` on each `TradeRecord`.
- `src/midas/results.py` — `_write_trades_csv` adds `purchase_date` column; `_write_equity_curve_csv` writes parallel `nav_after_tax` column when populated; `_write_summary_json` emits `after_tax_*` fields + `tax_summary` array; `BacktestResult` gains `after_tax_*` fields and `tax_summary`.
- `src/midas/output.py` — new after-tax block in `print_backtest_summary`.
- `src/midas/charts.py` — equity-curve renderer overlays `after_tax_equity_curve` when populated.
- `src/midas/cli.py` — `live` and `backtest` callers thread `tax_config` through; new `tax-report` subcommand.
- `src/midas/optimizer.py` — `write_strategies_yaml` round-trips the `tax:` block (mirrors how it round-trips `risk:`).
- `tests/test_config.py` — assert `tax_config` parsing.
- `tests/test_live_state.py` — update existing `apply_sell` tests for new return type; new tests cover `st_purchase_dates`/`lt_purchase_dates`.
- `tests/test_live_engine.py` — assert `<state>.trades.csv` is created and grows.
- `tests/test_backtest.py` — assert `purchase_date` column populated; assert after-tax fields with/without `TaxConfig`.
- `tests/test_live_backtest_parity.py` — assert trade-log CSVs are byte-identical between modes.
- `tests/test_charts.py` — snapshot the after-tax overlay.
- `docs/architecture.md` — pointer to the new tax doc.

---

## Task 1: Add `TaxConfig` and load it from the strategies YAML

**Files:**
- Modify: `src/midas/models.py`
- Modify: `src/midas/config.py`
- Modify: `src/midas/cli.py:163, 229, 352` (call sites)
- Modify: `src/midas/optimizer.py:785-840` (`write_strategies_yaml` round-trip)
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config.py`:

```python
def test_load_strategies_with_tax_block(tmp_path: Path) -> None:
    """Optional tax: block parses to a TaxConfig."""
    path = tmp_path / "strategies.yaml"
    path.write_text(
        "strategies:\n"
        "  - name: Momentum\n"
        "    params: {window: 20}\n"
        "tax:\n"
        "  short_term_rate: 0.32\n"
        "  long_term_rate: 0.15\n"
        "  deductible_loss_cap: 3000.0\n"
        "  payment_lag_days: 105\n"
    )
    _configs, _constraints, _risk, tax = load_strategies(path)
    assert tax is not None
    assert tax.short_term_rate == 0.32
    assert tax.long_term_rate == 0.15
    assert tax.deductible_loss_cap == 3000.0
    assert tax.payment_lag_days == 105


def test_load_strategies_without_tax_block(tmp_path: Path) -> None:
    """Omitting tax: yields tax_config=None — no behavior change for existing configs."""
    path = tmp_path / "strategies.yaml"
    path.write_text("strategies:\n  - name: Momentum\n    params: {window: 20}\n")
    _configs, _constraints, _risk, tax = load_strategies(path)
    assert tax is None
```

Update existing `tests/test_config.py:100-101, 147, 165` to unpack four values:

```python
configs, constraints, _risk, _tax = load_strategies(strategy_yaml)
# ...
_configs, _constraints, risk, _tax = load_strategies(path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ImportError` or `ValueError: not enough values to unpack (expected 4, got 3)`.

- [ ] **Step 3: Add `TaxConfig` to `src/midas/models.py`**

Add this dataclass alongside `RiskConfig`:

```python
@dataclass(frozen=True)
class TaxConfig:
    """Optional capital-gains-tax accounting policy.

    All rates are decimal fractions (0.37 == 37%). When None is passed in
    place of a TaxConfig, after-tax accounting is fully disabled — no extra
    BacktestResult fields, no equity_curve.csv column, no chart overlay.

    deductible_loss_cap: cap on net losses deducted against ordinary income
        per year (IRC §1211(b) — $3,000 single-filer / MFJ). Excess carries
        forward to the next year.
    payment_lag_days: calendar days from year-end to when tax for that year
        is deducted from the after-tax equity curve. Defaults to 105 ≈ Apr 15
        of the following year.
    """

    short_term_rate: float = 0.37
    long_term_rate: float = 0.20
    deductible_loss_cap: float = 3000.0
    payment_lag_days: int = 105

    def __post_init__(self) -> None:
        if self.short_term_rate < 0 or self.short_term_rate >= 1:
            msg = f"short_term_rate must be in [0, 1), got {self.short_term_rate}"
            raise ValueError(msg)
        if self.long_term_rate < 0 or self.long_term_rate >= 1:
            msg = f"long_term_rate must be in [0, 1), got {self.long_term_rate}"
            raise ValueError(msg)
        if self.deductible_loss_cap < 0:
            msg = f"deductible_loss_cap must be >= 0, got {self.deductible_loss_cap}"
            raise ValueError(msg)
        if self.payment_lag_days < 0:
            msg = f"payment_lag_days must be >= 0, got {self.payment_lag_days}"
            raise ValueError(msg)
```

- [ ] **Step 4: Update `src/midas/config.py`**

```python
from midas.models import (
    DEFAULT_MIN_BUY_DELTA,
    DEFAULT_MIN_CASH_PCT,
    DEFAULT_SOFTMAX_TEMPERATURE,
    DEFAULT_VOL_LOOKBACK_DAYS,
    AllocationConstraints,
    CashInfusion,
    Holding,
    PortfolioConfig,
    RiskConfig,
    StrategyConfig,
    TaxConfig,
    TradingRestrictions,
)


def load_strategies(
    path: Path,
) -> tuple[list[StrategyConfig], AllocationConstraints, RiskConfig, TaxConfig | None]:
    """Load strategy configs, allocation knobs, optional risk policy, and optional tax policy.

    Returns (strategies, constraints, risk_config, tax_config). Both risk and tax
    blocks are optional; omitting either yields the documented default (default
    RiskConfig for risk, None for tax — meaning after-tax accounting is disabled).
    """
    raw = _load_yaml(path)
    # ... existing strategies/constraints/risk parsing unchanged ...

    tax_raw = raw.get("tax")
    tax: TaxConfig | None = None
    if tax_raw is not None:
        tax = TaxConfig(
            short_term_rate=float(tax_raw.get("short_term_rate", 0.37)),
            long_term_rate=float(tax_raw.get("long_term_rate", 0.20)),
            deductible_loss_cap=float(tax_raw.get("deductible_loss_cap", 3000.0)),
            payment_lag_days=int(tax_raw.get("payment_lag_days", 105)),
        )

    return configs, constraints, risk, tax
```

- [ ] **Step 5: Update CLI call sites in `src/midas/cli.py`**

Three places (lines 162-163, 228-229, 352):

```python
strat_configs, constraints, risk_config, tax_config = (
    load_strategies(Path(strategies))
    if strategies
    else (None, AllocationConstraints(), RiskConfig(), None)
)
```

And at line 352:

```python
risk_config: RiskConfig = RiskConfig()
tax_config: TaxConfig | None = None
if strategies:
    strat_configs, strat_constraints, risk_config, tax_config = load_strategies(Path(strategies))
```

Add `TaxConfig` to the import block in `cli.py`:

```python
from midas.models import (
    AllocationConstraints,
    RiskConfig,
    TaxConfig,
    # ... existing imports
)
```

(Holding `tax_config` in the variable for now; downstream wiring lands in later tasks.)

- [ ] **Step 6: Round-trip the `tax:` block in `optimizer.write_strategies_yaml`**

Mirror the existing `risk_config` round-trip. In `src/midas/optimizer.py`:

```python
def write_strategies_yaml(
    params: dict[str, dict[str, float]],
    path: str,
    min_cash_pct: float = DEFAULT_MIN_CASH_PCT,
    risk_config: RiskConfig | None = None,
    tax_config: TaxConfig | None = None,
) -> None:
    """... existing docstring ...

    Args:
        ...
        tax_config: Preserved from the user's input config. When present,
            emitted as a ``tax:`` block. ``None`` is omitted.
    """
    # ... existing body up to the risk block ...
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
    # ... rest of body unchanged ...
```

Add `TaxConfig` to optimizer.py imports.

Update the two `write_strategies_yaml` call sites in `cli.py` (lines 388 and 473):

```python
write_strategies_yaml(
    wf_result.best_params,
    output,
    min_cash_pct=min_cash_pct,
    risk_config=risk_config,
    tax_config=tax_config,
)
```

(Same change at line 473 for the non-walk-forward path.)

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py tests/test_optimizer.py -v`
Expected: PASS for new and existing tests. Mypy & ruff both clean: `uv run ruff check . && uv run mypy src`

- [ ] **Step 8: Commit**

```bash
git add src/midas/models.py src/midas/config.py src/midas/cli.py src/midas/optimizer.py tests/test_config.py
git commit -m "Add optional TaxConfig loaded from strategies YAML

No behavior change yet — the loaded TaxConfig is passed to the live
and backtest entry points but not consumed. Wiring lands in later tasks
of the #66 implementation plan."
```

---

## Task 2: `consume_lots_fifo` tracks per-lot purchase dates in `SellBreakdown`

**Files:**
- Modify: `src/midas/live_state.py:270-352`
- Test: `tests/test_live_state.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_live_state.py`:

```python
def test_consume_lots_fifo_records_purchase_dates_per_bucket() -> None:
    """SellBreakdown should expose the purchase_date of each consumed lot,
    grouped by ST/LT bucket, so trade-log writers can resolve to a single
    date or the literal 'various'."""
    from midas.live_state import consume_lots_fifo

    lots = [
        PositionLot(shares=10.0, purchase_date=date(2024, 1, 1), cost_basis=10.0),  # LT
        PositionLot(shares=10.0, purchase_date=date(2024, 5, 1), cost_basis=12.0),  # LT
        PositionLot(shares=10.0, purchase_date=date(2026, 1, 1), cost_basis=20.0),  # ST
    ]
    # Sell 25 on 2026-05-08 → 20 LT + 5 ST consumed.
    breakdown = consume_lots_fifo(lots, shares=25.0, day=date(2026, 5, 8))
    assert breakdown.lt_purchase_dates == (date(2024, 1, 1), date(2024, 5, 1))
    assert breakdown.st_purchase_dates == (date(2026, 1, 1),)


def test_consume_lots_fifo_records_none_purchase_date_in_st_bucket() -> None:
    """A lot with purchase_date=None classifies as ST and surfaces None in the
    ST date tuple. The downstream resolver maps tuples containing None to the
    empty-string CSV value."""
    from midas.live_state import consume_lots_fifo

    lots = [PositionLot(shares=10.0, purchase_date=None, cost_basis=10.0)]
    breakdown = consume_lots_fifo(lots, shares=10.0, day=date(2026, 5, 8))
    assert breakdown.st_purchase_dates == (None,)
    assert breakdown.lt_purchase_dates == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_live_state.py::test_consume_lots_fifo_records_purchase_dates_per_bucket -v`
Expected: FAIL with `AttributeError: 'SellBreakdown' object has no attribute 'lt_purchase_dates'`.

- [ ] **Step 3: Extend `SellBreakdown` and `consume_lots_fifo`**

In `src/midas/live_state.py`, replace the `SellBreakdown` and `consume_lots_fifo` definitions:

```python
@dataclass(frozen=True)
class SellBreakdown:
    """Result of consuming lots FIFO for a sell.

    Reports shares and share-weighted cost basis separately for the short-term
    (held <365 days, or unknown purchase date) and long-term (held >=365 days)
    buckets. Either bucket may be zero. The ``*_weighted`` fields hold the raw
    ``sum(take * cost_basis)`` accumulator, useful for callers that need bit-
    identical reconstructions of total basis (since ``basis * shares`` is only
    algebraically — not bit-identically — equal in IEEE-754).

    The ``*_purchase_dates`` tuples list the purchase dates of the consumed lot
    slices in FIFO order; trade-log writers use them to populate the
    ``purchase_date`` column (single date when all lots in a bucket share one
    date, otherwise the literal string ``'various'``).
    """

    st_shares: float
    st_basis: float
    st_weighted: float
    lt_shares: float
    lt_basis: float
    lt_weighted: float
    st_purchase_dates: tuple[date | None, ...] = ()
    lt_purchase_dates: tuple[date | None, ...] = ()


def consume_lots_fifo(lots: list[PositionLot], shares: float, day: date) -> SellBreakdown:
    """Consume *shares* from *lots* in FIFO order, mutating in place.

    Lots whose ``purchase_date`` is at least 365 days before *day* contribute
    to the long-term bucket; everything else (including ``purchase_date=None``)
    goes to the short-term bucket. Each bucket reports a share-weighted cost
    basis over the consumed slices, plus the per-lot purchase dates of the
    slices in FIFO order.
    """
    if shares <= 0 or not lots:
        return SellBreakdown(
            st_shares=0.0,
            st_basis=0.0,
            st_weighted=0.0,
            lt_shares=0.0,
            lt_basis=0.0,
            lt_weighted=0.0,
        )

    st_shares = 0.0
    st_weighted = 0.0
    st_dates: list[date | None] = []
    lt_shares = 0.0
    lt_weighted = 0.0
    lt_dates: list[date | None] = []
    remaining = shares
    while remaining > 0 and lots:
        lot = lots[0]
        take = min(lot.shares, remaining)
        is_long_term = lot.purchase_date is not None and (day - lot.purchase_date).days >= 365
        if is_long_term:
            lt_shares += take
            lt_weighted += take * lot.cost_basis
            lt_dates.append(lot.purchase_date)
        else:
            st_shares += take
            st_weighted += take * lot.cost_basis
            st_dates.append(lot.purchase_date)

        if lot.shares <= remaining:
            remaining -= lot.shares
            lots.pop(0)
        else:
            lots[0] = PositionLot(
                shares=lot.shares - remaining,
                purchase_date=lot.purchase_date,
                cost_basis=lot.cost_basis,
            )
            remaining = 0

    st_basis = st_weighted / st_shares if st_shares > 0 else 0.0
    lt_basis = lt_weighted / lt_shares if lt_shares > 0 else 0.0
    return SellBreakdown(
        st_shares=st_shares,
        st_basis=st_basis,
        st_weighted=st_weighted,
        lt_shares=lt_shares,
        lt_basis=lt_basis,
        lt_weighted=lt_weighted,
        st_purchase_dates=tuple(st_dates),
        lt_purchase_dates=tuple(lt_dates),
    )
```

The two new fields default to `()` so any external code that constructs `SellBreakdown` directly (none in-tree, but a defensive choice) doesn't break.

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_live_state.py -v`
Expected: PASS for new tests + existing FIFO tests still pass.

- [ ] **Step 5: Commit**

```bash
git add src/midas/live_state.py tests/test_live_state.py
git commit -m "consume_lots_fifo: track per-lot purchase dates in SellBreakdown

SellBreakdown gains st_purchase_dates and lt_purchase_dates tuples
populated by consume_lots_fifo as it iterates. Trade-log writers use
them to populate the new purchase_date column on SELL bucket rows
(single date when all lots in a bucket share one, otherwise 'various')."
```

---

## Task 3: Add `purchase_date` to `TradeRecord`; populate in backtest

**Files:**
- Modify: `src/midas/models.py:119-127`
- Modify: `src/midas/backtest.py:907-1009`
- Test: `tests/test_backtest.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_backtest.py` (place near the existing `_execute` tests):

```python
def test_execute_buy_records_purchase_date_as_day() -> None:
    """BUY records carry the fill date as their purchase_date."""
    from datetime import date
    from midas.backtest import BacktestEngine
    from midas.models import Direction, Order, OrderContext

    engine = _build_minimal_engine()  # existing helper used elsewhere in the file
    state = _new_sim_state()
    order = Order(
        ticker="AAPL",
        direction=Direction.BUY,
        shares=10.0,
        price=20.0,
        estimated_value=200.0,
        context=OrderContext(
            contributions={"Momentum": 1.0},
            blended_score=1.0,
            target_weight=0.5,
            current_weight=0.0,
            reason="entry",
            source="Momentum",
        ),
    )
    records = engine._execute(order, day=date(2026, 5, 8), state=state)
    trade, _basis = records[0]
    assert trade.purchase_date == date(2026, 5, 8)


def test_execute_sell_single_lot_records_lot_purchase_date() -> None:
    """SELL bucket consuming one lot records that lot's purchase date."""
    from datetime import date
    from midas.backtest import BacktestEngine
    from midas.models import Direction, Order, OrderContext, PositionLot

    engine = _build_minimal_engine()
    state = _new_sim_state()
    state.lots["AAPL"] = [PositionLot(shares=10.0, purchase_date=date(2026, 1, 1), cost_basis=10.0)]
    state.positions["AAPL"] = 10.0
    order = Order(
        ticker="AAPL",
        direction=Direction.SELL,
        shares=10.0,
        price=15.0,
        estimated_value=150.0,
        context=OrderContext(
            contributions={"StopLoss": 1.0},
            blended_score=0.0,
            target_weight=0.0,
            current_weight=0.5,
            reason="exit",
            source="StopLoss",
        ),
    )
    records = engine._execute(order, day=date(2026, 5, 8), state=state)
    trade, _basis = records[0]
    assert trade.purchase_date == date(2026, 1, 1)


def test_execute_sell_mixed_lot_records_various() -> None:
    """SELL bucket spanning multiple lots with different dates records 'various'."""
    from datetime import date
    from midas.backtest import BacktestEngine
    from midas.models import Direction, Order, OrderContext, PositionLot

    engine = _build_minimal_engine()
    state = _new_sim_state()
    state.lots["AAPL"] = [
        PositionLot(shares=5.0, purchase_date=date(2026, 1, 1), cost_basis=10.0),
        PositionLot(shares=5.0, purchase_date=date(2026, 2, 1), cost_basis=11.0),
    ]
    state.positions["AAPL"] = 10.0
    order = Order(
        ticker="AAPL",
        direction=Direction.SELL,
        shares=10.0,
        price=15.0,
        estimated_value=150.0,
        context=OrderContext(
            contributions={"StopLoss": 1.0},
            blended_score=0.0,
            target_weight=0.0,
            current_weight=0.5,
            reason="exit",
            source="StopLoss",
        ),
    )
    records = engine._execute(order, day=date(2026, 5, 8), state=state)
    # Only one ST bucket since both lots are <365 days from sell day
    trade, _basis = records[0]
    assert trade.purchase_date == "various"
```

If `_build_minimal_engine` and `_new_sim_state` helpers don't already exist in `tests/test_backtest.py`, inline the construction. Read the top of `tests/test_backtest.py` to find the existing pattern and adapt.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_backtest.py::test_execute_buy_records_purchase_date_as_day -v`
Expected: FAIL with `AttributeError: 'TradeRecord' object has no attribute 'purchase_date'` or `TypeError: __init__() got an unexpected keyword argument`.

- [ ] **Step 3: Extend `TradeRecord` in `src/midas/models.py`**

```python
@dataclass(frozen=True)
class TradeRecord:
    date: date
    ticker: str
    direction: Direction
    shares: float
    price: float
    strategy_name: str
    holding_period: HoldingPeriod | None = None
    purchase_date: date | str | None = None
    """Purchase date of the consumed lots (SELL) or the fill date (BUY).

    On a SELL bucket row, ``'various'`` is the literal string sentinel for
    mixed-lot buckets where the consumed lots don't share a single purchase
    date — matches Schedule D convention. ``None`` indicates an unseeded
    live lot (purchase date never known).
    """
```

- [ ] **Step 4: Resolve and store `purchase_date` in backtest `_execute`**

In `src/midas/backtest.py`, add a helper near `_fifo_consumed_basis`:

```python
@staticmethod
def _resolve_purchase_date(dates: tuple[date | None, ...]) -> date | str | None:
    """Map a tuple of consumed lots' purchase dates to a single CSV value.

    Empty tuple → None. Single unique non-None date → that date. Anything
    else (multiple dates, any None present) → the literal string 'various'.
    """
    if not dates:
        return None
    unique = set(dates)
    if len(unique) == 1:
        only = next(iter(unique))
        return only  # may be a date or None
    return "various"
```

Update the BUY path in `_execute` to set `purchase_date=day`:

```python
return [
    (
        TradeRecord(
            date=day,
            ticker=ticker,
            direction=Direction.BUY,
            shares=order.shares,
            price=order.price,
            strategy_name=strategy_name,
            purchase_date=day,
        ),
        0.0,
    )
]
```

Update the SELL path to populate `purchase_date` per bucket:

```python
records: list[tuple[TradeRecord, float]] = []
if st_shares > 0:
    records.append(
        (
            TradeRecord(
                date=day,
                ticker=ticker,
                direction=Direction.SELL,
                shares=st_shares,
                price=order.price,
                strategy_name=strategy_name,
                holding_period=HoldingPeriod.SHORT_TERM,
                purchase_date=self._resolve_purchase_date(breakdown.st_purchase_dates),
            ),
            st_weighted_basis / st_shares,
        )
    )
if lt_shares > 0:
    records.append(
        (
            TradeRecord(
                date=day,
                ticker=ticker,
                direction=Direction.SELL,
                shares=lt_shares,
                price=order.price,
                strategy_name=strategy_name,
                holding_period=HoldingPeriod.LONG_TERM,
                purchase_date=self._resolve_purchase_date(breakdown.lt_purchase_dates),
            ),
            lt_weighted_basis / lt_shares,
        )
    )
return records
```

- [ ] **Step 5: Run tests to verify pass**

Run: `uv run pytest tests/test_backtest.py -v`
Expected: PASS for new and existing tests.

Run: `uv run mypy src` — clean.

- [ ] **Step 6: Commit**

```bash
git add src/midas/models.py src/midas/backtest.py tests/test_backtest.py
git commit -m "TradeRecord.purchase_date populated in backtest _execute

BUY rows carry the fill date. SELL bucket rows resolve to a single
purchase date when all consumed lots share one, the literal 'various'
when they span multiple dates, or None when the consumed lots have
no recorded date. Sets up the trades.csv column added in a later task."
```

---

## Task 4: `apply_sell` returns `SellBreakdown`

**Files:**
- Modify: `src/midas/live_state.py:365-390`
- Modify: `src/midas/live.py:319-323`
- Modify: `tests/test_live_state.py:339-355`

- [ ] **Step 1: Update existing `apply_sell` tests for the new return type**

Replace the body of `test_apply_sell_consumes_fifo_and_increments_cash` in `tests/test_live_state.py`:

```python
def test_apply_sell_consumes_fifo_and_increments_cash() -> None:
    state = LiveState(
        available_cash=0.0,
        cash_infusion_next_date=None,
        lots={
            "AAPL": [
                PositionLot(shares=30.0, purchase_date=date(2025, 4, 1), cost_basis=10.0),  # LT
                PositionLot(shares=20.0, purchase_date=date(2026, 4, 1), cost_basis=20.0),  # ST
            ]
        },
    )
    breakdown = apply_sell(state, "AAPL", shares=40.0, price=25.0, day=date(2026, 5, 7))
    assert state.available_cash == pytest.approx(40.0 * 25.0)
    assert state.lots["AAPL"] == [PositionLot(shares=10.0, purchase_date=date(2026, 4, 1), cost_basis=20.0)]

    lt_pnl = breakdown.lt_shares * 25.0 - breakdown.lt_weighted
    st_pnl = breakdown.st_shares * 25.0 - breakdown.st_weighted
    assert lt_pnl == pytest.approx(30.0 * (25.0 - 10.0))
    assert st_pnl == pytest.approx(10.0 * (25.0 - 20.0))
    assert breakdown.lt_purchase_dates == (date(2025, 4, 1),)
    assert breakdown.st_purchase_dates == (date(2026, 4, 1),)
```

The other apply_sell tests (`drops_empty_ticker_entry`, `keeps_ticker_with_remaining_shares`, `raises_on_oversell`) discard the return value entirely — they don't need updating, but verify they still call `apply_sell(...)` without unpacking a tuple. If any do unpack, replace with a single-name assignment.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_live_state.py::test_apply_sell_consumes_fifo_and_increments_cash -v`
Expected: FAIL with `TypeError` (still returns a tuple).

- [ ] **Step 3: Change `apply_sell` to return `SellBreakdown`**

In `src/midas/live_state.py`:

```python
def apply_sell(state: LiveState, ticker: str, shares: float, price: float, day: date) -> SellBreakdown:
    """Consume *shares* of *ticker* FIFO and increment cash by ``shares * price``.

    Mutates *state* in place. Returns the full ``SellBreakdown`` so callers
    can write trade-log rows with per-bucket shares, basis, and consumed-lot
    purchase dates. Realized P&L is recoverable as
    ``breakdown.st_shares * price - breakdown.st_weighted`` (analogous for LT).
    """
    lots = state.lots.get(ticker, [])
    breakdown = consume_lots_fifo(lots, shares, day)
    total_consumed = breakdown.st_shares + breakdown.lt_shares
    assert math.isclose(total_consumed, shares), (
        f"oversell on {ticker}: requested {shares}, consumed {total_consumed}"
    )
    state.available_cash += shares * price
    if not lots:
        state.lots.pop(ticker, None)
        state.high_water_marks.pop(ticker, None)
    return breakdown
```

- [ ] **Step 4: Update the `live.py` caller**

In `src/midas/live.py:319-323`, replace the apply_sell call (it was previously `apply_sell(self._state, ...)` discarding the return). Hold the breakdown for later:

```python
for order in filtered:
    if order.shares <= 0:
        continue
    if order.direction == Direction.BUY:
        apply_buy(self._state, order.ticker, order.shares, order.price, today)
    else:
        # apply_sell return now carries per-bucket detail used by the
        # trade-log append in Task 6 — hold for that wiring.
        _breakdown = apply_sell(self._state, order.ticker, order.shares, order.price, today)
    if self._restriction_tracker is not None:
        self._restriction_tracker.record_trade(order.ticker, order.direction, today)
```

- [ ] **Step 5: Run tests to verify pass**

Run: `uv run pytest tests/test_live_state.py tests/test_live_engine.py tests/test_live_backtest_parity.py -v`
Expected: PASS.

Run: `uv run mypy src` — clean.

- [ ] **Step 6: Commit**

```bash
git add src/midas/live_state.py src/midas/live.py tests/test_live_state.py
git commit -m "apply_sell returns SellBreakdown instead of (st_pnl, lt_pnl)

Strict superset of the old return: callers can recover realized P&L
as breakdown.st_shares * price - breakdown.st_weighted (analogous for
LT) while gaining access to per-bucket shares, basis, and consumed-lot
purchase dates needed by the trade log."
```

---

## Task 5: `trade_log` module — append-only CSV writer + strict reader

**Files:**
- Create: `src/midas/trade_log.py`
- Test: `tests/test_trade_log.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_trade_log.py`:

```python
"""Trade-log writer/reader unit tests."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from midas.models import Direction, HoldingPeriod, TradeRecord
from midas.trade_log import LoggedTrade, TradeLogError, append_trade, read_trades

TRADE_LOG_HEADER = (
    "date,ticker,direction,shares,price,strategy,"
    "holding_period,purchase_date,cost_basis,realized_pnl,return_pct\n"
)


def _buy(day: date) -> TradeRecord:
    return TradeRecord(
        date=day,
        ticker="AAPL",
        direction=Direction.BUY,
        shares=10.0,
        price=20.0,
        strategy_name="Momentum",
        purchase_date=day,
    )


def _sell(day: date, holding: HoldingPeriod, purchase: date | str | None) -> TradeRecord:
    return TradeRecord(
        date=day,
        ticker="AAPL",
        direction=Direction.SELL,
        shares=10.0,
        price=25.0,
        strategy_name="StopLoss",
        holding_period=holding,
        purchase_date=purchase,
    )


def test_first_append_writes_header(tmp_path: Path) -> None:
    path = tmp_path / "portfolio.state.yaml.trades.csv"
    append_trade(path, _buy(date(2026, 5, 8)), cost_basis=None, purchase_date=date(2026, 5, 8))
    contents = path.read_text()
    assert contents.startswith(TRADE_LOG_HEADER)
    # one data row after the header
    assert contents.count("\n") == 2


def test_subsequent_appends_do_not_duplicate_header(tmp_path: Path) -> None:
    path = tmp_path / "portfolio.state.yaml.trades.csv"
    append_trade(path, _buy(date(2026, 5, 8)), cost_basis=None, purchase_date=date(2026, 5, 8))
    append_trade(path, _buy(date(2026, 5, 9)), cost_basis=None, purchase_date=date(2026, 5, 9))
    text = path.read_text()
    assert text.count("date,ticker,direction") == 1
    assert text.count("\n") == 3  # header + 2 rows


def test_round_trip_buy_and_two_bucket_sell(tmp_path: Path) -> None:
    path = tmp_path / "portfolio.state.yaml.trades.csv"
    append_trade(path, _buy(date(2026, 5, 8)), cost_basis=None, purchase_date=date(2026, 5, 8))
    append_trade(
        path,
        _sell(date(2026, 6, 8), HoldingPeriod.SHORT_TERM, "various"),
        cost_basis=20.5,
        purchase_date="various",
    )
    append_trade(
        path,
        _sell(date(2026, 6, 8), HoldingPeriod.LONG_TERM, date(2024, 1, 1)),
        cost_basis=10.0,
        purchase_date=date(2024, 1, 1),
    )
    rows = read_trades(path)
    assert len(rows) == 3
    assert rows[0].direction == Direction.BUY
    assert rows[0].purchase_date == date(2026, 5, 8)
    assert rows[1].holding_period == HoldingPeriod.SHORT_TERM
    assert rows[1].purchase_date == "various"
    assert rows[1].cost_basis == 20.5
    assert rows[1].realized_pnl == pytest.approx((25.0 - 20.5) * 10.0)
    assert rows[2].purchase_date == date(2024, 1, 1)


def test_purchase_date_none_round_trips_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "portfolio.state.yaml.trades.csv"
    append_trade(
        path,
        _sell(date(2026, 6, 8), HoldingPeriod.SHORT_TERM, None),
        cost_basis=10.0,
        purchase_date=None,
    )
    rows = read_trades(path)
    assert rows[0].purchase_date is None


def test_header_drift_raises_with_named_divergence(tmp_path: Path) -> None:
    path = tmp_path / "portfolio.state.yaml.trades.csv"
    path.write_text("date,ticker,direction\n2026-05-08,AAPL,BUY\n")
    with pytest.raises(TradeLogError, match="header"):
        read_trades(path)


def test_partial_row_raises_with_line_number(tmp_path: Path) -> None:
    path = tmp_path / "portfolio.state.yaml.trades.csv"
    path.write_text(TRADE_LOG_HEADER + "2026-05-08,AAPL\n")  # 2 fields instead of 11
    with pytest.raises(TradeLogError, match="line 2"):
        read_trades(path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_trade_log.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'midas.trade_log'`.

- [ ] **Step 3: Implement `src/midas/trade_log.py`**

```python
"""Append-only trade-log writer and strict reader.

Used by both the live engine (writes to ``<state_path>.trades.csv``) and the
backtest result writer (writes to ``<output_dir>/trades.csv``). Single shape
across both modes so ``midas tax-report`` has one reader.

Header drift on read raises :class:`TradeLogError` rather than silently
returning empty / wrong rows; partial rows raise with the offending line
number. The log is intentionally permissive on content (negative shares,
future dates) but strict on shape — hand-edits are an explicit escape valve.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from midas.models import Direction, HoldingPeriod, TradeRecord

TRADE_LOG_COLUMNS: tuple[str, ...] = (
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
)


class TradeLogError(ValueError):
    """Raised on header drift, partial rows, or unparseable values."""


@dataclass(frozen=True)
class LoggedTrade:
    """In-memory representation of one trade-log row."""

    date: date
    ticker: str
    direction: Direction
    shares: float
    price: float
    strategy_name: str
    holding_period: HoldingPeriod | None
    purchase_date: date | str | None
    cost_basis: float | None
    realized_pnl: float | None
    return_pct: float | None


def _format_purchase_date(value: date | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return value.isoformat()


def _format_holding_period(value: HoldingPeriod | None) -> str:
    return value.value if value is not None else ""


def append_trade(
    path: Path,
    record: TradeRecord,
    cost_basis: float | None,
    purchase_date: date | str | None,
) -> None:
    """Append one row to *path*, creating the file with header on first write.

    For BUY rows pass ``cost_basis=None``; the ``cost_basis``, ``realized_pnl``,
    and ``return_pct`` columns are written empty. For SELL rows the bucket's
    share-weighted cost basis is required; ``realized_pnl`` and ``return_pct``
    are derived from it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if needs_header:
            writer.writerow(TRADE_LOG_COLUMNS)
        if record.direction == Direction.BUY or cost_basis is None:
            cost_basis_cell: str | float = ""
            pnl_cell: str | float = ""
            ret_cell: str | float = ""
        else:
            pnl = round((record.price - cost_basis) * record.shares, 4)
            ret = round((record.price - cost_basis) / cost_basis, 6) if cost_basis != 0 else 0.0
            cost_basis_cell = round(cost_basis, 4)
            pnl_cell = pnl
            ret_cell = ret
        writer.writerow(
            [
                record.date.isoformat(),
                record.ticker,
                record.direction.value,
                record.shares,
                record.price,
                record.strategy_name,
                _format_holding_period(record.holding_period),
                _format_purchase_date(purchase_date),
                cost_basis_cell,
                pnl_cell,
                ret_cell,
            ]
        )
        handle.flush()


def _parse_holding_period(raw: str) -> HoldingPeriod | None:
    if not raw:
        return None
    try:
        return HoldingPeriod(raw)
    except ValueError as exc:
        msg = f"unknown holding_period value {raw!r}"
        raise TradeLogError(msg) from exc


def _parse_purchase_date(raw: str) -> date | str | None:
    if not raw:
        return None
    if raw == "various":
        return raw
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        msg = f"unparseable purchase_date {raw!r}"
        raise TradeLogError(msg) from exc


def _parse_optional_float(raw: str) -> float | None:
    if raw == "":
        return None
    try:
        return float(raw)
    except ValueError as exc:
        msg = f"unparseable float {raw!r}"
        raise TradeLogError(msg) from exc


def read_trades(path: Path) -> list[LoggedTrade]:
    """Read all rows from *path* into ``LoggedTrade`` instances.

    Raises :class:`TradeLogError` on header drift or unparseable rows.
    """
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return []
        if tuple(header) != TRADE_LOG_COLUMNS:
            msg = f"trade-log header drift in {path}: expected {TRADE_LOG_COLUMNS}, got {tuple(header)}"
            raise TradeLogError(msg)

        out: list[LoggedTrade] = []
        for line_num, row in enumerate(reader, start=2):
            if len(row) != len(TRADE_LOG_COLUMNS):
                msg = f"trade-log row at line {line_num} has {len(row)} fields, expected {len(TRADE_LOG_COLUMNS)}"
                raise TradeLogError(msg)
            try:
                trade_date = date.fromisoformat(row[0])
                direction = Direction(row[2])
            except ValueError as exc:
                msg = f"trade-log row at line {line_num}: {exc}"
                raise TradeLogError(msg) from exc
            out.append(
                LoggedTrade(
                    date=trade_date,
                    ticker=row[1],
                    direction=direction,
                    shares=float(row[3]),
                    price=float(row[4]),
                    strategy_name=row[5],
                    holding_period=_parse_holding_period(row[6]),
                    purchase_date=_parse_purchase_date(row[7]),
                    cost_basis=_parse_optional_float(row[8]),
                    realized_pnl=_parse_optional_float(row[9]),
                    return_pct=_parse_optional_float(row[10]),
                )
            )
        return out
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_trade_log.py -v`
Expected: PASS for all 6 tests.

Run: `uv run mypy src` and `uv run ruff check .` — clean.

- [ ] **Step 5: Commit**

```bash
git add src/midas/trade_log.py tests/test_trade_log.py
git commit -m "Add trade_log module: append-only writer + strict reader

Single CSV shape used by both live and backtest. Header is written on
first append, never duplicated. Reader raises TradeLogError on header
drift or partial rows naming the offending line number. 'various' and
empty purchase_date both round-trip correctly."
```

---

## Task 6: Live engine appends fills to `<state>.trades.csv`

**Files:**
- Modify: `src/midas/live.py`
- Modify: `tests/test_live_engine.py`
- Create: `tests/test_live_trade_log.py`

- [ ] **Step 1: Write the failing integration test**

Create `tests/test_live_trade_log.py`:

```python
"""Integration test: live engine writes a complete trade log across ticks."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from midas.live_state import LiveState, save_atomic
from midas.models import PositionLot
from midas.trade_log import read_trades

# Reuse the existing live-engine harness from tests/test_live_engine.py
from tests.test_live_engine import (  # type: ignore[import-not-found]
    StubProvider,
    build_engine,
)


def test_live_engine_writes_trade_log(tmp_path: Path) -> None:
    state_path = tmp_path / "portfolio.state.yaml"
    log_path = tmp_path / "portfolio.state.yaml.trades.csv"

    # Seed state: AAPL held with one LT lot, ample cash.
    save_atomic(
        LiveState(
            available_cash=10000.0,
            cash_infusion_next_date=None,
            high_water_marks={"AAPL": 100.0},
            peak_equity=20000.0,
            lots={
                "AAPL": [PositionLot(shares=100.0, purchase_date=date(2024, 1, 1), cost_basis=50.0)],
            },
        ),
        state_path,
    )

    # Provider: feed three ticks with prices that trigger one full-exit SELL.
    provider = StubProvider(
        {
            "AAPL": pd.DataFrame(
                {"close": [100.0, 95.0, 30.0]},
                index=pd.to_datetime(["2026-05-06", "2026-05-07", "2026-05-08"]),
            )
        }
    )

    with build_engine(state_path, provider) as engine:
        for _ in range(3):
            engine._tick(["AAPL"])

    trades = read_trades(log_path)
    sells = [t for t in trades if t.direction.value == "SELL"]
    assert len(sells) >= 1, f"expected at least one SELL row, got {trades}"
    # All AAPL lots are LT → SELL bucket should be LT with the original purchase date.
    sell = sells[0]
    assert sell.holding_period is not None
    assert sell.holding_period.value == "long-term"
    assert sell.purchase_date == date(2024, 1, 1)
    assert sell.cost_basis == pytest.approx(50.0)
```

(`StubProvider` and `build_engine` already exist in `tests/test_live_engine.py`; if not, inline construction of a minimal engine matching the existing pattern there — read the file's existing fixtures first.)

Append to `tests/test_live_engine.py`:

```python
def test_engine_creates_trade_log_alongside_state(tmp_path: Path) -> None:
    """Smoke: trade-log file appears next to the state file after the first tick that produces fills."""
    state_path = tmp_path / "portfolio.state.yaml"
    expected_log = tmp_path / "portfolio.state.yaml.trades.csv"
    # ... use the existing fixture builders to seed a state that will trigger
    # at least one BUY on tick 1, then assert expected_log.exists().
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_live_trade_log.py -v`
Expected: FAIL because the engine doesn't write the log yet.

- [ ] **Step 3: Wire trade-log appends in `live.py`**

In `src/midas/live.py`:

Add to the imports:

```python
from midas.models import (
    AllocationConstraints,
    Direction,
    HoldingPeriod,
    PortfolioConfig,
    TradeRecord,
)
from midas.trade_log import append_trade
```

In `LiveEngine.__init__`, derive the log path from the state path:

```python
self._state_path = state_path
self._trade_log_path = state_path.with_suffix(state_path.suffix + ".trades.csv")
```

Replace the fill-application loop in `_tick` (currently around lines 316-325) with one that:
1. Captures `SellBreakdown` per SELL,
2. Mutates state via `apply_buy` / `apply_sell`,
3. Saves state atomically,
4. Then appends rows to the trade log.

```python
# Apply assumed fills to the in-memory state. Capture per-SELL breakdowns
# so the trade-log append below has shares/basis/dates per ST/LT bucket.
sell_breakdowns: dict[int, SellBreakdown] = {}
for order in filtered:
    if order.shares <= 0:
        continue
    if order.direction == Direction.BUY:
        apply_buy(self._state, order.ticker, order.shares, order.price, today)
    else:
        sell_breakdowns[id(order)] = apply_sell(
            self._state, order.ticker, order.shares, order.price, today
        )
    if self._restriction_tracker is not None:
        self._restriction_tracker.record_trade(order.ticker, order.direction, today)

# Update peak equity from the current portfolio value (post-fills).
positions_after = {
    ticker: sum(lot.shares for lot in self._state.lots.get(ticker, [])) for ticker in active_tickers
}
current_equity = self._state.available_cash + sum(
    positions_after[ticker] * current_prices[ticker] for ticker in active_tickers
)
self._state.peak_equity = max(self._state.peak_equity or 0.0, current_equity)

# Persist state at the end of the tick.
save_atomic(self._state, self._state_path)

# Append a row per BUY and per non-empty ST/LT SELL bucket to the trade log.
# Order matters: state is durable first; if the append fails, the operator
# sees the error and re-running re-attempts the append.
for order in filtered:
    if order.shares <= 0:
        continue
    if order.direction == Direction.BUY:
        record = TradeRecord(
            date=today,
            ticker=order.ticker,
            direction=Direction.BUY,
            shares=order.shares,
            price=order.price,
            strategy_name=order.context.source,
            purchase_date=today,
        )
        append_trade(self._trade_log_path, record, cost_basis=None, purchase_date=today)
    else:
        breakdown = sell_breakdowns[id(order)]
        if breakdown.st_shares > 0:
            purchase = _resolve_purchase_date(breakdown.st_purchase_dates)
            record = TradeRecord(
                date=today,
                ticker=order.ticker,
                direction=Direction.SELL,
                shares=breakdown.st_shares,
                price=order.price,
                strategy_name=order.context.source,
                holding_period=HoldingPeriod.SHORT_TERM,
                purchase_date=purchase,
            )
            append_trade(
                self._trade_log_path,
                record,
                cost_basis=breakdown.st_basis,
                purchase_date=purchase,
            )
        if breakdown.lt_shares > 0:
            purchase = _resolve_purchase_date(breakdown.lt_purchase_dates)
            record = TradeRecord(
                date=today,
                ticker=order.ticker,
                direction=Direction.SELL,
                shares=breakdown.lt_shares,
                price=order.price,
                strategy_name=order.context.source,
                holding_period=HoldingPeriod.LONG_TERM,
                purchase_date=purchase,
            )
            append_trade(
                self._trade_log_path,
                record,
                cost_basis=breakdown.lt_basis,
                purchase_date=purchase,
            )
```

Add the `_resolve_purchase_date` helper at module level (mirror of `BacktestEngine._resolve_purchase_date`):

```python
def _resolve_purchase_date(dates: tuple[date | None, ...]) -> date | str | None:
    """Map a tuple of consumed lots' purchase dates to a single CSV value.

    Empty tuple → None. Single unique non-None date → that date. Anything
    else (multiple dates, any None present) → the literal string 'various'.
    """
    if not dates:
        return None
    unique = set(dates)
    if len(unique) == 1:
        only = next(iter(unique))
        return only
    return "various"
```

Add `SellBreakdown` to the `live_state` import line in `live.py`:

```python
from midas.live_state import (
    LiveState,
    SellBreakdown,
    aggregate_cost_basis,
    apply_buy,
    apply_sell,
    load_or_seed,
    save_atomic,
)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_live_trade_log.py tests/test_live_engine.py tests/test_live_backtest_parity.py -v`
Expected: PASS.

Run: `uv run mypy src` — clean.

- [ ] **Step 5: Commit**

```bash
git add src/midas/live.py tests/test_live_engine.py tests/test_live_trade_log.py
git commit -m "Live engine writes append-only trade log next to state file

After fills are applied and state is saved atomically, append one row
per BUY and one per non-empty ST/LT SELL bucket to <state>.trades.csv.
Inherits the engine-wide flock; no new locking primitive."
```

---

## Task 7: Backtest `trades.csv` adds `purchase_date` column

**Files:**
- Modify: `src/midas/results.py:84-119`
- Test: `tests/test_backtest.py` or `tests/test_results.py` (use whichever exists; create a small results test if neither covers `_write_trades_csv`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_backtest.py`:

```python
def test_trades_csv_includes_purchase_date_column(tmp_path: Path) -> None:
    """Backtest output's trades.csv has a purchase_date column populated for BUYs and SELLs."""
    import csv
    from midas.results import BacktestResult, _write_trades_csv
    from midas.models import Direction, HoldingPeriod, TradeRecord

    trades = [
        TradeRecord(
            date=date(2026, 1, 5),
            ticker="AAPL",
            direction=Direction.BUY,
            shares=10.0,
            price=20.0,
            strategy_name="Momentum",
            purchase_date=date(2026, 1, 5),
        ),
        TradeRecord(
            date=date(2026, 4, 1),
            ticker="AAPL",
            direction=Direction.SELL,
            shares=10.0,
            price=25.0,
            strategy_name="StopLoss",
            holding_period=HoldingPeriod.SHORT_TERM,
            purchase_date=date(2026, 1, 5),
        ),
    ]
    result = _make_minimal_result(trades=trades, basis_per_sell=[20.0])
    out = tmp_path / "trades.csv"
    _write_trades_csv(result, out)
    rows = list(csv.DictReader(out.open()))
    assert "purchase_date" in rows[0]
    assert rows[0]["purchase_date"] == "2026-01-05"
    assert rows[1]["purchase_date"] == "2026-01-05"
```

(Inline `_make_minimal_result` if not present — a `BacktestResult` with placeholder zeros for everything except `trades` and `basis_per_sell`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_backtest.py::test_trades_csv_includes_purchase_date_column -v`
Expected: FAIL — column missing.

- [ ] **Step 3: Update `_write_trades_csv` in `src/midas/results.py`**

```python
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
            purchase_cell: str
            if trade.purchase_date is None:
                purchase_cell = ""
            elif isinstance(trade.purchase_date, str):
                purchase_cell = trade.purchase_date
            else:
                purchase_cell = trade.purchase_date.isoformat()
            common = [
                trade.date.isoformat(),
                trade.ticker,
                trade.direction.value,
                trade.shares,
                trade.price,
                trade.strategy_name,
                trade.holding_period.value if trade.holding_period else "",
                purchase_cell,
            ]
            if trade.direction == Direction.SELL:
                basis = sell_basis.get(id(trade), trade.price)
                pnl = round((trade.price - basis) * trade.shares, 4)
                ret = round((trade.price - basis) / basis, 6) if basis != 0 else 0.0
                writer.writerow([*common, round(basis, 4), pnl, ret])
            else:
                writer.writerow([*common, "", "", ""])
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_backtest.py -v`
Expected: PASS for new test + existing tests still pass.

- [ ] **Step 5: Commit**

```bash
git add src/midas/results.py tests/test_backtest.py
git commit -m "trades.csv: add purchase_date column for tax reporting

Same column shape now in backtest's trades.csv and live's
<state>.trades.csv. The new tax-report subcommand has one reader for
both modes."
```

---

## Task 8: `tax.py` netting math — `compute_tax_summary`

**Files:**
- Create: `src/midas/tax.py`
- Test: `tests/test_tax.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tax.py`:

```python
"""Unit tests for capital-gains tax computation."""

from __future__ import annotations

from datetime import date

import pytest

from midas.models import Direction, HoldingPeriod, TaxConfig, TradeRecord
from midas.tax import (
    AnnualTaxSummary,
    compute_after_tax_curve,
    compute_tax_summary,
)


CONFIG = TaxConfig(
    short_term_rate=0.30,
    long_term_rate=0.15,
    deductible_loss_cap=3000.0,
    payment_lag_days=105,
)


def _sell(year: int, period: HoldingPeriod, shares: float, price: float) -> TradeRecord:
    return TradeRecord(
        date=date(year, 6, 1),
        ticker="AAPL",
        direction=Direction.SELL,
        shares=shares,
        price=price,
        strategy_name="StopLoss",
        holding_period=period,
    )


def test_pure_st_gain_taxed_at_short_term_rate() -> None:
    trades = [_sell(2026, HoldingPeriod.SHORT_TERM, 10.0, 30.0)]
    basis = [20.0]  # gain = (30 - 20) * 10 = 100
    summary = compute_tax_summary(trades, basis, CONFIG, end_date=date(2026, 12, 31))
    assert len(summary) == 1
    s = summary[0]
    assert s.year == 2026
    assert s.st_realized == pytest.approx(100.0)
    assert s.lt_realized == 0.0
    assert s.tax_owed == pytest.approx(100.0 * 0.30)
    assert s.carry_forward == 0.0


def test_st_gain_lt_loss_cross_nets() -> None:
    """ST $100 gain, LT $40 loss → cross-net leaves $60 ST gain after offset.

    Per IRS Schedule D, an LT loss first offsets LT gains; with no LT gain,
    excess LT loss carries to ST. Result: $60 net ST gain taxed at ST rate.
    """
    trades = [
        _sell(2026, HoldingPeriod.SHORT_TERM, 10.0, 30.0),  # +$100
        _sell(2026, HoldingPeriod.LONG_TERM, 10.0, 16.0),   # -$40 vs basis=$20
    ]
    basis = [20.0, 20.0]
    summary = compute_tax_summary(trades, basis, CONFIG, end_date=date(2026, 12, 31))
    assert summary[0].net_after_cross == pytest.approx(60.0)
    assert summary[0].tax_owed == pytest.approx(60.0 * 0.30)


def test_net_loss_below_cap_full_deductible() -> None:
    """Net loss of $1,200 → full $1,200 deducted at ST rate; no carry."""
    trades = [_sell(2026, HoldingPeriod.SHORT_TERM, 10.0, 8.0)]  # -$1200
    basis = [20.0]
    summary = compute_tax_summary(trades, basis, CONFIG, end_date=date(2026, 12, 31))
    s = summary[0]
    assert s.net_after_cross == pytest.approx(-1200.0)
    assert s.deductible_loss == pytest.approx(1200.0)
    assert s.tax_owed == pytest.approx(-1200.0 * 0.30)  # negative → refund/credit
    assert s.carry_forward == 0.0


def test_net_loss_above_cap_caps_deductible_carries_remainder() -> None:
    """Net loss of $5,000 → $3,000 deducted, $2,000 carries forward."""
    trades = [_sell(2026, HoldingPeriod.SHORT_TERM, 100.0, 30.0)]  # -$5000 vs $80 basis
    basis = [80.0]
    summary = compute_tax_summary(trades, basis, CONFIG, end_date=date(2026, 12, 31))
    s = summary[0]
    assert s.net_after_cross == pytest.approx(-5000.0)
    assert s.deductible_loss == pytest.approx(3000.0)
    assert s.tax_owed == pytest.approx(-3000.0 * 0.30)
    assert s.carry_forward == pytest.approx(2000.0)


def test_carry_forward_absorbs_next_year_gain() -> None:
    """$5K loss in year 1 ($2K carries) absorbs $1.5K gain in year 2.

    Year 2's net post-carry = $1.5K - $2K = -$500 → fully deductible at ST,
    no remaining carry.
    """
    trades = [
        _sell(2026, HoldingPeriod.SHORT_TERM, 100.0, 30.0),  # -$5000
        _sell(2027, HoldingPeriod.SHORT_TERM, 100.0, 35.0),  # +$1500 vs $20 basis
    ]
    basis = [80.0, 20.0]
    summary = compute_tax_summary(trades, basis, CONFIG, end_date=date(2027, 12, 31))
    assert len(summary) == 2
    assert summary[0].carry_forward == pytest.approx(2000.0)
    assert summary[1].net_after_cross == pytest.approx(-500.0)
    assert summary[1].deductible_loss == pytest.approx(500.0)
    assert summary[1].tax_owed == pytest.approx(-500.0 * 0.30)
    assert summary[1].carry_forward == 0.0


def test_empty_trades_returns_empty_summary() -> None:
    summary = compute_tax_summary([], [], CONFIG, end_date=date(2026, 12, 31))
    assert summary == []


def test_payment_date_is_year_end_plus_lag() -> None:
    trades = [_sell(2026, HoldingPeriod.SHORT_TERM, 10.0, 30.0)]
    basis = [20.0]
    summary = compute_tax_summary(trades, basis, CONFIG, end_date=date(2027, 12, 31))
    # Dec 31 2026 + 105 days = Apr 15 2027
    assert summary[0].payment_date == date(2027, 4, 15)


def test_payment_date_clamped_to_end_date() -> None:
    """If the natural payment date falls past end_date, clamp to end_date.

    Year 2026 sale, end_date 2027-02-15 (before Apr 15) → deduct on 2027-02-15.
    """
    trades = [_sell(2026, HoldingPeriod.SHORT_TERM, 10.0, 30.0)]
    basis = [20.0]
    summary = compute_tax_summary(trades, basis, CONFIG, end_date=date(2027, 2, 15))
    assert summary[0].payment_date == date(2027, 2, 15)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tax.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/midas/tax.py` (netting half)**

```python
"""Capital-gains tax accounting (reporting layer, not strategy).

Pure functions: ``compute_tax_summary`` runs IRS Schedule D-style annual
netting + $3K deductible + carryforward across realized SELLs;
``compute_after_tax_curve`` applies each year's tax owed to a gross equity
curve at the configured payment date. Used by both ``BacktestResult``
construction and the ``midas tax-report`` subcommand so backtest after-tax
numbers and tax-report numbers are bit-identical when fed identical inputs.

Simplifications: carryforward is a single signed scalar that loses ST/LT
character on rollover (IRS preserves character). Wash-sale detection,
specific-lot accounting, state taxes, and bracketed rates are out of scope
— the engine is FIFO and the rates are flat.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

from midas.metrics import _pair_sells_with_basis
from midas.models import Direction, HoldingPeriod, TaxConfig, TradeRecord


@dataclass(frozen=True)
class AnnualTaxSummary:
    """Per-year realized-P&L tax breakdown.

    ``st_realized`` / ``lt_realized`` are pre-netting raw bucket sums
    (gross gain - gross loss within bucket). ``net_after_cross`` is the
    post-cross-bucket-netting figure. ``deductible_loss`` is the portion
    of an overall loss deducted against ordinary income this year (capped
    at ``TaxConfig.deductible_loss_cap``). ``carry_forward`` is the unused
    loss rolling into next year. ``tax_owed`` is signed: positive means
    cash deducted from the equity curve, negative means a refund/credit.
    """

    year: int
    st_realized: float
    lt_realized: float
    net_after_cross: float
    deductible_loss: float
    carry_forward: float
    tax_owed: float
    payment_date: date


def _payment_date_for_year(year: int, lag_days: int, end_date: date) -> date:
    """Year-end + lag, clamped to end_date if it would fall past it."""
    natural = date(year, 12, 31) + timedelta(days=lag_days)
    return min(natural, end_date)


def compute_tax_summary(
    trades: Sequence[TradeRecord],
    basis_per_sell: Sequence[float],
    config: TaxConfig,
    end_date: date,
) -> list[AnnualTaxSummary]:
    """Group SELLs by calendar year; net per-bucket then cross-bucket; carry losses.

    Args:
        trades: All TradeRecords from the backtest or live trade log. BUYs are
            ignored. SELL records must carry a non-None ``holding_period``.
        basis_per_sell: Parallel list of share-weighted cost basis per SELL,
            in the same order as SELL records appear in ``trades``.
        config: Tax-rate policy. None disables tax accounting; callers should
            check before invoking.
        end_date: Last bar of the backtest (or report-period end). Years whose
            natural payment date falls past this are clamped to it.

    Returns:
        One AnnualTaxSummary per calendar year that contained at least one
        SELL or that received a carryforward from the prior year. Sorted by
        year ascending.
    """
    paired = _pair_sells_with_basis(list(trades), list(basis_per_sell))
    by_year_st: dict[int, float] = defaultdict(float)
    by_year_lt: dict[int, float] = defaultdict(float)

    for trade, basis in paired:
        if trade.direction != Direction.SELL or trade.holding_period is None:
            continue
        pnl = (trade.price - basis) * trade.shares
        if trade.holding_period == HoldingPeriod.LONG_TERM:
            by_year_lt[trade.date.year] += pnl
        else:
            by_year_st[trade.date.year] += pnl

    years_with_activity = sorted(set(by_year_st) | set(by_year_lt))
    if not years_with_activity:
        return []

    summaries: list[AnnualTaxSummary] = []
    carry_in = 0.0  # signed; positive == accumulated loss available to offset gains
    for year in years_with_activity:
        st_raw = by_year_st.get(year, 0.0)
        lt_raw = by_year_lt.get(year, 0.0)

        # Per-bucket netting is the raw sum (already done by accumulation).
        # Cross-bucket: if signs differ, net them.
        st_after_cross = st_raw
        lt_after_cross = lt_raw
        if st_raw > 0 and lt_raw < 0:
            offset = min(st_raw, -lt_raw)
            st_after_cross -= offset
            lt_after_cross += offset
        elif lt_raw > 0 and st_raw < 0:
            offset = min(lt_raw, -st_raw)
            lt_after_cross -= offset
            st_after_cross += offset

        # Apply prior-year carry to remaining gains (loses ST/LT character).
        net_pre_carry = st_after_cross + lt_after_cross
        if carry_in > 0 and net_pre_carry > 0:
            offset = min(carry_in, net_pre_carry)
            # Shave the offset off whichever bucket is positive, ST first.
            if st_after_cross > 0:
                shave = min(offset, st_after_cross)
                st_after_cross -= shave
                offset -= shave
            if offset > 0 and lt_after_cross > 0:
                lt_after_cross -= offset
            carry_in -= min(carry_in, net_pre_carry)

        net_after_cross = st_after_cross + lt_after_cross
        deductible_loss = 0.0
        carry_out = 0.0

        if net_after_cross >= 0:
            tax_owed = st_after_cross * config.short_term_rate + lt_after_cross * config.long_term_rate
        else:
            absolute = -net_after_cross
            deductible_loss = min(absolute, config.deductible_loss_cap)
            carry_out = absolute - deductible_loss
            # Negative tax_owed == credit at ST rate (matches §1211(b) treatment
            # of net capital loss as an offset against ordinary income).
            tax_owed = -deductible_loss * config.short_term_rate

        # Roll any unused prior-year carry into this year's carry_out, since
        # carryforward accumulates indefinitely (loses character either way).
        carry_out += carry_in
        carry_in = carry_out

        summaries.append(
            AnnualTaxSummary(
                year=year,
                st_realized=st_raw,
                lt_realized=lt_raw,
                net_after_cross=net_after_cross,
                deductible_loss=deductible_loss,
                carry_forward=carry_out,
                tax_owed=tax_owed,
                payment_date=_payment_date_for_year(year, config.payment_lag_days, end_date),
            )
        )
    return summaries


def compute_after_tax_curve(
    equity_curve: Sequence[tuple[date, float]],
    summaries: Sequence[AnnualTaxSummary],
    end_date: date,
) -> list[tuple[date, float]]:
    """Apply each year's tax_owed to *equity_curve* at its payment_date.

    Refunds (negative tax_owed) increase the curve. The original curve is not
    mutated. If summaries is empty, returns the gross curve as a copy.
    """
    if not summaries:
        return list(equity_curve)
    pending = sorted(((s.payment_date, s.tax_owed) for s in summaries), key=lambda item: item[0])
    out: list[tuple[date, float]] = []
    cumulative_deduction = 0.0
    payment_idx = 0
    for day, value in equity_curve:
        while payment_idx < len(pending) and pending[payment_idx][0] <= day:
            cumulative_deduction += pending[payment_idx][1]
            payment_idx += 1
        out.append((day, value - cumulative_deduction))
    return out
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_tax.py -v`
Expected: PASS for all 8 tests.

Run: `uv run mypy src` and `uv run ruff check .` — clean.

- [ ] **Step 5: Commit**

```bash
git add src/midas/tax.py tests/test_tax.py
git commit -m "Add tax module: annual ST/LT netting + \$3K deductible + carryforward

compute_tax_summary groups SELLs by year, runs IRS Schedule D-style
per-bucket then cross-bucket netting, applies the \$3,000 ordinary-income
offset, and threads carryforward across years.

compute_after_tax_curve applies each year's tax_owed to a gross equity
curve at the configured payment date (Dec 31 + payment_lag_days,
defaulting to ~Apr 15 of the following year), clamped to backtest end."
```

---

## Task 9: After-tax fields in `BacktestResult` + result writers

**Files:**
- Modify: `src/midas/results.py` (`BacktestResult`, `write_backtest_results`, `_write_equity_curve_csv`, `_write_summary_json`)
- Modify: `src/midas/backtest.py:_build_result` (compute and populate the new fields)
- Modify: `src/midas/cli.py:179-191` (thread `tax_config` into `BacktestEngine.run`)
- Test: `tests/test_backtest.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_backtest.py`:

```python
def test_backtest_result_after_tax_fields_populated_with_tax_config() -> None:
    """End-to-end: a 2-year backtest with TaxConfig set populates after_tax_*."""
    from midas.models import TaxConfig
    # ... use the existing backtest harness used for full-engine tests in this file
    tax = TaxConfig(short_term_rate=0.30, long_term_rate=0.15)
    result = _run_backtest_with(tax_config=tax)  # implement this helper alongside
    assert result.after_tax_final_value is not None
    assert result.after_tax_cagr is not None
    assert result.tax_summary  # non-empty
    assert result.tax_cost_ratio is not None


def test_backtest_result_after_tax_fields_none_without_tax_config() -> None:
    """Backwards-compatible: without TaxConfig, all after-tax fields are None/empty."""
    result = _run_backtest_with(tax_config=None)
    assert result.after_tax_final_value is None
    assert result.after_tax_equity_curve == []
    assert result.tax_summary == []
    assert result.tax_cost_ratio is None
```

(Use the test patterns already in `tests/test_backtest.py` to construct `_run_backtest_with`. Read existing `BacktestEngine` calling tests as a template.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_backtest.py::test_backtest_result_after_tax_fields_populated_with_tax_config -v`
Expected: FAIL — fields don't exist.

- [ ] **Step 3: Add fields to `BacktestResult` in `src/midas/results.py`**

```python
from midas.tax import AnnualTaxSummary
from midas.models import Direction, TaxConfig, TradeRecord


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
    twr: float
    equity_curve: list[tuple[date, float]]
    total_days: int
    train_days: int
    test_days: int
    cagr: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    efficiency_ratio: float
    strategy_stats: list[StrategyStats]
    unrealized_pnl: float
    unrealized_pnl_by_ticker: dict[str, float]
    basis_per_sell: list[float]
    risk_metrics: RiskMetrics | None = None
    risk_history: RiskHistory | None = None
    bh_equity_curve: list[tuple[date, float]] = field(default_factory=list)
    """Per-bar buy-and-hold equity, parallel to ``equity_curve``."""
    after_tax_final_value: float | None = None
    after_tax_total_return: float | None = None
    after_tax_cagr: float | None = None
    after_tax_twr: float | None = None
    after_tax_equity_curve: list[tuple[date, float]] = field(default_factory=list)
    tax_cost_ratio: float | None = None
    tax_summary: list[AnnualTaxSummary] = field(default_factory=list)
```

- [ ] **Step 4: Compute the after-tax fields in `_build_result`**

Read `src/midas/backtest.py` to find `_build_result` (~line 1011). After all gross fields are computed, add:

```python
# After-tax accounting (opt-in via TaxConfig). When tax_config is None all
# fields stay at their dataclass defaults (None / []) — no behavior change
# for users who don't configure tax rates.
after_tax_final_value: float | None = None
after_tax_total_return: float | None = None
after_tax_cagr_value: float | None = None
after_tax_twr_value: float | None = None
after_tax_curve: list[tuple[date, float]] = []
tax_cost_ratio: float | None = None
tax_summary: list[AnnualTaxSummary] = []

if self._tax_config is not None:
    end_d = trading_days[-1]
    tax_summary = compute_tax_summary(
        state.trades,
        state.basis_per_sell,
        self._tax_config,
        end_date=end_d,
    )
    after_tax_curve = compute_after_tax_curve(state.equity_curve, tax_summary, end_d)
    if after_tax_curve:
        after_tax_final_value = after_tax_curve[-1][1]
        if portfolio.available_cash > 0:
            starting = state.starting_value
            after_tax_total_return = (
                (after_tax_final_value - starting) / starting if starting > 0 else 0.0
            )
            after_tax_cagr_value = compute_cagr(starting, after_tax_final_value, total_days)
            # After-tax TWR mirrors gross TWR construction: compound each
            # sub-period's return on the after-tax curve. Approximation:
            # piecewise scaling of the gross sub-periods by (after/gross) at
            # period close — acceptable since tax payments are pointwise.
            after_tax_twr_value = (
                ((1.0 + state.twr_pre_compound) * (after_tax_final_value / state.final_pre_tax))
                - 1.0
                if state.final_pre_tax > 0
                else twr
            )
            if cagr > 0:
                tax_cost_ratio = (cagr - after_tax_cagr_value) / cagr
            else:
                tax_cost_ratio = 0.0

return BacktestResult(
    # ... all existing fields ...
    after_tax_final_value=after_tax_final_value,
    after_tax_total_return=after_tax_total_return,
    after_tax_cagr=after_tax_cagr_value,
    after_tax_twr=after_tax_twr_value,
    after_tax_equity_curve=after_tax_curve,
    tax_cost_ratio=tax_cost_ratio,
    tax_summary=tax_summary,
)
```

(If `state.twr_pre_compound` and `state.final_pre_tax` aren't already tracked, replace the after-tax TWR with `compute_cagr(starting, after_tax_final_value, total_days)`-equivalent or simply `None` and document. Keep this scope tight — the implementer may simplify TWR to "same as CAGR for post-tax" if upstream state plumbing is invasive; the spec only requires the field to be populated.)

Add `tax_config` to `BacktestEngine.__init__`:

```python
def __init__(
    self,
    *,
    allocator: Allocator,
    order_sizer: OrderSizer,
    exit_rules: list[ExitRule] | None = None,
    constraints: AllocationConstraints | None = None,
    train_pct: float = DEFAULT_TRAIN_PCT,
    enable_split: bool = True,
    log_fn: Callable[[str], None] | None = None,
    execution_mode: ExecutionMode = "next_open",
    tax_config: TaxConfig | None = None,
) -> None:
    # existing body
    self._tax_config = tax_config
```

Add necessary imports:

```python
from midas.tax import AnnualTaxSummary, compute_after_tax_curve, compute_tax_summary
from midas.metrics import compute_cagr  # if not already imported
```

- [ ] **Step 5: Thread `tax_config` from `cli.py` into `BacktestEngine`**

In `src/midas/cli.py:179-188`:

```python
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
```

- [ ] **Step 6: Update `_write_equity_curve_csv` to include `nav_after_tax` when populated**

In `src/midas/results.py`:

```python
def _write_equity_curve_csv(result: BacktestResult, path: Path) -> None:
    drawdowns = _drawdown_series(result.equity_curve)
    has_after_tax = (
        len(result.after_tax_equity_curve) == len(result.equity_curve)
        and result.after_tax_equity_curve
    )
    after_tax_by_date = (
        {dt: nav for dt, nav in result.after_tax_equity_curve} if has_after_tax else {}
    )
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["date", "nav", "drawdown"]
        if has_after_tax:
            header.append("nav_after_tax")
        writer.writerow(header)
        for (dt, nav), drawdown in zip(result.equity_curve, drawdowns, strict=True):
            row = [dt.isoformat(), round(nav, 2), round(drawdown, 6)]
            if has_after_tax:
                row.append(round(after_tax_by_date.get(dt, nav), 2))
            writer.writerow(row)
```

- [ ] **Step 7: Update `_write_summary_json` for after-tax fields and tax_summary**

```python
def _write_summary_json(result: BacktestResult, path: Path) -> None:
    # ... existing summary dict construction ...

    if result.after_tax_final_value is not None:
        summary["after_tax_final_value"] = result.after_tax_final_value
        summary["after_tax_total_return"] = round(result.after_tax_total_return or 0.0, 6)
        summary["after_tax_cagr"] = round(result.after_tax_cagr or 0.0, 6)
        if result.after_tax_twr is not None:
            summary["after_tax_twr"] = round(result.after_tax_twr, 6)
        summary["tax_cost_ratio"] = round(result.tax_cost_ratio or 0.0, 6)

    if result.tax_summary:
        summary["tax_summary"] = [
            {
                "year": s.year,
                "st_realized": round(s.st_realized, 4),
                "lt_realized": round(s.lt_realized, 4),
                "net_after_cross": round(s.net_after_cross, 4),
                "deductible_loss": round(s.deductible_loss, 4),
                "carry_forward": round(s.carry_forward, 4),
                "tax_owed": round(s.tax_owed, 4),
                "payment_date": s.payment_date.isoformat(),
            }
            for s in result.tax_summary
        ]

    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
```

- [ ] **Step 8: Run tests to verify pass**

Run: `uv run pytest tests/ -v`
Expected: PASS for new and existing tests; existing `tests/test_backtest.py` may need touch-ups in fixture construction if it instantiates `BacktestEngine` directly — add `tax_config=None` to those calls.

Run: `uv run mypy src && uv run ruff check .` — clean.

- [ ] **Step 9: Commit**

```bash
git add src/midas/results.py src/midas/backtest.py src/midas/cli.py tests/test_backtest.py
git commit -m "BacktestResult: opt-in after-tax fields, equity-curve column, summary.json

When TaxConfig is set in the strategies YAML, BacktestResult gains
after_tax_final_value, after_tax_total_return, after_tax_cagr,
after_tax_twr, after_tax_equity_curve, tax_cost_ratio, and tax_summary.
equity_curve.csv gains a parallel nav_after_tax column; summary.json
gains the after-tax block and a per-year tax_summary array.

Without TaxConfig all fields stay None/empty — no behavior change for
existing configs."
```

---

## Task 10: After-tax block in summary print + chart overlay

**Files:**
- Modify: `src/midas/output.py:_return_row` (no change) and `print_backtest_summary`
- Modify: `src/midas/charts.py:_render_equity` to overlay after-tax curve when populated
- Test: `tests/test_charts.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_charts.py`:

```python
def test_render_equity_includes_after_tax_overlay_when_populated(capsys, monkeypatch) -> None:
    """When BacktestResult.after_tax_equity_curve is non-empty, the equity chart plots both."""
    import plotext as plt
    from midas.charts import _render_equity
    from midas.results import BacktestResult
    # ... build a minimal BacktestResult with equity_curve and a parallel after_tax_equity_curve

    plot_calls: list[str] = []
    real_plot = plt.plot
    def spy(*args, **kwargs):
        plot_calls.append(kwargs.get("label", ""))
        return real_plot(*args, **kwargs)
    monkeypatch.setattr(plt, "plot", spy)

    _render_equity(_make_result_with_after_tax())
    # Two series: gross + after-tax → two plot calls inside _render_equity.
    assert any("After-Tax" in label for label in plot_calls)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_charts.py::test_render_equity_includes_after_tax_overlay_when_populated -v`
Expected: FAIL — the renderer plots only the gross curve.

- [ ] **Step 3: Update `_render_equity` in `src/midas/charts.py`**

```python
def _render_equity(result: BacktestResult) -> None:
    dates = [dt.isoformat() for dt, _ in result.equity_curve]
    equity = [value for _, value in result.equity_curve]
    _setup_single_figure()
    has_after_tax = (
        len(result.after_tax_equity_curve) == len(result.equity_curve)
        and result.after_tax_equity_curve
    )
    if has_after_tax:
        plt.plot(dates, equity, color="cyan", label="Gross", marker="braille")
        after_tax = [value for _, value in result.after_tax_equity_curve]
        plt.plot(dates, after_tax, color="magenta", label="After-Tax", marker="braille")
    else:
        plt.plot(dates, equity, color="cyan", marker="braille")
    plt.ylabel("Portfolio $")
    title = "Equity Curve (Gross vs After-Tax)" if has_after_tax else "Equity Curve"
    _flush_centered(title)
```

- [ ] **Step 4: Add an after-tax block to `print_backtest_summary` in `src/midas/output.py`**

Insert after the existing Performance section (around line 175):

```python
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
```

- [ ] **Step 5: Run tests to verify pass**

Run: `uv run pytest tests/ -v`
Expected: PASS.

Manual smoke: run a 2-year backtest with a `tax:` block in the strategies YAML and visually inspect that the after-tax block prints under Performance and the chart overlays the two curves.

- [ ] **Step 6: Commit**

```bash
git add src/midas/output.py src/midas/charts.py tests/test_charts.py
git commit -m "Print summary: after-tax block + per-year tax table; chart overlay

print_backtest_summary now prints an After-Tax Performance block under
the gross Performance block when TaxConfig is set, plus a per-year
Realized Tax table (ST/LT/Net/Tax Owed/Carry Forward).

The equity-curve chart overlays the after-tax curve in magenta when
populated; otherwise renders the gross curve alone (no behavior change
for users without TaxConfig)."
```

---

## Task 11: `midas tax-report` CLI subcommand

**Files:**
- Modify: `src/midas/cli.py` (add new click command)
- Test: `tests/test_cli_tax_report.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli_tax_report.py`:

```python
"""Smoke tests for the `midas tax-report` subcommand."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from click.testing import CliRunner

from midas.cli import cli
from midas.models import Direction, HoldingPeriod, TradeRecord
from midas.trade_log import append_trade


def _seed_log(path: Path) -> None:
    append_trade(
        path,
        TradeRecord(
            date=date(2026, 6, 1),
            ticker="AAPL",
            direction=Direction.SELL,
            shares=10.0,
            price=30.0,
            strategy_name="StopLoss",
            holding_period=HoldingPeriod.SHORT_TERM,
            purchase_date=date(2026, 1, 1),
        ),
        cost_basis=20.0,
        purchase_date=date(2026, 1, 1),
    )
    append_trade(
        path,
        TradeRecord(
            date=date(2026, 7, 1),
            ticker="AAPL",
            direction=Direction.SELL,
            shares=5.0,
            price=40.0,
            strategy_name="StopLoss",
            holding_period=HoldingPeriod.LONG_TERM,
            purchase_date=date(2024, 1, 1),
        ),
        cost_basis=15.0,
        purchase_date=date(2024, 1, 1),
    )


def test_tax_report_emits_csv_and_prints_totals(tmp_path: Path) -> None:
    log = tmp_path / "trades.csv"
    output = tmp_path / "schedule_d_2026.csv"
    _seed_log(log)
    # Inline strategies file with tax block
    strategies = tmp_path / "strategies.yaml"
    strategies.write_text(
        "strategies:\n  - name: Momentum\n    params: {window: 20}\n"
        "tax:\n  short_term_rate: 0.30\n  long_term_rate: 0.15\n"
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "tax-report",
            "--from-trades",
            str(log),
            "--strategies",
            str(strategies),
            "--year",
            "2026",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Short-Term" in result.output
    assert "Long-Term" in result.output
    assert output.exists()
    csv_text = output.read_text()
    assert "AAPL" in csv_text


def test_tax_report_no_sells_in_year_prints_message(tmp_path: Path) -> None:
    log = tmp_path / "trades.csv"
    _seed_log(log)
    strategies = tmp_path / "strategies.yaml"
    strategies.write_text(
        "strategies:\n  - name: Momentum\n    params: {window: 20}\n"
        "tax:\n  short_term_rate: 0.30\n  long_term_rate: 0.15\n"
    )
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "tax-report",
            "--from-trades",
            str(log),
            "--strategies",
            str(strategies),
            "--year",
            "2099",
        ],
    )
    assert result.exit_code == 0
    assert "No realized sales" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli_tax_report.py -v`
Expected: FAIL with `Error: No such command 'tax-report'`.

- [ ] **Step 3: Implement the subcommand in `src/midas/cli.py`**

Add the click command (placement: after the `live` command, before `optimize`):

```python
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
    from midas.trade_log import LoggedTrade, read_trades

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
        assert start is not None and end is not None  # guarded above
        start_d = _to_date(start)
        end_d = _to_date(end)
        period_label = f"{start_d.isoformat()}_{end_d.isoformat()}"

    rows: list[LoggedTrade] = [
        row for row in read_trades(trades_path)
        if row.direction == Direction.SELL and start_d <= row.date <= end_d
    ]

    if not rows:
        click.echo(f"No realized sales in {period_label}.")
        if output is not None:
            Path(output).write_text(",".join(_TAX_REPORT_COLUMNS) + "\n")
        return

    out_path = Path(output) if output is not None else Path(f"schedule_d_{period_label}.csv")

    # Convert filtered LoggedTrade rows to TradeRecord for compute_tax_summary;
    # basis_per_sell parallel list comes from each row's cost_basis.
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
```

Add the column constants and helper functions at the bottom of `cli.py`:

```python
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
    from midas.output import make_wide_table, print_centered

    table = make_wide_table(f"Schedule D — {period_label}")
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
    print_centered(table)

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
        # Footer: per-year aggregates
        for s in summary:
            writer.writerow([])
            writer.writerow([f"Year {s.year}", "", "", "", "", "", "", "", ""])
            writer.writerow(["ST realized", "", "", "", "", "", round(s.st_realized, 4), "", ""])
            writer.writerow(["LT realized", "", "", "", "", "", round(s.lt_realized, 4), "", ""])
            writer.writerow(["Net (after netting)", "", "", "", "", "", round(s.net_after_cross, 4), "", ""])
            writer.writerow(["Deductible loss", "", "", "", "", "", round(s.deductible_loss, 4), "", ""])
            writer.writerow(["Tax owed", "", "", "", "", "", round(s.tax_owed, 4), "", ""])
            writer.writerow(["Carry forward", "", "", "", "", "", round(s.carry_forward, 4), "", ""])
```

Add the missing imports at the top of `cli.py`:

```python
from datetime import datetime  # if not already
from midas.models import Direction, TradeRecord
from midas.tax import AnnualTaxSummary
from midas.trade_log import LoggedTrade
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_cli_tax_report.py -v`
Expected: PASS for both tests.

Run the full suite: `uv run pytest -v && uv run mypy src && uv run ruff check .`

- [ ] **Step 5: Commit**

```bash
git add src/midas/cli.py tests/test_cli_tax_report.py
git commit -m "Add midas tax-report subcommand

Reads <state>.trades.csv (or --from-trades <path>) and emits a
Schedule D-shaped table to stdout plus a CSV at --output. Per-row
columns: ticker, shares, purchase/sale dates, basis, proceeds, P&L,
days held, classification. Per-year footer: ST/LT/Net/Tax Owed/
Carry-Forward computed via the same netting code backtest uses, so
backtest after-tax numbers and tax-report numbers agree exactly."
```

---

## Task 12: Documentation

**Files:**
- Create: `docs/tax-reporting.md`
- Modify: `docs/architecture.md` (add a one-line pointer)

- [ ] **Step 1: Write `docs/tax-reporting.md`**

```markdown
# Tax reporting

Midas can report realized capital gains in a Schedule D-shaped format and (in
backtest mode only) compute after-tax return metrics. Tax accounting is opt-in:
without a `tax:` block in the strategies YAML, all output is unchanged from
pre-#66 behavior.

## Configuring rates

Add a `tax:` block to the strategies YAML alongside the existing top-level
keys (`min_cash_pct`, `softmax_temperature`, optional `risk:`):

\`\`\`yaml
tax:
  short_term_rate: 0.37   # decimal fraction; default top federal bracket
  long_term_rate: 0.20    # decimal fraction; default top federal LTCG
  deductible_loss_cap: 3000.0
  payment_lag_days: 105   # Dec 31 → ~Apr 15 of following year
\`\`\`

All four fields have defaults; the block is parsed leniently.

## Backtest: after-tax metrics

Running `midas backtest` with a `tax:` block in the strategies YAML adds:

- `after_tax_final_value`, `after_tax_total_return`, `after_tax_cagr`,
  `after_tax_twr` to `summary.json`.
- A `nav_after_tax` column in `equity_curve.csv` parallel to `nav`.
- A `tax_summary` array in `summary.json` with one entry per calendar year
  containing realized sales (`st_realized`, `lt_realized`, `net_after_cross`,
  `deductible_loss`, `carry_forward`, `tax_owed`, `payment_date`).
- A `tax_cost_ratio` field equal to `(cagr - after_tax_cagr) / cagr`.

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

\`\`\`
midas tax-report --strategies strategies.yaml \\
                 --portfolio portfolio.yaml --year 2026 \\
                 [--output schedule_d_2026.csv]
\`\`\`

Or against an explicit log path (e.g. backtest output):

\`\`\`
midas tax-report --strategies strategies.yaml \\
                 --from-trades output/trades.csv --year 2026
\`\`\`

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
```

- [ ] **Step 2: Add a one-line pointer to `docs/architecture.md`**

Read the existing `docs/architecture.md` and append a short reference under the most appropriate section (likely a "Reporting & output" or similar). Example line:

```markdown
- **Tax reporting:** `<state>.trades.csv` (live) and `output/trades.csv` (backtest) share one shape and feed `midas tax-report`. Backtest gains opt-in after-tax metrics via a `tax:` block in the strategies YAML. See `docs/tax-reporting.md`.
```

- [ ] **Step 3: Commit**

```bash
git add docs/tax-reporting.md docs/architecture.md
git commit -m "Docs: add tax-reporting guide; reference from architecture.md"
```

---

## Final verification

- [ ] **Full test suite + lint + types**

Run all four checks; all must pass.

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

Expected: all green.

- [ ] **Manual smoke**

Run a 2-year backtest with a `tax:` block in the strategies YAML and verify:

1. `output/trades.csv` has the new `purchase_date` column populated for both buys and sells.
2. `output/equity_curve.csv` has a `nav_after_tax` column with values strictly less than or equal to `nav` after the first tax payment date.
3. `output/summary.json` has `after_tax_*` fields and a `tax_summary` array.
4. The terminal summary prints "After-Tax Performance" and "Realized Tax" tables.
5. The equity-curve chart shows two overlaid lines.

Run a brief `midas live --dry-run` against a portfolio with one BUY signal; verify `<state>.trades.csv` is created next to the state file with a single row + header.

Run `midas tax-report --year 2026 -p portfolio.yaml -s strategies.yaml`; verify it prints a Schedule D-style table and writes `schedule_d_2026.csv`.

- [ ] **Open the PR**

```bash
git push -u origin <branch>
gh pr create --title "Tax-shaped trade log + realized-P&L report (#66)" --body "$(cat <<'EOF'
Closes #66.

## Summary

Adds (1) an append-only `<state>.trades.csv` written by `midas live` next to the state file, (2) a new `midas tax-report --year YYYY` subcommand that emits a Schedule D-shaped table + CSV from any trade log, and (3) opt-in after-tax accounting for backtests — after-tax CAGR/TWR, nav_after_tax column in equity_curve.csv, equity-chart overlay, per-year tax_summary in summary.json, and a tax-cost-ratio metric.

## Spec & plan

- `docs/specs/2026-05-08-tax-trade-log-design.md`
- `docs/plans/2026-05-08-tax-trade-log.md`

## Out of scope (deferred)

- Tax-aware optimizer objective — follow-up issue #69.
- Wash-sale detection, specific-lot/LIFO, state taxes — explicit guardrails per #66.

## Test plan

- [ ] `uv run pytest` (all green)
- [ ] `uv run ruff check . && uv run mypy src` (clean)
- [ ] Manual smoke: backtest with `tax:` block produces after-tax block + chart overlay.
- [ ] Manual smoke: `midas live --dry-run` creates `<state>.trades.csv`.
- [ ] Manual smoke: `midas tax-report --year` against a fixture log emits Schedule D table + CSV.
EOF
)"
```
