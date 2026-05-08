# Live Per-Lot Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist per-lot positions, per-ticker HWM, peak equity, and available cash across `midas live` runs so the live engine matches backtest behavior. Today live mode synthesizes one lot per ticker and re-derives HWM each tick, structurally disabling `TrailingStop`, weighted-avg cost basis after multiple buys, ST/LT classification, and CPPI overlay.

**Architecture:** New `src/midas/live_state.py` module owns a `LiveState` dataclass and YAML serialization. `LiveEngine` loads or seeds at startup, mutates the in-memory `LiveState` as it generates orders (alerts == assumed fills) and updates HWM / peak / infusion, then atomic-writes back at the end of each tick. The FIFO sell-consumption logic that lives in `backtest.py` today is extracted into shared helpers in `live_state.py`; backtest imports from the new module with no behavior change.

**Tech Stack:** Python 3.14, dataclasses, PyYAML, pytest. No new deps.

**Spec:** `docs/specs/2026-05-07-live-per-lot-tracking-design.md`

---

## File Structure

**Created:**
- `src/midas/live_state.py` — `LiveState` dataclass, `load_or_seed`, `save_atomic`, `apply_buy`, `apply_sell`, plus the extracted FIFO primitives (`aggregate_cost_basis`, `consume_lots_fifo`).
- `tests/test_live_state.py` — unit tests for serialization, seeding, drift detection, FIFO helpers.
- `tests/test_live_engine.py` — integration tests for `LiveEngine` driving a `LiveState` across multiple synthetic ticks.
- `tests/test_live_backtest_parity.py` — bar-for-bar parity between backtest and live on a deterministic price series.

**Modified:**
- `src/midas/models.py` — add `state_file: Path | None = None` to `PortfolioConfig`.
- `src/midas/config.py` — parse optional top-level `state_file:` key.
- `src/midas/backtest.py` — replace `_aggregate_cost_basis` and the inline FIFO loop in `_execute` with imports from `live_state.py`. No behavior change.
- `src/midas/live.py` — constructor takes `state_path`, loads/seeds; `_tick` reads HWM and weighted-avg basis from `self._state`; emits `apply_buy` / `apply_sell` per alert; `save_atomic` per tick. Drop `TrailingStop` and `drawdown_penalty` inert-warnings.
- `src/midas/cli.py` — `live` command resolves `state_path` and passes to `LiveEngine`.

---

## Task 1: `LiveState` dataclass + YAML round-trip

**Files:**
- Create: `src/midas/live_state.py`
- Test: `tests/test_live_state.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_live_state.py
"""Unit tests for live state persistence."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from midas.live_state import LiveState, load_state, save_atomic
from midas.models import PositionLot


def test_save_load_round_trip(tmp_path: Path) -> None:
    state = LiveState(
        available_cash=4823.50,
        cash_infusion_next_date=date(2026, 5, 15),
        high_water_marks={"PLTR": 24.18, "NVDA": 142.5},
        peak_equity=18420.0,
        lots={
            "PLTR": [
                PositionLot(shares=100.0, purchase_date=None, cost_basis=10.0),
                PositionLot(shares=50.0, purchase_date=date(2026, 4, 12), cost_basis=22.5),
            ],
            "NVDA": [PositionLot(shares=100.0, purchase_date=None, cost_basis=49.76)],
        },
    )
    path = tmp_path / "portfolio.state.yaml"
    save_atomic(state, path)
    loaded = load_state(path)
    assert loaded == state
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_live_state.py::test_save_load_round_trip -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'midas.live_state'`.

- [ ] **Step 3: Implement `LiveState`, `save_atomic`, `load_state`**

```python
# src/midas/live_state.py
"""Persistent runtime state for the live engine.

After first seed, this state file is the runtime source of truth for
positions, available cash, per-ticker HWM, peak equity, and the cash-
infusion ``next_date``. The portfolio YAML continues to own everything
else (tickers, strategies, allocation constraints, infusion amount/
frequency, restrictions); see docs/specs/2026-05-07-live-per-lot-
tracking-design.md.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from midas.models import PositionLot

SCHEMA_VERSION = 1


class StateFileError(ValueError):
    """Raised on schema version mismatch, parse failure, or invalid state."""


@dataclass
class LiveState:
    """Mutable runtime state persisted between live ticks."""

    available_cash: float
    cash_infusion_next_date: date | None
    high_water_marks: dict[str, float] = field(default_factory=dict)
    peak_equity: float | None = None
    lots: dict[str, list[PositionLot]] = field(default_factory=dict)


def save_atomic(state: LiveState, path: Path) -> None:
    """Serialize *state* to *path* atomically (tempfile + os.replace)."""
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "last_updated": datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat(),
        "available_cash": state.available_cash,
        "cash_infusion": (
            {"next_date": state.cash_infusion_next_date}
            if state.cash_infusion_next_date is not None
            else None
        ),
        "high_water_marks": dict(state.high_water_marks),
        "peak_equity": state.peak_equity,
        "lots": {
            ticker: [
                {
                    "shares": lot.shares,
                    "purchase_date": lot.purchase_date,
                    "cost_basis": lot.cost_basis,
                }
                for lot in lots
            ]
            for ticker, lots in state.lots.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def load_state(path: Path) -> LiveState:
    """Load state from *path*. Raises ``StateFileError`` on invalid input."""
    try:
        with open(path) as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        msg = f"failed to parse state file at {path}: {exc}"
        raise StateFileError(msg) from exc
    if not isinstance(raw, dict):
        msg = f"state file at {path} is not a YAML mapping"
        raise StateFileError(msg)
    version = raw.get("schema_version")
    if version != SCHEMA_VERSION:
        msg = f"unsupported state version {version!r} in {path}; expected {SCHEMA_VERSION}"
        raise StateFileError(msg)

    infusion = raw.get("cash_infusion")
    next_date: date | None = None
    if infusion is not None:
        next_date = infusion.get("next_date")
        if isinstance(next_date, str):
            next_date = date.fromisoformat(next_date)

    lots: dict[str, list[PositionLot]] = {}
    for ticker, entries in (raw.get("lots") or {}).items():
        lots[ticker] = [
            PositionLot(
                shares=float(entry["shares"]),
                purchase_date=entry.get("purchase_date"),
                cost_basis=float(entry["cost_basis"]),
            )
            for entry in entries
        ]

    return LiveState(
        available_cash=float(raw["available_cash"]),
        cash_infusion_next_date=next_date,
        high_water_marks={k: float(v) for k, v in (raw.get("high_water_marks") or {}).items()},
        peak_equity=raw.get("peak_equity"),
        lots=lots,
    )
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/test_live_state.py::test_save_load_round_trip -v
```

Expected: PASS.

- [ ] **Step 5: Add tests for schema-version mismatch, parse failure, atomic-write safety**

```python
# tests/test_live_state.py — append
import pytest

from midas.live_state import StateFileError


def test_load_rejects_unknown_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "state.yaml"
    path.write_text("schema_version: 99\navailable_cash: 0\n")
    with pytest.raises(StateFileError, match="unsupported state version"):
        load_state(path)


def test_load_rejects_unparseable_yaml(tmp_path: Path) -> None:
    path = tmp_path / "state.yaml"
    path.write_text(":\n - this isn't\n  valid yaml: ::\n")
    with pytest.raises(StateFileError):
        load_state(path)


def test_save_atomic_leaves_canonical_intact_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = LiveState(available_cash=100.0, cash_infusion_next_date=None)
    path = tmp_path / "state.yaml"
    save_atomic(state, path)
    canonical_before = path.read_text()

    def boom(*args: object, **kwargs: object) -> None:
        msg = "simulated rename failure"
        raise OSError(msg)

    monkeypatch.setattr("os.replace", boom)
    with pytest.raises(OSError, match="simulated rename failure"):
        save_atomic(LiveState(available_cash=999.99, cash_infusion_next_date=None), path)
    assert path.read_text() == canonical_before
```

- [ ] **Step 6: Run all live_state tests**

```
uv run pytest tests/test_live_state.py -v
```

Expected: all PASS.

- [ ] **Step 7: Lint/format/type check**

```
uv run ruff check src/midas/live_state.py tests/test_live_state.py
uv run ruff format src/midas/live_state.py tests/test_live_state.py
uv run mypy src/midas/live_state.py
```

Expected: all clean.

- [ ] **Step 8: Commit**

```bash
git add src/midas/live_state.py tests/test_live_state.py
git commit -m "Add LiveState dataclass with atomic YAML round-trip (#36)"
```

---

## Task 2: Seed `LiveState` from `PortfolioConfig` on first run

**Files:**
- Modify: `src/midas/live_state.py`
- Modify: `tests/test_live_state.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_live_state.py — append
from midas.live_state import load_or_seed
from midas.models import CashInfusion, Holding, PortfolioConfig


def test_load_or_seed_creates_state_from_portfolio(tmp_path: Path) -> None:
    portfolio = PortfolioConfig(
        holdings=[
            Holding(ticker="AAPL", shares=100.0, cost_basis=150.0),
            Holding(ticker="NVDA", shares=50.0, cost_basis=49.76),
        ],
        available_cash=5000.0,
        cash_infusion=CashInfusion(amount=1500.0, next_date=date(2026, 5, 15), frequency="biweekly"),
    )
    path = tmp_path / "portfolio.state.yaml"
    assert not path.exists()

    state = load_or_seed(portfolio, path)

    assert path.exists()  # seed wrote the file
    assert state.available_cash == 5000.0
    assert state.cash_infusion_next_date == date(2026, 5, 15)
    assert state.peak_equity is None
    assert state.high_water_marks == {}
    assert state.lots == {
        "AAPL": [PositionLot(shares=100.0, purchase_date=None, cost_basis=150.0)],
        "NVDA": [PositionLot(shares=50.0, purchase_date=None, cost_basis=49.76)],
    }


def test_load_or_seed_skips_holdings_with_zero_shares(tmp_path: Path) -> None:
    portfolio = PortfolioConfig(
        holdings=[
            Holding(ticker="AAPL", shares=100.0, cost_basis=150.0),
            Holding(ticker="MSFT", shares=0.0, cost_basis=None),
        ],
        available_cash=1000.0,
    )
    path = tmp_path / "state.yaml"
    state = load_or_seed(portfolio, path)
    assert "AAPL" in state.lots
    assert "MSFT" not in state.lots
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_live_state.py::test_load_or_seed_creates_state_from_portfolio tests/test_live_state.py::test_load_or_seed_skips_holdings_with_zero_shares -v
```

Expected: FAIL with `ImportError: cannot import name 'load_or_seed'`.

- [ ] **Step 3: Implement `load_or_seed`'s seed branch**

```python
# src/midas/live_state.py — append
import logging

from midas.models import PortfolioConfig

logger = logging.getLogger(__name__)


def load_or_seed(portfolio: PortfolioConfig, state_path: Path) -> LiveState:
    """Load state from *state_path*, or seed it from *portfolio* if missing.

    The seed branch creates one ``PositionLot`` per ticker with ``shares > 0``,
    using the YAML's ``cost_basis`` and ``purchase_date=None`` (we don't know
    when the operator originally bought; affects ST/LT classification — they
    can hand-edit later if precision matters).
    """
    if state_path.exists():
        return load_state(state_path)

    lots: dict[str, list[PositionLot]] = {}
    for holding in portfolio.holdings:
        if holding.shares <= 0:
            continue
        lots[holding.ticker] = [
            PositionLot(
                shares=holding.shares,
                purchase_date=None,
                cost_basis=holding.cost_basis if holding.cost_basis is not None else 0.0,
            )
        ]

    state = LiveState(
        available_cash=portfolio.available_cash,
        cash_infusion_next_date=portfolio.cash_infusion.next_date if portfolio.cash_infusion else None,
        high_water_marks={},
        peak_equity=None,
        lots=lots,
    )
    save_atomic(state, state_path)
    logger.info("Seeded state at %s from portfolio config", state_path)
    return state
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/test_live_state.py::test_load_or_seed_creates_state_from_portfolio tests/test_live_state.py::test_load_or_seed_skips_holdings_with_zero_shares -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/midas/live_state.py tests/test_live_state.py
git commit -m "Seed LiveState from PortfolioConfig on first run (#36)"
```

---

## Task 3: Drift detection on subsequent loads

**Files:**
- Modify: `src/midas/live_state.py`
- Modify: `tests/test_live_state.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_live_state.py — append
def test_load_or_seed_warns_on_share_drift(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=100.0, cost_basis=150.0)],
        available_cash=1000.0,
    )
    path = tmp_path / "state.yaml"
    load_or_seed(portfolio, path)  # first call seeds

    portfolio_drifted = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=200.0, cost_basis=150.0)],  # state still has 100
        available_cash=1000.0,
    )
    with caplog.at_level("WARNING"):
        state = load_or_seed(portfolio_drifted, path)
    assert state.lots["AAPL"][0].shares == 100.0  # state file wins
    assert any("AAPL" in record.message and "200" in record.message for record in caplog.records)


def test_load_or_seed_warns_on_cash_drift(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=100.0, cost_basis=150.0)],
        available_cash=1000.0,
    )
    path = tmp_path / "state.yaml"
    load_or_seed(portfolio, path)

    portfolio_drifted = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=100.0, cost_basis=150.0)],
        available_cash=9999.99,
    )
    with caplog.at_level("WARNING"):
        state = load_or_seed(portfolio_drifted, path)
    assert state.available_cash == 1000.0  # state file wins
    assert any("available_cash" in record.message for record in caplog.records)


def test_load_or_seed_raises_when_held_ticker_removed_from_portfolio(tmp_path: Path) -> None:
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=100.0, cost_basis=150.0)],
        available_cash=1000.0,
    )
    path = tmp_path / "state.yaml"
    load_or_seed(portfolio, path)

    portfolio_without = PortfolioConfig(
        holdings=[],  # AAPL removed from portfolio while still held in state
        available_cash=1000.0,
    )
    with pytest.raises(StateFileError, match="AAPL.*100"):
        load_or_seed(portfolio_without, path)
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_live_state.py -k "drift or removed" -v
```

Expected: FAIL — drift detection not yet implemented.

- [ ] **Step 3: Add drift detection to `load_or_seed`**

```python
# src/midas/live_state.py — modify load_or_seed to add the load branch
def load_or_seed(portfolio: PortfolioConfig, state_path: Path) -> LiveState:
    """Load state from *state_path*, or seed it from *portfolio* if missing.

    On subsequent loads, warns if the YAML aggregates disagree with state
    (the state file wins). Refuses to start if the portfolio no longer lists
    a ticker for which lots are still held — that's almost certainly a config
    mistake.
    """
    if not state_path.exists():
        # ... existing seed code unchanged ...
        return state

    state = load_state(state_path)
    _check_for_drift(state, portfolio)
    return state


def _check_for_drift(state: LiveState, portfolio: PortfolioConfig) -> None:
    """Warn on aggregate drift; raise if a held ticker has no Holding."""
    yaml_tickers = {holding.ticker for holding in portfolio.holdings}
    for ticker, lots in state.lots.items():
        held = sum(lot.shares for lot in lots)
        if held <= 0:
            continue
        if ticker not in yaml_tickers:
            msg = (
                f"ticker {ticker} is held in state ({held} shares) but not "
                f"listed in portfolio.yaml; refusing to start. Either restore "
                f"the holding entry or close out the position in state."
            )
            raise StateFileError(msg)
        holding = portfolio.get_holding(ticker)
        assert holding is not None  # guarded by yaml_tickers membership
        if holding.shares != held:
            logger.warning(
                "share drift: portfolio.yaml has %s=%s but state has %s; trusting state",
                ticker,
                holding.shares,
                held,
            )
    if portfolio.available_cash != state.available_cash:
        logger.warning(
            "available_cash drift: portfolio.yaml has %s but state has %s; trusting state",
            portfolio.available_cash,
            state.available_cash,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/test_live_state.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/midas/live_state.py tests/test_live_state.py
git commit -m "Detect aggregate drift between portfolio.yaml and state (#36)"
```

---

## Task 4: Extract FIFO helpers from `backtest.py` into `live_state.py`

**Files:**
- Modify: `src/midas/backtest.py`
- Modify: `src/midas/live_state.py`
- Test: `tests/test_live_state.py`

- [ ] **Step 1: Write the failing test for the extracted primitives**

```python
# tests/test_live_state.py — append
from midas.live_state import aggregate_cost_basis, consume_lots_fifo


def test_aggregate_cost_basis_share_weighted() -> None:
    lots = [
        PositionLot(shares=100.0, purchase_date=None, cost_basis=10.0),
        PositionLot(shares=50.0, purchase_date=date(2026, 4, 12), cost_basis=22.0),
    ]
    # (100 * 10 + 50 * 22) / 150 = 14.0
    assert aggregate_cost_basis(lots) == pytest.approx(14.0)


def test_aggregate_cost_basis_empty_returns_zero() -> None:
    assert aggregate_cost_basis([]) == 0.0


def test_consume_lots_fifo_st_only() -> None:
    today = date(2026, 5, 7)
    lots = [PositionLot(shares=100.0, purchase_date=date(2026, 4, 1), cost_basis=10.0)]
    breakdown = consume_lots_fifo(lots, shares=40.0, day=today)
    assert breakdown.st_shares == 40.0
    assert breakdown.st_basis == pytest.approx(10.0)
    assert breakdown.lt_shares == 0.0
    assert lots[0].shares == 60.0  # mutated in place


def test_consume_lots_fifo_straddles_st_lt_boundary() -> None:
    today = date(2026, 5, 7)
    # First lot is >365 days old (LT); second is recent (ST).
    lots = [
        PositionLot(shares=30.0, purchase_date=date(2025, 4, 1), cost_basis=10.0),
        PositionLot(shares=20.0, purchase_date=date(2026, 4, 1), cost_basis=20.0),
    ]
    breakdown = consume_lots_fifo(lots, shares=40.0, day=today)
    assert breakdown.lt_shares == 30.0
    assert breakdown.lt_basis == pytest.approx(10.0)
    assert breakdown.st_shares == 10.0
    assert breakdown.st_basis == pytest.approx(20.0)
    assert lots == [PositionLot(shares=10.0, purchase_date=date(2026, 4, 1), cost_basis=20.0)]


def test_consume_lots_fifo_unknown_purchase_date_is_short_term() -> None:
    today = date(2026, 5, 7)
    lots = [PositionLot(shares=100.0, purchase_date=None, cost_basis=10.0)]
    breakdown = consume_lots_fifo(lots, shares=50.0, day=today)
    assert breakdown.st_shares == 50.0
    assert breakdown.lt_shares == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_live_state.py -k "aggregate_cost_basis or consume_lots_fifo" -v
```

Expected: FAIL — symbols not yet exported.

- [ ] **Step 3: Add the primitives to `live_state.py`**

```python
# src/midas/live_state.py — append
from collections.abc import Sequence


@dataclass(frozen=True)
class SellBreakdown:
    """Result of consuming lots FIFO for a sell.

    Each pair (shares, share-weighted basis) is reported separately for the
    short-term (held <365 days, or unknown purchase date) and long-term
    (held >=365 days) buckets. Either bucket can be zero.
    """

    st_shares: float
    st_basis: float
    lt_shares: float
    lt_basis: float


def aggregate_cost_basis(lots: Sequence[PositionLot]) -> float:
    """Share-weighted average cost basis across all open lots, or 0.0 if empty."""
    total_shares = sum(lot.shares for lot in lots)
    if total_shares <= 0:
        return 0.0
    return sum(lot.shares * lot.cost_basis for lot in lots) / total_shares


def consume_lots_fifo(lots: list[PositionLot], shares: float, day: date) -> SellBreakdown:
    """Consume *shares* from *lots* in FIFO order, mutating in place.

    Lots whose ``purchase_date`` is at least 365 days before *day* contribute
    to the long-term bucket; everything else (including ``purchase_date=None``)
    goes to the short-term bucket. Each bucket reports a share-weighted
    cost basis over the consumed slices.
    """
    if shares <= 0 or not lots:
        return SellBreakdown(0.0, 0.0, 0.0, 0.0)

    st_shares = 0.0
    st_weighted = 0.0
    lt_shares = 0.0
    lt_weighted = 0.0
    remaining = shares
    while remaining > 0 and lots:
        lot = lots[0]
        take = min(lot.shares, remaining)
        is_long_term = lot.purchase_date is not None and (day - lot.purchase_date).days >= 365
        if is_long_term:
            lt_shares += take
            lt_weighted += take * lot.cost_basis
        else:
            st_shares += take
            st_weighted += take * lot.cost_basis

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
    return SellBreakdown(st_shares=st_shares, st_basis=st_basis, lt_shares=lt_shares, lt_basis=lt_basis)
```

- [ ] **Step 4: Run new tests to verify they pass**

```
uv run pytest tests/test_live_state.py -v
```

Expected: all PASS.

- [ ] **Step 5: Replace backtest's inline FIFO logic with imports**

Locate and replace the static methods in `src/midas/backtest.py`:

- Delete `_aggregate_cost_basis` (lines 884-890) and replace its single call site at line 743 with `aggregate_cost_basis(state.lots.get(ticker, []))`.
- Delete `_fifo_consumed_basis` (lines 892-912). Its only call site is in `_update_buy_attribution` — keep using it there but route through `consume_lots_fifo` if appropriate, or leave the helper inlined as a backtest-internal computation (it does not mutate lots and is used for attribution only, so leave it as a static method on the backtest engine).
- Replace the inline lot-consumption while-loop in `_execute` (lines 957-982) with:

```python
breakdown = consume_lots_fifo(lots, order.shares, day)
```

Then derive the existing `st_shares`, `st_weighted_basis`, `lt_shares`, `lt_weighted_basis` from `breakdown` (multiply basis back out by shares to keep the rest of `_execute` unchanged):

```python
st_shares = breakdown.st_shares
st_weighted_basis = breakdown.st_basis * breakdown.st_shares
lt_shares = breakdown.lt_shares
lt_weighted_basis = breakdown.lt_basis * breakdown.lt_shares
```

Add the import at the top of `backtest.py`:

```python
from midas.live_state import aggregate_cost_basis, consume_lots_fifo
```

- [ ] **Step 6: Run the full backtest suite to verify no regressions**

```
uv run pytest tests/test_backtest.py tests/test_lookahead_regression.py tests/test_integration.py -v
```

Expected: all PASS — the refactor is mechanical, behavior unchanged.

- [ ] **Step 7: Commit**

```bash
git add src/midas/live_state.py src/midas/backtest.py tests/test_live_state.py
git commit -m "Extract FIFO helpers from backtest into live_state (#36)"
```

---

## Task 5: `apply_buy` and `apply_sell`

**Files:**
- Modify: `src/midas/live_state.py`
- Modify: `tests/test_live_state.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_live_state.py — append
from midas.live_state import apply_buy, apply_sell


def test_apply_buy_appends_lot_and_decrements_cash() -> None:
    state = LiveState(available_cash=1000.0, cash_infusion_next_date=None)
    apply_buy(state, "AAPL", shares=10.0, price=150.0, day=date(2026, 5, 7))
    assert state.available_cash == pytest.approx(1000.0 - 10.0 * 150.0)
    assert state.lots["AAPL"] == [
        PositionLot(shares=10.0, purchase_date=date(2026, 5, 7), cost_basis=150.0)
    ]


def test_apply_buy_appends_to_existing_lots() -> None:
    state = LiveState(
        available_cash=1000.0,
        cash_infusion_next_date=None,
        lots={"AAPL": [PositionLot(shares=5.0, purchase_date=date(2026, 4, 1), cost_basis=140.0)]},
    )
    apply_buy(state, "AAPL", shares=10.0, price=150.0, day=date(2026, 5, 7))
    assert len(state.lots["AAPL"]) == 2
    assert state.available_cash == pytest.approx(1000.0 - 10.0 * 150.0)


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
    st_pnl, lt_pnl = apply_sell(state, "AAPL", shares=40.0, price=25.0, day=date(2026, 5, 7))
    assert state.available_cash == pytest.approx(40.0 * 25.0)
    assert state.lots["AAPL"] == [
        PositionLot(shares=10.0, purchase_date=date(2026, 4, 1), cost_basis=20.0)
    ]
    assert lt_pnl == pytest.approx(30.0 * (25.0 - 10.0))
    assert st_pnl == pytest.approx(10.0 * (25.0 - 20.0))


def test_apply_sell_drops_empty_ticker_entry() -> None:
    state = LiveState(
        available_cash=0.0,
        cash_infusion_next_date=None,
        lots={"AAPL": [PositionLot(shares=10.0, purchase_date=None, cost_basis=10.0)]},
    )
    apply_sell(state, "AAPL", shares=10.0, price=20.0, day=date(2026, 5, 7))
    assert "AAPL" not in state.lots  # cleared
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_live_state.py -k "apply_buy or apply_sell" -v
```

Expected: FAIL — symbols not exported.

- [ ] **Step 3: Implement `apply_buy` and `apply_sell`**

```python
# src/midas/live_state.py — append
def apply_buy(state: LiveState, ticker: str, shares: float, price: float, day: date) -> None:
    """Append a new lot for *ticker* and decrement cash by ``shares * price``.

    Mutates *state* in place. Use after the live engine has emitted a buy
    alert and the operator is assumed to have filled at *price*.
    """
    state.lots.setdefault(ticker, []).append(
        PositionLot(shares=shares, purchase_date=day, cost_basis=price)
    )
    state.available_cash -= shares * price


def apply_sell(
    state: LiveState, ticker: str, shares: float, price: float, day: date
) -> tuple[float, float]:
    """Consume *shares* of *ticker* FIFO and increment cash by ``shares * price``.

    Mutates *state* in place. Returns ``(st_realized_pnl, lt_realized_pnl)``
    matching backtest's ST/LT classification (lots with purchase_date >=365
    days before *day* count as long-term; everything else, including
    ``purchase_date=None``, counts as short-term).
    """
    lots = state.lots.get(ticker, [])
    breakdown = consume_lots_fifo(lots, shares, day)
    state.available_cash += shares * price
    if not lots:
        state.lots.pop(ticker, None)
    st_pnl = breakdown.st_shares * (price - breakdown.st_basis) if breakdown.st_shares > 0 else 0.0
    lt_pnl = breakdown.lt_shares * (price - breakdown.lt_basis) if breakdown.lt_shares > 0 else 0.0
    return st_pnl, lt_pnl
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/test_live_state.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/midas/live_state.py tests/test_live_state.py
git commit -m "Add apply_buy and apply_sell mutators on LiveState (#36)"
```

---

## Task 6: Add `state_file` field to `PortfolioConfig` + config parsing

**Files:**
- Modify: `src/midas/models.py`
- Modify: `src/midas/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py — append at end of TestLoadPortfolio (or its equivalent)
def test_load_portfolio_parses_state_file_field(tmp_path: Path) -> None:
    yaml_path = tmp_path / "portfolio.yaml"
    yaml_path.write_text(
        "state_file: ./run/portfolio.state.yaml\n"
        "portfolio:\n"
        "  - ticker: AAPL\n"
        "    shares: 100\n"
        "    cost_basis: 150\n"
        "available_cash: 1000\n"
    )
    portfolio = load_portfolio(yaml_path)
    assert portfolio.state_file == Path("./run/portfolio.state.yaml")


def test_load_portfolio_state_file_field_optional(tmp_path: Path) -> None:
    yaml_path = tmp_path / "portfolio.yaml"
    yaml_path.write_text(
        "portfolio:\n"
        "  - ticker: AAPL\n"
        "    shares: 100\n"
        "    cost_basis: 150\n"
        "available_cash: 1000\n"
    )
    portfolio = load_portfolio(yaml_path)
    assert portfolio.state_file is None
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_config.py -k "state_file" -v
```

Expected: FAIL with `AttributeError: 'PortfolioConfig' object has no attribute 'state_file'`.

- [ ] **Step 3: Add the field to `PortfolioConfig`**

```python
# src/midas/models.py — modify
from pathlib import Path

@dataclass
class PortfolioConfig:
    holdings: list[Holding]
    available_cash: float
    cash_infusion: CashInfusion | None = None
    trading_restrictions: TradingRestrictions | None = None
    state_file: Path | None = None

    def __post_init__(self) -> None:
        self._by_ticker = {holding.ticker: holding for holding in self.holdings}

    def get_holding(self, ticker: str) -> Holding | None:
        return self._by_ticker.get(ticker)
```

- [ ] **Step 4: Parse the field in `load_portfolio`**

```python
# src/midas/config.py — modify load_portfolio body, add after existing parsing:
state_file_raw = raw.get("state_file")
state_file = Path(state_file_raw) if state_file_raw is not None else None

portfolio = PortfolioConfig(
    holdings=holdings,
    available_cash=float(raw["available_cash"]),
    cash_infusion=infusion,
    trading_restrictions=restrictions,
    state_file=state_file,
)
```

- [ ] **Step 5: Run tests to verify they pass**

```
uv run pytest tests/test_config.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/midas/models.py src/midas/config.py tests/test_config.py
git commit -m "Add state_file field to PortfolioConfig (#36)"
```

---

## Task 7: Wire `LiveEngine` to `LiveState` (load/seed at construction)

**Files:**
- Modify: `src/midas/live.py`
- Test: `tests/test_live_engine.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_live_engine.py — new file
"""Integration tests for LiveEngine driving LiveState across ticks."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from midas.allocator import Allocator
from midas.live import LiveEngine
from midas.live_state import load_state
from midas.models import (
    AllocationConstraints,
    CashInfusion,
    Holding,
    PortfolioConfig,
    StrategyConfig,
)
from midas.order_sizer import OrderSizer


def _make_provider(prices: dict[str, list[float]], dates: list[date]) -> MagicMock:
    """Build a fake DataProvider returning an OHLCV frame for each ticker."""
    provider = MagicMock()

    def get_history(ticker: str, start: date, end: date) -> pd.DataFrame:
        closes = prices[ticker]
        return pd.DataFrame(
            {
                "open": closes,
                "high": closes,
                "low": closes,
                "close": closes,
                "volume": [1000.0] * len(closes),
            },
            index=pd.DatetimeIndex([pd.Timestamp(d) for d in dates]),
        )

    provider.get_history.side_effect = get_history
    return provider


@pytest.fixture
def basic_portfolio() -> PortfolioConfig:
    return PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=10.0, cost_basis=150.0)],
        available_cash=1000.0,
    )


def test_live_engine_seeds_state_on_first_construction(basic_portfolio: PortfolioConfig, tmp_path: Path) -> None:
    state_path = tmp_path / "portfolio.state.yaml"
    assert not state_path.exists()

    allocator = Allocator(strategies=[], constraints=AllocationConstraints())
    sizer = OrderSizer()
    provider = _make_provider({"AAPL": [150.0]}, [date(2026, 5, 7)])

    LiveEngine(
        portfolio=basic_portfolio,
        allocator=allocator,
        order_sizer=sizer,
        provider=provider,
        state_path=state_path,
    )

    assert state_path.exists()
    state = load_state(state_path)
    assert state.available_cash == 1000.0
    assert "AAPL" in state.lots
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_live_engine.py::test_live_engine_seeds_state_on_first_construction -v
```

Expected: FAIL — `LiveEngine` doesn't accept `state_path` yet.

- [ ] **Step 3: Modify `LiveEngine.__init__` to load/seed state**

```python
# src/midas/live.py — modify imports
from pathlib import Path

from midas.live_state import LiveState, apply_buy, apply_sell, load_or_seed, save_atomic, aggregate_cost_basis

# src/midas/live.py — modify __init__ signature and body
class LiveEngine:
    def __init__(
        self,
        portfolio: PortfolioConfig,
        allocator: Allocator,
        order_sizer: OrderSizer,
        provider: DataProvider,
        state_path: Path,
        exit_rules: list[ExitRule] | None = None,
        constraints: AllocationConstraints | None = None,
        poll_interval: int = 60,
        dry_run: bool = False,
        history_days: int | None = None,
    ) -> None:
        self._portfolio = portfolio
        self._allocator = allocator
        self._order_sizer = order_sizer
        self._exit_rules = exit_rules or []
        self._constraints = constraints or AllocationConstraints()
        self._provider = provider
        self._poll_interval = poll_interval
        self._dry_run = dry_run
        self._state_path = state_path
        self._state: LiveState = load_or_seed(portfolio, state_path)
        # ... existing history_days, _last_order_keys, restriction_tracker setup ...
```

Also: **delete** the two warning blocks at `live.py:65-93` — both `TrailingStop` and CPPI conditions are now resolved.

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/test_live_engine.py::test_live_engine_seeds_state_on_first_construction -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/midas/live.py tests/test_live_engine.py
git commit -m "LiveEngine loads or seeds LiveState at construction (#36)"
```

---

## Task 8: Use persisted HWM and weighted-avg basis in `_tick`

**Files:**
- Modify: `src/midas/live.py`
- Modify: `tests/test_live_engine.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_live_engine.py — append
from midas.strategies.trailing_stop import TrailingStop


def test_persisted_hwm_lets_trailing_stop_fire_on_drawdown(tmp_path: Path) -> None:
    """Two ticks: tick 1 records peak at 200, tick 2 at 150 → 25% drawdown
    fires TrailingStop with trail_pct=0.10."""
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=10.0, cost_basis=100.0)],
        available_cash=0.0,
    )
    state_path = tmp_path / "state.yaml"
    allocator = Allocator(strategies=[], constraints=AllocationConstraints())
    sizer = OrderSizer()
    trailing = TrailingStop(trail_pct=0.10)

    # Tick 1: price at 200 (above cost basis → HWM advances).
    provider = _make_provider({"AAPL": [200.0]}, [date(2026, 5, 7)])
    engine = LiveEngine(
        portfolio=portfolio,
        allocator=allocator,
        order_sizer=sizer,
        provider=provider,
        state_path=state_path,
        exit_rules=[trailing],
    )
    engine._tick(["AAPL"])

    state = load_state(state_path)
    assert state.high_water_marks["AAPL"] == 200.0

    # Tick 2: price at 150 → 25% drawdown from HWM, exceeds 10% threshold.
    # Same engine instance with state already loaded; provider returns 150 now.
    provider2 = _make_provider({"AAPL": [150.0]}, [date(2026, 5, 8)])
    engine._provider = provider2
    engine._tick(["AAPL"])

    # The exit rule should have clamped the target to 0; sell order should
    # have been emitted; state should reflect the assumed sell.
    state2 = load_state(state_path)
    assert state2.lots.get("AAPL", []) == []  # sold out
    assert state2.available_cash > 0  # sell proceeds added
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_live_engine.py::test_persisted_hwm_lets_trailing_stop_fire_on_drawdown -v
```

Expected: FAIL — HWM not yet read from state.

- [ ] **Step 3: Read HWM and weighted-avg basis from `self._state` in `_tick`**

In `src/midas/live.py`, modify the exit-rule loop in `_tick` to:
- Update `self._state.high_water_marks[ticker]` from the current price *before* the exit-rule loop.
- Use `aggregate_cost_basis(self._state.lots.get(ticker, []))` instead of `holding.cost_basis`.
- Use `self._state.high_water_marks[ticker]` instead of `max(cost_basis, current_prices[ticker])`.

```python
# src/midas/live.py — replace the exit-rule cost_basis/hwm derivation
# (lines around 184-198 in the original)
for rule in self._exit_rules:
    for ticker in active_tickers:
        if positions.get(ticker, 0.0) <= 0:
            continue
        proposed = clamped_targets.get(ticker, 0.0)
        if proposed <= 0:
            continue
        cost_basis = aggregate_cost_basis(self._state.lots.get(ticker, []))
        hwm = self._state.high_water_marks.get(ticker, current_prices[ticker])
        clamped = rule.clamp_target(ticker, proposed, price_history[ticker], cost_basis, hwm)
        # ... rest unchanged ...
```

Also update `positions` to be derived from state lots, not from `holding.shares`:

```python
positions = {
    ticker: sum(lot.shares for lot in self._state.lots.get(ticker, [])) for ticker in active_tickers
}
```

And update HWM at the top of `_tick` (before the exit loop):

```python
# After current_prices is built, before the exit-rule loop:
for ticker in active_tickers:
    close = current_prices[ticker]
    self._state.high_water_marks[ticker] = max(
        self._state.high_water_marks.get(ticker, close), close
    )
```

- [ ] **Step 4: Run test to verify it passes** (will need Task 10's apply_sell wiring; for now, assert just up through HWM update)

Pause and split: this test asserts behavior that depends on Tasks 8 + 10. Reduce the assertion scope to just HWM persistence for now:

```python
# Replace the post-tick-2 assertions with:
state2 = load_state(state_path)
assert state2.high_water_marks["AAPL"] == 200.0  # HWM did not regress on lower price
```

The full sell-execution assertion lives in Task 10.

```
uv run pytest tests/test_live_engine.py::test_persisted_hwm_lets_trailing_stop_fire_on_drawdown -v
```

Expected: PASS (HWM behavior verified; sell-execution check is deferred).

- [ ] **Step 5: Commit**

```bash
git add src/midas/live.py tests/test_live_engine.py
git commit -m "LiveEngine reads HWM and weighted-avg basis from LiveState (#36)"
```

---

## Task 9: Per-tick mutations — apply assumed fills, advance peak, advance infusion, save

**Files:**
- Modify: `src/midas/live.py`
- Modify: `tests/test_live_engine.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_live_engine.py — append
def test_buy_alert_appends_lot_and_decrements_state_cash(tmp_path: Path) -> None:
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=0.0, cost_basis=None)],
        available_cash=2000.0,
    )
    state_path = tmp_path / "state.yaml"
    # Use a strategy stub that returns a strong buy signal for AAPL.
    # ... build allocator that produces a target weight for AAPL ...
    # (See "Test Helpers" section for build_strong_buy_engine helper)
    engine = build_engine_with_buy_signal(portfolio, state_path, ticker="AAPL", price=100.0)
    engine._tick(["AAPL"])

    state = load_state(state_path)
    # First tick should buy AAPL; lot list now has one entry.
    assert "AAPL" in state.lots
    assert state.available_cash < 2000.0


def test_peak_equity_advances_and_persists(tmp_path: Path) -> None:
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=10.0, cost_basis=100.0)],
        available_cash=0.0,
    )
    state_path = tmp_path / "state.yaml"
    provider = _make_provider({"AAPL": [200.0]}, [date(2026, 5, 7)])
    engine = LiveEngine(
        portfolio=portfolio,
        allocator=Allocator(strategies=[], constraints=AllocationConstraints()),
        order_sizer=OrderSizer(),
        provider=provider,
        state_path=state_path,
    )
    engine._tick(["AAPL"])

    state = load_state(state_path)
    assert state.peak_equity == pytest.approx(10 * 200.0)


def test_cash_infusion_advances_when_due(tmp_path: Path) -> None:
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=10.0, cost_basis=100.0)],
        available_cash=500.0,
        cash_infusion=CashInfusion(amount=1500.0, next_date=date(2026, 5, 1), frequency="biweekly"),
    )
    state_path = tmp_path / "state.yaml"
    provider = _make_provider({"AAPL": [100.0]}, [date(2026, 5, 7)])  # past next_date
    engine = LiveEngine(
        portfolio=portfolio,
        allocator=Allocator(strategies=[], constraints=AllocationConstraints()),
        order_sizer=OrderSizer(),
        provider=provider,
        state_path=state_path,
    )
    # Patch date.today() so the engine's "today" is 2026-05-07.
    # Use a fixture or freezegun if available; otherwise inject via a kwarg.
    engine._tick(["AAPL"])

    state = load_state(state_path)
    assert state.available_cash == pytest.approx(500.0 + 1500.0)
    assert state.cash_infusion_next_date == date(2026, 5, 15)  # advanced biweekly
```

> **Note on `build_engine_with_buy_signal`:** create a minimal helper at the top of `test_live_engine.py` that builds an `Allocator` with a single stub `EntrySignal` returning a fixed score of 1.0 for the given ticker. Reuse the pattern from `tests/test_allocator.py` if a similar fixture exists.

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_live_engine.py -v
```

Expected: FAIL — apply_buy/apply_sell not wired into `_tick`.

- [ ] **Step 3: Wire mutations into `_tick`**

In `src/midas/live.py`, after the alert-emission loop, add:

```python
# Apply assumed fills to the in-memory state.
for order in filtered:
    if order.shares <= 0:
        continue
    if order.direction == Direction.BUY:
        apply_buy(self._state, order.ticker, order.shares, order.price, today)
    else:
        apply_sell(self._state, order.ticker, order.shares, order.price, today)

# Update peak equity from the current portfolio value.
current_equity = self._state.available_cash + sum(
    sum(lot.shares for lot in self._state.lots.get(ticker, [])) * current_prices[ticker]
    for ticker in active_tickers
)
self._state.peak_equity = max(self._state.peak_equity or 0.0, current_equity)

# Advance cash infusion if due.
infusion = self._portfolio.cash_infusion
if (
    infusion is not None
    and self._state.cash_infusion_next_date is not None
    and today >= self._state.cash_infusion_next_date
):
    self._state.available_cash += infusion.amount
    infusion.next_date = self._state.cash_infusion_next_date  # for advance() to work on the right value
    infusion.advance()
    self._state.cash_infusion_next_date = infusion.next_date

# Persist state at the end of the tick.
save_atomic(self._state, self._state_path)
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/test_live_engine.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/midas/live.py tests/test_live_engine.py
git commit -m "LiveEngine applies assumed fills and persists state per tick (#36)"
```

---

## Task 10: CLI wiring — resolve `state_path` in `live` command

**Files:**
- Modify: `src/midas/cli.py`
- Test: extend an existing CLI test or add `tests/test_cli_live.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_live.py — new file
"""End-to-end smoke test for the live CLI command's state-path resolution."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from midas.cli import cli


def test_live_command_uses_sidecar_state_path_by_default(tmp_path: Path) -> None:
    portfolio_yaml = tmp_path / "portfolio.yaml"
    portfolio_yaml.write_text(
        "portfolio:\n"
        "  - ticker: AAPL\n"
        "    shares: 10\n"
        "    cost_basis: 150\n"
        "available_cash: 1000\n"
    )

    captured: dict[str, Path] = {}

    class _StubEngine:
        def __init__(self, *args: object, state_path: Path, **kwargs: object) -> None:
            captured["state_path"] = state_path

        def run(self) -> None:
            return

    with patch("midas.live.LiveEngine", _StubEngine):
        runner = CliRunner()
        result = runner.invoke(cli, ["live", "-p", str(portfolio_yaml), "--dry-run"])
        assert result.exit_code == 0, result.output
    assert captured["state_path"] == portfolio_yaml.with_suffix(".state.yaml")


def test_live_command_honors_state_file_field(tmp_path: Path) -> None:
    state_target = tmp_path / "run" / "explicit.state.yaml"
    portfolio_yaml = tmp_path / "portfolio.yaml"
    portfolio_yaml.write_text(
        f"state_file: {state_target}\n"
        "portfolio:\n"
        "  - ticker: AAPL\n"
        "    shares: 10\n"
        "    cost_basis: 150\n"
        "available_cash: 1000\n"
    )

    captured: dict[str, Path] = {}

    class _StubEngine:
        def __init__(self, *args: object, state_path: Path, **kwargs: object) -> None:
            captured["state_path"] = state_path

        def run(self) -> None:
            return

    with patch("midas.live.LiveEngine", _StubEngine):
        runner = CliRunner()
        result = runner.invoke(cli, ["live", "-p", str(portfolio_yaml), "--dry-run"])
        assert result.exit_code == 0, result.output
    assert captured["state_path"] == state_target
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_cli_live.py -v
```

Expected: FAIL — `LiveEngine` invocation in `cli.py` does not pass `state_path` yet.

- [ ] **Step 3: Resolve and pass `state_path` in the CLI**

```python
# src/midas/cli.py — modify the live() command body
def live(portfolio: str, strategies: str | None, interval: int, dry_run: bool) -> None:
    """Run live analysis with real-time price polling."""
    from midas.live import LiveEngine

    portfolio_path = Path(portfolio)
    port = load_portfolio(portfolio_path)
    state_path = port.state_file or portfolio_path.with_suffix(".state.yaml")
    # ... existing strategy/component setup unchanged ...

    engine = LiveEngine(
        portfolio=port,
        allocator=allocator,
        order_sizer=order_sizer,
        provider=provider,
        state_path=state_path,
        exit_rules=exit_rules,
        constraints=constraints,
        poll_interval=interval,
        dry_run=dry_run,
    )
    engine.run()
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/test_cli_live.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/midas/cli.py tests/test_cli_live.py
git commit -m "CLI live command resolves state_path from portfolio config (#36)"
```

---

## Task 11: Backtest parity test — same prices, same state evolution

**Files:**
- Test: `tests/test_live_backtest_parity.py`

- [ ] **Step 1: Write the parity test**

```python
# tests/test_live_backtest_parity.py — new file
"""Bar-for-bar parity between the backtest and live engines.

Feeds a deterministic synthetic price series through both engines and asserts
that lot lists, HWMs, peak equity, and available cash agree at the end of
the run. The live engine reads from a fake DataProvider that yields the
same OHLCV frame the backtest sees.

NOTE: Backtest executes orders at the next bar's open (lag=1), so the live
harness emits at the previous bar's close to match. This is the only
intentional offset.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from midas.allocator import Allocator
from midas.backtest import Backtest
from midas.live import LiveEngine
from midas.live_state import load_state
from midas.models import (
    AllocationConstraints,
    Holding,
    PortfolioConfig,
    StrategyConfig,
)
from midas.order_sizer import OrderSizer


@pytest.fixture
def deterministic_price_series() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(42)
    n_bars = 60
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n_bars)))
    index = pd.bdate_range(start="2026-01-02", periods=n_bars)
    return {
        "AAPL": pd.DataFrame(
            {
                "open": closes,
                "high": closes,
                "low": closes,
                "close": closes,
                "volume": [1000.0] * n_bars,
            },
            index=index,
        )
    }


def test_live_and_backtest_agree_on_state_trajectory(
    deterministic_price_series: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    """Run backtest and live on the same series; state at the end matches."""
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=10.0, cost_basis=90.0)],
        available_cash=1000.0,
    )

    # ... build identical Allocator / strategies / OrderSizer for both engines ...
    # ... run Backtest end-to-end, capture final state.lots / state.cash / state.peak_value ...
    # ... drive LiveEngine tick-by-tick over the same price series with a fake DataProvider ...
    # ... at end, load_state(state_path) and compare:
    #       - lot lists per ticker (shares, purchase_date, cost_basis)
    #       - high_water_marks
    #       - peak_equity
    #       - available_cash

    # Concrete assertion shape:
    # final_live = load_state(tmp_path / "p.state.yaml")
    # assert final_live.lots == backtest_final.lots
    # assert final_live.high_water_marks == backtest_final.high_water_marks
    # assert final_live.peak_equity == pytest.approx(backtest_final.peak_value)
    # assert final_live.available_cash == pytest.approx(backtest_final.cash)
    pytest.skip("parity harness scaffold — concrete strategies/sizer setup TBD in implementation")
```

- [ ] **Step 2: Implement the harness (replace the `pytest.skip` with a concrete run)**

The harness builds:
- A `Backtest` driven by `deterministic_price_series` with one `EntrySignal` strategy, no exit rules. Run end-to-end via `Backtest.run` (or whatever the project's entry point is).
- A `LiveEngine` with a fake `DataProvider` that, on each `_tick` invocation for date `d`, returns the slice of `deterministic_price_series` up through the bar before `d` (matching backtest's lag=1 execution).
- Iterate over the trading days; for each day, monkey-patch `date.today()` (or pass a `today` argument; if the engine doesn't accept one, add it as part of this task) and call `engine._tick`.

Compare the live state file to the backtest's final `_SimState` field-by-field as documented in Step 1.

- [ ] **Step 3: Run the parity test**

```
uv run pytest tests/test_live_backtest_parity.py -v
```

Expected: PASS — the two engines see the same prices and produce the same state.

- [ ] **Step 4: Commit**

```bash
git add tests/test_live_backtest_parity.py
git commit -m "Add backtest/live parity test for state evolution (#36)"
```

---

## Task 12: Documentation updates

**Files:**
- Modify: `docs/architecture.md`
- Modify: `docs/strategies.md` (the live-mode warning section, if any)

- [ ] **Step 1: Update `docs/architecture.md`**

Add a section describing the new state model:

```markdown
## Live State Persistence

The live engine persists runtime state to a YAML sidecar (`portfolio.state.yaml`
by default, or the path under the optional `state_file:` field in `portfolio.yaml`).
Schema is owned by `src/midas/live_state.py` and documented in
`docs/specs/2026-05-07-live-per-lot-tracking-design.md`.

After first seed, `portfolio.yaml` is read-only seed config. The state file owns
positions (per-lot), available cash, per-ticker high-water marks, peak equity,
and the cash-infusion `next_date`. Edits to `portfolio.yaml`'s aggregate
`shares`, `cost_basis`, or `available_cash` after seed have no effect; the engine
warns on drift but trusts the state file.

The engine writes the state file atomically (tempfile + os.replace) at the end
of every tick, on the assumption that emitted alerts are filled at the alert
price. Operators who need to reflect slippage or manual overrides can hand-edit
the state file (it is plain YAML).
```

- [ ] **Step 2: Update `docs/strategies.md`**

If the doc currently warns that `TrailingStop` does not fire in live mode (it does, per the original issue), remove or update that warning.

- [ ] **Step 3: Run the full test suite as a regression sweep**

```
uv run pytest -q
uv run ruff check .
uv run mypy src
```

Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add docs/architecture.md docs/strategies.md
git commit -m "Document live state persistence (#36)"
```

---

## Self-Review

**Spec coverage:**
- Goal (live behaves like backtest re HWM, weighted-avg basis, ST/LT classification, CPPI peak): Tasks 7, 8, 9, 11.
- State file location & schema: Task 1.
- Source-of-truth principle (portfolio.yaml seeds, state owns thereafter): Tasks 2, 3, 7.
- Lifecycle — first run seeds, subsequent runs detect drift: Tasks 2, 3.
- Per-tick HWM/peak/infusion updates and atomic save: Tasks 8, 9.
- Crash safety (atomic writes, parse failure, schema mismatch): Task 1.
- Migration / strict ticker-removed: Task 3.
- Engine integration (`live_state.py` module, `LiveEngine` rework, FIFO refactor, `config.py`, `cli.py`): Tasks 1, 4, 5, 6, 7, 8, 9, 10.
- Removal of `TrailingStop` and `drawdown_penalty` warnings: Task 7 (deletion in same change as wiring).
- Testing — unit, integration, parity: Tasks 1-5 (unit), 7-9 (integration), 11 (parity).

**Placeholder scan:** the parity test in Task 11 has a partial-skeleton implementation (Step 2 says "implement the harness" rather than showing the full code). This is a deliberate scope decision — the harness shape depends on small details of `Backtest.run`'s API that are easier to lock in when the engineer has Tasks 1-10 in hand. Acceptable, but flagging.

**Type consistency:** `apply_sell` returns `tuple[float, float]` (st_pnl, lt_pnl) consistently. `consume_lots_fifo` returns `SellBreakdown`. `LiveState` field names match across all tasks. `state_path` consistently typed as `Path`.

**Engineer-may-be-out-of-order assumption:** every task that depends on a previous task names the symbol(s) it imports from `midas.live_state`, so reading Task 7 in isolation, the engineer can still write the import line correctly.
