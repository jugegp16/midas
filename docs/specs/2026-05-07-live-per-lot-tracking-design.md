# Live Per-Lot Tracking Design

Closes #36.

## Goal

Make the live engine behave like the backtest engine with respect to per-lot accounting and stateful exit rules. Today live mode synthesizes a single one-element `PositionLot` per ticker on every tick, which collapses three things that work correctly in backtest:

1. **Weighted-average cost basis after multiple buys** — the YAML's static `cost_basis` doesn't update when the operator buys more, so exit-rule clamps fire against a stale basis.
2. **Per-ticker high-water mark** — re-derived as `max(cost_basis, current_price)` on every tick, which makes `TrailingStop`'s drawdown-from-HWM check structurally always-zero or never-fire.
3. **Short-term vs long-term holding-period classification on sells** — meaningless with one synthetic lot.

Persistence also unblocks the CPPI overlay (`drawdown_penalty`/`drawdown_floor`), which is currently inert in live because the running peak isn't carried across runs.

## Scope

**In:**

1. A single mutable state file (sidecar to `portfolio.yaml`, YAML, atomic writes) that is the runtime source of truth for positions, cash, HWM, and peak equity.
2. `LiveEngine` loads or seeds the state file at startup, mutates an in-memory representation as it generates orders (alerts == assumed fills) and updates HWM / peak / infusion, writes back at the end of each tick.
3. Drift detection on subsequent loads (warn if `portfolio.yaml` aggregates disagree with state).
4. Removal of the `TrailingStop`-disabled and `drawdown_penalty`-inert warnings — both conditions are now resolved.

**Out (deferred to follow-up issues):**

- **Per-lot HWM and lot-aware exit clamping** (#36 follow-up). Backtest doesn't have per-lot HWM today either; per-ticker HWM is enough to match. Lot-aware clamping is an `ExitRule` API change, not a persistence change.
- **`record-fill` CLI** for slippage / manual override. This iteration assumes every emitted alert is filled at the alert price; operators who need slippage handling can hand-edit the state file as an escape valve. Real fill recording is its own UX surface.
- **`RestrictionTracker` round-trip-days persistence**. The 30-day no-buy-after-sell window currently resets on `midas live` restart. Adjacent state but a separate concern; keep the tracker in-memory for v1.
- **Tax-shaped P&L reporting** (tracked separately under #66).

## Source-of-Truth Principle

After first seed, `portfolio.yaml` is read-only seed config and the state file owns all mutable runtime state. Edits to the YAML's `shares`, `cost_basis`, `available_cash`, or `cash_infusion.next_date` after seed have no effect on live behavior; the engine warns on first load if it detects drift but trusts the state file.

The YAML keeps owning everything that isn't mutable runtime state: tickers, strategies, allocation constraints, restrictions, infusion *amount* and *frequency* (only `next_date` moves to state).

## State File

**Location:** sidecar to `portfolio.yaml` by default. `<portfolio>.yaml` → `<portfolio>.state.yaml` next to it. An optional `state_file:` top-level field in `portfolio.yaml` overrides the path (use case: read-only `portfolio.yaml` checked into git, mutable state in a writable runtime dir).

**Format:** YAML, atomic writes (temp file + `os.replace`). One canonical file per portfolio, continuously updated; no append-only history, no rotating backups.

**Schema:**

```yaml
schema_version: 1
last_updated: 2026-05-07T15:23:00Z

available_cash: 4823.50

cash_infusion:
  next_date: 2026-05-15
  # amount + frequency stay in portfolio.yaml (immutable policy);
  # next_date moves here because advance() mutates it.

high_water_marks:
  PLTR: 24.18
  NVDA: 142.50

peak_equity: 18420.00     # for CPPI overlay; current_drawdown = 1 - equity/peak

lots:
  PLTR:
    - shares: 100
      purchase_date: null      # seeded; operator can hand-edit
      cost_basis: 10.00
    - shares: 50
      purchase_date: 2026-04-12
      cost_basis: 22.50
  NVDA:
    - shares: 100
      purchase_date: null
      cost_basis: 49.76
```

**Schema versioning:** `schema_version: 1` is reserved for forward compatibility. Any other value refuses to load with a clear "unsupported state version" error. No migration code in v1.

**Hand-edits:** allowed but unusual. Expected use cases are correcting a seed lot's `purchase_date` for accurate ST/LT classification, or adjusting cash to reflect external transfers the engine doesn't know about. Not a primary workflow.

## Lifecycle

### First run (state file does not exist)

1. Resolve state path: `portfolio.yaml`'s `state_file:` field if set, else sidecar.
2. Seed from `portfolio.yaml`:
   - One `PositionLot(shares=Holding.shares, cost_basis=Holding.cost_basis, purchase_date=None)` per ticker with `shares > 0`.
   - `available_cash` from YAML.
   - `cash_infusion.next_date` from YAML.
   - Empty `high_water_marks` dict.
   - `peak_equity = None` (filled on first tick).
3. Atomic-write the state file.
4. Log one info line: `"Seeded state at <path> from <portfolio.yaml>"`.
5. Continue the tick loop normally.

### Subsequent runs

1. Load state file.
2. Ignore `Holding.shares`, `Holding.cost_basis`, `available_cash`, and `cash_infusion.next_date` from `portfolio.yaml`.
3. Drift check:
   - For each ticker in state, sum `lots[ticker].shares` and compare to `Holding.shares`. If different, warn naming both values.
   - Compare state `available_cash` to YAML's. If different, warn.
   - Don't refuse to run; trust the state file. Warning is the signal.

### Per tick

1. **Update HWM:** `hwm[ticker] = max(hwm.get(ticker, 0.0), close)` for each held ticker.
2. **Update peak equity:** `peak_equity = max(peak_equity or 0.0, current_equity)` where `current_equity = available_cash + Σ shares * close`.
3. **Allocator + exit rules + sizer** run as today, but:
   - `cost_basis` for `clamp_target` is the weighted-avg over the in-memory lot list (matches backtest's `_aggregate_cost_basis`).
   - `high_water_mark` for `clamp_target` is the persisted per-ticker HWM (not the synthesized `max(cost_basis, current_price)`).
4. **Apply assumed fills** to the in-memory state for each emitted alert:
   - `apply_buy(state, ticker, shares, price, today)` appends a `PositionLot(shares, today, price)` and decrements cash by `shares * price`.
   - `apply_sell(state, ticker, shares, price)` consumes lots FIFO, increments cash by `shares * price`, returns `(st_realized_pnl, lt_realized_pnl)` matching backtest's classification.
5. **Advance cash infusion:** if `today >= cash_infusion.next_date`, add `amount` to cash and call `advance()` (matches backtest behavior).
6. **Atomic write:** serialize to temp file, `os.replace` over the canonical path. Update `last_updated`.

The write happens every tick. HWM and peak almost always advance on a non-flat day, so skipping writes when "nothing changed" would rarely apply and isn't worth the complexity. Optimizing further is a v1.1 concern if I/O becomes a bottleneck.

## Crash Safety

- **Mid-write crash:** atomic write (`tempfile + os.replace`) guarantees the canonical file is either the old version or the new version, never partial.
- **Mid-tick crash after alert emission, before write:** the next tick re-reads the previous state. The alert was emitted but not recorded — same divergence the operator sees if they ignore an alert. Acceptable for the "assume immediate execution" model.
- **State file corruption / parse failure:** refuse to start with a clear error pointing at the file. Don't auto-recover or re-seed; that would silently overwrite real positions.
- **Schema version mismatch:** refuse to load.

## Migration & Edge Cases

**Tickers added to `portfolio.yaml` after seed.** `Holding` exists with `shares=0` (or non-zero, doesn't matter — state ignores it) and no entry in `state.lots`. Engine treats it as a fresh ticker; allocator can buy into it; first buy creates the lot list for that ticker.

**Tickers removed from `portfolio.yaml` after seed but still held in state.** Refuse to start with a clear error naming the ticker and the share count in state. Removing a ticker while still holding shares is almost certainly a config mistake; failing loudly prevents silent forks of state.

**`state_file:` field present in YAML but the named file doesn't exist.** Treat as first run; seed there.

**Two `midas live` processes against the same state file.** Out of scope. The engine is not designed for concurrent execution. (Future hardening could use `flock` or a lockfile, but not in v1.)

## Engine Integration

### New module: `src/midas/live_state.py` (~150 lines)

```python
@dataclass
class LiveState:
    available_cash: float
    cash_infusion_next_date: date | None
    high_water_marks: dict[str, float]
    peak_equity: float | None
    lots: dict[str, list[PositionLot]]

def load_or_seed(portfolio: PortfolioConfig, state_path: Path) -> LiveState: ...
def save_atomic(state: LiveState, path: Path) -> None: ...
def apply_buy(state: LiveState, ticker: str, shares: float, price: float, day: date) -> None: ...
def apply_sell(
    state: LiveState, ticker: str, shares: float, price: float, day: date
) -> tuple[float, float]:
    """Consume lots FIFO; returns (st_realized_pnl, lt_realized_pnl)."""
```

`apply_buy` / `apply_sell` reuse the FIFO logic that lives in `backtest.py:893-1004` today. Extract that logic into shared helpers in `live_state.py`; `backtest.py` imports from the new module. The refactor is mechanical — no behavior change for backtest.

### `LiveEngine` (`src/midas/live.py`)

- Constructor takes a `state_path: Path`. Loads or seeds at construction.
- The CPPI and `TrailingStop` warnings (`live.py:73-93`) are removed — both conditions are now resolved.
- `_tick` reads HWM / weighted-avg cost basis from `self._state` instead of synthesizing them.
- After alerts are emitted, `apply_buy` / `apply_sell` mutate `self._state`, then `save_atomic` flushes.
- The duplicate-suppression set `_last_order_keys` stays in-memory: re-emitting on restart is correct so the operator doesn't miss a still-relevant signal.

### `config.py`

- `load_portfolio` learns to parse an optional `state_file:` top-level field. No change to the `portfolio:` block schema.

### `cli.py`

- The `live` command resolves `state_path = port.state_file or portfolio_path.with_suffix('.state.yaml')` and passes it to `LiveEngine`.

## Testing

### Unit (`tests/test_live_state.py`)

- `load_or_seed` creates a state file from `portfolio.yaml` when none exists; seed lots have `purchase_date=None`.
- `load_or_seed` reads an existing state file and ignores aggregate drift in `Holding.shares`, with a warning.
- `apply_buy` appends a lot, decrements cash by `shares * price`.
- `apply_sell` consumes FIFO, returns ST/LT P&L split matching backtest's classification (boundary at 365 days).
- `save_atomic` produces a parseable file; partial-write failure (mocked) leaves the canonical file intact.
- `schema_version` mismatch refuses to load.
- Strict mode: ticker removed from `portfolio.yaml` while lots remain → load raises with a clear error.

### Integration (`tests/test_live_engine.py`)

- HWM persists across two synthetic ticks: tick 1 records peak, tick 2 with lower price → `TrailingStop` fires.
- Two assumed buys at different prices produce a correct weighted-avg cost basis on the next tick's clamp.
- Sell consumes lots FIFO; the next tick's allocator sees reduced shares.
- Cash decrements/increments correctly across a buy + sell sequence.
- `peak_equity` advances on a winning tick; on a subsequent losing tick that produces a non-zero `current_drawdown`, the CPPI exposure scale falls below 1.0 (matching the formula in `risk.apply_drawdown_overlay`).

### Backtest parity (`tests/test_live_backtest_parity.py`)

A deterministic synthetic price series fed through both the backtest engine and a `LiveEngine` driven by a fake `DataProvider`. Assert lot lists, HWMs, peak equity, available cash, and emitted-order shapes match bar-for-bar (modulo backtest's `lag=1` execution offset, which the test harness replicates by emitting on the previous bar's close).

### No regressions

The refactor of FIFO helpers from `backtest.py` into `live_state.py` is mechanical — backtest imports from the new module, behavior unchanged. All existing tests pass.

## YAML Interface

```yaml
# portfolio.yaml — unchanged except for the optional state_file field
state_file: ./run/portfolio.state.yaml   # optional; default is sidecar

portfolio:
  - ticker: PLTR
    shares: 100
    cost_basis: 10
  # ...

available_cash: 5000.00
# (cash_infusion, trading_restrictions unchanged)
```

The `state_file:` field is optional and the `portfolio:` block schema does not change. Existing portfolios continue to work; first `midas live` invocation seeds the state file alongside the portfolio.
