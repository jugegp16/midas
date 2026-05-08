"""Integration tests for LiveEngine driving LiveState across ticks."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from midas.allocator import Allocator
from midas.live import LiveEngine
from midas.live_state import LiveState, StateFileError, aggregate_cost_basis, load_state, save_atomic
from midas.models import (
    AllocationConstraints,
    CashInfusion,
    Direction,
    Holding,
    Order,
    OrderContext,
    PortfolioConfig,
    PositionLot,
)
from midas.order_sizer import OrderSizer

ProviderFactory = Callable[[dict[str, list[float]], list[date]], MagicMock]


@pytest.fixture
def basic_portfolio() -> PortfolioConfig:
    return PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=10.0, cost_basis=150.0)],
        available_cash=1000.0,
    )


def test_live_engine_seeds_state_on_first_construction(
    basic_portfolio: PortfolioConfig, tmp_path: Path, make_provider: ProviderFactory
) -> None:
    state_path = tmp_path / "portfolio.state.yaml"
    assert not state_path.exists()

    allocator = Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1)
    sizer = OrderSizer()
    provider = make_provider({"AAPL": [150.0]}, [date(2026, 5, 7)])

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


def test_tick_advances_in_memory_hwm(tmp_path: Path, make_provider: ProviderFactory) -> None:
    """First tick should advance per-ticker HWM in self._state to the latest close."""
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=10.0, cost_basis=100.0)],
        available_cash=0.0,
    )
    state_path = tmp_path / "state.yaml"
    allocator = Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1)
    sizer = OrderSizer()
    provider = make_provider({"AAPL": [200.0]}, [date(2026, 5, 7)])

    engine = LiveEngine(
        portfolio=portfolio,
        allocator=allocator,
        order_sizer=sizer,
        provider=provider,
        state_path=state_path,
    )
    engine._tick(["AAPL"])

    assert engine._state.high_water_marks["AAPL"] == 200.0


def test_tick_uses_weighted_avg_basis_from_state(tmp_path: Path, make_provider: ProviderFactory) -> None:
    """If the operator has two lots at different prices, the exit-rule loop
    should see the share-weighted average basis, not the YAML's static value."""
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=20.0, cost_basis=999.0)],  # YAML lies
        available_cash=0.0,
    )
    state_path = tmp_path / "state.yaml"
    # Pre-populate state with two real lots so the engine sees the right basis.
    seeded = LiveState(
        available_cash=0.0,
        cash_infusion_next_date=None,
        lots={
            "AAPL": [
                PositionLot(shares=10.0, purchase_date=None, cost_basis=100.0),
                PositionLot(shares=10.0, purchase_date=None, cost_basis=200.0),
            ]
        },
    )
    save_atomic(seeded, state_path)

    allocator = Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1)
    sizer = OrderSizer()
    provider = make_provider({"AAPL": [150.0]}, [date(2026, 5, 7)])
    engine = LiveEngine(
        portfolio=portfolio,
        allocator=allocator,
        order_sizer=sizer,
        provider=provider,
        state_path=state_path,
    )

    # Verify positions/basis as the engine would compute them inside _tick.
    assert aggregate_cost_basis(engine._state.lots["AAPL"]) == 150.0
    # And after a tick, HWM should advance to 150.0 (the close).
    engine._tick(["AAPL"])
    assert engine._state.high_water_marks["AAPL"] == 150.0


def test_peak_equity_advances_and_persists(tmp_path: Path, make_provider: ProviderFactory) -> None:
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=10.0, cost_basis=100.0)],
        available_cash=0.0,
    )
    state_path = tmp_path / "state.yaml"
    provider = make_provider({"AAPL": [200.0]}, [date(2026, 5, 7)])
    engine = LiveEngine(
        portfolio=portfolio,
        allocator=Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1),
        order_sizer=OrderSizer(),
        provider=provider,
        state_path=state_path,
    )
    engine._tick(["AAPL"])

    state = load_state(state_path)
    assert state.peak_equity == pytest.approx(10 * 200.0)


def test_state_is_persisted_to_disk_each_tick(tmp_path: Path, make_provider: ProviderFactory) -> None:
    """Even on a no-op tick (no orders), HWM and peak advance get saved."""
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=10.0, cost_basis=100.0)],
        available_cash=0.0,
    )
    state_path = tmp_path / "state.yaml"
    provider = make_provider({"AAPL": [150.0]}, [date(2026, 5, 7)])
    engine = LiveEngine(
        portfolio=portfolio,
        allocator=Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1),
        order_sizer=OrderSizer(),
        provider=provider,
        state_path=state_path,
    )
    engine._tick(["AAPL"])

    on_disk = load_state(state_path)
    assert on_disk.high_water_marks["AAPL"] == 150.0
    assert on_disk.peak_equity == pytest.approx(10 * 150.0)


def test_cash_infusion_advances_when_due(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_provider: ProviderFactory
) -> None:
    """If today >= cash_infusion.next_date, cash increases by amount and
    next_date advances."""
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=10.0, cost_basis=100.0)],
        available_cash=500.0,
        cash_infusion=CashInfusion(amount=1500.0, next_date=date(2026, 5, 1), frequency="biweekly"),
    )
    state_path = tmp_path / "state.yaml"
    provider = make_provider({"AAPL": [100.0]}, [date(2026, 5, 7)])
    engine = LiveEngine(
        portfolio=portfolio,
        allocator=Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1),
        order_sizer=OrderSizer(),
        provider=provider,
        state_path=state_path,
    )

    # Patch date.today() to 2026-05-07 (past next_date 2026-05-01).
    class _FakeDate:
        @staticmethod
        def today() -> date:
            return date(2026, 5, 7)

    monkeypatch.setattr("midas.live.date", _FakeDate)
    engine._tick(["AAPL"])

    state = load_state(state_path)
    assert state.available_cash == pytest.approx(500.0 + 1500.0)
    assert state.cash_infusion_next_date == date(2026, 5, 15)  # +14 days for biweekly


def test_alert_cash_display_does_not_double_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_provider: ProviderFactory
) -> None:
    """``apply_buy``/``apply_sell`` mutate ``state.available_cash`` in place,
    so the per-alert running subtotal must seed from the *pre-fill* baseline.
    Otherwise a single $100 BUY against $1000 cash would print "$800" instead
    of "$900" — both cash mutation and the printed delta apply.
    """
    portfolio = PortfolioConfig(
        holdings=[],  # no held positions; only a buy will fire
        available_cash=1000.0,
    )
    state_path = tmp_path / "state.yaml"
    provider = make_provider({"AAPL": [100.0]}, [date(2026, 5, 7)])
    engine = LiveEngine(
        portfolio=portfolio,
        allocator=Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1),
        order_sizer=OrderSizer(),
        provider=provider,
        state_path=state_path,
    )

    # Inject a synthetic $100 BUY directly into the order sizer's buy pass; this
    # bypasses allocator/strategy plumbing while exercising the real apply-fills
    # and alert-emission paths in ``_tick``.
    fake_buy = Order(
        ticker="AAPL",
        direction=Direction.BUY,
        shares=1.0,
        price=100.0,
        estimated_value=100.0,
        context=OrderContext(
            contributions={"fake": 1.0},
            blended_score=1.0,
            target_weight=1.0,
            current_weight=0.0,
            reason="test",
            source="fake",
        ),
    )
    monkeypatch.setattr(engine._order_sizer, "size_buys", lambda *a, **kw: [fake_buy])

    captured: list[float] = []

    def capture_alert(order: Order, remaining_cash: float, *_args: object, **_kw: object) -> None:
        captured.append(remaining_cash)

    monkeypatch.setattr("midas.live.print_alert", capture_alert)

    engine._tick(["AAPL"])

    # Pre-fill cash $1000 minus $100 buy = $900. If the bug returns,
    # state.available_cash (already $900 after apply_buy) would be debited again
    # to $800.
    assert captured == [pytest.approx(900.0)]
    assert engine._state.available_cash == pytest.approx(900.0)


def test_tick_passes_current_drawdown_to_allocator(tmp_path: Path, make_provider: ProviderFactory) -> None:
    """``_tick`` must compute ``(peak_equity - total_value) / peak_equity`` from
    persisted state and pass it as ``current_drawdown`` to the allocator. Without
    this, the CPPI overlay is inert in live regardless of how far below peak the
    portfolio sits — the headline scope-1 fix this PR claims to deliver. (Earlier
    iterations let it default to 0.0, leaving CPPI a no-op even when peak_equity
    persistence was present.)
    """
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=10.0, cost_basis=100.0)],
        available_cash=0.0,
    )
    state_path = tmp_path / "state.yaml"
    # Pre-seed state with peak_equity=$2000 so the current $800 portfolio is
    # 60% below peak — non-trivial drawdown that should flow through.
    seeded = LiveState(
        available_cash=0.0,
        cash_infusion_next_date=None,
        peak_equity=2000.0,
        lots={"AAPL": [PositionLot(shares=10.0, purchase_date=None, cost_basis=100.0)]},
    )
    save_atomic(seeded, state_path)

    provider = make_provider({"AAPL": [80.0]}, [date(2026, 5, 7)])
    captured_kwargs: dict[str, object] = {}
    allocator = Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1)

    original_allocate = allocator.allocate

    def capturing_allocate(*args: object, **kwargs: object) -> object:
        captured_kwargs.update(kwargs)
        return original_allocate(*args, **kwargs)  # type: ignore[arg-type]

    allocator.allocate = capturing_allocate  # type: ignore[method-assign]

    engine = LiveEngine(
        portfolio=portfolio,
        allocator=allocator,
        order_sizer=OrderSizer(),
        provider=provider,
        state_path=state_path,
    )
    engine._tick(["AAPL"])

    # total_value = $0 cash + 10 shares * $80 = $800. peak = $2000.
    # current_drawdown = (2000 - 800) / 2000 = 0.6
    assert captured_kwargs["current_drawdown"] == pytest.approx(0.6)


def test_peak_equity_survives_engine_restart_and_drives_drawdown(
    tmp_path: Path, make_provider: ProviderFactory
) -> None:
    """Tick A advances peak via the engine's own logic, then a fresh engine
    constructed from the same state path reads peak from disk and the next
    tick's current_drawdown reflects the persisted value.

    Without persistence, the second engine would re-seed peak from the YAML's
    seed_equity and compute drawdown=0.0 on the same total_value that was a
    real drawdown vs the persisted peak. Same dead-code-shape as the original
    C1 finding (peak_equity wired but never restored across processes).
    """
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=10.0, cost_basis=100.0)],
        available_cash=0.0,
    )
    state_path = tmp_path / "state.yaml"

    # Tick A: price at 200 → equity at $2000, peak ratchets to $2000.
    provider_a = make_provider({"AAPL": [200.0]}, [date(2026, 5, 7)])
    engine_a = LiveEngine(
        portfolio=portfolio,
        allocator=Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1),
        order_sizer=OrderSizer(),
        provider=provider_a,
        state_path=state_path,
    )
    engine_a._tick(["AAPL"])
    persisted_peak = engine_a._state.peak_equity
    engine_a.close()

    # On disk the peak should now be $2000 (state was written by tick A).
    on_disk = load_state(state_path)
    assert on_disk.peak_equity == pytest.approx(2000.0)
    assert persisted_peak == pytest.approx(2000.0)

    # Tick B: fresh engine, lower price → current_drawdown reflects the persisted peak.
    captured: dict[str, float] = {}
    allocator_b = Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1)
    original_allocate = allocator_b.allocate

    def capturing_allocate(*args: object, current_drawdown: float = 0.0, **kwargs: object) -> object:
        captured["current_drawdown"] = current_drawdown
        return original_allocate(*args, current_drawdown=current_drawdown, **kwargs)  # type: ignore[arg-type]

    allocator_b.allocate = capturing_allocate  # type: ignore[method-assign]

    provider_b = make_provider({"AAPL": [80.0]}, [date(2026, 5, 8)])
    engine_b = LiveEngine(
        portfolio=portfolio,
        allocator=allocator_b,
        order_sizer=OrderSizer(),
        provider=provider_b,
        state_path=state_path,
    )
    engine_b._tick(["AAPL"])
    engine_b.close()

    # total_value on tick B = 0 cash + 10 * 80 = $800. peak (persisted) = $2000.
    # current_drawdown = (2000 - 800) / 2000 = 0.6
    assert captured["current_drawdown"] == pytest.approx(0.6)


def test_tick_does_not_advance_hwm_for_unheld_tickers(tmp_path: Path, make_provider: ProviderFactory) -> None:
    """HWM must only ratchet for tickers we actually hold. Otherwise a ticker
    that ran up before the strategy first bought it would seed ``TrailingStop``
    against a pre-purchase peak — silently more conservative than backtest,
    which only tracks HWM on positions with shares > 0.
    """
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=0.0, cost_basis=None)],
        available_cash=10000.0,
    )
    state_path = tmp_path / "state.yaml"
    seeded = LiveState(
        available_cash=10000.0,
        cash_infusion_next_date=None,
        # No lots → not held. AAPL is on the watchlist but has zero shares.
        lots={},
    )
    save_atomic(seeded, state_path)

    provider = make_provider({"AAPL": [200.0]}, [date(2026, 5, 7)])
    engine = LiveEngine(
        portfolio=portfolio,
        allocator=Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1),
        order_sizer=OrderSizer(),
        provider=provider,
        state_path=state_path,
    )
    engine._tick(["AAPL"])

    assert "AAPL" not in engine._state.high_water_marks, (
        "HWM should not be set for an unheld ticker; otherwise TrailingStop seeds against pre-purchase peaks"
    )


def test_lockfile_prevents_concurrent_engines(tmp_path: Path, make_provider: ProviderFactory) -> None:
    """Two ``LiveEngine``s against the same state file would otherwise both
    load, both compute, both write — ``os.replace`` would pick a winner and
    silently drop the loser's fills. The advisory lock turns this into a clear
    "another midas live is already running" error.
    """
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=10.0, cost_basis=100.0)],
        available_cash=0.0,
    )
    state_path = tmp_path / "state.yaml"
    provider = make_provider({"AAPL": [100.0]}, [date(2026, 5, 7)])

    first = LiveEngine(
        portfolio=portfolio,
        allocator=Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1),
        order_sizer=OrderSizer(),
        provider=provider,
        state_path=state_path,
    )
    try:
        with pytest.raises(RuntimeError, match="another midas live process"):
            LiveEngine(
                portfolio=portfolio,
                allocator=Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1),
                order_sizer=OrderSizer(),
                provider=provider,
                state_path=state_path,
            )
    finally:
        first.close()

    # After close(), a fresh engine should acquire the lock cleanly.
    second = LiveEngine(
        portfolio=portfolio,
        allocator=Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1),
        order_sizer=OrderSizer(),
        provider=provider,
        state_path=state_path,
    )
    second.close()


def test_lock_released_when_init_fails_after_acquisition(tmp_path: Path, make_provider: ProviderFactory) -> None:
    """If load_or_seed raises after the lock is acquired (e.g., corrupt state file),
    the lock fd must be released. Otherwise a retry in the same process hits a
    bogus 'another midas live process' RuntimeError.
    """
    # Pre-write a corrupt state file so load_or_seed raises StateFileError.
    state_path = tmp_path / "state.yaml"
    state_path.write_text("schema_version: 99\navailable_cash: 0\n")  # unsupported

    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=10.0, cost_basis=100.0)],
        available_cash=0.0,
    )
    provider = make_provider({"AAPL": [100.0]}, [date(2026, 5, 7)])

    # First attempt: load_or_seed raises on the unsupported schema_version.
    with pytest.raises(StateFileError):
        LiveEngine(
            portfolio=portfolio,
            allocator=Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1),
            order_sizer=OrderSizer(),
            provider=provider,
            state_path=state_path,
        )

    # Fix the state file and retry — should succeed without "another midas live"
    # RuntimeError because the lock was released on the first failure.
    state_path.write_text("schema_version: 1\navailable_cash: 0.0\n")
    second = LiveEngine(
        portfolio=portfolio,
        allocator=Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1),
        order_sizer=OrderSizer(),
        provider=provider,
        state_path=state_path,
    )
    second.close()


def test_engine_creates_trade_log_alongside_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_provider: ProviderFactory
) -> None:
    """Smoke: a tick that produces a fill writes ``<state>.trades.csv`` with a BUY row."""
    from midas.trade_log import read_trades

    portfolio = PortfolioConfig(
        holdings=[],
        available_cash=1000.0,
    )
    state_path = tmp_path / "portfolio.state.yaml"
    expected_log = state_path.with_suffix(state_path.suffix + ".trades.csv")
    provider = make_provider({"AAPL": [100.0]}, [date(2026, 5, 7)])

    engine = LiveEngine(
        portfolio=portfolio,
        allocator=Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1),
        order_sizer=OrderSizer(),
        provider=provider,
        state_path=state_path,
    )

    fake_buy = Order(
        ticker="AAPL",
        direction=Direction.BUY,
        shares=1.0,
        price=100.0,
        estimated_value=100.0,
        context=OrderContext(
            contributions={"fake": 1.0},
            blended_score=1.0,
            target_weight=1.0,
            current_weight=0.0,
            reason="test",
            source="fake",
        ),
    )
    monkeypatch.setattr(engine._order_sizer, "size_buys", lambda *a, **kw: [fake_buy])

    try:
        engine._tick(["AAPL"])
    finally:
        engine.close()

    assert expected_log.exists(), "trade log should be created next to the state file"
    trades = read_trades(expected_log)
    buys = [t for t in trades if t.direction == Direction.BUY]
    assert buys, "first tick should record at least one BUY row"


def test_drawdown_overlay_produces_smaller_exposure_under_real_drawdown() -> None:
    """End-to-end check that current_drawdown actually affects exposure scaling.

    The Task 9 fix wires current_drawdown from state.peak_equity through to
    Allocator.allocate. Paired with ``test_tick_passes_current_drawdown_to_allocator``
    (which pins the kwarg arriving), this test pins the downstream formula —
    ``apply_drawdown_overlay`` reduces exposure when drawdown is non-zero, with
    the expected formula and clamping. Together they establish the chain works
    end-to-end: kwarg arrives + overlay produces expected scale = behavior is
    correct (and not silently dead code).
    """
    from midas.risk import apply_drawdown_overlay

    # No drawdown: exposure scale stays at 1.0.
    no_dd = apply_drawdown_overlay(current_drawdown=0.0, penalty=2.0, floor=0.5)
    assert no_dd == 1.0

    # 20% drawdown with penalty=2.0: exposure = max(1 - 2*0.2, 0.5) = 0.6.
    moderate = apply_drawdown_overlay(current_drawdown=0.2, penalty=2.0, floor=0.5)
    assert moderate == pytest.approx(0.6)

    # 60% drawdown (matches the live-engine kwarg-arrival test): raw =
    # 1 - 1.2 = -0.2, clamped to floor 0.5.
    deep = apply_drawdown_overlay(current_drawdown=0.6, penalty=2.0, floor=0.5)
    assert deep == pytest.approx(0.5)

    # Confirm the chain: deeper drawdown produces smaller-or-equal exposure.
    assert deep <= moderate <= no_dd
