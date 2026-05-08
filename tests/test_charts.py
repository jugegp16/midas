"""Smoke tests for the terminal charts module.

These tests exercise the render path with realistic shapes; they assert the
function doesn't crash and produces some non-empty output. The goal is
regression protection against shape mismatches between ``RiskHistory`` and
``equity_curve``, not pixel-level chart correctness.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from midas.charts import render_charts
from midas.results import BacktestResult
from midas.risk_metrics import RiskHistory, RiskMetrics


def _empty_result() -> BacktestResult:
    return BacktestResult(
        trades=[],
        final_value=0,
        starting_value=0,
        buy_and_hold_value=0,
        train_trades=[],
        test_trades=[],
        train_return=0,
        test_return=0,
        train_bh_return=0,
        test_bh_return=0,
        split_date=None,
        twr=0,
        equity_curve=[],
        total_days=0,
        train_days=0,
        test_days=0,
        cagr=0,
        max_drawdown=0,
        sharpe_ratio=0,
        sortino_ratio=0,
        win_rate=0,
        profit_factor=0,
        avg_win=0,
        avg_loss=0,
        efficiency_ratio=0,
        strategy_stats=[],
        unrealized_pnl=0,
        unrealized_pnl_by_ticker={},
        basis_per_sell=[],
    )


def _populated_result(*, vol_target: float | None = None, with_history: bool = True) -> BacktestResult:
    # 100 bars: above the rolling-Sharpe skip threshold (SHARPE_LOOKBACK_BARS//4
    # ≈ 63), short enough to keep tests fast. The CPPI-active range starting
    # at i=30 stays valid (still 70 active bars in the second half).
    bar_count = 100
    start = date(2024, 1, 1)
    dates = [start + timedelta(days=i) for i in range(bar_count)]
    equity = [100.0 + i for i in range(bar_count)]
    history: RiskHistory | None = None
    if with_history:
        history = RiskHistory(
            dates=list(dates),
            gross_exposure=[0.95] * bar_count,
            cppi_scale=[1.0 if i < 30 else 0.85 for i in range(bar_count)],
            vol_target_scale=[1.0 if vol_target is None else 0.9] * bar_count,
            vol_target_predicted_vol=[0.0 if vol_target is None else 0.12] * bar_count,
            drawdown=[0.0] * bar_count,
        )
    metrics = RiskMetrics(
        realized_vol_60d=0.15,
        vol_target=vol_target,
        drawdown_from_peak=0.0,
        rolling_sharpe_252d=1.2,
    )
    result = _empty_result()
    result.equity_curve = list(zip(dates, equity, strict=True))
    result.risk_metrics = metrics
    result.risk_history = history
    return result


def test_render_charts_no_crash_on_empty_curve() -> None:
    render_charts(_empty_result())


def test_render_charts_no_crash_without_history(capsys: pytest.CaptureFixture[str]) -> None:
    result = _populated_result(with_history=False)
    render_charts(result)
    out = capsys.readouterr().out
    assert "Equity Curve" in out


def test_render_charts_with_cppi_history(capsys: pytest.CaptureFixture[str]) -> None:
    result = _populated_result()
    render_charts(result)
    out = capsys.readouterr().out
    assert "Equity Curve" in out
    assert "Drawdown" in out
    assert "Gross Exposure" in out


def test_render_charts_with_vol_target_panel(capsys: pytest.CaptureFixture[str]) -> None:
    result = _populated_result(vol_target=0.10)
    render_charts(result)
    out = capsys.readouterr().out
    assert "Predicted vs Target" in out


def test_render_charts_excess_return(capsys: pytest.CaptureFixture[str]) -> None:
    """When bh_equity_curve is populated, the excess-return chart renders."""
    result = _populated_result()
    # B&H slightly underperforms strategy so excess is positive and growing.
    result.bh_equity_curve = [(dt, 100.0 + 0.5 * i) for i, (dt, _) in enumerate(result.equity_curve)]
    result.starting_value = 100.0
    render_charts(result)
    out = capsys.readouterr().out
    assert "Excess Return" in out
    # B&H now lives only in the dedicated Excess Return chart, not as an
    # overlay on the equity chart. "Buy & Hold" should appear exactly once
    # (in the excess chart's title); a regression that re-overlays B&H on
    # the equity chart would push the count to 2+.
    assert out.count("Buy & Hold") == 1, (
        f"Buy & Hold should appear once (in Excess Return title); got {out.count('Buy & Hold')}"
    )


def test_render_charts_excess_return_skipped_on_length_mismatch(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Curve-length mismatch must not crash; chart simply skips."""
    result = _populated_result()
    # Half-length B&H curve simulates a future shape regression.
    result.bh_equity_curve = [(dt, 100.0) for dt, _ in result.equity_curve[: len(result.equity_curve) // 2]]
    result.starting_value = 100.0
    render_charts(result)
    out = capsys.readouterr().out
    assert "Excess Return" not in out
    # Equity chart still renders even though excess was skipped.
    assert "Equity Curve" in out


def test_render_charts_excess_return_skipped_without_bh_curve(capsys: pytest.CaptureFixture[str]) -> None:
    """No B&H curve → no excess chart, but other panels still render."""
    result = _populated_result()
    result.bh_equity_curve = []
    render_charts(result)
    out = capsys.readouterr().out
    assert "Excess Return" not in out
    assert "Equity Curve" in out


def test_render_charts_excess_return_skipped_when_starting_value_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """starting_value <= 0 short-circuits the excess chart's rendering."""
    result = _populated_result()
    result.bh_equity_curve = list(result.equity_curve)
    result.starting_value = 0.0
    render_charts(result)
    out = capsys.readouterr().out
    assert "Excess Return" not in out


def test_render_charts_rolling_sharpe(capsys: pytest.CaptureFixture[str]) -> None:
    """Rolling Sharpe chart renders unconditionally when there's an equity curve."""
    result = _populated_result()
    render_charts(result)
    out = capsys.readouterr().out
    assert "Rolling Sharpe" in out


def _make_result_with_curves(
    *,
    equity: list[tuple[date, float]],
    after_tax: list[tuple[date, float]],
) -> BacktestResult:
    """Minimal BacktestResult with the two equity curves populated.

    Everything else gets zero/empty defaults — just enough to drive
    ``_render_equity`` without exercising any other code path.
    """
    result = _empty_result()
    result.equity_curve = list(equity)
    result.after_tax_equity_curve = list(after_tax)
    return result


def test_render_equity_includes_after_tax_overlay_when_populated(monkeypatch: pytest.MonkeyPatch) -> None:
    """When BacktestResult.after_tax_equity_curve is non-empty, the equity chart
    plots both gross and after-tax series."""
    import plotext as plt

    from midas.charts import _render_equity

    plot_calls: list[dict] = []
    real_plot = plt.plot

    def spy(*args: object, **kwargs: object) -> object:
        plot_calls.append(kwargs)
        return real_plot(*args, **kwargs)

    monkeypatch.setattr(plt, "plot", spy)

    result = _make_result_with_curves(
        equity=[(date(2026, 1, 1), 10000.0), (date(2026, 12, 31), 12000.0)],
        after_tax=[(date(2026, 1, 1), 10000.0), (date(2026, 12, 31), 11500.0)],
    )
    _render_equity(result)

    labels = [c.get("label", "") for c in plot_calls]
    assert any("After-Tax" in lbl for lbl in labels), f"expected After-Tax label, got {labels}"
