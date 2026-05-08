"""Terminal ASCII charts for backtest summaries.

Built via ``plotext`` and emitted through Rich's ``Console`` so the chart
block aligns with the centered summary tables. Each function is a no-op
when its inputs are empty, so callers can render unconditionally without
guarding on ``risk_history`` shape.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

import plotext as plt  # type: ignore[import-untyped]
from rich.console import Console
from rich.text import Text

from midas.risk import TRADING_DAYS_PER_YEAR
from midas.risk_metrics import SHARPE_LOOKBACK_BARS

if TYPE_CHECKING:
    from midas.results import BacktestResult


CHART_HEIGHT = 18
CHART_WIDTH = 100

console = Console()

# Strips CSI escape sequences (colors, styles) so visible width can be measured
# independently of the ANSI codes plotext emits.
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def render_charts(result: BacktestResult) -> None:
    """Render the full chart panel — equity, drawdown, exposure, and vol target.

    Each chart is its own single-figure plot at the same ``CHART_WIDTH`` so all
    panels line up vertically. The drawdown chart always renders alongside the
    equity curve. The excess-return chart is emitted whenever a parallel
    ``bh_equity_curve`` is populated. The gross-exposure chart renders whenever
    ``risk_history`` is populated; the predicted-vs-target vol chart only when
    the run was configured with a vol target and the history contains at least
    one non-zero predicted-vol sample.
    """
    if not result.equity_curve:
        return
    _render_equity(result)
    if result.bh_equity_curve:
        _render_excess_return(result)
    _render_drawdown(result)
    _render_rolling_sharpe(result)
    if result.risk_history is None or not result.risk_history.dates:
        return
    _render_gross_exposure(result)
    if (result.risk_metrics is not None and result.risk_metrics.vol_target is not None) and any(
        value > 0 for value in result.risk_history.vol_target_predicted_vol
    ):
        _render_predicted_vs_target_vol(result)


def _flush_centered(title: str) -> None:
    """Print *title* on its own centered line, then the centered chart block.

    plotext anchors its own title flush-left within the chart frame, so we skip
    ``plt.title()`` and emit the heading separately through Rich — that way the
    title is centered relative to the terminal, not stuck to the left edge of
    the plot area. Chart lines vary in visible width (axis labels, legend,
    etc.); padding to a uniform width first keeps the figure as a single
    rectangular block when Rich centers it.
    """
    chart = plt.build().rstrip("\n")
    if not chart:
        return
    console.print()
    console.print(f"[bold]{title}[/bold]", justify="center")
    lines = chart.split("\n")
    visible_widths = [len(ANSI_RE.sub("", line)) for line in lines]
    max_width = max(visible_widths) if visible_widths else 0
    padded = [line + " " * (max_width - width) for line, width in zip(lines, visible_widths, strict=True)]
    console.print(Text.from_ansi("\n".join(padded)), justify="center")


def _trailing_mean(values: list[float], window: int) -> list[float]:
    """Trailing simple mean over a fixed-bar window, parallel to ``values``.

    Bars before the window has filled use whatever bars are available —
    so the curve starts at ``values[0]`` and converges to a true ``window``-bar
    mean once enough bars have accumulated. This avoids leaving leading bars
    blank on the chart.
    """
    out: list[float] = []
    running_sum = 0.0
    for i, value in enumerate(values):
        running_sum += value
        if i >= window:
            running_sum -= values[i - window]
            out.append(running_sum / window)
        else:
            out.append(running_sum / (i + 1))
    return out


def _drawdown_pct_series(result: BacktestResult, dates: list[str]) -> list[float]:
    """Drawdown as a negative percentage so the chart reads as a downward dip."""
    if result.risk_history is not None and len(result.risk_history.drawdown) == len(dates):
        return [-value * 100.0 for value in result.risk_history.drawdown]
    equity = [value for _, value in result.equity_curve]
    peak = equity[0] if equity else 0.0
    out: list[float] = []
    for value in equity:
        peak = max(peak, value)
        out.append(-((peak - value) / peak) * 100.0 if peak > 0 else 0.0)
    return out


def _setup_single_figure() -> None:
    plt.clear_figure()
    plt.plot_size(CHART_WIDTH, CHART_HEIGHT)
    plt.theme("clear")
    plt.date_form("Y-m-d")


def _render_equity(result: BacktestResult) -> None:
    dates = [dt.isoformat() for dt, _ in result.equity_curve]
    equity = [value for _, value in result.equity_curve]
    _setup_single_figure()
    has_after_tax = len(result.after_tax_equity_curve) == len(result.equity_curve) and bool(
        result.after_tax_equity_curve
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


def _render_excess_return(result: BacktestResult) -> None:
    """Per-bar (strategy minus buy-and-hold) / starting_value, in percent.

    Positive values are bars where the strategy is ahead of B&H; negative
    bars are where it's behind. A flat line at zero means the strategy is
    tracking B&H exactly. Renders only when ``bh_equity_curve`` is populated
    and parallel to ``equity_curve``.
    """
    starting = result.starting_value
    if starting <= 0:
        return
    if len(result.bh_equity_curve) != len(result.equity_curve):
        # Curves should be parallel by construction; defensive against test
        # fixtures or future shape regressions so chart rendering doesn't
        # crash the whole summary output.
        return
    dates = [dt.isoformat() for dt, _ in result.equity_curve]
    excess_pct = [
        ((eq - bh) / starting) * 100.0
        for (_, eq), (_, bh) in zip(result.equity_curve, result.bh_equity_curve, strict=True)
    ]
    _setup_single_figure()
    plt.plot(dates, excess_pct, color="magenta", marker="braille")
    plt.ylabel("Excess %")
    _flush_centered("Excess Return vs Buy & Hold (%)")


def _rolling_sharpe_series(equity: list[float], lookback: int) -> list[float]:
    """Per-bar annualized Sharpe over a trailing-``lookback`` log-return window.

    Bars before the window has filled emit ``0.0`` so the array stays parallel
    to ``equity``. Within the window, mean / stdev of log returns are
    annualized via ``sqrt(252)``. A degenerate window (zero stdev) emits
    ``0.0`` rather than ``NaN``/inf.
    """
    if len(equity) < 2:
        return [0.0] * len(equity)
    log_returns: list[float] = []
    prev = equity[0]
    for value in equity[1:]:
        if prev > 0 and value > 0:
            log_returns.append(math.log(value / prev))
        else:
            log_returns.append(0.0)
        prev = value
    out = [0.0]  # bar 0 has no return yet
    for i in range(len(log_returns)):
        start = max(0, i + 1 - lookback)
        window = log_returns[start : i + 1]
        if len(window) < 2:
            out.append(0.0)
            continue
        mean = sum(window) / len(window)
        var = sum((x - mean) ** 2 for x in window) / (len(window) - 1)
        if var <= 0:
            out.append(0.0)
            continue
        stdev = math.sqrt(var)
        out.append(mean / stdev * math.sqrt(TRADING_DAYS_PER_YEAR))
    return out


def _render_rolling_sharpe(result: BacktestResult) -> None:
    """Annualized rolling Sharpe (252-bar window) per bar.

    The summary table reports a single end-of-run scalar; this chart shows when
    the strategy was producing risk-adjusted returns vs underperforming. Same
    log-return window the spec uses for ``rolling_sharpe_252d``.

    Skipped on backtests too short for the lookback to fill — rendering a
    near-flat-zero curve would be visually indistinguishable from a real
    Sharpe-zero strategy. Threshold is a quarter of the lookback (~63 bars
    of a 252-bar window): below that, the chart is mostly noise from
    too-small windows.
    """
    if len(result.equity_curve) < SHARPE_LOOKBACK_BARS // 4:
        return
    dates = [dt.isoformat() for dt, _ in result.equity_curve]
    equity = [value for _, value in result.equity_curve]
    sharpe = _rolling_sharpe_series(equity, SHARPE_LOOKBACK_BARS)
    _setup_single_figure()
    plt.plot(dates, sharpe, color="yellow", marker="braille")
    plt.ylabel("Sharpe")
    _flush_centered("Rolling Sharpe (252d, annualized)")


def _render_drawdown(result: BacktestResult) -> None:
    dates = [dt.isoformat() for dt, _ in result.equity_curve]
    drawdown_pct = _drawdown_pct_series(result, dates)
    _setup_single_figure()
    plt.plot(dates, drawdown_pct, color="red", marker="braille")
    plt.ylabel("DD %")
    _flush_centered("Drawdown (%)")


def _render_gross_exposure(result: BacktestResult) -> None:
    history = result.risk_history
    assert history is not None  # checked by caller
    dates = [dt.isoformat() for dt in history.dates]
    gross_pct = [value * 100.0 for value in history.gross_exposure]

    _setup_single_figure()
    plt.plot(dates, gross_pct, color="green", label="Gross Exposure", marker="braille")

    # 252-day trailing mean so the eye can track how typical deployment
    # shifts across regimes (e.g. higher during bull years, lower during
    # corrections). A static average would flatten this into one number.
    rolling_avg = _trailing_mean(gross_pct, window=252)
    plt.plot(dates, rolling_avg, color="white", label="Rolling Avg (252d)", marker="braille")

    cppi_active = any(scale < 1.0 for scale in history.cppi_scale)
    if cppi_active:
        cppi_pct = [scale * 100.0 for scale in history.cppi_scale]
        plt.plot(dates, cppi_pct, color="orange", label="CPPI Scale", marker="braille")

    # Set the Y-window to ``[min - 5pp, max + 5pp]`` clipped to ``[0, 105]``
    # so the line shows real variation without autoscaling into a noisy 0.5pp
    # window for a near-flat run. With a 5pp buffer, a strategy at 95% renders
    # in ``[90, 100]`` — clearly reading as "near full investment" rather than
    # noise hugging the chart floor.
    all_pct = list(gross_pct)
    if cppi_active:
        all_pct.extend(cppi_pct)
    lo = max(0.0, min(all_pct) - 5.0)
    hi = min(105.0, max(all_pct) + 5.0)
    plt.ylim(lo, hi)
    plt.ylabel("%")
    _flush_centered("Gross Exposure (%)")


def _render_predicted_vs_target_vol(result: BacktestResult) -> None:
    history = result.risk_history
    assert history is not None  # checked by caller
    metrics = result.risk_metrics
    assert metrics is not None and metrics.vol_target is not None

    dates = [dt.isoformat() for dt in history.dates]
    predicted_pct = [value * 100.0 for value in history.vol_target_predicted_vol]
    target_pct = [metrics.vol_target * 100.0] * len(dates)

    _setup_single_figure()
    plt.plot(dates, predicted_pct, color="cyan", label="Predicted Vol", marker="braille")
    plt.plot(dates, target_pct, color="red", label="Target Vol", marker="braille")
    plt.ylabel("Vol %")
    _flush_centered("Predicted vs Target Annualized Vol (%)")
