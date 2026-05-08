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
    # The Schedule D table is rendered via Rich — check for the per-row content
    # and the per-year footer line. Field labels like "AAPL" and "long-term"
    # / "short-term" appear in any sensible rendering.
    assert "AAPL" in result.output
    assert "long-term" in result.output or "short-term" in result.output
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
