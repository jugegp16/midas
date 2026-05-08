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
        _sell(2026, HoldingPeriod.LONG_TERM, 10.0, 16.0),  # -$40 vs basis=$20
    ]
    basis = [20.0, 20.0]
    summary = compute_tax_summary(trades, basis, CONFIG, end_date=date(2026, 12, 31))
    assert summary[0].net_after_cross == pytest.approx(60.0)
    assert summary[0].tax_owed == pytest.approx(60.0 * 0.30)


def test_net_loss_below_cap_full_deductible() -> None:
    """Net loss of $1,200 → full $1,200 deducted at ST rate; no carry."""
    trades = [_sell(2026, HoldingPeriod.SHORT_TERM, 100.0, 8.0)]  # -$1200
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


def test_after_tax_curve_subtracts_owed_at_payment_date() -> None:
    """compute_after_tax_curve subtracts each summary's tax_owed at payment_date."""
    summaries = [
        AnnualTaxSummary(
            year=2026,
            st_realized=100.0,
            lt_realized=0.0,
            net_after_cross=100.0,
            deductible_loss=0.0,
            carry_forward=0.0,
            tax_owed=30.0,
            payment_date=date(2027, 4, 15),
        )
    ]
    gross = [
        (date(2027, 4, 14), 1000.0),
        (date(2027, 4, 15), 1010.0),
        (date(2027, 4, 16), 1020.0),
    ]
    after_tax = compute_after_tax_curve(gross, summaries)
    assert after_tax[0] == (date(2027, 4, 14), 1000.0)
    assert after_tax[1] == (date(2027, 4, 15), 1010.0 - 30.0)
    assert after_tax[2] == (date(2027, 4, 16), 1020.0 - 30.0)
    # Original not mutated.
    assert gross[1] == (date(2027, 4, 15), 1010.0)


def test_after_tax_curve_empty_summaries_returns_gross_copy() -> None:
    gross = [(date(2026, 1, 1), 100.0), (date(2026, 6, 1), 110.0)]
    result = compute_after_tax_curve(gross, [])
    assert result == gross
    assert result is not gross
