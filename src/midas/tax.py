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

from midas.metrics import pair_sells_with_basis
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
        config: Tax-rate policy.
        end_date: Last bar of the backtest (or report-period end). Years whose
            natural payment date falls past this are clamped to it.

    Returns:
        One AnnualTaxSummary per calendar year that contained at least one
        SELL or that received a carryforward from the prior year. Sorted by
        year ascending.

        Years with no SELL activity are omitted even if a non-zero
        ``carry_forward`` is alive — the carry threads forward silently to the
        next year that has activity.
    """
    paired = pair_sells_with_basis(list(trades), list(basis_per_sell))
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
    # carry_in is non-negative: it represents accumulated unused loss rolling
    # in from prior years. Applied as an additional ST-character loss to the
    # current year's net (loses ST/LT character on rollover; IRS preserves it,
    # we don't — see module docstring).
    carry_in = 0.0
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

        # Apply prior-year carry as an additional loss. It nets first against
        # any remaining ST gain, then any remaining LT gain, then becomes pure
        # loss that pushes net_after_cross negative.
        remaining_carry = carry_in
        if remaining_carry > 0 and st_after_cross > 0:
            shave = min(remaining_carry, st_after_cross)
            st_after_cross -= shave
            remaining_carry -= shave
        if remaining_carry > 0 and lt_after_cross > 0:
            shave = min(remaining_carry, lt_after_cross)
            lt_after_cross -= shave
            remaining_carry -= shave
        # Any still-remaining carry is pure loss applied to ST bucket.
        # Residual carry after exhausting gains in both buckets is pure ST loss
        # (carryforward already lost ST/LT character on rollover).
        st_after_cross -= remaining_carry

        net_after_cross = st_after_cross + lt_after_cross
        deductible_loss = 0.0
        carry_out = 0.0

        if net_after_cross >= 0:
            tax_owed = st_after_cross * config.short_term_rate + lt_after_cross * config.long_term_rate
        else:
            absolute = -net_after_cross
            deductible_loss = min(absolute, config.deductible_loss_cap)
            carry_out = absolute - deductible_loss
            # Negative tax_owed == reduction in tax otherwise owed on ordinary income.
            # Deduction is capped at TaxConfig.deductible_loss_cap (default $3,000)
            # per IRC §1211(b); we model the resulting savings as `cap * short_term_rate`.
            tax_owed = -deductible_loss * config.short_term_rate

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
