"""Persistent runtime state for the live engine.

After first seed, this state file is the runtime source of truth for
positions, available cash, per-ticker HWM, peak equity, and the cash-
infusion ``next_date``. The portfolio YAML continues to own everything
else (tickers, strategies, allocation constraints, infusion amount/
frequency, restrictions); see docs/specs/2026-05-07-live-per-lot-
tracking-design.md.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml

from midas.models import PortfolioConfig, PositionLot

logger = logging.getLogger(__name__)

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
        "last_updated": datetime.now(tz=UTC).replace(microsecond=0).isoformat(),
        "available_cash": state.available_cash,
        "cash_infusion": (
            {"next_date": state.cash_infusion_next_date} if state.cash_infusion_next_date is not None else None
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
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def load_state(path: Path) -> LiveState:
    """Load state from *path*. Raises ``StateFileError`` on invalid input."""
    try:
        with open(path, encoding="utf-8") as handle:
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
        if not isinstance(infusion, dict):
            msg = f"cash_infusion in {path} must be a mapping, got {type(infusion).__name__}"
            raise StateFileError(msg)
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


def load_or_seed(portfolio: PortfolioConfig, state_path: Path) -> LiveState:
    """Load state from *state_path*, or seed it from *portfolio* if missing.

    The seed branch creates one ``PositionLot`` per ticker with ``shares > 0``,
    using the YAML's ``cost_basis`` and ``purchase_date=None`` (we don't know
    when the operator originally bought; affects ST/LT classification — they
    can hand-edit later if precision matters).

    On subsequent loads, warns if the YAML aggregates disagree with state
    (the state file wins). Refuses to start if the portfolio no longer lists
    a ticker for which lots are still held — that's almost certainly a config
    mistake.

    Args:
        portfolio: Portfolio configuration to seed from when no state exists.
        state_path: Filesystem path of the persisted state YAML.

    Returns:
        The loaded or freshly-seeded ``LiveState``.
    """
    if state_path.exists():
        state = load_state(state_path)
        _check_for_drift(state, portfolio)
        return state

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

    # Seed peak_equity to starting equity so CPPI/drawdown calcs match backtest,
    # which seeds state.peak_value = state.starting_value at _initialize_state.
    # The YAML's cost_basis is the operator's basis — for live, that IS the
    # starting equity baseline. Defensive ``> 0`` keeps zero/negative as None.
    seed_equity = portfolio.available_cash + sum(
        holding.shares * (holding.cost_basis or 0.0) for holding in portfolio.holdings if holding.shares > 0
    )

    state = LiveState(
        available_cash=portfolio.available_cash,
        cash_infusion_next_date=portfolio.cash_infusion.next_date if portfolio.cash_infusion else None,
        high_water_marks={},
        peak_equity=seed_equity if seed_equity > 0 else None,
        lots=lots,
    )
    save_atomic(state, state_path)
    logger.info("Seeded state at %s from portfolio config", state_path)
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


@dataclass(frozen=True)
class SellBreakdown:
    """Result of consuming lots FIFO for a sell.

    Reports shares and share-weighted cost basis separately for the short-term
    (held <365 days, or unknown purchase date) and long-term (held >=365 days)
    buckets. Either bucket may be zero. The ``*_weighted`` fields hold the raw
    ``sum(take * cost_basis)`` accumulator, useful for callers that need bit-
    identical reconstructions of total basis (since ``basis * shares`` is only
    algebraically — not bit-identically — equal in IEEE-754).
    """

    st_shares: float
    st_basis: float
    st_weighted: float
    lt_shares: float
    lt_basis: float
    lt_weighted: float


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
    return SellBreakdown(
        st_shares=st_shares,
        st_basis=st_basis,
        st_weighted=st_weighted,
        lt_shares=lt_shares,
        lt_basis=lt_basis,
        lt_weighted=lt_weighted,
    )


def apply_buy(state: LiveState, ticker: str, shares: float, price: float, day: date) -> None:
    """Append a new lot for *ticker* and decrement cash by ``shares * price``.

    Mutates *state* in place. Use after the live engine has emitted a buy
    alert and the operator is assumed to have filled at *price*.
    """
    state.lots.setdefault(ticker, []).append(PositionLot(shares=shares, purchase_date=day, cost_basis=price))
    state.available_cash -= shares * price


def apply_sell(state: LiveState, ticker: str, shares: float, price: float, day: date) -> tuple[float, float]:
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
    st_pnl = breakdown.st_shares * price - breakdown.st_weighted
    lt_pnl = breakdown.lt_shares * price - breakdown.lt_weighted
    return st_pnl, lt_pnl
