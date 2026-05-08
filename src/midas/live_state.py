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

    Args:
        portfolio: Portfolio configuration to seed from when no state exists.
        state_path: Filesystem path of the persisted state YAML.

    Returns:
        The loaded or freshly-seeded ``LiveState``.
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
