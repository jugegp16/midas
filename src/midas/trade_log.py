"""Append-only trade-log writer and strict reader.

Used by both the live engine (writes to ``<state_path>.trades.csv``) and the
backtest result writer (writes to ``<output_dir>/trades.csv``). Single shape
across both modes so ``midas tax-report`` has one reader.

Header drift on read raises :class:`TradeLogError` rather than silently
returning empty / wrong rows; partial rows raise with the offending line
number. The log is intentionally permissive on content (negative shares,
future dates) but strict on shape — hand-edits are an explicit escape valve.
"""

from __future__ import annotations

import csv
import datetime
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

from midas.models import Direction, HoldingPeriod, TradeRecord

type PurchaseDate = date | Literal["various"] | None

TRADE_LOG_COLUMNS: tuple[str, ...] = (
    "date",
    "ticker",
    "direction",
    "shares",
    "price",
    "strategy",
    "holding_period",
    "purchase_date",
    "cost_basis",
    "realized_pnl",
    "return_pct",
)


class TradeLogError(ValueError):
    """Raised on header drift, partial rows, or unparseable values."""


@dataclass(frozen=True)
class LoggedTrade:
    """In-memory representation of one trade-log row."""

    date: datetime.date
    ticker: str
    direction: Direction
    shares: float
    price: float
    strategy_name: str
    holding_period: HoldingPeriod | None
    purchase_date: PurchaseDate
    cost_basis: float | None
    realized_pnl: float | None
    return_pct: float | None


def _format_purchase_date(value: PurchaseDate) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return value.isoformat()


def _format_holding_period(value: HoldingPeriod | None) -> str:
    return value.value if value is not None else ""


def append_trade(
    path: Path,
    record: TradeRecord,
    *,
    cost_basis: float | None,
    purchase_date: PurchaseDate,
) -> None:
    """Append one row to *path*, creating the file with header on first write.

    For BUY rows pass ``cost_basis=None``; the ``cost_basis``, ``realized_pnl``,
    and ``return_pct`` columns are written empty. For SELL rows the bucket's
    share-weighted cost basis is required; ``realized_pnl`` and ``return_pct``
    are derived from it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        needs_header = handle.tell() == 0
        writer = csv.writer(handle)
        if needs_header:
            writer.writerow(TRADE_LOG_COLUMNS)
        if record.direction == Direction.BUY or cost_basis is None:
            cost_basis_cell: str | float = ""
            pnl_cell: str | float = ""
            ret_cell: str | float = ""
        else:
            pnl = round((record.price - cost_basis) * record.shares, 4)
            ret = round((record.price - cost_basis) / cost_basis, 6) if cost_basis != 0 else 0.0
            cost_basis_cell = round(cost_basis, 4)
            pnl_cell = pnl
            ret_cell = ret
        writer.writerow(
            [
                record.date.isoformat(),
                record.ticker,
                record.direction.value,
                record.shares,
                record.price,
                record.strategy_name,
                _format_holding_period(record.holding_period),
                _format_purchase_date(purchase_date),
                cost_basis_cell,
                pnl_cell,
                ret_cell,
            ]
        )
        handle.flush()


def _parse_holding_period(raw: str) -> HoldingPeriod | None:
    if not raw:
        return None
    try:
        return HoldingPeriod(raw)
    except ValueError as exc:
        msg = f"unknown holding_period value {raw!r}"
        raise TradeLogError(msg) from exc


def _parse_purchase_date(raw: str) -> PurchaseDate:
    if not raw:
        return None
    if raw == "various":
        return "various"
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        msg = f"unparseable purchase_date {raw!r}"
        raise TradeLogError(msg) from exc


def _parse_optional_float(raw: str) -> float | None:
    if raw == "":
        return None
    try:
        return float(raw)
    except ValueError as exc:
        msg = f"unparseable float {raw!r}"
        raise TradeLogError(msg) from exc


def read_trades(path: Path) -> list[LoggedTrade]:
    """Read all rows from *path* into ``LoggedTrade`` instances.

    Raises :class:`TradeLogError` on header drift or unparseable rows.
    """
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            msg = f"trade-log file {path} exists but is empty"
            raise TradeLogError(msg) from exc
        if tuple(header) != TRADE_LOG_COLUMNS:
            msg = f"trade-log header drift in {path}: expected {TRADE_LOG_COLUMNS}, got {tuple(header)}"
            raise TradeLogError(msg)

        out: list[LoggedTrade] = []
        for line_num, row in enumerate(reader, start=2):
            if len(row) != len(TRADE_LOG_COLUMNS):
                msg = f"trade-log row at line {line_num} has {len(row)} fields, expected {len(TRADE_LOG_COLUMNS)}"
                raise TradeLogError(msg)
            try:
                trade_date = date.fromisoformat(row[0])
                direction = Direction(row[2])
            except ValueError as exc:
                msg = f"trade-log row at line {line_num}: {exc}"
                raise TradeLogError(msg) from exc
            out.append(
                LoggedTrade(
                    date=trade_date,
                    ticker=row[1],
                    direction=direction,
                    shares=float(row[3]),
                    price=float(row[4]),
                    strategy_name=row[5],
                    holding_period=_parse_holding_period(row[6]),
                    purchase_date=_parse_purchase_date(row[7]),
                    cost_basis=_parse_optional_float(row[8]),
                    realized_pnl=_parse_optional_float(row[9]),
                    return_pct=_parse_optional_float(row[10]),
                )
            )
        return out
