"""YAML config loading for portfolio and strategy definitions."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from midas.models import (
    DEFAULT_MIN_BUY_DELTA,
    DEFAULT_MIN_CASH_PCT,
    DEFAULT_SOFTMAX_TEMPERATURE,
    DEFAULT_VOL_LOOKBACK_DAYS,
    AllocationConstraints,
    CashInfusion,
    Holding,
    PortfolioConfig,
    RiskConfig,
    StrategyConfig,
    TradingRestrictions,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        msg = f"Expected a YAML mapping in {path}"
        raise ValueError(msg)
    return data


def load_portfolio(path: Path) -> PortfolioConfig:
    """Load portfolio config from YAML."""
    raw = _load_yaml(path)

    holdings = [
        Holding(
            ticker=entry["ticker"],
            shares=float(entry["shares"]),
            cost_basis=float(entry["cost_basis"]) if "cost_basis" in entry else None,
        )
        for entry in raw["portfolio"]
    ]

    infusion = None
    if "cash_infusion" in raw:
        ci = raw["cash_infusion"]
        next_date = ci["next_date"]
        if isinstance(next_date, str):
            next_date = date.fromisoformat(next_date)
        elif isinstance(next_date, datetime):
            next_date = next_date.date()
        infusion = CashInfusion(
            amount=float(ci["amount"]),
            next_date=next_date,
            frequency=ci.get("frequency"),
        )

    restrictions = None
    if "trading_restrictions" in raw:
        tr = raw["trading_restrictions"]
        restrictions = TradingRestrictions(
            round_trip_days=int(tr.get("round_trip_days", 0)),
        )

    state_file_raw = raw.get("state_file")
    state_file = Path(state_file_raw) if state_file_raw is not None else None

    portfolio = PortfolioConfig(
        holdings=holdings,
        available_cash=float(raw["available_cash"]),
        cash_infusion=infusion,
        trading_restrictions=restrictions,
        state_file=state_file,
    )

    return portfolio


def load_strategies(
    path: Path,
) -> tuple[list[StrategyConfig], AllocationConstraints, RiskConfig]:
    """Load strategy configs, allocation knobs, and optional risk policy from YAML.

    Returns (strategies, constraints, risk_config). The risk block is optional;
    omitting it yields a default ``RiskConfig`` (all features off, current behavior).
    """
    raw = _load_yaml(path)

    configs = []
    for strat in raw["strategies"]:
        configs.append(
            StrategyConfig(
                name=strat["name"],
                params=strat.get("params", {}),
                tickers=strat.get("tickers"),
                weight=float(strat.get("weight", 1.0)),
            )
        )

    max_pos = raw.get("max_position_pct")
    constraints = AllocationConstraints(
        max_position_pct=float(max_pos) if max_pos is not None else None,
        min_cash_pct=float(raw.get("min_cash_pct", DEFAULT_MIN_CASH_PCT)),
        softmax_temperature=float(
            raw.get("softmax_temperature", DEFAULT_SOFTMAX_TEMPERATURE),
        ),
        min_buy_delta=float(
            raw.get("min_buy_delta", DEFAULT_MIN_BUY_DELTA),
        ),
    )

    risk_raw = raw.get("risk") or {}
    risk = RiskConfig(
        weighting=str(risk_raw.get("weighting", "equal")),
        vol_lookback_days=int(risk_raw.get("vol_lookback_days", DEFAULT_VOL_LOOKBACK_DAYS)),
        vol_target=float(risk_raw["vol_target"]) if risk_raw.get("vol_target") is not None else None,
        drawdown_penalty=(
            float(risk_raw["drawdown_penalty"]) if risk_raw.get("drawdown_penalty") is not None else None
        ),
        drawdown_floor=(float(risk_raw["drawdown_floor"]) if risk_raw.get("drawdown_floor") is not None else None),
    )

    return configs, constraints, risk
