"""Shared test fixtures — synthetic price data and configs."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from midas.data.price_history import PriceHistory
from midas.models import CashInfusion, Holding, PortfolioConfig


@pytest.fixture
def make_provider() -> Callable[[dict[str, list[float]], list[date]], MagicMock]:
    """Factory fixture producing a fake DataProvider with controlled OHLCV data."""

    def _make(prices: dict[str, list[float]], dates: list[date]) -> MagicMock:
        provider = MagicMock()

        def get_history(ticker: str, start: date, end: date) -> pd.DataFrame:
            closes = prices[ticker]
            return pd.DataFrame(
                {
                    "open": closes,
                    "high": closes,
                    "low": closes,
                    "close": closes,
                    "volume": [1000.0] * len(closes),
                },
                index=pd.DatetimeIndex([pd.Timestamp(d) for d in dates]),
            )

        provider.get_history.side_effect = get_history
        return provider

    return _make


@pytest.fixture
def sample_portfolio() -> PortfolioConfig:
    return PortfolioConfig(
        holdings=[
            Holding(ticker="AAPL", shares=10, cost_basis=150.0),
            Holding(ticker="VOO", shares=5, cost_basis=400.0),
        ],
        available_cash=5000.0,
        cash_infusion=CashInfusion(
            amount=1500.0,
            next_date=date(2025, 1, 10),
        ),
    )


def make_price_frame(
    start: date,
    days: int,
    base_price: float,
    daily_returns: list[float] | None = None,
    name: str = "TEST",
) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame with a date index.

    Used by backtest/optimizer tests that need a provider-style DataFrame.
    Open/high/low all equal close; volume is a flat constant. Strategies
    that depend on real H/L data get degenerate bars, which is fine for
    tests that only exercise close-based logic.
    """
    dates: list[date] = []
    prices: list[float] = []
    price = base_price
    current = start
    for i in range(days):
        while current.weekday() >= 5:
            current += timedelta(days=1)
        dates.append(current)
        if daily_returns and i < len(daily_returns):
            price *= 1 + daily_returns[i]
        prices.append(round(price, 2))
        current += timedelta(days=1)
    close = np.asarray(prices, dtype=float)
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.full(days, 1_000_000.0),
        },
        index=dates,
    )
    frame.index.name = "date"
    return frame


# Backwards-compatible alias used by existing backtest/optimizer tests.
make_price_series = make_price_frame


def make_price_history(
    days: int,
    base_price: float,
    daily_returns: list[float] | None = None,
) -> PriceHistory:
    """Generate a synthetic PriceHistory for strategy/allocator tests."""
    prices: list[float] = []
    price = base_price
    for i in range(days):
        if daily_returns and i < len(daily_returns):
            price *= 1 + daily_returns[i]
        prices.append(round(price, 2))
    close = np.asarray(prices, dtype=float)
    dates = np.asarray([date(2024, 1, 1) + timedelta(days=i) for i in range(days)], dtype=object)
    return PriceHistory.from_close_only(dates, close)


def ph(close: np.ndarray) -> PriceHistory:
    """Wrap a close-only numpy array as a PriceHistory for strategy tests."""
    close_arr = np.asarray(close, dtype=float)
    dates = np.asarray(
        [date(2024, 1, 1) + timedelta(days=i) for i in range(len(close_arr))],
        dtype=object,
    )
    return PriceHistory.from_close_only(dates, close_arr)


def ph_ohlc(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray | None = None,
) -> PriceHistory:
    """Wrap independent OHLC(V) numpy arrays as a PriceHistory."""
    close_arr = np.asarray(close, dtype=float)
    dates = np.asarray(
        [date(2024, 1, 1) + timedelta(days=i) for i in range(len(close_arr))],
        dtype=object,
    )
    return PriceHistory(
        dates=dates,
        open=np.asarray(open_, dtype=float),
        high=np.asarray(high, dtype=float),
        low=np.asarray(low, dtype=float),
        close=close_arr,
        volume=np.asarray(volume, dtype=float) if volume is not None else None,
    )


@pytest.fixture
def flat_prices() -> PriceHistory:
    """100 days of flat $100 price."""
    return make_price_history(100, 100.0)


@pytest.fixture
def dropping_prices() -> PriceHistory:
    """Price is stable then drops sharply at the end — triggers mean reversion.

    The last few days are a sharp drop while the 30-day MA still includes
    the stable period, creating a gap between current price and MA.
    """
    returns = [0.0] * 90 + [-0.02] * 10
    return make_price_history(100, 100.0, returns)


@pytest.fixture
def rising_prices() -> PriceHistory:
    """Price rises steadily — triggers profit taking."""
    returns = [0.003] * 100
    return make_price_history(100, 100.0, returns)


@pytest.fixture
def crossover_prices() -> PriceHistory:
    """Price dips below MA then crosses back above — triggers momentum."""
    returns = [0.0] * 20 + [-0.008] * 15 + [0.015] * 10 + [0.0] * 55
    return make_price_history(100, 100.0, returns)


@pytest.fixture
def volatile_dropping_prices() -> PriceHistory:
    """Price with sustained losses at the end — triggers RSI oversold."""
    returns = [0.001] * 80 + [-0.02] * 20
    return make_price_history(100, 100.0, returns)


@pytest.fixture
def volatile_rising_prices() -> PriceHistory:
    """Strong sustained gains at the end — triggers RSI overbought."""
    returns = [0.001] * 80 + [0.02] * 20
    return make_price_history(100, 100.0, returns)


@pytest.fixture
def gap_down_recovery_prices() -> PriceHistory:
    """100 bars flat at $100 then a day that opens 5% down and recovers partway.

    Day index 95: real gap — open=$95 (5% below prior close $100),
    intraday low dips to $94, then rallies to close at $97 (halfway back
    into the gap). ``GapDownRecovery`` reads the real OPEN vs prior CLOSE
    to detect the gap, so this fixture uses independent OHLC arrays
    instead of the degenerate close-only path.
    """
    n = 100
    open_ = np.full(n, 100.0)
    high = np.full(n, 100.0)
    low = np.full(n, 100.0)
    close = np.full(n, 100.0)
    gap_day = 95
    open_[gap_day] = 95.0
    low[gap_day] = 94.0
    high[gap_day] = 97.5
    close[gap_day] = 97.0
    # Subsequent flat days at the new level
    for i in range(gap_day + 1, n):
        open_[i] = 97.0
        high[i] = 97.0
        low[i] = 97.0
        close[i] = 97.0
    return ph_ohlc(open_, high, low, close)


@pytest.fixture
def peak_then_drop_prices() -> PriceHistory:
    """Price rises then falls — triggers trailing stop."""
    returns = [0.01] * 30 + [-0.005] * 40 + [0.0] * 30
    return make_price_history(100, 100.0, returns)


@pytest.fixture
def ma_crossover_prices() -> PriceHistory:
    """Long decline followed by recovery — triggers golden cross."""
    returns = [-0.002] * 60 + [0.008] * 30 + [0.0] * 10
    return make_price_history(100, 100.0, returns)
