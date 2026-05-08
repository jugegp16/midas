"""Integration test — full pipeline from config to backtest output."""

import json
from datetime import date
from pathlib import Path

import yaml
from conftest import make_price_series

from midas.allocator import Allocator
from midas.backtest import BacktestEngine
from midas.config import load_portfolio, load_strategies
from midas.models import Direction
from midas.order_sizer import OrderSizer
from midas.results import write_backtest_results
from midas.strategies import STRATEGY_REGISTRY
from midas.strategies.base import EntrySignal, ExitRule


def test_full_pipeline(tmp_path: Path) -> None:
    """End-to-end: YAML configs -> strategies -> allocator -> backtest -> CSV."""

    # 1. Write portfolio config
    portfolio_data = {
        "portfolio": [
            {"ticker": "VOO", "shares": 5, "cost_basis": 95.0},
            {"ticker": "AAPL", "shares": 10, "cost_basis": 90.0},
        ],
        "available_cash": 3000.0,
        "cash_infusion": {
            "amount": 1500.0,
            "next_date": "2025-01-10",
        },
    }
    portfolio_path = tmp_path / "portfolio.yaml"
    portfolio_path.write_text(yaml.dump(portfolio_data))

    # 2. Write strategy config
    strategy_data = {
        "min_cash_pct": 0.05,
        "min_buy_delta": 0.01,
        "softmax_temperature": 0.5,
        "strategies": [
            {"name": "MeanReversion", "params": {"window": 20, "threshold": 0.05}},
            {"name": "ProfitTaking", "params": {"gain_threshold": 0.15}},
            {"name": "Momentum", "params": {"window": 15}},
        ],
    }
    strategy_path = tmp_path / "strategies.yaml"
    strategy_path.write_text(yaml.dump(strategy_data))

    # 3. Load configs
    portfolio = load_portfolio(portfolio_path)
    assert len(portfolio.holdings) == 2
    assert portfolio.available_cash == 3000.0

    strat_configs, constraints, _risk_config, _tax_config = load_strategies(strategy_path)
    assert len(strat_configs) == 3

    # 4. Build allocator + order_sizer + exit rules
    entries: list[tuple[EntrySignal, float]] = []
    exits: list[ExitRule] = []
    for cfg in strat_configs:
        cls = STRATEGY_REGISTRY[cfg.name]
        strategy = cls(**cfg.params)
        if isinstance(strategy, ExitRule):
            exits.append(strategy)
        elif isinstance(strategy, EntrySignal):
            entries.append((strategy, cfg.weight))

    n_tickers = sum(1 for h in portfolio.holdings if h.shares > 0)
    allocator = Allocator(entries, constraints, n_tickers)
    order_sizer = OrderSizer()

    # 5. Generate synthetic price data
    voo_returns = [0.0] * 20 + [-0.006] * 20 + [0.008] * 30 + [0.0] * 30
    voo_prices = make_price_series(date(2024, 1, 2), 100, 100.0, voo_returns, name="VOO")

    aapl_returns = [0.004] * 100
    aapl_prices = make_price_series(date(2024, 1, 2), 100, 100.0, aapl_returns, name="AAPL")

    price_data = {"VOO": voo_prices, "AAPL": aapl_prices}

    # 6. Run backtest
    engine = BacktestEngine(
        allocator=allocator,
        order_sizer=order_sizer,
        exit_rules=exits,
        constraints=constraints,
        train_pct=0.7,
        enable_split=True,
    )

    start = date(2024, 1, 2)
    end = max(max(s.index) for s in price_data.values())
    result = engine.run(portfolio, price_data, start, end)

    # 7. Verify results
    assert result.starting_value > 0
    assert result.final_value > 0
    assert result.split_date is not None
    assert len(result.trades) > 0

    directions = {t.direction for t in result.trades}
    assert Direction.BUY in directions

    # 8. Write results directory and verify
    out_dir = tmp_path / "integration_results"
    write_backtest_results(result, out_dir)

    assert out_dir.is_dir()
    assert (out_dir / "trades.csv").exists()
    assert (out_dir / "equity_curve.csv").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "strategy_breakdown.csv").exists()

    with open(out_dir / "summary.json") as f:
        summary = json.load(f)
    assert "split" in summary  # split was enabled


def test_strategy_registry_complete() -> None:
    """All strategies should be registered and instantiable."""
    expected = {
        # entry signals
        "BollingerBand",
        "DonchianBreakout",
        "GapDownRecovery",
        "KeltnerChannel",
        "MACDCrossover",
        "MeanReversion",
        "Momentum",
        "MovingAverageCrossover",
        "RSIOversold",
        "VWAPReversion",
        # exit rules
        "ChandelierStop",
        "MACDExit",
        "MovingAverageCrossoverExit",
        "ParabolicSARExit",
        "ProfitTaking",
        "StopLoss",
        "TrailingStop",
    }
    assert set(STRATEGY_REGISTRY.keys()) == expected

    for _name, cls in STRATEGY_REGISTRY.items():
        instance = cls()
        assert instance.name
        assert instance.description
        assert len(instance.suitability) > 0
        assert isinstance(instance, (EntrySignal, ExitRule))
