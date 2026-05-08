"""Tests for YAML config loading."""

from datetime import date
from pathlib import Path

import pytest
import yaml

from midas.config import load_portfolio, load_strategies


@pytest.fixture
def portfolio_yaml(tmp_path: Path) -> Path:
    data = {
        "portfolio": [
            {"ticker": "VOO", "shares": 5, "cost_basis": 420.0},
            {"ticker": "AAPL", "shares": 10},
        ],
        "available_cash": 2000.0,
        "cash_infusion": {
            "amount": 1500.0,
            "next_date": "2026-04-03",
            "frequency": "biweekly",
        },
    }
    p = tmp_path / "portfolio.yaml"
    p.write_text(yaml.dump(data))
    return p


@pytest.fixture
def strategy_yaml(tmp_path: Path) -> Path:
    data = {
        "softmax_temperature": 0.25,
        "min_buy_delta": 0.03,
        "min_cash_pct": 0.10,
        "strategies": [
            {
                "name": "MeanReversion",
                "weight": 1.5,
                "params": {"window": 20, "threshold": 0.08},
            },
            {
                "name": "StopLoss",
                "params": {"loss_threshold": 0.10},
            },
            {"name": "Momentum"},
        ],
    }
    p = tmp_path / "strategies.yaml"
    p.write_text(yaml.dump(data))
    return p


def test_load_portfolio(portfolio_yaml: Path) -> None:
    port = load_portfolio(portfolio_yaml)
    assert len(port.holdings) == 2
    assert port.holdings[0].ticker == "VOO"
    assert port.holdings[0].cost_basis == 420.0
    assert port.holdings[1].cost_basis is None
    assert port.available_cash == 2000.0
    assert port.cash_infusion is not None
    assert port.cash_infusion.amount == 1500.0
    assert port.cash_infusion.next_date == date(2026, 4, 3)
    assert port.cash_infusion.frequency == "biweekly"


def test_load_portfolio_minimal(tmp_path: Path) -> None:
    data = {
        "portfolio": [{"ticker": "VOO", "shares": 5}],
        "available_cash": 1000.0,
    }
    p = tmp_path / "portfolio.yaml"
    p.write_text(yaml.dump(data))
    port = load_portfolio(p)
    assert port.available_cash == 1000.0


def test_load_portfolio_parses_state_file_field(tmp_path: Path) -> None:
    yaml_path = tmp_path / "portfolio.yaml"
    yaml_path.write_text(
        "state_file: ./run/portfolio.state.yaml\n"
        "portfolio:\n"
        "  - ticker: AAPL\n"
        "    shares: 100\n"
        "    cost_basis: 150\n"
        "available_cash: 1000\n"
    )
    portfolio = load_portfolio(yaml_path)
    assert portfolio.state_file == Path("./run/portfolio.state.yaml")


def test_load_portfolio_state_file_field_optional(tmp_path: Path) -> None:
    yaml_path = tmp_path / "portfolio.yaml"
    yaml_path.write_text("portfolio:\n  - ticker: AAPL\n    shares: 100\n    cost_basis: 150\navailable_cash: 1000\n")
    portfolio = load_portfolio(yaml_path)
    assert portfolio.state_file is None


def test_load_strategies(strategy_yaml: Path) -> None:
    configs, constraints, _risk, _tax = load_strategies(strategy_yaml)
    assert len(configs) == 3

    assert configs[0].name == "MeanReversion"
    assert configs[0].params["window"] == 20
    assert configs[0].weight == 1.5

    assert configs[1].name == "StopLoss"
    assert configs[1].params["loss_threshold"] == 0.10

    assert configs[2].name == "Momentum"
    assert configs[2].params == {}
    assert configs[2].weight == 1.0  # default

    # Allocation knobs
    assert constraints.softmax_temperature == 0.25
    assert constraints.min_buy_delta == 0.03
    assert constraints.min_cash_pct == 0.10
    assert constraints.max_position_pct is None  # not specified -> None


# ---------------------------------------------------------------------------
# Risk: block parsing
# ---------------------------------------------------------------------------

import textwrap  # noqa: E402

from midas.models import RiskConfig  # noqa: E402


def _write_strategies(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "s.yaml"
    p.write_text(textwrap.dedent(body))
    return p


class TestLoadStrategiesRisk:
    def test_no_risk_block_returns_default(self, tmp_path: Path) -> None:
        path = _write_strategies(
            tmp_path,
            """
            strategies:
              - name: BollingerBand
                params: {window: 20}
            """,
        )
        _configs, _constraints, risk, _tax = load_strategies(path)
        assert risk == RiskConfig()

    def test_full_risk_block(self, tmp_path: Path) -> None:
        path = _write_strategies(
            tmp_path,
            """
            strategies:
              - name: BollingerBand
                params: {window: 20}
            risk:
              weighting: inverse_vol
              vol_lookback_days: 90
              vol_target: 0.20
              drawdown_penalty: 1.5
              drawdown_floor: 0.5
            """,
        )
        _configs, _constraints, risk, _tax = load_strategies(path)
        assert risk.weighting == "inverse_vol"
        assert risk.vol_lookback_days == 90
        assert risk.vol_target == 0.20
        assert risk.drawdown_penalty == 1.5
        assert risk.drawdown_floor == 0.5

    def test_partial_risk_block_only_vol_target(self, tmp_path: Path) -> None:
        path = _write_strategies(
            tmp_path,
            """
            strategies:
              - name: BollingerBand
                params: {window: 20}
            risk:
              vol_target: 0.18
            """,
        )
        _configs, _constraints, risk, _tax = load_strategies(path)
        assert risk.vol_target == 0.18
        assert risk.weighting == "equal"
        assert risk.drawdown_penalty is None
        assert risk.drawdown_floor is None

    def test_drawdown_one_sided_raises_at_load_time(self, tmp_path: Path) -> None:
        path = _write_strategies(
            tmp_path,
            """
            strategies:
              - name: BollingerBand
                params: {window: 20}
            risk:
              drawdown_penalty: 1.5
            """,
        )
        with pytest.raises(ValueError, match="drawdown_floor"):
            load_strategies(path)


# ---------------------------------------------------------------------------
# Tax: block parsing
# ---------------------------------------------------------------------------


def test_load_strategies_with_tax_block(tmp_path: Path) -> None:
    """Optional tax: block parses to a TaxConfig."""
    path = tmp_path / "strategies.yaml"
    path.write_text(
        "strategies:\n"
        "  - name: Momentum\n"
        "    params: {window: 20}\n"
        "tax:\n"
        "  short_term_rate: 0.32\n"
        "  long_term_rate: 0.15\n"
        "  deductible_loss_cap: 3000.0\n"
        "  payment_lag_days: 105\n"
    )
    _configs, _constraints, _risk, tax = load_strategies(path)
    assert tax is not None
    assert tax.short_term_rate == 0.32
    assert tax.long_term_rate == 0.15
    assert tax.deductible_loss_cap == 3000.0
    assert tax.payment_lag_days == 105


def test_load_strategies_without_tax_block(tmp_path: Path) -> None:
    """Omitting tax: yields tax_config=None — no behavior change for existing configs."""
    path = tmp_path / "strategies.yaml"
    path.write_text("strategies:\n  - name: Momentum\n    params: {window: 20}\n")
    _configs, _constraints, _risk, tax = load_strategies(path)
    assert tax is None
