"""Tests for core data models."""

from datetime import date

import pytest

from midas.models import (
    AllocationConstraints,
    CashInfusion,
    Direction,
    Holding,
    HoldingPeriod,
    Order,
    OrderContext,
    PortfolioConfig,
    PositionLot,
    RiskConfig,
    StrategyConfig,
    TaxConfig,
    TradeRecord,
)


def test_portfolio_get_holding() -> None:
    port = PortfolioConfig(
        holdings=[
            Holding(ticker="AAPL", shares=10, cost_basis=150.0),
            Holding(ticker="VOO", shares=5),
        ],
        available_cash=1000.0,
    )
    assert port.get_holding("AAPL") is not None
    assert port.get_holding("AAPL").shares == 10  # type: ignore[union-attr]
    assert port.get_holding("VOO").cost_basis is None  # type: ignore[union-attr]
    assert port.get_holding("MSFT") is None


def test_holding_period_values() -> None:
    assert HoldingPeriod.SHORT_TERM.value == "short-term"
    assert HoldingPeriod.LONG_TERM.value == "long-term"


def test_order_frozen() -> None:
    ctx = OrderContext(
        contributions={"TestStrategy": 0.5},
        blended_score=0.5,
        target_weight=0.10,
        current_weight=0.05,
        reason="test buy",
        source="TestStrategy",
    )
    order = Order(
        ticker="VOO",
        direction=Direction.BUY,
        shares=2,
        price=500.0,
        estimated_value=1000.0,
        context=ctx,
    )
    assert order.context.blended_score == 0.5
    assert order.price == 500.0


def test_trade_record() -> None:
    tr = TradeRecord(
        date=date(2024, 6, 1),
        ticker="AAPL",
        direction=Direction.SELL,
        shares=5,
        price=180.0,
        strategy_name="ProfitTaking",
        holding_period=HoldingPeriod.LONG_TERM,
    )
    assert tr.holding_period == HoldingPeriod.LONG_TERM


def test_allocation_constraints_defaults() -> None:
    c = AllocationConstraints()
    assert c.max_position_pct is None
    assert c.min_cash_pct == 0.05
    assert c.min_buy_delta == 0.02
    assert c.softmax_temperature == 0.5


def test_position_lot() -> None:
    lot = PositionLot(shares=10.0, purchase_date=date(2024, 1, 5), cost_basis=150.0)
    assert lot.shares == 10.0
    assert lot.cost_basis == 150.0


def test_strategy_config_defaults() -> None:
    cfg = StrategyConfig(name="TestStrategy")
    assert cfg.weight == 1.0


class TestCashInfusion:
    def test_advance_biweekly(self) -> None:
        infusion = CashInfusion(amount=1500.0, next_date=date(2025, 1, 3), frequency="biweekly")
        infusion.advance()
        assert infusion.next_date == date(2025, 1, 17)

    def test_advance_weekly(self) -> None:
        infusion = CashInfusion(amount=500.0, next_date=date(2025, 1, 3), frequency="weekly")
        infusion.advance()
        assert infusion.next_date == date(2025, 1, 10)

    def test_advance_monthly(self) -> None:
        infusion = CashInfusion(amount=2000.0, next_date=date(2025, 1, 3), frequency="monthly")
        infusion.advance()
        assert infusion.next_date == date(2025, 2, 2)

    def test_advance_no_frequency_is_noop(self) -> None:
        infusion = CashInfusion(amount=1500.0, next_date=date(2025, 1, 3))
        infusion.advance()
        assert infusion.next_date == date(2025, 1, 3)

    def test_advance_unknown_frequency_raises(self) -> None:
        infusion = CashInfusion(amount=1500.0, next_date=date(2025, 1, 3), frequency="quarterly")
        with pytest.raises(ValueError, match="Unknown cash_infusion frequency"):
            infusion.advance()


class TestRiskConfig:
    def test_defaults_disable_everything(self) -> None:
        cfg = RiskConfig()
        assert cfg.weighting == "equal"
        assert cfg.vol_lookback_days == 60
        assert cfg.vol_target is None
        assert cfg.drawdown_penalty is None
        assert cfg.drawdown_floor is None

    def test_drawdown_both_or_neither_neither(self) -> None:
        RiskConfig(drawdown_penalty=None, drawdown_floor=None)  # ok

    def test_drawdown_both_or_neither_both(self) -> None:
        RiskConfig(drawdown_penalty=1.5, drawdown_floor=0.5)  # ok

    def test_drawdown_penalty_without_floor_raises(self) -> None:
        with pytest.raises(ValueError, match="drawdown_floor"):
            RiskConfig(drawdown_penalty=1.5)

    def test_drawdown_floor_without_penalty_raises(self) -> None:
        with pytest.raises(ValueError, match="drawdown_penalty"):
            RiskConfig(drawdown_floor=0.5)

    def test_weighting_validation(self) -> None:
        with pytest.raises(ValueError, match="weighting"):
            RiskConfig(weighting="bogus")


class TestTaxConfig:
    def test_defaults_instantiation(self) -> None:
        cfg = TaxConfig()
        assert cfg.short_term_rate == 0.37
        assert cfg.long_term_rate == 0.20
        assert cfg.deductible_loss_cap == 3000.0
        assert cfg.payment_lag_days == 105

    def test_short_term_rate_zero_valid(self) -> None:
        TaxConfig(short_term_rate=0.0, long_term_rate=0.0)  # ok

    def test_short_term_rate_one_invalid(self) -> None:
        with pytest.raises(ValueError, match="short_term_rate"):
            TaxConfig(short_term_rate=1.0)

    def test_short_term_rate_negative_invalid(self) -> None:
        with pytest.raises(ValueError, match="short_term_rate"):
            TaxConfig(short_term_rate=-0.01)

    def test_long_term_rate_zero_valid(self) -> None:
        TaxConfig(long_term_rate=0.0)  # ok

    def test_long_term_rate_one_invalid(self) -> None:
        with pytest.raises(ValueError, match="long_term_rate"):
            TaxConfig(long_term_rate=1.0)

    def test_long_term_rate_negative_invalid(self) -> None:
        with pytest.raises(ValueError, match="long_term_rate"):
            TaxConfig(long_term_rate=-0.01)

    def test_deductible_loss_cap_zero_valid(self) -> None:
        TaxConfig(deductible_loss_cap=0.0)  # ok

    def test_deductible_loss_cap_negative_invalid(self) -> None:
        with pytest.raises(ValueError, match="deductible_loss_cap"):
            TaxConfig(deductible_loss_cap=-1.0)

    def test_payment_lag_days_zero_valid(self) -> None:
        TaxConfig(payment_lag_days=0)  # ok

    def test_payment_lag_days_negative_invalid(self) -> None:
        with pytest.raises(ValueError, match="payment_lag_days"):
            TaxConfig(payment_lag_days=-1)

    def test_long_term_exceeds_short_term_raises(self) -> None:
        with pytest.raises(ValueError, match="long_term_rate"):
            TaxConfig(short_term_rate=0.20, long_term_rate=0.37)
