import numpy as np
import pytest
import torch

from psann.metrics import equity_curve, portfolio_metrics, sharpe_ratio
from psann.rewards import (
    RewardStrategyBundle,
    get_reward_strategy,
    register_reward_strategy,
)
from psann.rewards import _STRATEGY_REGISTRY as REWARD_REGISTRY
from psann.episodes import multiplicative_return_reward


def test_equity_curve_handles_short_series_gracefully():
    allocations = np.array([[0.6, 0.4]])
    prices = np.array([[1.0, 1.0]])

    curve = equity_curve(allocations, prices)

    assert curve.shape == (1,)
    assert np.allclose(curve, np.ones_like(curve))


def test_equity_curve_clamps_growth_with_transaction_costs():
    allocations = torch.tensor(
        [
            [0.5, 0.5],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    prices = torch.tensor(
        [
            [1.0, 1.0],
            [1.1, 0.9],
            [1.0, 1.2],
        ],
        dtype=torch.float32,
    )

    curve = equity_curve(allocations, prices, trans_cost=10.0)

    assert curve.shape == (3,)
    assert np.all(curve > 0.0)


def test_sharpe_ratio_returns_zero_for_single_observation():
    ratio = sharpe_ratio(np.array([0.01]))
    assert ratio == pytest.approx(0.0)


def test_sharpe_ratio_is_finite_for_constant_returns():
    ratio = sharpe_ratio(np.array([0.02, 0.02, 0.02]))
    assert np.isfinite(ratio)


def test_portfolio_metrics_handles_minimal_history():
    allocations = np.array([[1.0, 0.0]])
    prices = np.array([[1.0, 1.0]])

    metrics = portfolio_metrics(allocations, prices)

    assert metrics["cum_return"] == pytest.approx(0.0)
    assert metrics["log_return"] == pytest.approx(0.0)
    assert metrics["sharpe"] == pytest.approx(0.0)
    assert metrics["max_drawdown"] == pytest.approx(0.0)
    assert metrics["turnover"] == pytest.approx(0.0)


def test_portfolio_metrics_handles_underflow_without_nan():
    steps = 160
    allocations = np.ones((steps, 1), dtype=np.float64)
    # Prices decay exponentially to drive the equity curve towards underflow.
    prices = (1e-4) ** np.arange(steps, dtype=np.float64)
    prices = prices.reshape(-1, 1)

    metrics = portfolio_metrics(allocations, prices)

    assert np.isfinite(metrics["sharpe"])
    assert metrics["sharpe"] <= 0.0
    assert np.isfinite(metrics["log_return"])


def test_reward_strategy_registry_requires_explicit_overwrite():
    original_registry = dict(REWARD_REGISTRY)
    bundle = RewardStrategyBundle(
        reward_fn=multiplicative_return_reward,
        metrics_fn=None,
        description="test bundle",
    )
    replacement = RewardStrategyBundle(
        reward_fn=multiplicative_return_reward,
        metrics_fn=portfolio_metrics,
        description="replacement bundle",
    )

    try:
        register_reward_strategy("Custom", bundle)
        assert get_reward_strategy("custom") is bundle

        with pytest.raises(ValueError):
            register_reward_strategy("custom", bundle)

        register_reward_strategy("  CUSTOM  ", replacement, overwrite=True)
        assert get_reward_strategy("CuStOm") is replacement
    finally:
        REWARD_REGISTRY.clear()
        REWARD_REGISTRY.update(original_registry)
