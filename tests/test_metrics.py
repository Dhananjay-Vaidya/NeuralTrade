"""Unit tests for metrics.py."""

from __future__ import annotations

import math

import numpy as np

from neuraltrade.metrics import (
    buy_and_hold_return,
    max_drawdown,
    sharpe_ratio,
    total_return,
    win_rate,
)


class TestTotalReturn:
    def test_double(self):
        assert math.isclose(total_return([100, 200]), 1.0)

    def test_flat(self):
        assert total_return([100, 100, 100]) == 0.0

    def test_loss(self):
        assert math.isclose(total_return([100, 50]), -0.5)

    def test_single_element(self):
        assert total_return([100]) == 0.0


class TestMaxDrawdown:
    def test_no_drawdown(self):
        assert max_drawdown([100, 110, 120, 130]) == 0.0

    def test_full_loss(self):
        result = max_drawdown([100, 50, 0.001])
        assert result > 0.99

    def test_known_drawdown(self):
        # Peak 100 → trough 50 → drawdown 50%
        result = max_drawdown([100, 80, 50, 60, 70])
        assert math.isclose(result, 0.5, rel_tol=1e-6)


class TestSharpeRatio:
    def test_flat_returns_zero_sharpe(self):
        equity = [100.0] * 50
        assert sharpe_ratio(equity) == 0.0

    def test_positive_trend_positive_sharpe(self):
        equity = np.linspace(100, 200, 252).tolist()
        assert sharpe_ratio(equity) > 0

    def test_negative_trend_negative_sharpe(self):
        equity = np.linspace(200, 100, 252).tolist()
        assert sharpe_ratio(equity) < 0


class TestWinRate:
    def test_all_winning(self):
        log = [
            {"action": "BUY", "price": 100},
            {"action": "SELL", "price": 110},
            {"action": "BUY", "price": 105},
            {"action": "SELL", "price": 120},
        ]
        assert win_rate(log) == 1.0

    def test_all_losing(self):
        log = [
            {"action": "BUY", "price": 110},
            {"action": "SELL", "price": 100},
        ]
        assert win_rate(log) == 0.0

    def test_empty_log(self):
        assert win_rate([]) == 0.0

    def test_half_win_rate(self):
        log = [
            {"action": "BUY", "price": 100},
            {"action": "SELL", "price": 110},  # win
            {"action": "BUY", "price": 120},
            {"action": "SELL", "price": 110},  # loss
        ]
        assert win_rate(log) == 0.5


class TestBuyAndHoldReturn:
    def test_double(self):
        assert math.isclose(buy_and_hold_return([50, 100]), 1.0)

    def test_flat(self):
        assert buy_and_hold_return([100, 100]) == 0.0
