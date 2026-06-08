"""Shared fixtures for NeuralTrade tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neuraltrade.config import EnvConfig
from neuraltrade.feature_engineering import FEATURE_COLUMNS


def _make_price_series(n: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.01, size=n)
    prices = 100.0 * np.cumprod(1 + returns)
    return prices


def make_feature_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Return a synthetic DataFrame with FEATURE_COLUMNS columns.

    The 'close' column is a realistic price series; other columns are small
    standardised values that satisfy the env's shape contract.
    """
    prices = _make_price_series(n, seed)
    rng = np.random.default_rng(seed + 1)
    data = {col: rng.standard_normal(n) for col in FEATURE_COLUMNS}
    data["close"] = prices
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(data, index=dates)[FEATURE_COLUMNS]


@pytest.fixture()
def price_series(n: int = 200) -> np.ndarray:
    return _make_price_series(n)


@pytest.fixture()
def feature_df() -> pd.DataFrame:
    return make_feature_df()


@pytest.fixture()
def default_env_cfg() -> EnvConfig:
    return EnvConfig(
        initial_cash=10_000.0,
        transaction_cost_bps=0,
        slippage_bps=0,
        sizing="all_in",
        reward_type="log_return",
    )


@pytest.fixture()
def fee_env_cfg() -> EnvConfig:
    return EnvConfig(
        initial_cash=10_000.0,
        transaction_cost_bps=10,
        slippage_bps=5,
        sizing="all_in",
        reward_type="log_return",
    )
