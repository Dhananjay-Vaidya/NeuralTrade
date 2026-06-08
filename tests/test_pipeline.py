"""Pipeline contract tests — feature engineering and data splits.

These tests run against the *actual* engineered data if present,
and fall back to synthetic data so the CI suite works without internet access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neuraltrade.config import DataConfig
from neuraltrade.data_splits import DataSplit, split_and_scale
from neuraltrade.feature_engineering import FEATURE_COLUMNS, N_FEATURES, add_indicators
from tests.conftest import make_feature_df

# ──────────────────────────────────────────────────────────────────────────────
# Feature engineering contracts
# ──────────────────────────────────────────────────────────────────────────────

class TestAddIndicators:
    def _make_ohlcv(self, n: int = 200) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        close = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        df = pd.DataFrame(
            {
                "open": close * rng.uniform(0.99, 1.0, n),
                "high": close * rng.uniform(1.0, 1.02, n),
                "low": close * rng.uniform(0.98, 1.0, n),
                "close": close,
                "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
            },
            index=dates,
        )
        return df

    def test_output_columns_match_schema(self):
        df = self._make_ohlcv()
        result = add_indicators(df)
        assert list(result.columns) == FEATURE_COLUMNS, (
            f"Column mismatch.\nExpected: {FEATURE_COLUMNS}\nGot: {list(result.columns)}"
        )

    def test_no_nans_after_engineering(self):
        df = self._make_ohlcv()
        result = add_indicators(df)
        nan_count = result.isna().sum().sum()
        assert nan_count == 0, f"add_indicators produced {nan_count} NaN values"

    def test_column_count_matches_N_FEATURES(self):
        df = self._make_ohlcv()
        result = add_indicators(df)
        assert result.shape[1] == N_FEATURES, (
            f"Expected {N_FEATURES} feature columns, got {result.shape[1]}"
        )

    def test_rows_reduced_due_to_warmup(self):
        """Indicator warmup (SMA_50 needs 50 rows) must shrink row count."""
        df = self._make_ohlcv(200)
        result = add_indicators(df)
        assert len(result) < len(df), "NaN rows from indicator warmup must be dropped"

    def test_high_always_gte_low(self):
        """Sanity: high >= low in the input survives through (no transformation done)."""
        df = self._make_ohlcv()
        result = add_indicators(df)
        assert (result["high"] >= result["low"]).all(), "high must be >= low throughout"


# ──────────────────────────────────────────────────────────────────────────────
# Data-split contracts
# ──────────────────────────────────────────────────────────────────────────────

class TestDataSplits:
    @pytest.fixture()
    def engineered_df(self) -> pd.DataFrame:
        return make_feature_df(300)

    @pytest.fixture()
    def split(self, engineered_df) -> DataSplit:
        cfg = DataConfig(train_ratio=0.7, val_ratio=0.15)
        return split_and_scale(engineered_df, cfg)

    def test_no_nans_in_any_split(self, split: DataSplit):
        for name, df in [("train", split.train), ("val", split.val), ("test", split.test)]:
            nans = df.isna().sum().sum()
            assert nans == 0, f"{name} split has {nans} NaN values after scaling"

    def test_chronological_non_overlap(self, split: DataSplit):
        assert split.train.index.max() < split.val.index.min(), "Train/val date overlap"
        assert split.val.index.max() < split.test.index.min(), "Val/test date overlap"

    def test_test_never_overlaps_train(self, split: DataSplit):
        train_dates = set(split.train.index)
        test_dates = set(split.test.index)
        overlap = train_dates & test_dates
        assert len(overlap) == 0, f"Train and test share {len(overlap)} dates — lookahead leak"

    def test_split_sizes_sum_to_total(self, engineered_df: pd.DataFrame, split: DataSplit):
        total = len(split.train) + len(split.val) + len(split.test)
        assert total == len(engineered_df), (
            f"Split sizes {len(split.train)}+{len(split.val)}+{len(split.test)}"
            f"={total} != {len(engineered_df)}"
        )

    def test_scaler_fitted_on_train_only(self, engineered_df: pd.DataFrame):
        """Scaler center must equal train median (RobustScaler), not full-data median."""
        cfg = DataConfig(train_ratio=0.7, val_ratio=0.15)
        split = split_and_scale(engineered_df, cfg)

        n_train = int(len(engineered_df) * 0.7)
        train_raw = engineered_df.iloc[:n_train]
        train_medians = train_raw[FEATURE_COLUMNS].median().values

        np.testing.assert_allclose(
            split.scaler.center_,
            train_medians,
            rtol=1e-5,
            err_msg="Scaler center_ must match training-set medians (no lookahead)",
        )

    def test_val_not_scaled_with_val_statistics(self, engineered_df: pd.DataFrame):
        """Val data scaled with train scaler must differ from val-fitted scaling."""
        cfg = DataConfig(train_ratio=0.7, val_ratio=0.15)
        split = split_and_scale(engineered_df, cfg)

        from sklearn.preprocessing import RobustScaler

        n_train = int(len(engineered_df) * 0.7)
        n_val = int(len(engineered_df) * 0.15)
        val_raw = engineered_df.iloc[n_train : n_train + n_val][FEATURE_COLUMNS]

        val_scaler = RobustScaler()
        val_self_scaled = val_scaler.fit_transform(val_raw)

        # The two scaled versions must be different (different scaler statistics)
        assert not np.allclose(split.val.values, val_self_scaled), (
            "Val split appears to have been scaled with its own statistics — lookahead leak"
        )

    def test_column_order_preserved(self, split: DataSplit):
        for name, df in [("train", split.train), ("val", split.val), ("test", split.test)]:
            assert list(df.columns) == FEATURE_COLUMNS, (
                f"{name} split column order differs from FEATURE_COLUMNS"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Env ↔ pipeline integration
# ──────────────────────────────────────────────────────────────────────────────

class TestEnvPipelineContract:
    def test_env_accepts_split_output(self):
        """TradingEnv must initialise without error from split_and_scale output."""
        from neuraltrade.environment import TradingEnv

        df = make_feature_df(200)
        cfg = DataConfig(train_ratio=0.7, val_ratio=0.15)
        split = split_and_scale(df, cfg)

        env = TradingEnv(df=split.train, price_col="close", seed=0)
        obs, info = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape
        assert not np.isnan(obs).any(), "Initial observation contains NaN"

    def test_obs_contains_no_nans_throughout_episode(self):
        from neuraltrade.environment import HOLD, TradingEnv

        df = make_feature_df(100)
        env = TradingEnv(df=df, price_col="close", seed=1)
        obs, _ = env.reset(seed=1)
        assert not np.isnan(obs).any()

        while True:
            obs, _, terminated, truncated, _ = env.step(HOLD)
            assert not np.isnan(obs).any(), "NaN in observation during episode"
            if terminated or truncated:
                break
