"""Chronological train / val / test split with no lookahead leakage.

Scaler is fitted ONLY on the training slice and applied to val/test.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import RobustScaler

from neuraltrade.config import DataConfig, cfg
from neuraltrade.feature_engineering import FEATURE_COLUMNS, build_engineered

_DEFAULT_DATA_CFG = cfg.data


@dataclass
class DataSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    scaler: RobustScaler
    # Unscaled price series for each split (used by env and evaluation)
    train_prices: pd.Series
    val_prices: pd.Series
    test_prices: pd.Series


def split_and_scale(
    df: pd.DataFrame,
    data_cfg: DataConfig | None = None,
    price_col: str = "close",
) -> DataSplit:
    """Split *df* chronologically; fit scaler on train, transform all splits.

    Parameters
    ----------
    df:
        Engineered DataFrame with DatetimeIndex and columns = FEATURE_COLUMNS.
    data_cfg:
        DataConfig; used for train_ratio and val_ratio.
    price_col:
        Column to preserve as raw prices alongside scaled features.
    """
    dcfg = data_cfg or _DEFAULT_DATA_CFG

    df = df.sort_index()
    n = len(df)
    n_train = int(n * dcfg.train_ratio)
    n_val = int(n * dcfg.val_ratio)
    # test gets the remainder
    n_test = n - n_train - n_val

    if n_test <= 0:
        raise ValueError(
            f"Not enough rows for test split (n={n}, train={n_train}, val={n_val})"
        )

    train_raw = df.iloc[:n_train].copy()
    val_raw = df.iloc[n_train : n_train + n_val].copy()
    test_raw = df.iloc[n_train + n_val :].copy()

    # --- Fit scaler on train only ---
    scaler = RobustScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_raw[FEATURE_COLUMNS]),
        index=train_raw.index,
        columns=FEATURE_COLUMNS,
    )
    val_scaled = pd.DataFrame(
        scaler.transform(val_raw[FEATURE_COLUMNS]),
        index=val_raw.index,
        columns=FEATURE_COLUMNS,
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_raw[FEATURE_COLUMNS]),
        index=test_raw.index,
        columns=FEATURE_COLUMNS,
    )

    # Sanity checks — no NaNs after scaling
    for name, frame in [("train", train_scaled), ("val", val_scaled), ("test", test_scaled)]:
        nan_count = frame.isna().sum().sum()
        assert nan_count == 0, f"{name} split has {nan_count} NaN values after scaling"

    # Chronological non-overlap assertions
    assert train_raw.index.max() < val_raw.index.min(), "Train/val date overlap detected"
    assert val_raw.index.max() < test_raw.index.min(), "Val/test date overlap detected"

    return DataSplit(
        train=train_scaled,
        val=val_scaled,
        test=test_scaled,
        scaler=scaler,
        train_prices=train_raw[price_col],
        val_prices=val_raw[price_col],
        test_prices=test_raw[price_col],
    )


def load_and_split(
    data_cfg: DataConfig | None = None,
    ticker: str | None = None,
) -> DataSplit:
    """Load engineered CSV/parquet for a ticker and return a DataSplit."""
    dcfg = data_cfg or _DEFAULT_DATA_CFG
    t = ticker or dcfg.primary_ticker

    parquet_path = dcfg.abs(dcfg.engineered_path).with_suffix(".parquet")
    csv_path = dcfg.abs(dcfg.engineered_path)

    if parquet_path.exists():
        combined = pd.read_parquet(parquet_path)
        df = combined[combined["ticker"] == t].drop(columns="ticker")
    elif csv_path.exists():
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if not set(FEATURE_COLUMNS).issubset(df.columns):
            df = _recover_engineered_from_legacy(data_cfg=dcfg, ticker=t, csv_path=csv_path)
    else:
        raise FileNotFoundError(
            f"Engineered data not found at {parquet_path} or {csv_path}. "
            "Run `python -m neuraltrade.data_collection` and "
            "`python -m neuraltrade.feature_engineering` first."
        )

    df.index = pd.to_datetime(df.index)
    df = df[FEATURE_COLUMNS]  # ensure consistent column order
    return split_and_scale(df, dcfg)


def _recover_engineered_from_legacy(
    data_cfg: DataConfig,
    ticker: str,
    csv_path: Path,
) -> pd.DataFrame:
    """Recover when the engineered CSV is from the pre-package legacy schema."""
    parquet_path = data_cfg.abs(data_cfg.raw_path).with_suffix(".parquet")
    if not parquet_path.exists():
        raise FileNotFoundError(
            "The default engineered dataset is in a legacy format and cannot be used by the "
            "current package code. Missing raw OHLCV parquet at "
            f"{parquet_path}. Run `python -m neuraltrade.data_collection` and then "
            "`python -m neuraltrade.feature_engineering` to rebuild the dataset."
        )

    build_engineered(data_cfg=data_cfg)
    rebuilt_csv = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if not set(FEATURE_COLUMNS).issubset(rebuilt_csv.columns):
        raise ValueError(
            f"Rebuilt engineered data at {csv_path} still does not match the expected schema: "
            f"{FEATURE_COLUMNS}"
        )
    return rebuilt_csv


def save_scaler(scaler: RobustScaler, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)


def load_scaler(path: Path) -> RobustScaler:
    with open(path, "rb") as f:
        return pickle.load(f)
