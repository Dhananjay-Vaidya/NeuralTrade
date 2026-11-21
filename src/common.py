"""Shared settings and data helpers for the NeuralTrade pipeline.

All paths are resolved relative to the repo root, so every script works
regardless of the current working directory.
"""

import json
from pathlib import Path

import pandas as pd

TICKER = "AAPL"  # also works with "MSFT" or "GOOGL"

START_DATE = "2019-01-01"
END_DATE = "2025-01-01"  # exclusive
TRAIN_END = "2022-12-31"  # train: 2019-2022
TEST_START = "2023-01-01"  # test: 2023-2024

INITIAL_CASH = 100_000.0
TRANSACTION_COST = 0.001  # 0.1% per trade
SEED = 42

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
RESULTS_DIR = ROOT / "results"

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def raw_path(ticker=TICKER):
    return DATA_DIR / f"raw_{ticker}.csv"


def features_path(ticker=TICKER):
    return DATA_DIR / f"features_{ticker}.csv"


def model_path(ticker=TICKER):
    return MODEL_DIR / f"ppo_{ticker}.zip"


def norm_path(ticker=TICKER):
    return MODEL_DIR / f"ppo_{ticker}_norm.json"


def load_features(ticker=TICKER):
    path = features_path(ticker)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run src/feature_engineering.py first.")
    return pd.read_csv(path, index_col="Date", parse_dates=True)


def feature_columns(df):
    """Observation features = every column that is not raw price/volume."""
    return [c for c in df.columns if c not in PRICE_COLUMNS]


def split_train_test(df):
    return df.loc[:TRAIN_END], df.loc[TEST_START:]


def compute_norm_stats(train_df):
    cols = feature_columns(train_df)
    std = train_df[cols].std().replace(0, 1.0)
    return {"feature_columns": cols, "mean": train_df[cols].mean().to_dict(), "std": std.to_dict()}


def save_norm_stats(stats, ticker=TICKER):
    MODEL_DIR.mkdir(exist_ok=True)
    norm_path(ticker).write_text(json.dumps(stats, indent=2))


def load_norm_stats(ticker=TICKER):
    return json.loads(norm_path(ticker).read_text())
