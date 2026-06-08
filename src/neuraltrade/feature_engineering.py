"""Feature engineering: real OHLCV → technical indicators.

Schema produced per ticker
--------------------------
date (index), open, high, low, close, volume,
sma_10, sma_50, ema_20,
rsi_14,
macd, macd_signal, macd_hist,
bb_middle, bb_upper, bb_lower,
atr_14, adx_14
= 18 columns (6 OHLCV + 12 indicators)

The Date index is NOT saved as a column; callers must use index_col=0 / parquet.
"""

from __future__ import annotations

import warnings

import pandas as pd
import ta

from neuraltrade.config import DataConfig, cfg

# Suppress ta's FutureWarnings from older pandas internals
warnings.filterwarnings("ignore", category=FutureWarning)

_DEFAULT_DATA_CFG = cfg.data

FEATURE_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "sma_10", "sma_50", "ema_20",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_middle", "bb_upper", "bb_lower",
    "atr_14", "adx_14",
]
N_FEATURES = len(FEATURE_COLUMNS)  # 17


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to a single-ticker OHLCV DataFrame.

    Input must have lowercase columns: open, high, low, close, volume.
    Returns DataFrame with indicator columns appended; NaN rows dropped.
    """
    df = df.copy()

    # --- Moving averages ---
    df["sma_10"] = ta.trend.sma_indicator(df["close"], window=10)
    df["sma_50"] = ta.trend.sma_indicator(df["close"], window=50)
    df["ema_20"] = ta.trend.ema_indicator(df["close"], window=20)

    # --- Momentum ---
    df["rsi_14"] = ta.momentum.rsi(df["close"], window=14)

    # --- MACD ---
    df["macd"] = ta.trend.macd(df["close"])
    df["macd_signal"] = ta.trend.macd_signal(df["close"])
    df["macd_hist"] = ta.trend.macd_diff(df["close"])

    # --- Bollinger Bands ---
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()

    # --- Volatility / trend ---
    df["atr_14"] = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], window=14
    )
    df["adx_14"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)

    df.dropna(inplace=True)
    return df[FEATURE_COLUMNS]


def build_engineered(
    raw_data: dict[str, pd.DataFrame] | None = None,
    data_cfg: DataConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Build and persist per-ticker engineered DataFrames.

    Parameters
    ----------
    raw_data:
        Output of data_collection.run(); if None, reads from raw_path parquet.
    data_cfg:
        DataConfig instance; defaults to global cfg.data.

    Returns
    -------
    Dict mapping ticker → engineered DataFrame (DatetimeIndex, FEATURE_COLUMNS).
    """
    dcfg = data_cfg or _DEFAULT_DATA_CFG

    if raw_data is None:
        parquet_path = dcfg.abs(dcfg.raw_path).with_suffix(".parquet")
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Raw data not found at {parquet_path}. Run data_collection first."
            )
        combined = pd.read_parquet(parquet_path)
        raw_data = {
            ticker: combined[combined["ticker"] == ticker].drop(columns="ticker")
            for ticker in combined["ticker"].unique()
        }

    engineered: dict[str, pd.DataFrame] = {}
    for ticker, df in raw_data.items():
        eng = add_indicators(df)
        engineered[ticker] = eng
        print(f"[{ticker}] engineered shape: {eng.shape}")

    # Save primary ticker as the main engineered CSV (for backwards compat)
    primary = dcfg.primary_ticker
    if primary in engineered:
        out_path = dcfg.abs(dcfg.engineered_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        engineered[primary].to_csv(out_path)
        print(f"Engineered data ({primary}) saved → {out_path}")

    # Save all tickers as a multi-ticker parquet
    all_frames = []
    for ticker, df in engineered.items():
        df = df.copy()
        df["ticker"] = ticker
        all_frames.append(df)
    multi_path = dcfg.abs(dcfg.engineered_path).with_suffix(".parquet")
    pd.concat(all_frames).to_parquet(multi_path)
    print(f"Multi-ticker parquet saved → {multi_path}")

    return engineered


def run(data_cfg: DataConfig | None = None) -> dict[str, pd.DataFrame]:
    return build_engineered(data_cfg=data_cfg)


if __name__ == "__main__":
    run()
