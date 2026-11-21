"""Compute technical indicators from real High/Low/Close data.

Price-level indicators (SMA, EMA, Bollinger, MACD, ATR) are expressed relative
to the close price so they stay comparable between the train period (2019-22)
and the test period (2023-24), when AAPL trades at a very different price level.

Usage: python src/feature_engineering.py
"""

import pandas as pd
import ta

from common import TICKER, features_path, raw_path


def add_indicators(df):
    close, high, low = df["Close"], df["High"], df["Low"]
    out = df.copy()

    out["SMA_10"] = ta.trend.sma_indicator(close, window=10) / close - 1
    out["SMA_50"] = ta.trend.sma_indicator(close, window=50) / close - 1
    out["EMA_20"] = ta.trend.ema_indicator(close, window=20) / close - 1
    out["RSI_14"] = ta.momentum.rsi(close, window=14)

    macd = ta.trend.MACD(close)
    out["MACD"] = macd.macd() / close
    out["MACD_Signal"] = macd.macd_signal() / close
    out["MACD_Hist"] = macd.macd_diff() / close

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    out["BB_Upper"] = bb.bollinger_hband() / close - 1
    out["BB_Lower"] = bb.bollinger_lband() / close - 1
    out["BB_PctB"] = bb.bollinger_pband()

    out["ATR_14"] = ta.volatility.average_true_range(high, low, close, window=14) / close
    out["ADX_14"] = ta.trend.adx(high, low, close, window=14)

    return out.dropna()


def main():
    path = raw_path()
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run src/datacollection.py first.")
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    feats = add_indicators(df)
    feats.to_csv(features_path())
    print(f"Saved {len(feats)} rows x {feats.shape[1]} columns for {TICKER} to {features_path()}")


if __name__ == "__main__":
    main()
