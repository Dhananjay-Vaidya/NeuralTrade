"""Download real OHLCV data from Yahoo Finance."""

from __future__ import annotations

import inspect
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from neuraltrade.config import DataConfig, cfg

_DEFAULT_DATA_CFG = cfg.data


def _download_kwargs() -> dict:
    """Build version-compatible kwargs for `yf.download`."""
    kwargs = {
        "auto_adjust": True,
        "progress": False,
    }
    signature = inspect.signature(yf.download)
    if "multi_level_index" in signature.parameters:
        kwargs["multi_level_index"] = False
    return kwargs


def _normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten and normalize Yahoo Finance output to lowercase OHLCV columns."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(c).lower().strip() for c in df.columns]
    df.index.name = "date"

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Downloaded frame missing OHLCV columns: {sorted(missing)}")

    return df[["open", "high", "low", "close", "volume"]].copy()


def download_ohlcv(
    tickers: list[str],
    start: str,
    end: str,
    retries: int = 3,
    pause: float = 2.0,
) -> dict[str, pd.DataFrame]:
    """Return {ticker: OHLCV DataFrame} with DatetimeIndex."""
    results: dict[str, pd.DataFrame] = {}
    kwargs = _download_kwargs()
    for ticker in tickers:
        for attempt in range(retries):
            try:
                df = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    **kwargs,
                )
                if df.empty:
                    raise ValueError(f"Empty download for {ticker}")
                df = _normalize_ohlcv_columns(df)
                results[ticker] = df
                break
            except Exception as exc:
                print(f"[{ticker}] attempt {attempt + 1} failed: {exc}")
                if attempt < retries - 1:
                    time.sleep(pause)
        else:
            print(f"[{ticker}] all retries exhausted - skipping")
    return results


def save_raw(data: dict[str, pd.DataFrame], data_cfg: DataConfig | None = None) -> Path:
    """Persist raw data as a multi-ticker parquet, return path."""
    dcfg = data_cfg or _DEFAULT_DATA_CFG
    out_path = dcfg.abs(dcfg.raw_path).with_suffix(".parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not data:
        raise ValueError(
            "No ticker data was downloaded. Check your internet connection, ticker symbols, "
            "or yfinance compatibility before running feature engineering."
        )

    frames = []
    for ticker, df in data.items():
        df = df.copy()
        df["ticker"] = ticker
        frames.append(df)

    combined = pd.concat(frames).sort_index()
    combined.to_parquet(out_path)
    print(f"Raw OHLCV saved -> {out_path}  ({combined.shape})")
    return out_path


def run(data_cfg: DataConfig | None = None) -> dict[str, pd.DataFrame]:
    dcfg = data_cfg or _DEFAULT_DATA_CFG
    data = download_ohlcv(dcfg.tickers, dcfg.start_date, dcfg.end_date)
    save_raw(data, dcfg)
    return data


if __name__ == "__main__":
    run()
