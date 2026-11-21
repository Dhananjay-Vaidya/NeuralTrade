"""Download real daily OHLCV data from Yahoo Finance.

Usage: python src/datacollection.py
"""

import time

import yfinance as yf

from common import DATA_DIR, END_DATE, PRICE_COLUMNS, START_DATE, TICKER, raw_path


def download_ohlcv(ticker, retries=3):
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                ticker,
                start=START_DATE,
                end=END_DATE,
                auto_adjust=True,
                progress=False,
                multi_level_index=False,
            )
            if not df.empty:
                return df
            print(f"Attempt {attempt}: empty response for {ticker}")
        except Exception as e:  # network errors etc.
            print(f"Attempt {attempt}: failed to download {ticker}: {e}")
        time.sleep(2)
    raise RuntimeError(f"Could not download data for {ticker}")


def main():
    DATA_DIR.mkdir(exist_ok=True)
    df = download_ohlcv(TICKER)[PRICE_COLUMNS].dropna()
    df.index.name = "Date"
    df.to_csv(raw_path())
    print(f"Saved {len(df)} rows ({df.index[0].date()} to {df.index[-1].date()}) to {raw_path()}")


if __name__ == "__main__":
    main()
