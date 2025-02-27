import yfinance as yf
import pandas as pd
import time

stock_symbols = ["AAPL", "MSFT", "GOOGL"]

def download_data(ticker):
    """Download stock data with retries in case of failure."""
    for i in range(3):  # Retry up to 3 times
        try:
            stock_data = yf.download(ticker, start="2019-01-01", end="2024-01-01", auto_adjust=True, progress=False)
            return stock_data
        except Exception as e:
            print(f"Retry {i+1}: Failed to download {ticker} due to {e}")
            time.sleep(2)  # Wait before retrying
    return None  # Return None if all retries fail

# Fetch data for each stock separately
data_dict = {ticker: download_data(ticker) for ticker in stock_symbols}

# Combine into a single DataFrame using "Close" prices
data = pd.concat({k: v["Close"] for k, v in data_dict.items()}, axis=1)

# Display the first few rows
print(data.head())

# Save to CSV
data.to_csv("data/stock_data1.csv")
