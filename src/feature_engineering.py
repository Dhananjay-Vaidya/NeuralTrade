import pandas as pd
import ta

def add_technical_indicators(df, column='Stock_Price'):
    """
    Adds advanced technical indicators to the dataset using the `ta` library.
    :param df: Pandas DataFrame containing stock data.
    :param column: The stock price column to use for calculations.
    :return: DataFrame with additional technical indicators.
    """
    # Ensure Date column is datetime format
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df.set_index('Date', inplace=True)

    # Validate if selected column exists
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame. Available columns: {df.columns}")

    # Simulating High and Low Prices (Assumption)
    df['High'] = df[column] * 1.02  # Assume high is 2% higher than close
    df['Low'] = df[column] * 0.98   # Assume low is 2% lower than close

    # Moving Averages
    df['SMA_10'] = ta.trend.sma_indicator(df[column], window=10)
    df['SMA_50'] = ta.trend.sma_indicator(df[column], window=50)
    df['EMA_20'] = ta.trend.ema_indicator(df[column], window=20)

    # Relative Strength Index (RSI)
    df['RSI_14'] = ta.momentum.rsi(df[column], window=14)

    # Moving Average Convergence Divergence (MACD)
    df['MACD'] = ta.trend.macd(df[column])
    df['MACD_Signal'] = ta.trend.macd_signal(df[column])
    df['MACD_Hist'] = ta.trend.macd_diff(df[column])

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df[column], window=20, window_dev=2)
    df['BB_Middle'] = bb.bollinger_mavg()
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()

    # Corrected Average True Range (ATR)
    df['ATR_14'] = ta.volatility.average_true_range(df['High'], df['Low'], df[column], window=14)

    # Average Directional Index (ADX)
    df['ADX_14'] = ta.trend.adx(df['High'], df['Low'], df[column], window=14)

    # Drop NaN values generated by indicators (initial periods)
    df.dropna(inplace=True)

    return df

# Load cleaned stock data
input_file = "clean_stock_data.csv"
output_file = "engineered_stock_data.csv"

# Read CSV with correct headers
df = pd.read_csv(input_file)

# Fix column naming issue
df.columns = ['Date', 'AAPL', 'MSFT', 'GOOGL']

# Select stock column dynamically (modify based on available data)
target_stock = "AAPL"

# Apply feature engineering
df = add_technical_indicators(df, column=target_stock)

# Save the engineered dataset
df.to_csv(output_file)

print("✅ Feature engineering complete. Saved to:", output_file)
