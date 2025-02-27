import pandas as pd

def clean_stock_data(input_file, output_file):
    # Read CSV while skipping the first row
    df = pd.read_csv(input_file, skiprows=1, parse_dates=['Ticker'])
    
    # Rename 'Ticker' to 'Date' since it's actually the date column
    df.rename(columns={'Ticker': 'Date'}, inplace=True)
    
    # Sort by Date
    df.sort_values(by='Date', inplace=True)
    
    # Handle missing values
    df.fillna(method='ffill', inplace=True)  # Forward-fill missing values
    df.dropna(inplace=True)  # Drop remaining NaN values
    
    # Save cleaned data
    df.to_csv(output_file, index=False)
    print("Data cleaning complete. Saved to:", output_file)

# Usage
input_file = "data/stock_data1.csv"  # Adjust path if needed
output_file = "clean_stock_data.csv"
clean_stock_data(input_file, output_file)
