# src/data_loader.py

import yfinance as yf
import pandas as pd

def download_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download historical stock data for a given ticker between start and end dates.
    
    Parameters:
      ticker (str): Stock ticker symbol (e.g. "AAPL").
      start (str): Start date in 'YYYY-MM-DD' format.
      end (str): End date in 'YYYY-MM-DD' format.
    
    Returns:
      pd.DataFrame: DataFrame containing the stock data.
    """
    data = yf.download(ticker, start=start, end=end)
    data.reset_index(inplace=True)  # Convert the index (date) to a column.
    return data

if __name__ == "__main__":
    # Example: Download data for Apple from 2015 to 2023.
    df = download_stock_data("AAPL", "2015-01-01", "2023-01-01")
    # Save the data to the 'data' folder.
    df.to_csv("data/AAPL.csv", index=False)
    print("Data downloaded and saved to ../data/AAPL.csv")
