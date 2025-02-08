# src/feature_engineering.py

import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add new features to the DataFrame, such as moving averages and daily returns.
    
    Parameters:
      df (pd.DataFrame): Original DataFrame with stock data.
    
    Returns:
      pd.DataFrame: DataFrame with additional feature columns.
    """
    # Calculate a 10-day moving average of the closing price.
    df['MA10'] = df['Close'].rolling(window=10).mean()
    # Calculate a 50-day moving average of the closing price.
    df['MA50'] = df['Close'].rolling(window=50).mean()
    # Calculate daily returns (percentage change).
    df['Return'] = df['Close'].pct_change()
    # Drop rows with missing values (which appear because of rolling calculations).
    df = df.dropna().reset_index(drop=True)
    return df

if __name__ == "__main__":
    # Load raw data from CSV.
    df = pd.read_csv("data/AAPL.csv")
    # Add new features.
    df_features = add_features(df)
    # Save the processed data.
    df_features.to_csv("data/AAPL_features.csv", index=False)
    print("Features added and saved to data/AAPL_features.csv")
