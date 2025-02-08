# src/model.py

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def prepare_data(df: pd.DataFrame, feature: str, window_size: int = 60):
    """
    Prepare data for the LSTM model by scaling and creating sequences.
    
    Parameters:
      df (pd.DataFrame): DataFrame containing the data (e.g. 'Close' prices).
      feature (str): The column name to forecast.
      window_size (int): The number of previous time steps to use as input.
    
    Returns:
      tuple: (X, y, scaler) where X is input sequences, y are target values, and scaler is the fitted scaler.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(df[[feature]].values)
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y), scaler

def build_lstm_model(input_shape):
    """
    Build and compile an LSTM model.
    
    Parameters:
      input_shape (tuple): The shape of the input data (timesteps, features).
    
    Returns:
      model: A compiled Keras Sequential model.
    """
    model = Sequential()
    # First LSTM layer (50 units, returning sequences).
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    # Second LSTM layer (50 units).
    model.add(LSTM(50))
    # Dense layer to output a single prediction.
    model.add(Dense(1))
    # Compile the model with mean squared error loss and Adam optimiser.
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

if __name__ == "__main__":
    # Load processed data.
    df = pd.read_csv("data/AAPL_features.csv")
    # Prepare data using the 'Close' column.
    X, y, scaler = prepare_data(df, "Close", window_size=60)
    # Reshape X for LSTM input: (samples, timesteps, features).
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Build the LSTM model.
    model = build_lstm_model((X.shape[1], 1))
    # Display the model summary.
    model.summary()
    # Train the model.
    model.fit(X, y, epochs=20, batch_size=32)
    # Save the trained model.
    model.save("experiments/lstm_model.h5")
    print("Model trained and saved to experiments/lstm_model.h5")
