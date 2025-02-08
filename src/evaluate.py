# src/evaluate.py

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from model import prepare_data

def evaluate_model(model_path: str, data_path: str, feature: str, window_size: int = 60):
    """
    Evaluate the trained model by computing RMSE and plotting predictions vs. actual values.
    
    Parameters:
      model_path (str): Path to the saved model.
      data_path (str): Path to the CSV file with processed data.
      feature (str): The column to forecast.
      window_size (int): The number of time steps per sample.
    """
    # Load the processed data.
    df = pd.read_csv(data_path)
    # Prepare the data.
    X, y, scaler = prepare_data(df, feature, window_size)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    # Load the model.
    model = load_model(model_path)
    # Generate predictions.
    predictions = model.predict(X)
    # Inverse transform the predictions and actual values.
    predictions = scaler.inverse_transform(predictions)
    y_true = scaler.inverse_transform(y.reshape(-1, 1))
    # Calculate RMSE.
    rmse = np.sqrt(mean_squared_error(y_true, predictions))
    print(f"RMSE: {rmse}")
    
    # Plot actual vs. predicted values.
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='Actual Price', color='blue')
    plt.plot(predictions, label='Predicted Price', color='red')
    plt.xlabel("Time Steps")
    plt.ylabel("Stock Price")
    plt.title("Forecast vs Actual")
    plt.legend()
    plt.savefig("experiments/evaluation_plot.png")
    plt.show()
    
if __name__ == "__main__":
    evaluate_model("experiments/lstm_model.h5", "data/AAPL_features.csv", "Close")
