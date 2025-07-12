# predict/run_predict.py

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from datetime import date

from .data_collect import data_collection

def run_prediction(stock_ticker, start_date, end_date):
    """
    Main function to run the entire pipeline:
    - Download data
    - Preprocess data
    - Train the ANN model
    - Make predictions
    - Generate and save plots
    """
    # Download data using yfinance
    df = yf.download(
        tickers=stock_ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        print("No data downloaded. Check the ticker or date range.")
        return None, None

    # Preprocess data
    X_train, y_train, X_val, y_val, val_df, features = data_collection(df)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Build and train the ANN model
    model = Sequential([
        Dense(64, activation='relu', input_dim=X_train_scaled.shape[1]),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=500,
        batch_size=32,
        verbose=0
    )

    # Make predictions
    y_pred_ann = model.predict(X_val_scaled).flatten()

    # Generate and save the first plot (Actual vs. Predicted)
    plt.figure(figsize=(12, 6))
    plt.plot(val_df.index, y_val.values, marker='o', linestyle='-', label='Actual')
    plt.plot(val_df.index, y_pred_ann, marker='x', linestyle='--', label='Predicted (ANN)')
    plt.title(f'Actual vs. Predicted Close Price for {stock_ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig('static/plot1.png')
    plt.close()

    # Generate and save the second plot (Last 7 days)
    last7_df_index = val_df.index[-7:]
    last7_actual = y_val.values[-7:]
    last7_pred = y_pred_ann[-7:]

    plt.figure(figsize=(12, 6))
    plt.plot(last7_df_index, last7_actual, marker='o', linestyle='-', label='Actual')
    plt.plot(last7_df_index, last7_pred, marker='x', linestyle='--', label='Predicted (ANN)')
    plt.title(f'Actual vs. Predicted Close Price for {stock_ticker} (Last 7 Days)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig('static/plot2.png')
    plt.close()

    return 'static/plot1.png', 'static/plot2.png'
