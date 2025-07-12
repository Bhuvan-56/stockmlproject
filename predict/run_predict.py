# predict/run_predict.py

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import plotly.graph_objects as go

# Make sure you have this import
from .data_collect import data_collection

def run_prediction(stock_ticker, start_date, end_date):
    """
    Main function to run the entire pipeline using Plotly for visualization.
    """
    # Download data
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

    # --- Plotly Chart 1: Full Period ---
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=val_df.index, y=y_val.values, mode='lines+markers', name='Actual'))
    fig1.add_trace(go.Scatter(x=val_df.index, y=y_pred_ann, mode='lines+markers', name='Predicted (ANN)'))
    fig1.update_layout(
        title=f'Actual vs. Predicted Close Price for {stock_ticker}',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white'
    )
    # Convert plot to HTML
    plot1_html = fig1.to_html(full_html=False, include_plotlyjs='cdn')


    # # --- Plotly Chart 2: Last 7 Days ---
    # last7_df_index = val_df.index[-7:]
    # last7_actual = y_val.values[-7:]
    # last7_pred = y_pred_ann[-7:]
    
    # fig2 = go.Figure()
    # fig2.add_trace(go.Scatter(x=last7_df_index, y=last7_actual, mode='lines+markers', name='Actual'))
    # fig2.add_trace(go.Scatter(x=last7_df_index, y=last7_pred, mode='lines+markers', name='Predicted (ANN)'))
    # fig2.update_layout(
    #     title=f'Last 7 Days: Actual vs. Predicted for {stock_ticker}',
    #     xaxis_title='Date',
    #     yaxis_title='Price',
    #     template='plotly_white'
    # )
    # # Convert plot to HTML (no need to include plotly.js again)
    # plot2_html = fig2.to_html(full_html=False, include_plotlyjs=False)

    return plot1_html, None
