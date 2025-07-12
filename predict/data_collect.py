# predict/data_collect.py

import pandas as pd

def data_collection(df):
    """
    Preprocesses the raw stock data frame.
    - Cleans column names
    - Creates new features
    - Splits data into training and validation sets
    """
    df.columns = df.columns.get_level_values(0)
    df["H_L"] = df["High"] - df["Low"]
    df["O_C"] = df["Open"] - df["Close"]

    for window in (7, 14, 21):
        df[f"MA_{window}"] = df["Close"].rolling(window).mean()

    df["STDDEV_7"] = df["Close"].rolling(7).std()
    df = df.dropna()

    n = len(df)
    split_index = int(n * 0.8)

    train_df = df.iloc[:split_index]
    val_df = df.iloc[split_index:]

    features = ["H_L", "O_C", "MA_7", "MA_14", "MA_21", "STDDEV_7", "Volume"]
    X_train, y_train = train_df[features], train_df["Close"]
    X_val, y_val = val_df[features], val_df["Close"]

    return X_train, y_train, X_val, y_val, val_df, features
