import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath='electricitydemand.csv'):
    # Load the dataset
    df = pd.read_csv(filepath)

    # Convert 'Timestamp' to datetime and set as index
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.set_index('Timestamp')

    # Drop rows where all values are missing
    df = df.dropna(how='all')

    # Fill remaining missing values if any (simple forward fill for time series)
    df = df.ffill()

    # Feature Engineering
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['Quarter'] = df.index.quarter
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)

    # Lagged Features
    df['Demand_lag_24hr'] = df['Demand'].shift(24)
    df['Demand_lag_168hr'] = df['Demand'].shift(168)

    # Drop rows with NaN values created by lagging
    df = df.dropna()

    return df

def get_split_data(df):
    # Features and Target
    X = df.drop(columns=['Demand'])
    Y = df['Demand']

    # Split the data (Shuffle=False for time series)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

    return X_train, X_test, Y_train, Y_test

def get_data_for_modeling():
    df = load_and_preprocess_data()
    X_train, X_test, Y_train, Y_test = get_split_data(df)
    return df, X_train, X_test, Y_train, Y_test

if __name__ == "__main__":
    df, X_train, X_test, Y_train, Y_test = get_data_for_modeling()
    print("Data Preprocessing Complete.")
    print(f"Dataset shape: {df.shape}")
    print(f"X_train shape: {X_train.shape}")
