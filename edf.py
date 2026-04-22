import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import holidays

def load_and_preprocess_data(filepath='electricitydemand.csv'):
    # Load the dataset
    df = pd.read_csv(filepath)

    # Convert 'Timestamp' to datetime and combine with hour for proper index
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d-%b-%y', errors='coerce') + \
                      pd.to_timedelta(df['hour'], unit='h', errors='coerce')
    df = df.dropna(subset=['Timestamp'])
    df = df.set_index('Timestamp')

    # Drop rows where all values are missing
    df = df.dropna(how='all')

    # Fill remaining missing values if any (simple forward fill for time series)
    df = df.ffill()
    df = df.dropna()

    # Feature Engineering
    df['hour'] = df.index.hour.values.astype(int)
    df['dayofweek'] = df.index.dayofweek.values.astype(int)
    df['month'] = df.index.month.values.astype(int)
    df['year'] = df.index.year.values.astype(int)
    df['dayofyear'] = df.index.dayofyear.values.astype(int)
    df['Quarter'] = df.index.quarter.values.astype(int)
    # Use a safe way to get week of year without NAs
    df['weekofyear'] = df.index.to_series().dt.isocalendar().week.values.astype(int)
    df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)

    # Cyclical Features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)

    # Holiday Feature (assuming India as per edf.ipynb)
    years = df.index.year.unique().astype(int).tolist()
    holiday_calendar = holidays.IN(years=years)
    df['is_holiday'] = df.index.to_series().dt.date.isin(holiday_calendar).astype(int)

    # Lagged Features
    df['Demand_lag_24hr'] = df['Demand'].shift(24)
    df['Demand_lag_168hr'] = df['Demand'].shift(168)

    # Rolling Statistics
    df['demand_rolling_mean_24hr'] = df['Demand'].rolling(window=24).mean()
    df['demand_rolling_std_24hr'] = df['Demand'].rolling(window=24).std()

    # Drop rows with NaN values created by lagging and rolling
    df = df.dropna()

    return df

def get_season(month):
    """Maps month to season."""
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring' # Or Summer depending on region, prompt used Summer/Winter/Spring/Autumn
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

def calculate_normalized_error(actual, predicted, range_val=None):
    """
    Calculates error normalized to 0-1 range.
    If range_val is not provided, it uses the range of the actual values provided.
    """
    if range_val is None:
        range_val = actual.max() - actual.min()

    if range_val == 0:
        # Fallback to mean if range is 0 (e.g., single point)
        denom = actual.mean() if actual.mean() != 0 else 1
        norm_err = np.abs(actual - predicted) / denom
    else:
        norm_err = np.abs(actual - predicted) / range_val

    # Ensure it's in (0-1) range as requested
    return np.clip(norm_err, 0, 1)

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
