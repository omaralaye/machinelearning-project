import pandas as pd
import numpy as np
import joblib
import holidays
from datetime import timedelta
import sys

def prepare_input_features(timestamp, temperature, humidity, historical_df):
    """
    Prepares the feature vector for a given timestamp and exogenous variables,
    using historical data for lags and rolling statistics.
    """
    try:
        ts = pd.to_datetime(timestamp)
    except:
        # Try custom format if standard fails
        ts = pd.to_datetime(timestamp, format='%d-%b-%y %H:%M')

    # Base features
    data = {
        'hour': ts.hour,
        'dayofweek': ts.dayofweek,
        'month': ts.month,
        'year': ts.year,
        'dayofyear': ts.dayofyear,
        'Temperature': temperature,
        'Humidity': humidity,
        'Quarter': (ts.month - 1) // 3 + 1,
        'weekofyear': ts.isocalendar().week,
        'is_weekend': 1 if ts.dayofweek >= 5 else 0
    }

    # Cyclical Features
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    data['month_sin'] = np.sin(2 * np.pi * (data['month'] - 1) / 12)
    data['month_cos'] = np.cos(2 * np.pi * (data['month'] - 1) / 12)

    # Holiday Feature
    in_holidays = holidays.IN(years=[ts.year])
    data['is_holiday'] = 1 if ts.date() in in_holidays else 0

    # Lagged and Rolling Features from history
    # We need Demand values for these.
    # Demand_lag_24hr: Demand at ts - 24h
    # Demand_lag_168hr: Demand at ts - 168h (1 week)
    # demand_rolling_mean_24hr: mean Demand from (ts - 24h) to (ts - 1h)

    target_24h = ts - timedelta(hours=24)
    target_168h = ts - timedelta(hours=168)

    # Look up in historical_df
    try:
        lag_24 = historical_df.loc[target_24h, 'Demand']
        if isinstance(lag_24, pd.Series): lag_24 = lag_24.iloc[0]
    except KeyError:
        print(f"Warning: No historical data for 24h lag at {target_24h}. Using mean.")
        lag_24 = historical_df['Demand'].mean()

    try:
        lag_168 = historical_df.loc[target_168h, 'Demand']
        if isinstance(lag_168, pd.Series): lag_168 = lag_168.iloc[0]
    except KeyError:
        print(f"Warning: No historical data for 168h lag at {target_168h}. Using mean.")
        lag_168 = historical_df['Demand'].mean()

    # Rolling stats for last 24 hours
    start_rolling = ts - timedelta(hours=24)
    end_rolling = ts - timedelta(hours=1)
    last_24h_data = historical_df.loc[start_rolling:end_rolling, 'Demand']

    if len(last_24h_data) < 12: # Heuristic: if we have less than half the expected data points
        print(f"Warning: Insufficient data for rolling statistics at {ts}. Using global stats.")
        rolling_mean = historical_df['Demand'].mean()
        rolling_std = historical_df['Demand'].std()
    else:
        rolling_mean = last_24h_data.mean()
        rolling_std = last_24h_data.std()

    data['Demand_lag_24hr'] = lag_24
    data['Demand_lag_168hr'] = lag_168
    data['demand_rolling_mean_24hr'] = rolling_mean
    data['demand_rolling_std_24hr'] = rolling_std

    return pd.DataFrame([data])

def main():
    print("--- Electricity Demand Forecast ---")

    # Load model and history
    try:
        model = joblib.load('xgb_electricity_demand_model.pkl')
        historical_df = pd.read_csv('electricitydemand.csv')
        # We need to combine date and hour to create proper timestamp for index
        # Looking at csv format: Timestamp,hour,... where Timestamp is 01-Jan-20
        historical_df['full_ts'] = pd.to_datetime(historical_df['Timestamp'], format='%d-%b-%y') + \
                                  pd.to_timedelta(historical_df['hour'], unit='h')
        historical_df = historical_df.dropna(subset=['full_ts'])
        historical_df.set_index('full_ts', inplace=True)
        historical_df.sort_index(inplace=True)
    except Exception as e:
        print(f"Error loading resources: {e}")
        return

    # User Input
    print("\nPlease enter the details for the forecast:")
    try:
        ts_input = input("Timestamp (YYYY-MM-DD HH:MM): ")
        temp_input = float(input("Temperature (Celsius): "))
        hum_input = float(input("Humidity (%): "))
    except ValueError:
        print("Invalid input. Please enter numerical values for temperature and humidity.")
        return

    # Prepare features
    features_df = prepare_input_features(ts_input, temp_input, hum_input, historical_df)

    # Ensure feature order matches model
    features_df = features_df[model.feature_names_in_]

    # Predict
    prediction = model.predict(features_df)[0]

    print(f"\nForecasted Electricity Demand: {prediction:.2f}")

if __name__ == "__main__":
    main()
