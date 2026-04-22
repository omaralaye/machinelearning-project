import pandas as pd
import numpy as np
import joblib
import holidays
from datetime import timedelta
import sys
import edf
import tensorflow as tf

def prepare_input_features(timestamp, temperature, humidity, historical_df, feature_names):
    """
    Prepares the feature vector for a given timestamp and exogenous variables,
    using historical data for lags and rolling statistics.
    """
    try:
        ts = pd.to_datetime(timestamp)
    except:
        ts = pd.to_datetime(timestamp, format='%d-%b-%y %H:%M')

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

    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    data['month_sin'] = np.sin(2 * np.pi * (data['month'] - 1) / 12)
    data['month_cos'] = np.cos(2 * np.pi * (data['month'] - 1) / 12)

    in_holidays = holidays.IN(years=[ts.year])
    data['is_holiday'] = 1 if ts.date() in in_holidays else 0

    target_24h = ts - timedelta(hours=24)
    target_168h = ts - timedelta(hours=168)

    try:
        lag_24 = historical_df.loc[target_24h, 'Demand']
        if isinstance(lag_24, pd.Series): lag_24 = lag_24.iloc[0]
    except KeyError:
        lag_24 = historical_df['Demand'].mean()

    try:
        lag_168 = historical_df.loc[target_168h, 'Demand']
        if isinstance(lag_168, pd.Series): lag_168 = lag_168.iloc[0]
    except KeyError:
        lag_168 = historical_df['Demand'].mean()

    start_rolling = ts - timedelta(hours=24)
    end_rolling = ts - timedelta(hours=1)
    last_24h_data = historical_df.loc[start_rolling:end_rolling, 'Demand']

    if len(last_24h_data) < 12:
        rolling_mean = historical_df['Demand'].mean()
        rolling_std = historical_df['Demand'].std()
    else:
        rolling_mean = last_24h_data.mean()
        rolling_std = last_24h_data.std()

    data['Demand_lag_24hr'] = lag_24
    data['Demand_lag_168hr'] = lag_168
    data['demand_rolling_mean_24hr'] = rolling_mean
    data['demand_rolling_std_24hr'] = rolling_std

    df = pd.DataFrame([data])
    return df[feature_names]

def prepare_lstm_sequence(timestamp, temperature, humidity, historical_df, lstm_features, scaler):
    """
    Prepares a 24-hour sequence for LSTM prediction.
    """
    ts = pd.to_datetime(timestamp)
    sequence_data = []

    # We need 24 hours of data leading up to the target timestamp
    for i in range(24, 0, -1):
        curr_ts = ts - timedelta(hours=i)

        # For the target timestamp's exogenous variables, we use inputs.
        # But for the sequence, we either need historical exogenous or assume they are available.
        # Since we only have historical 'Demand', 'Temperature', 'Humidity' in the CSV,
        # let's try to get them from historical_df.

        try:
            row = historical_df.loc[curr_ts]
            if isinstance(row, pd.DataFrame): row = row.iloc[0]
            curr_temp = row['Temperature']
            curr_hum = row['Humidity']
        except KeyError:
            # Fallback to current input if history missing
            curr_temp = temperature
            curr_hum = humidity

        features_df = prepare_input_features(curr_ts, curr_temp, curr_hum, historical_df, lstm_features)

        # Get Demand for this historical point to include in scaling
        try:
            curr_demand = historical_df.loc[curr_ts, 'Demand']
            if isinstance(curr_demand, pd.Series): curr_demand = curr_demand.iloc[0]
        except KeyError:
            curr_demand = historical_df['Demand'].mean()

        # Combine features and demand for scaling (scaler expects all features + demand)
        combined_row = features_df.iloc[0].tolist() + [curr_demand]
        scaled_row = scaler.transform([combined_row])[0]

        # LSTM input sequence is only the features (last column is target)
        sequence_data.append(scaled_row[:-1])

    return np.array([sequence_data])

def main():
    print("--- Electricity Demand Forecast (Multi-Model) ---")

    # Load resources
    try:
        xgb_model = joblib.load('xgb_electricity_demand_model.pkl')
        lstm_model = tf.keras.models.load_model('lstm_model.keras')
        scaler = joblib.load('scaler.pkl')
        lstm_features = joblib.load('lstm_features.pkl')

        historical_df = pd.read_csv('electricitydemand.csv')
        historical_df['full_ts'] = pd.to_datetime(historical_df['Timestamp'], format='%d-%b-%y') + \
                                  pd.to_timedelta(historical_df['hour'], unit='h')
        historical_df = historical_df.dropna(subset=['full_ts']).set_index('full_ts').sort_index()
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
        print("Invalid input.")
        return

    # XGBoost Prediction
    xgb_features = prepare_input_features(ts_input, temp_input, hum_input, historical_df, xgb_model.feature_names_in_)
    xgb_pred = xgb_model.predict(xgb_features)[0]

    # LSTM Prediction
    lstm_seq = prepare_lstm_sequence(ts_input, temp_input, hum_input, historical_df, lstm_features, scaler)
    lstm_pred_scaled = lstm_model.predict(lstm_seq, verbose=0)[0][0]

    # Inverse transform LSTM prediction
    dummy = np.zeros((1, len(lstm_features) + 1))
    dummy[0, -1] = lstm_pred_scaled
    lstm_pred = scaler.inverse_transform(dummy)[0, -1]

    print(f"\nForecast Results:")
    print(f"XGBoost Forecast: {xgb_pred:.2f} MW")
    print(f"LSTM Forecast:    {lstm_pred:.2f} MW")
    print(f"Average Forecast: {(xgb_pred + lstm_pred)/2:.2f} MW")

    # Comparison logic
    actual_input = input("\nActual demand value (optional, press Enter to skip): ")
    if actual_input.strip():
        try:
            actual_val = float(actual_input)
            hist_range = historical_df['Demand'].max() - historical_df['Demand'].min()

            print(f"\n--- Performance Comparison ---")
            for name, pred in [("XGBoost", xgb_pred), ("LSTM", lstm_pred)]:
                error = actual_val - pred
                norm_err = edf.calculate_normalized_error(actual_val, pred, range_val=hist_range)
                print(f"{name:8} | Pred: {pred:7.2f} | Error: {error:7.2f} | Norm Error: {norm_err:.4f}")

            # Anomaly check
            if abs((actual_val - xgb_pred)/actual_val) > 0.10 or abs((actual_val - lstm_pred)/actual_val) > 0.10:
                print("\nRED FLAG - ANOMALY DETECTED: Significant deviation in one or both models.")
        except ValueError:
            pass

if __name__ == "__main__":
    main()
