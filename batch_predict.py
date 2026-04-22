import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
import holidays
import os
import sys
import edf
import tensorflow as tf

def edf_preprocess(df):
    """
    Applies the same preprocessing as in edf.py but on a provided DataFrame.
    """
    df = df.copy()
    df = df.sort_index()
    df = df.ffill()

    # Feature Engineering
    df['hour'] = df.index.hour.values.astype(int)
    df['dayofweek'] = df.index.dayofweek.values.astype(int)
    df['month'] = df.index.month.values.astype(int)
    df['year'] = df.index.year.values.astype(int)
    df['dayofyear'] = df.index.dayofyear.values.astype(int)
    df['Quarter'] = df.index.quarter.values.astype(int)
    df['weekofyear'] = df.index.to_series().dt.isocalendar().week.values.astype(int)
    df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)

    # Cyclical Features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)

    # Holiday Feature
    years = df.index.year.unique().astype(int).tolist()
    holiday_calendar = holidays.IN(years=years)
    df['is_holiday'] = df.index.to_series().dt.date.isin(holiday_calendar).astype(int)

    # Lagged Features
    df['Demand_lag_24hr'] = df['Demand'].shift(24)
    df['Demand_lag_168hr'] = df['Demand'].shift(168)

    # Rolling Statistics
    df['demand_rolling_mean_24hr'] = df['Demand'].rolling(window=24).mean()
    df['demand_rolling_std_24hr'] = df['Demand'].rolling(window=24).std()

    return df

def generate_plots(df):
    """
    Generates comparison plots for both models.
    """
    plt.style.use('seaborn-v0_8-darkgrid')

    try:
        hist_df = pd.read_csv('electricitydemand.csv')
        thresholds = hist_df['Demand'].quantile([0.333, 0.666]).values
    except:
        thresholds = df['Actual Demand'].quantile([0.333, 0.666]).values if 'Actual Demand' in df.columns else [4400, 5600]

    if 'Actual Demand' in df.columns:
        # Comparison plot
        plt.figure(figsize=(15, 7))
        plt.plot(df.index, df['Actual Demand'], label='Actual Demand', color='black', alpha=0.6, linewidth=2)
        plt.plot(df.index, df['XGB Forecast'], label='XGBoost', color='blue', linestyle='--')
        plt.plot(df.index, df['LSTM Forecast'], label='LSTM', color='red', linestyle=':')

        plt.title('Multi-Model Electricity Demand Forecast Comparison', fontsize=16)
        plt.xlabel('Timestamp')
        plt.ylabel('Demand (MW)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('multi_model_comparison.png')
        print("Comparison plot saved to multi_model_comparison.png")

        # Metrics for both
        metrics = []
        for model_name in ['XGB', 'LSTM']:
            pred_col = f'{model_name} Forecast'
            mae = mean_absolute_error(df['Actual Demand'], df[pred_col])
            rmse = np.sqrt(mean_squared_error(df['Actual Demand'], df[pred_col]))
            mape = np.mean(np.abs((df['Actual Demand'] - df[pred_col]) / df['Actual Demand'])) * 100
            r2 = r2_score(df['Actual Demand'], df[pred_col])
            metrics.append([model_name, f"{mae:.2f}", f"{rmse:.2f}", f"{mape:.2f}%", f"{r2:.4f}"])

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('off')
        table = ax.table(cellText=metrics,
                         colLabels=['Model', 'MAE', 'RMSE', 'MAPE', 'R² Score'],
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)
        plt.title('Model Performance Comparison', fontsize=14)
        plt.savefig('batch_metrics.png')
        print("Metrics table saved to batch_metrics.png")
    else:
        # Forecast only
        plt.figure(figsize=(15, 7))
        plt.plot(df.index, df['XGB Forecast'], label='XGBoost', color='blue')
        plt.plot(df.index, df['LSTM Forecast'], label='LSTM', color='red')
        plt.title('Electricity Demand Forecasts')
        plt.legend()
        plt.savefig('forecast_only.png')

def main(input_csv, output_csv='predictions.csv'):
    # Load models
    try:
        xgb_model = joblib.load('xgb_electricity_demand_model.pkl')
        lstm_model = tf.keras.models.load_model('lstm_model.keras')
        scaler = joblib.load('scaler.pkl')
        lstm_features_list = joblib.load('lstm_features.pkl')
        print("Models and resources loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # Load data
    try:
        input_df = pd.read_csv(input_csv)
        input_df['full_ts'] = pd.to_datetime(input_df['Timestamp'], format='%d-%b-%y', errors='coerce') + \
                              pd.to_timedelta(input_df['hour'], unit='h', errors='coerce')
        if input_df['full_ts'].isnull().any():
            input_df['full_ts'] = pd.to_datetime(input_df['Timestamp'], errors='coerce')
        input_df = input_df.dropna(subset=['full_ts']).set_index('full_ts').sort_index()

        hist_df = pd.read_csv('electricitydemand.csv')
        hist_df['full_ts'] = pd.to_datetime(hist_df['Timestamp'], format='%d-%b-%y', errors='coerce') + \
                             pd.to_timedelta(hist_df['hour'], unit='h', errors='coerce')
        hist_df = hist_df.dropna(subset=['full_ts']).set_index('full_ts').sort_index()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    combined_df = pd.concat([hist_df, input_df])
    combined_df = combined_df[~combined_df.index.duplicated(keep='last')].sort_index()
    processed_df = edf_preprocess(combined_df)

    # Prediction set
    predict_df = processed_df.loc[input_df.index].copy()

    # XGBoost
    predict_df['XGB Forecast'] = xgb_model.predict(predict_df[xgb_model.feature_names_in_])

    # Optimized LSTM Prediction
    print("Preparing data for LSTM (vectorized)...")
    # Pre-scale the entire combined dataframe for efficiency
    lstm_data_to_scale = processed_df[lstm_features_list + ['Demand']].values
    scaled_full_data = scaler.transform(lstm_data_to_scale)

    all_sequences = []
    for ts in predict_df.index:
        idx = processed_df.index.get_loc(ts)
        if idx >= 24:
            # Extract the 24 hours preceding the current timestamp
            seq = scaled_full_data[idx-24:idx, :-1]
        else:
            # Pad with zeros if 24h history is not available
            needed = 24
            available = idx
            padding = needed - available
            seq = np.zeros((needed, len(lstm_features_list)))
            if available > 0:
                seq[padding:] = scaled_full_data[:idx, :-1]
        all_sequences.append(seq)

    all_sequences_np = np.array(all_sequences)
    print(f"Generating LSTM predictions for {len(all_sequences_np)} rows...")
    lstm_preds_scaled = lstm_model.predict(all_sequences_np, verbose=0)

    # Inverse transform all predictions at once
    dummy = np.zeros((len(lstm_preds_scaled), len(lstm_features_list) + 1))
    dummy[:, -1] = lstm_preds_scaled.flatten()
    predict_df['LSTM Forecast'] = scaler.inverse_transform(dummy)[:, -1]

    if 'Demand' in predict_df.columns:
        predict_df['Actual Demand'] = predict_df['Demand']

    # Output
    cols = ['XGB Forecast', 'LSTM Forecast']
    if 'Actual Demand' in predict_df.columns:
        cols = ['Actual Demand'] + cols

    predict_df[cols].to_csv(output_csv)
    print(f"Predictions saved to {output_csv}")

    generate_plots(predict_df)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python batch_predict.py <input_csv> [output_csv]")
    else:
        main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'predictions.csv')
