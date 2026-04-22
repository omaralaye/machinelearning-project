import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import holidays
import os
import sys

def edf_preprocess(df):
    """
    Applies the same preprocessing as in edf.py but on a provided DataFrame.
    """
    df = df.copy()
    # Sort by index to ensure time-based operations work correctly
    df = df.sort_index()

    # Fill remaining missing values if any (simple forward fill for time series)
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
    Generates forecast comparison plot and metrics diagram.
    """
    plt.style.use('seaborn-v0_8-darkgrid')

    if 'Actual Demand' not in df.columns or df['Actual Demand'].isnull().all():
        # Just plot forecast
        plt.figure(figsize=(15, 7))
        plt.plot(df.index, df['Forecast Demand'], label='Forecasted Demand', color='red', linewidth=2)
        plt.title('Electricity Demand Forecast', fontsize=16)
        plt.xlabel('Timestamp', fontsize=12)
        plt.ylabel('Demand (MW)', fontsize=12)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('forecast_plot.png')
        print("Forecast plot saved to forecast_plot.png")
        return

    # Actual vs Forecast
    plt.figure(figsize=(15, 7))
    plt.plot(df.index, df['Actual Demand'], label='Actual Demand', color='blue', alpha=0.7)
    plt.plot(df.index, df['Forecast Demand'], label='Forecasted Demand', color='red', linestyle='--', linewidth=2)

    # Calculate confidence interval (approximate using 95% of residuals if available)
    error = df['Actual Demand'] - df['Forecast Demand']
    std_error = error.std()
    plt.fill_between(df.index,
                     df['Forecast Demand'] - 1.96 * std_error,
                     df['Forecast Demand'] + 1.96 * std_error,
                     color='pink', alpha=0.3, label='95% Confidence Interval')

    plt.title('Actual vs Forecasted Electricity Demand', fontsize=16)
    plt.xlabel('Timestamp', fontsize=12)
    plt.ylabel('Demand (MW)', fontsize=12)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('forecast_vs_actual.png')
    print("Comparison plot saved to forecast_vs_actual.png")

    # Metrics Diagram
    mae = mean_absolute_error(df['Actual Demand'], df['Forecast Demand'])
    rmse = np.sqrt(mean_squared_error(df['Actual Demand'], df['Forecast Demand']))
    mape = np.mean(np.abs((df['Actual Demand'] - df['Forecast Demand']) / df['Actual Demand'])) * 100
    r2 = r2_score(df['Actual Demand'], df['Forecast Demand'])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    metric_names = ['Mean Absolute Error (MAE)', 'Root Mean Squared Error (RMSE)', 'Mean Absolute Percentage Error (MAPE)', 'R² Score (Accuracy)']
    metric_values = [f"{mae:.2f} MW", f"{rmse:.2f} MW", f"{mape:.2f}%", f"{r2:.4f}"]

    table_data = []
    for n, v in zip(metric_names, metric_values):
        table_data.append([n, v])

    table = ax.table(cellText=table_data,
                     colLabels=['Metric', 'Value'],
                     loc='center', cellLoc='left',
                     colColours=['#f2f2f2', '#f2f2f2'])

    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 2.5)

    # Color coding based on MAPE
    status = "EXCELLENT" if mape < 5 else "GOOD" if mape < 10 else "FAIR" if mape < 20 else "POOR"
    color = "green" if mape < 10 else "orange" if mape < 20 else "red"

    plt.text(0.5, 0.95, f'Forecast Performance: {status}',
             horizontalalignment='center', fontsize=18, fontweight='bold', color=color, transform=ax.transAxes)

    plt.title('Summary of Prediction Metrics', fontsize=16, pad=30)
    plt.tight_layout()
    plt.savefig('metrics_diagram.png')
    print("Metrics diagram saved to metrics_diagram.png")

def main(input_csv, output_csv='predictions.csv'):
    # Load model
    try:
        model = joblib.load('xgb_electricity_demand_model.pkl')
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load input data
    try:
        input_df = pd.read_csv(input_csv)
        print(f"Loaded input data from {input_csv} ({len(input_df)} rows).")
        # Standardize timestamp
        input_df['full_ts'] = pd.to_datetime(input_df['Timestamp'], format='%d-%b-%y', errors='coerce') + \
                          pd.to_timedelta(input_df['hour'], unit='h', errors='coerce')
        if input_df['full_ts'].isnull().any():
            # Try alternate format
            input_df['full_ts'] = pd.to_datetime(input_df['Timestamp'], errors='coerce')

        input_df = input_df.dropna(subset=['full_ts']).set_index('full_ts')
        input_df = input_df.sort_index()
    except Exception as e:
        print(f"Error processing input CSV: {e}")
        return

    # Load historical data to calculate lags/rolling
    try:
        hist_df = pd.read_csv('electricitydemand.csv')
        hist_df['full_ts'] = pd.to_datetime(hist_df['Timestamp'], format='%d-%b-%y', errors='coerce') + \
                          pd.to_timedelta(hist_df['hour'], unit='h', errors='coerce')
        hist_df = hist_df.dropna(subset=['full_ts']).set_index('full_ts')
        hist_df = hist_df.sort_index()
    except Exception as e:
        print(f"Error loading historical data: {e}")
        return

    # Combine history and input to ensure lags are correct
    # If input data is far in the future, we might have gaps, but ffill() handles it somewhat
    combined_df = pd.concat([hist_df, input_df])
    # Remove duplicates if any
    combined_df = combined_df[~combined_df.index.duplicated(keep='last')].sort_index()

    # Preprocess
    processed_df = edf_preprocess(combined_df)

    # Extract only the rows that were in input_df for prediction
    predict_df = processed_df.loc[input_df.index].copy()

    # Features
    try:
        X = predict_df[model.feature_names_in_]
    except KeyError as e:
        print(f"Error: Missing required features in processed data: {e}")
        # Identify missing features
        missing = [f for f in model.feature_names_in_ if f not in predict_df.columns]
        print(f"Missing: {missing}")
        return

    # Predict
    predict_df['Forecast Demand'] = model.predict(X)

    # Actual demand and error
    if 'Demand' in predict_df.columns:
        predict_df['Actual Demand'] = predict_df['Demand']
        predict_df['Error'] = predict_df['Actual Demand'] - predict_df['Forecast Demand']

    # Prepare Output CSV
    output_cols = []
    if 'Actual Demand' in predict_df.columns:
        output_cols.append('Actual Demand')
    output_cols.append('Forecast Demand')
    if 'Actual Demand' in predict_df.columns:
        output_cols.append('Error')

    predict_df[output_cols].to_csv(output_csv)
    print(f"Predictions saved to {output_csv}")

    # Visualizations
    generate_plots(predict_df)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python batch_predict.py <input_csv> [output_csv]")
    else:
        input_csv = sys.argv[1]
        output_csv = sys.argv[2] if len(sys.argv) > 2 else 'predictions.csv'
        main(input_csv, output_csv)
