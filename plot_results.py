import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
import edf
import os

def main():
    print("Generating Demand Forecast Graph...")

    # 1. Load resources
    try:
        xgb_model = joblib.load('xgb_electricity_demand_model.pkl')
        lstm_model = tf.keras.models.load_model('lstm_model.keras')
        scaler = joblib.load('scaler.pkl')
        lstm_features_list = joblib.load('lstm_features.pkl')
        print("Models and resources loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # 2. Get data
    df, X_train, X_val, X_test, Y_train, Y_val, Y_test = edf.get_data_for_modeling()

    # 3. XGBoost Predictions
    xgb_preds = xgb_model.predict(X_test)

    # Calculate Confidence Interval for XGBoost
    residuals = Y_test - xgb_preds
    std_resid = np.std(residuals)
    ci_95 = 1.96 * std_resid

    # 4. LSTM Predictions (Optimized Batch Processing)
    # To get LSTM predictions for X_test, we need the 24h sequences.
    # edf.load_and_preprocess_data returns the full processed dataframe.
    full_df = edf.load_and_preprocess_data()

    # We need to scale the data first
    lstm_data_to_scale = full_df[lstm_features_list + ['Demand']].values
    scaled_full_data = scaler.transform(lstm_data_to_scale)

    all_sequences = []
    for ts in Y_test.index:
        idx = full_df.index.get_loc(ts)
        # Extract the 24 hours preceding the current timestamp
        if idx >= 24:
            seq = scaled_full_data[idx-24:idx, :-1]
        else:
            needed = 24
            available = idx
            padding = needed - available
            seq = np.zeros((needed, len(lstm_features_list)))
            if available > 0:
                seq[padding:] = scaled_full_data[:idx, :-1]
        all_sequences.append(seq)

    all_sequences_np = np.array(all_sequences)
    lstm_preds_scaled = lstm_model.predict(all_sequences_np, verbose=0)

    # Inverse transform LSTM predictions
    dummy = np.zeros((len(lstm_preds_scaled), len(lstm_features_list) + 1))
    dummy[:, -1] = lstm_preds_scaled.flatten()
    lstm_preds = scaler.inverse_transform(dummy)[:, -1]

    # 5. Plotting
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(15, 8))

    # Actual Demand
    plt.plot(Y_test.index, Y_test, label='Actual Demand', color='black', alpha=0.7, linewidth=1.5)

    # XGBoost
    plt.plot(Y_test.index, xgb_preds, label='XGBoost Forecast', color='blue', linestyle='--', alpha=0.8)
    plt.fill_between(Y_test.index,
                     xgb_preds - ci_95,
                     xgb_preds + ci_95,
                     color='blue', alpha=0.1, label='XGBoost 95% CI')

    # LSTM
    plt.plot(Y_test.index, lstm_preds, label='LSTM Forecast', color='red', linestyle=':', alpha=0.8)

    plt.title('Electricity Demand: Actual vs Forecasted (Test Set)', fontsize=16)
    plt.xlabel('Timestamp', fontsize=12)
    plt.ylabel('Demand (MW)', fontsize=12)
    plt.legend(fontsize=11)
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_path = 'demand_forecast_graph.png'
    plt.savefig(output_path)
    print(f"Graph saved successfully to {output_path}")

if __name__ == "__main__":
    main()
