import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import joblib
import edf
import os

def train_and_save_lstm():
    print("Loading and preprocessing data...")
    original_df, X_train_raw, X_val_raw, X_test_raw, Y_train_raw, Y_val_raw, Y_test_raw = edf.get_data_for_modeling()

    # Prepare the data for LSTM model (Multivariate)
    features = original_df.columns.tolist()
    target = 'Demand'
    feature_cols = [c for c in features if c != target]

    data = original_df[feature_cols + [target]].values
    training_data_len = int(np.ceil(len(data) * 0.8))

    # Preprocessing: Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    train_data = scaled_data[0:len(X_train_raw), :]
    val_data = scaled_data[len(X_train_raw):len(X_train_raw)+len(X_val_raw), :]

    X_train = []
    Y_train = []
    X_val = []
    Y_val = []

    # Create a sliding window of 24 hours to predict the next hour's demand
    window_size = 24
    for i in range(window_size, len(train_data)):
        X_train.append(train_data[i-window_size:i, :-1]) # All columns except last (target)
        Y_train.append(train_data[i, -1])     # Last column (target)

    # For validation, we need the last 24 hours of training data to predict the first hour of validation
    combined_train_val = np.vstack([train_data[-window_size:], val_data])
    for i in range(window_size, len(combined_train_val)):
        X_val.append(combined_train_val[i-window_size:i, :-1])
        Y_val.append(combined_train_val[i, -1])

    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_val, Y_val = np.array(X_val), np.array(Y_val)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")

    # Building the LSTM model
    model = keras.models.Sequential([
        keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.LSTM(64, return_sequences=False),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1)
    ])

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5, # Reduced patience for quicker execution if needed, notebook used 10
        restore_best_weights=True
    )

    model.compile(optimizer='adam', loss='mae', metrics=[keras.metrics.RootMeanSquaredError()])

    print("Starting training...")
    model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_data=(X_val, Y_val), callbacks=[early_stop])

    # Save the model and the scaler
    model.save('lstm_model.keras')
    joblib.dump(scaler, 'scaler.pkl')

    # Also save the feature names to ensure consistency during inference
    joblib.dump(feature_cols, 'lstm_features.pkl')

    print("Model, scaler, and feature names saved successfully.")

if __name__ == "__main__":
    train_and_save_lstm()
