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
    original_df, _, _, _, _ = edf.get_data_for_modeling()

    # Prepare the data for LSTM model (Multivariate)
    features = original_df.columns.tolist()
    target = 'Demand'
    feature_cols = [c for c in features if c != target]

    data = original_df[feature_cols + [target]].values
    training_data_len = int(np.ceil(len(data) * 0.8))

    # Preprocessing: Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    train_data = scaled_data[0:training_data_len, :]

    X_train = []
    Y_train = []

    # Create a sliding window of 24 hours to predict the next hour's demand
    window_size = 24
    for i in range(window_size, len(train_data)):
        X_train.append(train_data[i-window_size:i, :-1]) # All columns except last (target)
        Y_train.append(train_data[i, -1])     # Last column (target)

    X_train, Y_train = np.array(X_train), np.array(Y_train)

    print(f"X_train shape: {X_train.shape}")

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
    model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_split=0.1, callbacks=[early_stop])

    # Save the model and the scaler
    model.save('lstm_model.keras')
    joblib.dump(scaler, 'scaler.pkl')

    # Also save the feature names to ensure consistency during inference
    joblib.dump(feature_cols, 'lstm_features.pkl')

    print("Model, scaler, and feature names saved successfully.")

if __name__ == "__main__":
    train_and_save_lstm()
