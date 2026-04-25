import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
import edf
import joblib

def create_sequences(X_data, Y_data, window):
    shape = (X_data.shape[0] - window, window, X_data.shape[1])
    strides = (X_data.strides[0], X_data.strides[0], X_data.strides[1])
    X_seq = np.lib.stride_tricks.as_strided(X_data, shape=shape, strides=strides)
    Y_seq = Y_data[window:]
    return X_seq, Y_seq

def get_prepared_data():
    df_raw = edf.load_and_preprocess_data()
    df = df_raw.copy()
    target_col = 'Demand'

    # Fix 4 - Outlier Handling
    mean = df[target_col].mean()
    std = df[target_col].std()
    upper_limit = mean + 3 * std
    lower_limit = mean - 3 * std
    clipped_count = ((df[target_col] > upper_limit) | (df[target_col] < lower_limit)).sum()
    df[target_col] = df[target_col].clip(lower=lower_limit, upper=upper_limit)
    print(f"[FIX 4]: Clipped {clipped_count} outliers.")

    # Fix 2 - Chronological Split (Last 20% for val)
    X = df.drop(columns=[target_col])
    Y = df[target_col]
    split_idx_test = int(len(df) * 0.8)
    split_idx_val = int(len(df) * 0.6)
    X_train, Y_train = X.iloc[:split_idx_val], Y.iloc[:split_idx_val]
    X_val, Y_val = X.iloc[split_idx_val:split_idx_test], Y.iloc[split_idx_val:split_idx_test]
    print(f"[FIX 2]: Chronological split applied. Train size: {len(X_train)}, Val size: {len(X_val)}.")

    # Fix 3 - Data Normalization
    scaler_X, scaler_Y = StandardScaler(), StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    Y_train_scaled = scaler_Y.fit_transform(Y_train.values.reshape(-1, 1))
    Y_val_scaled = scaler_Y.transform(Y_val.values.reshape(-1, 1))
    print("[FIX 3]: Normalization applied (StandardScaler fit on train only).")

    window_size = 24
    X_val_combined = np.vstack([X_train_scaled[-window_size:], X_val_scaled])
    Y_val_combined = np.vstack([Y_train_scaled[-window_size:], Y_val_scaled])
    X_train_final, Y_train_final = create_sequences(X_train_scaled, Y_train_scaled, window_size)
    X_val_final, Y_val_final = create_sequences(X_val_combined, Y_val_combined, window_size)
    return X_train_final, Y_train_final, X_val_final, Y_val_final, scaler_X, scaler_Y

def build_model(input_shape):
    # Fix 6 - Regularization
    l2_reg = keras.regularizers.l2(1e-4)
    model = keras.models.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.LSTM(64, return_sequences=True, kernel_regularizer=l2_reg),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(64, return_sequences=False, kernel_regularizer=l2_reg),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128, activation='relu', kernel_regularizer=l2_reg),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation='relu', kernel_regularizer=l2_reg),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1)
    ])
    print("[FIX 6]: Dropout(0.2) and L2 weight decay added.")
    return model

def train_fixed_lstm():
    X_train, Y_train, X_val, Y_val, scaler_X, scaler_Y = get_prepared_data()
    model = build_model((X_train.shape[1], X_train.shape[2]))

    # Fix 1 & 5
    lr = 1e-4
    print(f"[FIX 1]: Initial learning rate reduced to {lr}.")
    optimizer = keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    print("[FIX 5]: Gradient clipping (max_norm=1.0) added.")

    model.compile(optimizer=optimizer, loss='mae', metrics=[keras.metrics.RootMeanSquaredError()])

    # Fix 1 scheduler & Fix 7 early stopping
    lr_scheduler = ReduceLROnPlateau(monitor='val_root_mean_squared_error', factor=0.5, patience=5, min_lr=1e-7)
    early_stop = EarlyStopping(monitor='val_root_mean_squared_error', patience=15, restore_best_weights=True)
    print("[FIX 7]: Early stopping (patience=15) and ReduceLROnPlateau added.")

    # Fix 8
    batch_size = 64
    print(f"[FIX 8]: Batch size increased to {batch_size}.")

    print("Starting re-training with all fixes...")
    history = model.fit(
        X_train, Y_train,
        epochs=30, # Sufficient for demonstration, usually more
        batch_size=batch_size,
        validation_data=(X_val, Y_val),
        callbacks=[lr_scheduler, early_stop],
        verbose=1
    )
    model.save('lstm_model_fixed.keras')
    joblib.dump(scaler_X, 'scaler_X_fixed.pkl')
    joblib.dump(scaler_Y, 'scaler_Y_fixed.pkl')
    return history, model
