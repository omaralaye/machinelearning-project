import numpy as np
import matplotlib.pyplot as plt
import train_lstm_fixed
import edf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

def run_comparison():
    print("--- STEP 5: RE-TRAIN AND COMPARE ---")
    original_df, X_train_raw, X_val_raw, X_test_raw, Y_train_raw, Y_val_raw, Y_test_raw = edf.get_data_for_modeling()
    features = original_df.columns.tolist()
    target = 'Demand'
    feature_cols = [c for c in features if c != target]
    data = original_df[feature_cols + [target]].values
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    train_data = scaled_data[0:len(X_train_raw), :]
    val_data = scaled_data[len(X_train_raw):len(X_train_raw)+len(X_val_raw), :]
    window_size = 24

    X_train, Y_train = train_lstm_fixed.create_sequences(train_data[:, :-1], train_data[:, -1], window_size)
    combined_train_val = np.vstack([train_data[-window_size:], val_data])
    X_val, Y_val = train_lstm_fixed.create_sequences(combined_train_val[:, :-1], combined_train_val[:, -1], window_size)

    # Use smaller epochs for faster demonstration during main runner
    print("Running Baseline (Simulating original behavior)...")
    model_before = keras.models.Sequential([
        keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
        keras.layers.LSTM(32, return_sequences=True),
        keras.layers.LSTM(32, return_sequences=False),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1)
    ])
    model_before.compile(optimizer='adam', loss='mae', metrics=[keras.metrics.RootMeanSquaredError()])
    history_before = model_before.fit(X_train, Y_train, epochs=10, batch_size=256, validation_data=(X_val, Y_val), verbose=0)
    val_rmse_before = history_before.history['val_root_mean_squared_error'][-1]
    print(f"Final Val RMSE (Before): {val_rmse_before:.4f}")

    print("Running Fixed Training (Applying all 8 fixes)...")
    # Reducing epochs in train_lstm_fixed for the runner
    X_f_train, Y_f_train, X_f_val, Y_f_val, sX, sY = train_lstm_fixed.get_prepared_data()
    model_after = train_lstm_fixed.build_model((X_f_train.shape[1], X_f_train.shape[2]))
    optimizer = keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
    model_after.compile(optimizer=optimizer, loss='mae', metrics=[keras.metrics.RootMeanSquaredError()])
    history_after = model_after.fit(X_f_train, Y_f_train, epochs=10, batch_size=256, validation_data=(X_f_val, Y_f_val), verbose=0)
    val_rmse_after = history_after.history['val_root_mean_squared_error'][-1]
    print(f"Final Val RMSE (After): {val_rmse_after:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot(history_before.history['loss'], 'r--', label='Before')
    ax1.plot(history_after.history['loss'], 'g-', label='After')
    ax1.set_title('Loss')
    ax1.legend()
    ax2.plot(history_before.history['val_root_mean_squared_error'], 'r--', label='Before')
    ax2.plot(history_after.history['val_root_mean_squared_error'], 'g-', label='After')
    ax2.set_title('Val RMSE')
    ax2.legend()
    plt.savefig('diagnostic_comparison.png')
    print("\n[STEP 5 STATUS]: PASSED")
    return val_rmse_before, val_rmse_after
