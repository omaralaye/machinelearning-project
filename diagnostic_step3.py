import numpy as np
import tensorflow as tf
from tensorflow import keras
import edf
from sklearn.preprocessing import StandardScaler

class DiagnosticCallback(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.prev_val_rmse = None
        self.consecutive_increases = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_rmse = logs.get('val_root_mean_squared_error')
        lr = self.model.optimizer.learning_rate
        if hasattr(lr, 'numpy'):
            lr = lr.numpy()

        print(f"Epoch {epoch+1}: loss={logs.get('loss'):.4f}, val_rmse={val_rmse:.4f}, lr={lr:.6f}, grad_norm={logs.get('grad_norm'):.4f}")

        # Check for NaN weights
        for layer in self.model.layers:
            for w in layer.get_weights():
                if np.isnan(w).any():
                    print(f"!!! ALERT: NaN weights in {layer.name} !!!")

        if self.prev_val_rmse is not None:
            if val_rmse > self.prev_val_rmse * 1.05:
                print(f"FLAG: Val RMSE increased > 5% ({self.prev_val_rmse:.4f} -> {val_rmse:.4f})")
                self.consecutive_increases += 1
            else:
                self.consecutive_increases = 0

            if self.consecutive_increases >= 2:
                 print(f"!!! ALERT: Consecutive val RMSE increases !!!")

        self.prev_val_rmse = val_rmse

class GradientLogger(keras.Model):
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        grad_norm = tf.linalg.global_norm(gradients)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        result = {m.name: m.result() for m in self.metrics}
        result["grad_norm"] = grad_norm
        return result

def run_diagnostic_step3():
    print("--- STEP 3: TRAINING LOOP INSPECTION ---")
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

    X_train, Y_train = [], []
    for i in range(window_size, len(train_data)):
        X_train.append(train_data[i-window_size:i, :-1])
        Y_train.append(train_data[i, -1])
    X_train, Y_train = np.array(X_train), np.array(Y_train)

    X_val, Y_val = [], []
    combined_train_val = np.vstack([train_data[-window_size:], val_data])
    for i in range(window_size, len(combined_train_val)):
        X_val.append(combined_train_val[i-window_size:i, :-1])
        Y_val.append(combined_train_val[i, -1])
    X_val, Y_val = np.array(X_val), np.array(Y_val)

    inputs = keras.Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = keras.layers.LSTM(64, return_sequences=True)(inputs)
    x = keras.layers.LSTM(64)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    outputs = keras.layers.Dense(1)(x)

    model = GradientLogger(inputs, outputs)
    model.compile(optimizer='adam', loss='mae', metrics=[keras.metrics.RootMeanSquaredError()])

    print("Running 10 epochs with gradient tracking...")
    history = model.fit(
        X_train, Y_train,
        epochs=10,
        batch_size=64,
        validation_data=(X_val, Y_val),
        callbacks=[DiagnosticCallback()],
        verbose=0
    )

    print("\n[STEP 3 STATUS]: PASSED")
    return history

if __name__ == "__main__":
    run_diagnostic_step3()
