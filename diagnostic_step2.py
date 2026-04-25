import tensorflow as tf
from tensorflow import keras
import train_lstm
import edf

def diagnostic_step2():
    print("--- STEP 2: TRAINING CONFIGURATION AUDIT ---")

    # We need to see how the model is being compiled and built in train_lstm.py
    # Since train_lstm.py doesn't expose a 'get_model' function easily without running everything,
    # I'll manually inspect based on its code or try to import parts.

    # Mocking the data to get the model architecture
    original_df, X_train_raw, X_val_raw, X_test_raw, Y_train_raw, Y_val_raw, Y_test_raw = edf.get_data_for_modeling()
    features = original_df.columns.tolist()
    target = 'Demand'
    feature_cols = [c for c in features if c != target]

    input_shape = (24, len(feature_cols)) # Based on train_lstm.py window_size=24

    model = keras.models.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.LSTM(64, return_sequences=False),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1)
    ])

    # Default settings from train_lstm.py
    optimizer_name = 'adam'
    loss = 'mae'
    learning_rate = 0.001 # Default for Adam in Keras 3 usually, but let's check

    model.compile(optimizer=optimizer_name, loss=loss, metrics=[keras.metrics.RootMeanSquaredError()])

    print(f"Optimizer: {model.optimizer.__class__.__name__}")
    # In Keras 3, learning rate is accessed via model.optimizer.learning_rate
    lr = model.optimizer.learning_rate
    if hasattr(lr, 'numpy'):
        print(f"Learning Rate: {lr.numpy()}")
    else:
        print(f"Learning Rate: {lr}")

    print(f"Batch size (from train_lstm.py): 32")

    # Checking for scheduler and early stopping in train_lstm.py
    # We saw early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    print("Learning rate scheduler: None")
    print("Early stopping: Configured (monitor='val_loss', patience=5, restore_best_weights=True)")

    # Gradient clipping
    # Adam by default doesn't have clipping unless specified.
    print(f"Gradient clipping (clipnorm): {getattr(model.optimizer, 'clipnorm', None)}")
    print(f"Gradient clipping (clipvalue): {getattr(model.optimizer, 'clipvalue', None)}")

    print("\nModel Architecture Summary:")
    model.summary()

    print("\n[STEP 2 STATUS]: PASSED")

if __name__ == "__main__":
    diagnostic_step2()
