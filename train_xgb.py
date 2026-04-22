import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import edf

def train_and_save_xgb():
    print("Loading and preprocessing data...")
    df, X_train, X_val, X_test, Y_train, Y_val, Y_test = edf.get_data_for_modeling()

    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")

    # Initializing the XGBoost model
    model_xgb = XGBRegressor(
        n_estimators=1000,
        early_stopping_rounds=50,
        learning_rate=0.01,
        random_state=42,
        objective='reg:squarederror'
    )

    print("Starting training with validation set for early stopping...")
    # Training the model using the training data, and validation set for early stopping
    model_xgb.fit(
        X_train, Y_train,
        eval_set=[(X_train, Y_train), (X_val, Y_val)],
        verbose=100
    )

    # make predictions using the trained model on the test data
    predictions_xgb = model_xgb.predict(X_test)

    # Evaluating the model
    mae_xgb = mean_absolute_error(Y_test, predictions_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(Y_test, predictions_xgb))
    mape_xgb = np.mean(np.abs((Y_test - predictions_xgb) / Y_test)) * 100

    print('XGBoost RMSE:', rmse_xgb)
    print('XGBoost MAE:', mae_xgb)
    print(f'XGBoost MAPE: {mape_xgb:.2f}%')

    # save the model
    joblib.dump(model_xgb, 'xgb_electricity_demand_model.pkl')
    print("XGBoost model saved successfully.")

if __name__ == "__main__":
    train_and_save_xgb()
