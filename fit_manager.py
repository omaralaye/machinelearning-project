import numpy as np
import edf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def check_model_fit(model, X_train, Y_train, X_test, Y_test):
    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Metrics
    train_mape = np.mean(np.abs((Y_train - train_pred) / Y_train)) * 100
    test_mape = np.mean(np.abs((Y_test - test_pred) / Y_test)) * 100

    train_r2 = r2_score(Y_train, train_pred)
    test_r2 = r2_score(Y_test, test_pred)

    gap = test_mape - train_mape

    print("=== MODEL FIT ANALYSIS ===")
    print(f"Training MAPE: {train_mape:.2f}%")
    print(f"Testing MAPE:  {test_mape:.2f}%")
    print(f"Training R2:   {train_r2:.4f}")
    print(f"Testing R2:    {test_r2:.4f}")
    print(f"MAPE Gap:      {gap:.2f}%")
    print("-" * 26)

    # Logic for Overfitting/Underfitting
    if train_mape > 10:
        print("DIAGNOSIS: UNDERFITTING DETECTED")
        print("Reason: High error on both training and test sets.")
        print("Recommendations:")
        print("1. Increase model complexity (e.g., more estimators, deeper trees in XGBoost).")
        print("2. Add more relevant features (e.g., more lags, weather variables).")
        print("3. Decrease regularization parameters.")
    elif gap > 2.0:
        print("DIAGNOSIS: OVERFITTING DETECTED")
        print("Reason: Significantly better performance on training set than test set.")
        print("Recommendations:")
        print("1. Increase regularization (e.g., reg_alpha, reg_lambda).")
        print("2. Reduce model complexity (e.g., decrease max_depth, increase min_child_weight).")
        print("3. Increase the number of early stopping rounds or decrease learning rate.")
        print("4. Use more training data or perform cross-validation.")
    elif train_mape < 5 and test_mape < 5:
        print("DIAGNOSIS: GOOD FIT")
        print("The model generalizes well and has low error.")
    else:
        print("DIAGNOSIS: ACCEPTABLE FIT")
        print("The model shows balanced performance, though there might be room for improvement.")

if __name__ == "__main__":
    try:
        model = joblib.load('xgb_electricity_demand_model.pkl')
        _, X_train, X_test, Y_train, Y_test = edf.get_data_for_modeling()
        check_model_fit(model, X_train, Y_train, X_test, Y_test)
    except Exception as e:
        print(f"Error: {e}")
