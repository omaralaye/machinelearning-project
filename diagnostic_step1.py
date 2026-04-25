import pandas as pd
import numpy as np
import edf
from scipy import stats

def diagnostic_step1():
    print("--- STEP 1: DATA AUDIT ---")

    # Load and inspect the dataset
    df = edf.load_and_preprocess_data()
    print(f"Dataset loaded with {len(df)} rows.")

    # Check for NaN, null, or infinite values
    null_counts = df.isnull().sum().sum()
    inf_counts = np.isinf(df).values.sum()
    print(f"Total NaN/Null values: {null_counts}")
    print(f"Total Infinite values: {inf_counts}")

    # Detect outliers in electricity demand
    demand = df['Demand']
    q1 = demand.quantile(0.25)
    q3 = demand.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers_iqr = demand[(demand < lower_bound) | (demand > upper_bound)]

    z_scores = np.abs(stats.zscore(demand))
    outliers_z = demand[z_scores > 3]

    print(f"Outliers detected (IQR): {len(outliers_iqr)}")
    print(f"Outliers detected (Z-score > 3): {len(outliers_z)}")

    # Verify the train/validation split is CHRONOLOGICAL
    X_train, X_val, X_test, Y_train, Y_val, Y_test = edf.get_split_data(df)

    train_min, train_max = X_train.index.min(), X_train.index.max()
    val_min, val_max = X_val.index.min(), X_val.index.max()
    test_min, test_max = X_test.index.min(), X_test.index.max()

    print(f"Train date range: {train_min} to {train_max}")
    print(f"Val date range: {val_min} to {val_max}")
    print(f"Test date range: {test_min} to {test_max}")

    is_chronological = (train_max < val_min) and (val_max < test_min)
    status = "PASSED" if is_chronological else "WARNING"
    print(f"Chronological split: {status}")

    # Check that no future data leaks into the training set
    # (By checking if any index in train is after min index in val)
    leak = train_max >= val_min
    print(f"Data leak detected: {leak}")

    # Verify all features are normalized/standardized
    print("\nFeature statistics (Min, Max, Mean):")
    stats_df = pd.DataFrame({
        'Min': df.min(),
        'Max': df.max(),
        'Mean': df.mean()
    })
    print(stats_df)

    # Summary
    step_status = "PASSED"
    if null_counts > 0 or inf_counts > 0 or not is_chronological or leak:
        step_status = "WARNING"

    # Check normalization: if max > 10 or min < -10 (rough check)
    if stats_df['Max'].max() > 10 or stats_df['Min'].min() < -10:
        print("\nWARNING: Features do not appear to be normalized.")
        step_status = "WARNING"

    print(f"\n[STEP 1 STATUS]: {step_status}")

if __name__ == "__main__":
    diagnostic_step1()
