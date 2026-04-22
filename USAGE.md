# Usage Guide: Electricity Demand Forecasting

This guide provides step-by-step instructions on how to use the electricity demand forecasting system, including interactive predictions, batch processing, and performance analysis.

## 1. Environment Setup

Before running any scripts, ensure you have the necessary dependencies installed:

```bash
pip install -r requirements.txt
```

---

## 2. Interactive Prediction (`predict.py`)

Use the `predict.py` script to get a forecast for a specific timestamp, temperature, and humidity. This script uses the trained XGBoost model and historical context from `electricitydemand.csv` to calculate lagged features and rolling statistics automatically.

### Steps:
1. Run the script:
   ```bash
   python predict.py
   ```
2. Enter the required information when prompted:
   - **Timestamp**: (Format: `YYYY-MM-DD HH:MM`) e.g., `2020-03-01 14:00`
   - **Temperature**: (Celsius) e.g., `28.5`
   - **Humidity**: (%) e.g., `60`
3. (Optional) Enter the actual demand value if available to see a comparison report and anomaly check.

**Example Output:**
```text
Forecasted Electricity Demand: 4250.32 MW
```

---

## 3. Batch Prediction (`batch_predict.py`)

Use the `batch_predict.py` script to generate forecasts for a set of timestamps provided in a CSV file.

### Input CSV Format:
The input CSV should have at least the following columns: `Timestamp` (e.g., `01-Jan-20`), `hour`, `Temperature`, and `Humidity`. If a `Demand` column is present, the script will also calculate error metrics and generate comparison plots.

### Steps:
1. Prepare your input CSV (e.g., `test_input.csv`).
2. Run the script:
   ```bash
   python batch_predict.py test_input.csv output_predictions.csv
   ```
3. The script will generate:
   - `output_predictions.csv`: Contains the forecasted demand and errors (if actual demand was provided).
   - `forecast_vs_actual.png`: A visualization comparing actual vs. forecasted demand with 95% confidence intervals.
   - `metrics_diagram.png`: A summary table of performance metrics (MAE, RMSE, MAPE, R²).

---

## 4. Comprehensive Forecast Reporting (`forecast_report.py`)

To generate a detailed statistical report on the model's performance, run `forecast_report.py`. This is useful for auditing the model's reliability and checking for statistical anomalies.

### Steps:
1. Run the script:
   ```bash
   python forecast_report.py
   ```
2. The script will output a detailed report to the console and save it to `forecasting_report.txt`.

The report includes:
- Forecast summary and uncertainty analysis (Prediction Intervals).
- Error metrics and accuracy interpretation.
- Diagnostic test results (ACF/PACF, Ljung-Box, ADF, Normality).
- Performance breakdown by Season and Time of Day.
- Operational alerts (Peak alerts, Anomaly flags, Retraining triggers).

---

## 5. Model Fit Analysis (`fit_manager.py`)

If you want to check if the model is over-fitting or under-fitting, use `fit_manager.py`.

### Steps:
1. Run the script:
   ```bash
   python fit_manager.py
   ```
2. The script analyzes the gap between training and testing MAPE and provides a diagnosis with specific recommendations for improvement.

---

## 6. Training and Experimentation (Jupyter Notebooks)

If you wish to retrain the models or explore the data further:

- **`edf.ipynb`**: Recommended for Exploratory Data Analysis (EDA).
- **`XGBOOST.ipynb`**: Used for training and evaluating the XGBoost model. It saves the final model as `xgb_electricity_demand_model.pkl`.
- **`LSTM.ipynb`**: Used for training and evaluating the LSTM neural network. Note that the LSTM model requires standardized features and uses a 24-hour sliding window.

### Centralized Logic:
All scripts and notebooks use `edf.py` for consistent data loading and feature engineering. If you change the feature engineering logic, update `edf.py` to ensure all components remain synchronized.
