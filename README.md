# Electricity Demand Forecasting

This project implements and compares two different machine learning models, **XGBoost** and **LSTM (Long Short-Term Memory)**, to forecast electricity demand based on historical usage patterns and environmental factors like temperature and humidity.

## Project Structure

- `edf.py`: Contains centralized functions for data loading, preprocessing, and feature engineering to ensure consistency across models.
- `edf.ipynb`: Exploratory Data Analysis (EDA) and detailed preprocessing steps.
- `XGBOOST.ipynb`: Implementation, training, and evaluation of the XGBoost regression model.
- `LSTM.ipynb`: Implementation, training, and evaluation of the LSTM neural network.
- `electricitydemand.csv`: The dataset containing hourly electricity demand, temperature, and humidity.
- `requirements.txt`: List of Python dependencies required for the project.
- `xgb_electricity_demand_model.pkl`: The serialized trained XGBoost model.

## Dataset

The dataset provides hourly observations with the following features:
- `Timestamp`: The date and time of the observation.
- `Demand`: Hourly electricity demand (Target variable).
- `Temperature`: Ambient temperature.
- `Humidity`: Relative humidity.

## Preprocessing and Feature Engineering

Data preprocessing is crucial for time-series forecasting. The following steps are performed:
1.  **Missing Value Handling**: Forward filling is used to handle gaps in the time series data.
2.  **Time-Based Features**: Extraction of features like `hour`, `dayofweek`, `month`, `year`, `Quarter`, and `is_weekend`.
3.  **Cyclical Encoding**: Applied sine/cosine transformations to `hour` and `month` to help models understand their periodic nature.
4.  **Holiday Indicators**: Added a feature to identify public holidays using the `holidays` library.
5.  **Lagged Features**: Inclusion of past demand values (`Demand_lag_24hr` and `Demand_lag_168hr`) to capture daily and weekly seasonality.
6.  **Rolling Statistics**: Calculation of rolling mean and standard deviation over a 24-hour window to capture local trends.
7.  **Data Splitting**: The data is split into 80% training and 20% testing sets using a time-series split (no shuffling) to preserve chronological order.

## Models

### 1. XGBoost (Extreme Gradient Boosting)
- **Type**: Gradient Boosted Decision Trees.
- **Approach**: Uses the full set of engineered features (time-based, cyclical, lagged, rolling) to predict demand.
- **Configuration**: Trained with 1000 estimators, a learning rate of 0.01, and early stopping to prevent overfitting.
- **Performance**: High accuracy with an R² score of approximately 0.976.

### 2. LSTM (Long Short-Term Memory)
- **Type**: Recurrent Neural Network (RNN).
- **Approach**: Captures long-term dependencies in the sequence of demand and other exogenous variables (Multivariate forecasting).
- **Architecture**:
    - Input window of 24 hours.
    - Multi-variate input including demand, temperature, humidity, and engineered features.
    - Two LSTM layers with 64 units each.
    - Two Dense layers (128 and 64 units) with ReLU activation.
    - Single output layer for demand prediction.
- **Preprocessing**: All input features and target are standardized using `StandardScaler`.

## Evaluation Metrics

The models are evaluated using:
- **Mean Absolute Error (MAE)**: Measures the average magnitude of errors.
- **Root Mean Squared Error (RMSE)**: Penalizes larger errors more heavily.
- **Mean Absolute Percentage Error (MAPE)**: Provides error as a percentage of actual values, facilitating cross-dataset comparison.
- **R-squared (R²)**: Indicates the proportion of variance explained by the model.

## Getting Started

### Prerequisites
Ensure you have Python installed. You can install the dependencies using:

```bash
pip install -r requirements.txt
```

### Running the Project
1.  Run the `XGBOOST.ipynb` notebook to train and evaluate the XGBoost model.
2.  Run the `LSTM.ipynb` notebook to train and evaluate the LSTM model.
3.  Refer to `edf.ipynb` for detailed data exploration.
