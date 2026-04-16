# ⚡ Power Demand Forecasting with XGBoost

A machine learning pipeline for short-term electricity demand forecasting using historical power grid data, weather conditions, and macroeconomic indicators. The model predicts next-hour power demand (in MW) for the Bangladesh Power Grid Company (PGCB) using an optimized XGBoost regressor.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Data Sources](#data-sources)
- [Project Structure](#project-structure)
- [Pipeline Walkthrough](#pipeline-walkthrough)
  - [1. Data Loading](#1-data-loading)
  - [2. Data Cleaning & Integration](#2-data-cleaning--integration)
  - [3. Feature Engineering](#3-feature-engineering)
  - [4. Model Training & Hyperparameter Tuning](#4-model-training--hyperparameter-tuning)
  - [5. Evaluation & Visualization](#5-evaluation--visualization)
- [Features Used](#features-used)
- [Model Details](#model-details)
- [Outputs](#outputs)
- [Requirements](#requirements)
- [Getting Started](#getting-started)

---

## Project Overview

Short-term electricity demand forecasting is critical for grid stability, resource planning, and cost reduction. This project builds a robust, end-to-end forecasting pipeline that:

- Cleans and integrates multi-source time series data (grid, weather, economic)
- Engineers time-aware and lag-based features
- Tunes an XGBoost model using Bayesian optimization (Optuna)
- Evaluates predictions using MAPE and visual diagnostics

The target variable is **next-hour power demand (MW)**, achieved by shifting the `demand_mw` column one step forward.

---

## Data Sources

| File | Description |
|------|-------------|
| `PGCB_date_power_demand.xlsx` | Hourly power grid data from PGCB including total demand, generation, and import sources (India, Nepal, etc.) |
| `weather_data.xlsx` | Hourly weather observations (temperature, wind speed, precipitation, etc.) |
| `economic_full_1.csv` | Annual World Bank macroeconomic indicators for Bangladesh |

---

## Project Structure

```
├── Power_demand_predict.ipynb   # Main notebook
├── PGCB_date_power_demand.xlsx  # Raw grid data
├── weather_data.xlsx            # Raw weather data
├── economic_full_1.csv          # Raw economic data
├── data_final.xlsx              # Processed feature-engineered dataset (generated)
├── parity_plot.png              # Actual vs. predicted scatter plot (generated)
└── README.md
```

---

## Pipeline Walkthrough

### 1. Data Loading

Three datasets are loaded:

```python
df  = pd.read_excel('PGCB_date_power_demand.xlsx')      # Grid data
wea = pd.read_excel('weather_data.xlsx', skiprows=3)    # Weather data
eco = pd.read_csv('economic_full_1.csv')                # Economic indicators
```

### 2. Data Cleaning & Integration

**Grid Data (`df`):**
- Datetime index is rounded to the nearest hour and deduplicated.
- Missing values for import sources (`india_adani`, `nepal`) and renewables (`solar`, `wind`) are filled with `0` (indicating no generation/import).
- **Outlier detection** uses the Modified Z-Score method (robust to skewed distributions): values with |modified Z| > 3 are flagged and replaced via linear interpolation, followed by forward/backward fill.

**Weather Data (`wea`):**
- Indexed by time, sorted, and deduplicated.

**Economic Data (`eco`):**
- Filtered to 6 relevant World Bank indicators:
  | Indicator Code | Description |
  |---|---|
  | `NY.GDP.MKTP.KD.ZG` | GDP growth (annual %) |
  | `SP.POP.TOTL` | Total population |
  | `NV.IND.TOTL.ZS` | Industry value added (% of GDP) |
  | `FP.CPI.TOTL.ZG` | Inflation, consumer prices (annual %) |
  | `EG.USE.PCAP.KG.OE` | Energy use per capita |
  | `EG.EGY.PRIM.PP.KD` | Energy intensity level of primary energy |
- Transposed so years become rows; missing values interpolated.

**Integration:**
- Weather data is joined to the grid data on the datetime index.
- Economic data is merged on the year extracted from the datetime index.

### 3. Feature Engineering

Several columns are dropped (granular generation sources, weather variables with high correlation/low predictive value, or columns already encoded via lags).

New features created:

| Feature | Description |
|---------|-------------|
| `hour` (then dropped, encoded via sin/cos) | Hour of day |
| `dayofweek` | Day of week (0=Monday, 6=Sunday) |
| `month` | Month of year |
| `weekend` | Binary flag: 1 if Saturday or Sunday |
| `lag_1` | Demand 1 hour ago |
| `lag_24` | Demand 24 hours ago (same hour, previous day) |
| `lag_168` | Demand 168 hours ago (same hour, previous week) |
| `roll_mean_24` | 24-hour rolling mean of past demand |
| `roll_std_24` | 24-hour rolling std of past demand |
| `generation_mw` | Previous hour's generation (shifted by 1) |
| `hour_sin` | Sine encoding of hour (captures cyclical nature) |
| `hour_cos` | Cosine encoding of hour |
| `is_peak_hour` | 1 if hour is in {10, 11, 12, 18, 19, 20} |
| `target` | **Next-hour demand (MW)** — prediction target |

> All lag and rolling features use `.shift(1)` to prevent data leakage. Rows with any remaining NaN values are dropped.

The processed dataset is saved to `data_final.xlsx`.

### 4. Model Training & Hyperparameter Tuning

**Train/Validation Split:**
- Training data: all records before `2024-01-01`
- Validation data: all records from `2024-01-01` onward
- Split is **time-based** (no shuffling) to respect temporal ordering.

**Hyperparameter Optimization with Optuna:**

[Optuna](https://optuna.org/) performs Bayesian optimization over 100 trials to minimize MAPE on the validation set.

| Hyperparameter | Search Range |
|---|---|
| `n_estimators` | 100 – 1000 |
| `max_depth` | 3 – 10 |
| `learning_rate` | 0.01 – 0.3 |
| `subsample` | 0.5 – 1.0 |
| `colsample_bytree` | 0.5 – 1.0 |

The best parameters are printed after optimization completes.

### 5. Evaluation & Visualization

**Feature Importance:** A ranked DataFrame of all input features by XGBoost importance score is displayed.

**Parity Plot:** Scatter plot of actual vs. predicted demand with a red 1:1 reference line. Saved as `parity_plot.png`.

**Time Series Plot:** Line chart overlaying actual and predicted demand across the full validation period.

**MAPE Score:**
```
MAPE for the best model: X.XX%
```

---

## Features Used

The final model uses these input features (after dropping the target):

- **Temporal:** `dayofweek`, `month`, `weekend`, `hour_sin`, `hour_cos`, `is_peak_hour`
- **Lag/Rolling:** `lag_1`, `lag_24`, `lag_168`, `roll_mean_24`, `roll_std_24`
- **Grid:** `generation_mw` (lagged), and remaining generation/import columns
- **Weather:** Wind speed, precipitation, humidity, and other retained weather variables
- **Economic:** GDP growth, population, inflation, energy use per capita, energy intensity

---

## Model Details

| Property | Value |
|---|---|
| Algorithm | XGBoost Regressor |
| Objective | Regression (minimize MAPE) |
| Tuning | Optuna (100 trials, Bayesian optimization) |
| Evaluation Metric | Mean Absolute Percentage Error (MAPE) |
| Prediction Horizon | 1 hour ahead |

---

## Outputs

| Output | Description |
|--------|-------------|
| `data_final.xlsx` | Fully processed and feature-engineered dataset |
| `parity_plot.png` | Actual vs. predicted scatter plot (validation set) |
| Console output | Best hyperparameters, feature importances, MAPE score |

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
optuna
openpyxl
```

Install all dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost optuna openpyxl
```

---

## Getting Started

1. **Clone the repository** and place the three raw data files in the same directory as the notebook:
   - `PGCB_date_power_demand.xlsx`
   - `weather_data.xlsx`
   - `economic_full_1.csv`

2. **Install dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost optuna openpyxl
   ```

3. **Run the notebook:**
   ```bash
   jupyter notebook Power_demand_predict.ipynb
   ```

4. **Review outputs:**
   - Check the printed best hyperparameters and MAPE score in the final cells.
   - View `parity_plot.png` for a visual assessment of model accuracy.
   - Inspect `data_final.xlsx` to explore the engineered feature set.

---

## Notes

- The modified Z-score method (threshold = 3) is used for outlier detection instead of standard Z-score, making it more robust to non-normal distributions common in power demand data.
- Cyclical hour encoding (`hour_sin`, `hour_cos`) preserves the continuity between hour 23 and hour 0, which standard integer encoding would break.
- All lag features are computed with a minimum shift of 1 to ensure no future information leaks into the training data.
- The time-based train/validation split (cutoff: 2024-01-01) is essential for realistic evaluation of a forecasting model.
