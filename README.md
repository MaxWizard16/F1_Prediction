# F1 Race Prediction Engine

## Overview

This project predicts Formula 1 race finishing positions using historical race data and machine learning models.

The project uses the Formula 1 World Championship dataset (1950–2024) and builds a feature engineering pipeline that captures:

* Driver recent performance
* Constructor recent performance
* Circuit-specific performance
* Track-type performance
* Qualifying performance
* Reliability (DNF rate)
* Momentum trends
* Grid position effects
* Teammate comparisons

The goal is to generate pre-race predictions for any Formula 1 Grand Prix.

---

## Project Structure

```text
F1 Prediction/
│
├── Data/
│   ├── results.csv
│   ├── races.csv
│   ├── circuits.csv
│   ├── qualifying.csv
│   ├── status.csv
│   ├── master_dataset.csv
│   ├── master_dataset_features.csv
│
├── src/
│   ├── training_dataset.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── model_benchmark.py
│
└── README.md
```

---

## Dataset Construction

### training_dataset.py

Creates the master dataset by combining:

* results.csv
* races.csv
* circuits.csv
* qualifying.csv
* status.csv

Generated columns include:

| Column              | Description                     |
| ------------------- | ------------------------------- |
| raceId              | Race identifier                 |
| year                | Season                          |
| round               | Race round                      |
| circuitId           | Circuit identifier              |
| circuitRef          | Circuit reference               |
| track_type          | Street / High Speed / Technical |
| alt                 | Circuit altitude                |
| driverId            | Driver identifier               |
| constructorId       | Constructor identifier          |
| grid                | Starting position               |
| qualifying_position | Qualifying result               |
| positionOrder       | Final race position             |
| statusId            | Race status                     |
| is_dnf              | Did Not Finish flag             |

---

## Feature Engineering

### Driver Features

* driver_recent_form
* driver_recent_3
* driver_recent_10
* driver_momentum
* driver_circuit_form
* driver_tracktype_form
* driver_recent_qualifying_form
* driver_dnf_rate

### Constructor Features

* constructor_recent_form
* constructor_recent_3
* constructor_recent_10
* constructor_momentum
* constructor_circuit_form
* constructor_tracktype_form

### Race Features

* grid
* relative_grid

### Comparison Features

* driver_vs_teammate

---

## Machine Learning Models Tested

The project benchmarks multiple regression models:

1. CatBoost
2. LightGBM
3. XGBoost
4. ExtraTrees
5. Random Forest
6. Gradient Boosting
7. HistGradientBoosting
8. Ridge Regression
9. Linear Regression
10. AdaBoost
11. Decision Tree

Models are evaluated using Mean Absolute Error (MAE).

---

## Benchmark Results

Current benchmark results:

| Rank | Model            |
| ---- | ---------------- |
| 1    | ExtraTrees       |
| 2    | XGBoost          |
| 3    | LightGBM         |
| 4    | GradientBoosting |
| 5    | RandomForest     |
| 6    | CatBoost         |
| 7    | Ridge            |
| 8    | LinearRegression |
| 9    | AdaBoost         |
| 10   | DecisionTree     |

Best MAE achieved so far:

```text
2.8443
``

using ExtraTreesRegressor.

---

## Training Pipeline

### Step 1

Generate master dataset:

```bash
python src/training_dataset.py
```

### Step 2

Generate engineered features:

```bash
python src/feature_engineering.py
```

### Step 3

Train benchmark models:

```bash
python src/model_benchmark.py
```

### Step 4

Review model comparison results:

```text
Data/model_comparison.csv
```

---

## Future Improvements

Planned enhancements:

* Driver Elo ratings
* Constructor Elo ratings
* Sprint race features
* Championship position features
* Weather features
* Stacking ensemble models
* Hyperparameter optimization
* Monte Carlo race simulations
* Winner probability estimation
* Podium probability estimation

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* CatBoost
* XGBoost
* LightGBM

---

## Author

Tanish Bansal

B.E. Robotics & Artificial Intelligence

Thapar Institute of Engineering and Technology

Formula 1 Race Prediction Project
