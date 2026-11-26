# 🏎️ Formula 1 Race Prediction Engine

A machine-learning-based Formula 1 race prediction system built using Python and CatBoost. The project analyses historical F1 race data to estimate driver win likelihoods for a selected circuit and generate a ranked prediction leaderboard for the 2026 season.

## 📌 Project Overview

The objective of this project is to build a predictive system that can:

- Analyse historical Formula 1 race performances.
- Engineer driver, constructor, and circuit-specific features.
- Train and evaluate multiple machine-learning models.
- Predict the relative win likelihood of 2026 drivers.
- Allow users to select a circuit and generate a prediction leaderboard.
- Visualise model performance and race predictions.

The prediction engine uses historical race data available through the 2024 season.

## ⚙️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical computations |
| Scikit-learn | Model evaluation and machine-learning utilities |
| CatBoost | Winner prediction classifier |
| Matplotlib | Data visualisation |
| Seaborn | Statistical visualisation |

## 🧠 Machine Learning Approach

### 1. Historical Dataset Construction

Historical Formula 1 datasets are merged to create a master dataset containing information about:

- Drivers
- Constructors
- Circuits
- Race results
- Starting grid positions
- Qualifying positions
- Race finishing positions
- DNF status

### 2. Feature Engineering

The system generates features to capture historical performance and trends.

Some important features include:

- Driver recent form
- Constructor recent form
- Driver circuit-specific performance
- Constructor circuit-specific performance
- Recent qualifying performance
- Driver DNF rate
- Recent 3-race and 10-race performance
- Performance momentum
- Track-type performance
- Relative grid position
- Driver performance against teammates

### 3. Model Benchmarking

Multiple regression algorithms are benchmarked to predict finishing positions.

The benchmark includes:

- CatBoost
- LightGBM
- XGBoost
- Random Forest
- Extra Trees
- Gradient Boosting
- AdaBoost
- Decision Tree
- Ridge Regression
- Linear Regression
- HistGradientBoosting
- Lasso
- ElasticNet

### 4. Winner Prediction Model

A separate CatBoost classification model is trained to estimate whether a driver wins a race.

The target variable is:

- `1` – Race winner
- `0` – All other finishing positions

The classifier is evaluated using:
- ROC-AUC
- Log Loss
- Brier Score

The final model is trained using the available historical dataset.

## 🏁 Generating 2026 Predictions

The prediction engine allows users to select a circuit using its circuit ID.

It then:

1. Loads the 2026 driver and constructor lineup.
2. Calculates historical driver and constructor features.
3. Loads the trained CatBoost classifier.
4. Estimates win likelihood scores for each entrant.
5. Ranks drivers from highest to lowest score.
6. Exports the leaderboard as a CSV file.
7. Generates a visual ranking graph.

The prediction scores are normalised across the selected driver lineup to sum to 100%.

**Important:** These percentages are relative model ranking scores, not calibrated real-world probabilities or betting odds.

## 📂 Project Structure

```text
F1 Prediction/
│
├── Data/
│   ├── drivers.csv
│   ├── drivers_2026.csv
│   ├── constructors.csv
│   ├── circuits.csv
│   ├── races.csv
│   ├── results.csv
│   ├── qualifying.csv
│   ├── status.csv
│   ├── master_dataset.csv
│   └── master_dataset_features.csv
│
├── Models/
│   ├── best_model.pkl
│   ├── feature_columns.pkl
│   ├── model_name.txt
│   ├── win_probability_model.cbm
│   └── win_model_features.pkl
│
├── graphs/
│   ├── model_comparison.png
│   ├── feature_importance.png
│   ├── actual_vs_predicted.png
│   ├── driver_predictions.png
│   └── 2026_win_ranking_*.png
│
├── src/
│   ├── training_dataset.py
│   ├── feature_engineering.py
│   ├── model_benchmark.py
│   ├── train_model.py
│   ├── train_win_model.py
│   ├── predict_2026.py
│   ├── prediction_dataset.py
│   ├── code.py
│   └── run_pipeline.py
│
└── README.md
```

## 🚀 Installation and Setup

### Prerequisites

- Python 3.10 or newer
- Git
- Visual Studio Code (recommended)

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd "F1 Prediction"
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn catboost
```

Install LightGBM and XGBoost if you intend to run the full model benchmark:

```bash
pip install lightgbm xgboost
```

### 3. Prepare the datasets

Place the required historical CSV files and the 2026 driver lineup inside the `Data/` directory.

### 4. Run the complete prediction pipeline

From the project root directory:

```bash
python src/run_pipeline.py
```

The pipeline executes the following scripts in order:

1. `training_dataset.py`
2. `feature_engineering.py`
3. `train_win_model.py`
4. `predict_2026.py`

When prompted, enter the circuit ID for which you want predictions.

## 📊 Generated Outputs

The project generates several visualisations and data files.

| Output | Description |
|---|---|
| `model_comparison.png` | Comparison of regression model MAE |
| `feature_importance.png` | Importance of model features |
| `actual_vs_predicted.png` | Actual vs predicted finishing positions |
| `driver_predictions.png` | Historical driver prediction comparison |
| `2026_win_ranking_*.png` | Circuit-specific 2026 prediction graph |
| `2026_predictions_*.csv` | Circuit-specific prediction leaderboard |

## ⚠️ Limitations

- Historical race data currently ends in 2024.
- The system does not incorporate actual 2025 or 2026 race results.
- Drivers without historical entries receive fallback feature values.
- Grid position is estimated using recent qualifying performance rather than actual event qualifying results.
- Normalised win scores should not be interpreted as calibrated probabilities.
- Predictions are experimental and should not be treated as guaranteed race outcomes.

## 🔮 Future Improvements

Potential areas for further development:

- Integrate updated 2025 and 2026 race results.
- Incorporate actual qualifying results before race-day predictions.
- Improve probability calibration.
- Include weather and track conditions.
- Incorporate tyre strategies and pit-stop performance.
- Develop circuit-specific models.
- Build an interactive web dashboard for predictions.
- Add automated data updates.

## 📚 Dataset

Historical Formula 1 data:

[Formula 1 World Championship (1950–2024) – Kaggle](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2024)

Dataset by Rohan Rao.

## 👨‍💻 Author

**Tanish Bansal**

B.E. Robotics and Artificial Intelligence

Thapar Institute of Engineering and Technology

---

*Built as a machine-learning project exploring historical Formula 1 performance, feature engineering, model benchmarking, and race outcome prediction.*
