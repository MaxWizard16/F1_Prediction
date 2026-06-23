# 🏎️ Formula 1 Race Prediction Engine

A Data Analysis and Machine Learning project built in Python to predict Formula 1 race finishing positions using historical Formula 1 data from 1950–2024.

## Project Status

🚧 In Development

Current phase:

* Data collection
* Data cleaning
* Machine learning baseline development
* Feature engineering
* Model evaluation

---

## Objective

Build a machine learning system capable of predicting Formula 1 race results before a race weekend begins.

The long-term goal is to create a prediction engine that combines:

* Historical race results
* Driver performance trends
* Constructor performance trends
* Circuit-specific performance
* Qualifying performance
* Reliability metrics
* Recent form indicators

The project began as a circuit-based statistical prediction system and has now evolved into a machine learning pipeline.

---

## Dataset

Dataset Used:

**Formula 1 World Championship (1950–2024)**

Source:

https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2024

Main files used:

* races.csv
* results.csv
* drivers.csv
* constructors.csv
* qualifying.csv

---

## Current Architecture

### Phase 1 — Statistical Prediction Engine

Initial implementation used historical circuit data to generate predictions.

Pipeline:

1. Select circuit using `circuitId`
2. Extract historical races for that circuit
3. Filter for current drivers
4. Calculate driver average finish
5. Calculate constructor average finish
6. Generate weighted prediction score

Prediction Formula:

Prediction Score =

0.7 × Driver Average Finish

*

0.3 × Constructor Average Finish

Lower scores indicate stronger predicted performance.

---

### Phase 2 — Machine Learning Pipeline

The project now includes a supervised learning workflow.

A master training dataset was created using historical race results.

Each row represents:

Driver + Race

Training features currently include:

* driverId
* constructorId
* circuitId
* year

Optional feature:

* grid position (used for post-qualifying predictions)

Target variable:

* positionOrder (final race finishing position)

---

## Machine Learning Workflow

### Step 1

Build master dataset from:

* races.csv
* results.csv

Dataset size:

* 26,759 race entries
* 7 core columns

Columns:

* raceId
* year
* circuitId
* driverId
* constructorId
* grid
* positionOrder

---

### Step 2

Chronological train/test split

Training Data:

* 1950–2022

Testing Data:

* 2023–2024

Rows:

* Training: 25,840
* Testing: 919

---

### Step 3

Train a CatBoost Regressor

Model:

* CatBoostRegressor

Reason:

* Handles categorical variables directly
* No one-hot encoding required
* Strong performance on tabular datasets

---

### Step 4

Evaluate Model Performance

Baseline Model Features:

* driverId
* constructorId
* circuitId
* year

Results:

MAE (Mean Absolute Error)

4.02 positions

This means the model predicts finishing position within approximately four places on average.

---

### Post-Qualifying Experiment

Including:

* grid position

Improved performance:

MAE = 3.31 positions

This demonstrates the strong predictive power of qualifying position.

---

## Current Features

### Statistical Engine

✅ Circuit-based prediction

✅ Driver average finish

✅ Constructor average finish

✅ Weighted scoring model

✅ Reusable `build_prediction(circuitId)` function

---

### Machine Learning Engine

✅ Master dataset generation

✅ Chronological train/test split

✅ CatBoost implementation

✅ Baseline regression model

✅ Performance evaluation using MAE

✅ Pre-weekend prediction baseline

---

## Recent Progress

### 2026-05-30

* Project created
* Monaco GP analysis completed
* Driver performance pipeline completed
* Constructor performance pipeline completed
* Weighted prediction model completed
* GitHub repository created

### 2026-06-02

* Refactored Monaco-specific code into reusable prediction engine
* Added support for any circuit
* Improved project architecture

### 2026-06-06

* Created master machine learning dataset
* Generated 26,759 historical training examples
* Implemented chronological train/test split
* Trained first CatBoost model
* Evaluated model using Mean Absolute Error
* Achieved MAE of 4.02 using only:

  * driverId
  * constructorId
  * circuitId
  * year
* Achieved MAE of 3.31 when including grid position
* Established first machine learning baseline for future improvements

---

## Planned Features

### Feature Engineering

* Driver recent form
* Constructor recent form
* Circuit-specific driver performance
* Circuit-specific constructor performance
* Driver podium rate
* Driver win rate
* Reliability metrics
* DNF rate

### Race Features

* Historical qualifying strength
* Weather effects
* Era-adjusted performance
* Circuit categorization

### Machine Learning

* Random Forest
* XGBoost
* Hyperparameter tuning
* Ensemble models

### Deployment

* Streamlit dashboard
* Interactive race selection
* Prediction visualizations
* Race simulation interface

---

## Technologies Used

### Current

* Python
* Pandas
* CatBoost
* Git
* GitHub

### Planned

* NumPy
* Scikit-Learn
* XGBoost
* Matplotlib
* Streamlit

---

## Learning Goals

This project is being developed as a practical exploration of:

* Data Analysis
* Feature Engineering
* Sports Analytics
* Machine Learning
* Predictive Modeling
* Time-Series Aware Evaluation
* Data Science Workflows

The goal is to build a complete machine learning system, progressing from raw historical Formula 1 data to deployable predictive models.

---

## Repository Structure

```text
F1_Prediction/
│
├── races.csv
├── results.csv
├── drivers.csv
├── constructors.csv
├── qualifying.csv
├── drivers_2026.csv
│
├── prediction.py
├── build_training_dataset.py
├── train_model.py
│
├── master_dataset.csv
│
└── README.md
```

---

## Future Roadmap

### Phase 1 — Data Preparation

✅ Data collection

✅ Data cleaning

✅ Master dataset generation

---

### Phase 2 — Baseline Machine Learning

✅ CatBoost baseline model

✅ Model evaluation

🔄 Feature engineering

---

### Phase 3 — Advanced Modeling

* Driver form metrics
* Constructor form metrics
* Circuit-aware features
* Hyperparameter optimization

---

### Phase 4 — Deployment

* Streamlit web application
* Interactive prediction dashboard
* Race simulation tools

---

## Author

**Tanish Bansal**

B.E. Robotics & Artificial Intelligence

Thapar Institute of Engineering & Technology, Patiala

GitHub: https://github.com/MaxWizard16

---

## About This Project

This project serves as a hands-on exploration of Formula 1 analytics, machine learning, and predictive modeling.

The objective is not only to predict race outcomes, but also to develop a complete end-to-end machine learning workflow, from raw historical data to production-ready prediction systems.
