# 🏎️ Formula 1 Race Prediction Engine

A Data Analysis and Machine Learning project built in Python to predict Formula 1 race winners using historical race data, driver performance, and constructor performance.

## Project Status

🚧 In Development

Current phase:

* Data collection
* Data cleaning
* Feature engineering
* Circuit-based prediction engine
* Exploratory analysis

Machine learning models will be added after a strong analytics pipeline is established.

---

## Objective

Build a race prediction engine capable of generating predictions for any Formula 1 circuit using:

* Historical race results
* Driver performance history
* Constructor performance history
* Qualifying performance
* Reliability metrics
* Current season form

The project began with Monaco Grand Prix analysis and has since been refactored into a reusable circuit-based prediction system.

---

## Dataset

Dataset used:

**Formula 1 World Championship (1950–2024)**

Source:

https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2024

Main files used:

* circuits.csv
* races.csv
* results.csv
* drivers.csv
* constructors.csv
* qualifying.csv

---

## Current Pipeline

### Step 1

Select a circuit using its `circuitId`.

### Step 2

Extract all historical races held at that circuit.

### Step 3

Filter race results for current Formula 1 drivers.

### Step 4

Calculate driver-specific circuit performance:

* Average finishing position

### Step 5

Calculate constructor-specific circuit performance:

* Average finishing position

### Step 6

Combine driver and constructor performance into a weighted prediction score.

### Current Prediction Formula

Prediction Score =

0.7 × Driver Circuit Performance

*

0.3 × Constructor Circuit Performance

Lower scores indicate stronger predicted performance.

---

## Current Features

✅ Circuit-based prediction engine

✅ Historical race filtering

✅ Current driver filtering

✅ Driver average finishing position

✅ Constructor average finishing position

✅ Weighted prediction score

✅ Reusable `build_prediction(circuit_id)` function

---

## Recent Progress

### 2026-05-30

* Project created
* Monaco filtering pipeline completed
* Driver performance model completed
* Constructor performance model completed
* First weighted prediction model completed
* Project published to GitHub

### 2026-06-02

* Refactored Monaco-only code into a reusable prediction engine
* Created `build_prediction(circuit_id)`
* Added support for predictions at any circuit in the dataset
* Separated driver and constructor feature generation
* Improved project architecture for future UI and ML integration

---

## Planned Features

### Driver Features

* Circuit wins
* Circuit podiums
* Average qualifying position
* DNF rate
* Street circuit performance

### Constructor Features

* Historical circuit performance
* Reliability metrics
* Current season pace

### Race Features

* Qualifying weighting
* Recent form weighting
* Weather effects
* Era-adjusted performance

### Machine Learning

* Linear Regression
* Random Forest
* XGBoost

### User Interface

* Race selection dropdown
* Streamlit dashboard
* Interactive prediction visualizations

---

## Technologies Used

### Current

* Python
* Pandas
* Git
* GitHub

### Planned

* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Streamlit

---

## Learning Goals

This project is being built as a practical exercise in:

* Data Analysis
* Feature Engineering
* Relational Datasets
* Data Pipelines
* Machine Learning Workflows
* Sports Analytics
* Predictive Modeling

The goal is to understand the complete data science process, from raw data collection and cleaning to feature engineering, model development, and deployment.

---

## Repository Structure

```text
F1_Prediction/
│
├── circuits.csv
├── races.csv
├── results.csv
├── drivers.csv
├── constructors.csv
├── qualifying.csv
├── drivers_2026.csv
│
├── prediction.py
│
└── README.md
```

---

## Future Roadmap

### Phase 1 — Data Analysis

* Historical race filtering
* Driver statistics
* Constructor statistics
* Feature engineering

### Phase 2 — Prediction Engine

* Circuit-based predictions
* Enhanced scoring system
* Qualifying integration
* Reliability metrics

### Phase 3 — Machine Learning

* Feature selection
* Model training
* Performance evaluation
* Hyperparameter tuning

### Phase 4 — Deployment

* Streamlit web application
* Interactive race selection
* Prediction dashboard
* Model visualization

---

## Author

**Tanish Bansal**

B.E. Robotics & Artificial Intelligence
Thapar Institute of Engineering & Technology, Patiala

### About This Project

This project is being developed as a hands-on exploration of:

* Data Analysis
* Feature Engineering
* Sports Analytics
* Machine Learning
* Predictive Modeling

The objective is not only to predict Formula 1 race outcomes but also to build a complete data science workflow from raw historical data to deployable prediction systems.

GitHub: https://github.com/MaxWizard16
