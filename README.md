# 🏎️ Monaco Grand Prix Prediction Engine

A Machine Learning and Data Analysis project built in Python to predict the winner of the Monaco Grand Prix using historical Formula 1 data.

## Project Status

🚧 In Development

Current phase:
- Data collection
- Data cleaning
- Feature engineering
- Exploratory analysis

Machine learning models will be added after a strong analytics pipeline is established.

---

## Objective

Predict the most likely winner of the Monaco Grand Prix using:

- Historical Monaco race results
- Driver performance history
- Constructor performance history
- Qualifying performance
- Reliability metrics
- Current season form

The goal is to build a progressively more sophisticated prediction engine rather than immediately jumping into machine learning.

---

## Dataset

Dataset used:

**Formula 1 World Championship (1950–2024)**

Source:
https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2024

Main files used:

- circuits.csv
- races.csv
- results.csv
- drivers.csv
- constructors.csv
- qualifying.csv

---

## Current Pipeline

### Step 1
Identify Monaco's circuit ID from `circuits.csv`.

### Step 2
Extract all historical Monaco races from `races.csv`.

### Step 3
Filter race results to include only:
- Monaco Grand Prix races
- Current 2026 drivers

### Step 4
Calculate historical driver performance:
- Average Monaco finishing position

### Step 5
Calculate historical constructor performance:
- Average Monaco finishing position

### Step 6
Create a weighted prediction score:

Prediction Score =
70% Driver Monaco Performance +
30% Constructor Monaco Performance

Lower score indicates a stronger predicted performance.

---

## Current Features

✅ Monaco race filtering

✅ Current driver filtering

✅ Driver average Monaco finish

✅ Constructor average Monaco finish

✅ Weighted prediction score

---

## Planned Features

### Driver Features

- Monaco wins
- Monaco podiums
- Average qualifying position
- DNF rate
- Street circuit performance

### Constructor Features

- Monaco team performance
- Team reliability
- Current season pace

### Race Features

- Qualifying importance weighting
- Recent form weighting
- Weather effects
- Era-adjusted performance

---

## Technologies Used

- Python
- Pandas
- Git
- GitHub

Future:
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn

---

## Learning Goals

This project is being built as a practical exercise in:

- Data analysis
- Feature engineering
- Relational datasets
- Data pipelines
- Machine learning workflows
- Sports analytics

---

## Timeline

### 2026-05-30
- Project created
- Monaco filtering pipeline completed
- Driver performance model completed
- Constructor performance model completed
- First weighted prediction model completed
- Project published to GitHub

---

## Author

Tanish Bansal

Robotics & Artificial Intelligence
Thapar Institute of Engineering & Technology
