from pathlib import Path
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"

df = pd.read_csv(DATA_DIR / "master_dataset_features.csv")


train_df = df[df["year"] <= 2022]
test_df = df[df["year"] >= 2023]

print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))


features = [
    "driverId","constructorId","circuitId","year","grid","driver_recent_form","constructor_recent_form","driver_recent_qualifying_form","driver_circuit_form","constructor_circuit_form","driver_dnf_rate","driver_recent_3","driver_recent_10","driver_recent_momentum","constructor_recent_3","constructor_recent_10","constructor_recent_momentum","driver_tracktype_form","constructor_tracktype_form","relative_grid","driver_vs_teammate"
]

X_train = train_df[features]
y_train = train_df["positionOrder"]

X_test = test_df[features]
y_test = test_df["positionOrder"]


model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    verbose=100
)

model.fit(
    X_train,
    y_train,
    cat_features=["driverId", "constructorId", "circuitId"]
)


print("\n===== FEATURE IMPORTANCE =====")

feature_importance = model.get_feature_importance()

for feature, importance in zip(features, feature_importance):
    print(f"{feature}: {importance:.2f}")


predictions = model.predict(X_test)


mae = mean_absolute_error(y_test, predictions)

print("\n===== MODEL PERFORMANCE =====")
print("MAE:", round(mae, 4))


results_df = X_test.copy()

results_df["actual"] = y_test.values
results_df["predicted"] = predictions.round(0)

results_df["error"] = abs(
    results_df["actual"] - results_df["predicted"]
)

print("\n===== TOP 20 BIGGEST ERRORS =====")

print(
    results_df.sort_values(
        "error",
        ascending=False
    ).head(20)
)



baseline_predictions = X_test["driver_recent_form"]

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

print("\n===== BASELINE COMPARISON =====")
print("Baseline MAE:", round(baseline_mae, 4))
print("CatBoost MAE:", round(mae, 4))