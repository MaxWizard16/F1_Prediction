import joblib
import os
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet
)

from sklearn.ensemble import(
    RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor,AdaBoostRegressor,HistGradientBoostingRegressor
)

from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR/"Data"
MODEL_DIR  = BASE_DIR/ "Models"
MODEL_DIR.mkdir(exist_ok=True)
df = pd.read_csv(DATA_DIR/ "master_dataset_features.csv")

train_df = df[df["year"]<=2022]
test_df = df[df["year"]>=2023]

features = [
    "driverId","constructorId","circuitId","year","grid","driver_recent_form","constructor_recent_form","driver_recent_qualifying_form","driver_circuit_form","constructor_circuit_form","driver_dnf_rate","driver_recent_3","driver_recent_10","driver_recent_momentum","constructor_recent_3","constructor_recent_10","constructor_recent_momentum","driver_tracktype_form","constructor_tracktype_form","relative_grid","driver_vs_teammate"
]

X_train = train_df[features]
y_train = train_df["positionOrder"]

X_test = test_df[features]
y_test = test_df["positionOrder"]

models = {
    "CatBoost": CatBoostRegressor(
        iterations=3000, learning_rate=0.02, depth=8, verbose=0,loss_function="MAE"
    ),

    "LightGBM": LGBMRegressor(
        n_estimators=3000, learning_rate=0.02,num_leaves=63,max_depth=-1,subsample=0.8,colsample_bytree=0.8,random_state=42,force_col_wise=True
    ),

    "XGBoost": XGBRegressor(
        n_estimators=3000, learning_rate=0.02, max_depth = 8,subsample = 0.8, colsample_bytree = 0.8, objective = "reg:squarederror", random_state=42,n_jobs=-1
    ),

    "RandomForest": RandomForestRegressor(
        n_estimators=500,n_jobs=-1,random_state=42
    ),

    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=1500,min_samples_leaf=2, n_jobs=-1, random_state=42
    ),

    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=500,learning_rate=0.05
    ),

    "AdaBoost": AdaBoostRegressor(
        n_estimators=500,learning_rate=0.05,random_state=42
    ),

    "DecisionTree": DecisionTreeRegressor(
        random_state=42
    ),
    
    "Ridge": Ridge(),

    "LinearRegression": LinearRegression(),

    "HistGradientBoosting": HistGradientBoostingRegressor(
        learning_rate=0.02,max_depth=8,max_iter=1500,random_state=42
    ),

    "Lasso" : Lasso(),

    "ElasticNet" : ElasticNet()
}

categorical_cols= [
    "driverId","constructorId","circuitId"
]

X_train_encoded = pd.get_dummies(
    X_train,columns=categorical_cols
)

X_test_encoded = pd.get_dummies(
    X_test,columns=categorical_cols
)

X_train_encoded,X_test_encoded = (
    X_train_encoded.align(
        X_test_encoded,join="left",axis=1,fill_value=0
    )
)

results = []
trained_models = {}

results = []

for name,model in models.items():
    print(f"\nTraining {name}")
    if name == "CatBoost":
        model.fit(
            X_train,y_train,cat_features=categorical_cols
        )
        predictions = model.predict(X_test)
        
    else:
        model.fit(
            X_train_encoded,y_train
        )
        predictions = model.predict(X_test_encoded)
    trained_models[name] = model
    mae = mean_absolute_error(
        y_test,predictions
    )

    results.append({
        "Model":name, "MAE" : mae
    })
    print(
        f"{name} MAE = {mae: .4f}"
    )

results_df = pd.DataFrame(results)
results_df = (
    results_df.sort_values("MAE")
)


print("\nFinal Ranking")
print(results_df)
results_df.to_csv(
    DATA_DIR/"model_comparison.csv",index=False
)

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

print(f"Best Model : {best_model_name}")
print(f"Best MAE: {results_df.iloc[0]['MAE']:.4f}")

joblib.dump(best_model, MODEL_DIR / "best_model.pkl")
joblib.dump(features, MODEL_DIR/ "feature_columns.pkl")

with open(MODEL_DIR / "model_name.txt", "w") as f:
    f.write(best_model_name)

print("\nSaved Successfully!")
print(f"Model : {MODEL_DIR/ 'best_model.pkl'}")
print(f"Features : {MODEL_DIR/ 'features.pkl'}")
print(f"Model Name : {MODEL_DIR/ 'model_name.txt'}")

print("ENSEMBLE TEST")
et_pred = trained_models["ExtraTrees"].predict(X_test_encoded)
xgb_pred = trained_models["XGBoost"].predict(X_test_encoded)
lgb_pred = trained_models["LightGBM"].predict(X_test_encoded)
ensemble_pred = (
    0.4*et_pred + 0.3*xgb_pred + 0.3*lgb_pred
)
ensemble_mae = mean_absolute_error(y_test,ensemble_pred)
print(
    f"Ensemble MAE = {ensemble_mae:.4f}"
)