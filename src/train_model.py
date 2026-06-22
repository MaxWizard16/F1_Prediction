import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error

qualifying = pd.read_csv("qualifying.csv")
df = pd.read_csv("master_dataset_features.csv")

train_df = df[df["year"]<= 2022]
test_df = df[df["year"]>=2023]

# print("Training rows:", len(train_df))
# print("Testing rows:", len(test_df))

features = ["driverId","constructorId", "circuitId","year","driver_recent_form","constructor_recent_form","driver_recent_qualifying_form","driver_circuit_form",
"constructor_circuit_form",
"driver_dnf_rate"]

X_train = train_df[features]
y_train = train_df["positionOrder"]

X_test = test_df[features]
y_test = test_df["positionOrder"]



model = CatBoostRegressor(iterations=500, learning_rate= 0.05, depth=6, verbose=100)
model.fit(X_train,y_train,cat_features=["driverId","constructorId","circuitId"])
# print(X_train)
# print()
# print(y_train)

predictions = model.predict(X_test)
# print(predictions[:10])

mae = mean_absolute_error(y_test,predictions)
print("MAE:",mae)

results_df = X_test.copy()
results_df["actual"] = y_test.values
results_df["predicted"] = predictions.round(0)

# print(results_df[["driverId","constructorId","circuitId","grid","actual","predicted"]])
# print(qualifying.columns)
# print(qualifying.head())