from pathlib import Path
import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
MODEL_DIR = BASE_DIR / "Models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR / "master_dataset_features.csv"
FEATURES = ["driverId", "constructorId", "circuitId", "year", "grid","driver_recent_form", "constructor_recent_form", "driver_recent_qualifying_form", "driver_circuit_form","constructor_circuit_form","driver_dnf_rate","driver_recent_3","driver_recent_10", "driver_recent_momentum", "constructor_recent_3", "constructor_recent_10","constructor_recent_momentum","driver_tracktype_form","constructor_tracktype_form","relative_grid","driver_vs_teammate"]
CATEGORICAL_FEATURES = ["driverId", "constructorId", "circuitId"]
df = pd.read_csv(DATA_FILE)

required = FEATURES + ["positionOrder"]
missing = [column for column in required if column not in df.columns]
if missing:
    raise ValueError(f"Missing columns in master_dataset_features.csv: {missing}")

df = df.dropna(subset=FEATURES + ["positionOrder"]).copy()
df["winner"] = (df["positionOrder"] == 1).astype(int)

train_df = df[df["year"] <= 2022].copy()
test_df = df[df["year"] >= 2023].copy()

if train_df.empty or test_df.empty:
    raise ValueError("The year-based train/test split is empty. Check the dataset years.")

X_train = train_df[FEATURES]
y_train = train_df["winner"]
X_test = test_df[FEATURES]
y_test = test_df["winner"]

# First fit: evaluate on seasons not used for training.
validation_model = CatBoostClassifier(iterations=1200,learning_rate=0.03,depth=7, loss_function="Logloss",eval_metric="AUC", random_seed=42,  verbose=200, allow_writing_files=False,)

print("Training win classifier for validation...")
validation_model.fit( X_train, y_train, cat_features=CATEGORICAL_FEATURES,)

test_probabilities = validation_model.predict_proba(X_test)[:, 1]

print("\nHELD-OUT VALIDATION (seasons 2023 onwards)")
print("ROC-AUC:", round(roc_auc_score(y_test, test_probabilities), 4))
print("Log loss:", round(log_loss(y_test, test_probabilities), 4))
print("Brier score:", round(brier_score_loss(y_test, test_probabilities), 4))

# Final fit: use all historical data for the 2026 prediction model.
final_model = CatBoostClassifier(iterations=1200, learning_rate=0.03,depth=7,loss_function="Logloss",eval_metric="AUC",random_seed=42,verbose=200, allow_writing_files=False,)

print("\nTraining final model on all available historical seasons...")
final_model.fit( df[FEATURES], df["winner"],cat_features=CATEGORICAL_FEATURES,)

model_path = MODEL_DIR / "win_probability_model.cbm"
final_model.save_model(str(model_path))
joblib.dump(FEATURES, MODEL_DIR / "win_model_features.pkl")

with open(MODEL_DIR / "win_model_info.txt", "w", encoding="utf-8") as file:
    file.write("Model: CatBoostClassifier\n")
    file.write("Target: positionOrder == 1\n")
    file.write(f"Training rows: {len(df)}\n")
    file.write(f"Latest historical season: {int(df['year'].max())}\n")
    file.write("Note: prediction probabilities are normalized across the selected 2026 driver lineup.")

print("\nSaved successfully:")
print(model_path)
print(MODEL_DIR / "win_model_features.pkl")
