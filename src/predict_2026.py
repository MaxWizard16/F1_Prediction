from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR/ "Data"
MODEL_DIR = BASE_DIR / "Models"
GRAPH_DIR = BASE_DIR / "graphs"
GRAPH_DIR.mkdir(parents=True, exist_ok= True)

drivers_2026 = pd.read_csv(DATA_DIR / "drivers_2026.csv")
drivers = pd.read_csv(DATA_DIR / "drivers.csv")
constructors = pd.read_csv(DATA_DIR / "constructors.csv")
circuits = pd.read_csv(DATA_DIR / "circuits.csv")
history = pd.read_csv(DATA_DIR / "master_dataset.csv")

def make_driver_name_lookup():
    name_lookup = drivers_2026[["driverId"]].copy()

    # First try names from the 2026 driver file
    if {"forename", "surname"}.issubset(drivers_2026.columns):
        names = drivers_2026["forename"].fillna("") + " " + drivers_2026["surname"].fillna("")
    elif "name" in drivers_2026.columns:
        names = drivers_2026["name"].fillna("")
    else:
        names = pd.Series("", index=drivers_2026.index)

    name_lookup["driver"] = names.astype(str).str.strip()

    # Fill missing names using historical drivers.csv
    if {"forename", "surname"}.issubset(drivers.columns):
        historical_names = (
            drivers["forename"].fillna("") + " " +
            drivers["surname"].fillna("")
        ).str.strip()

        historical_lookup = pd.DataFrame({
            "driverId": drivers["driverId"],
            "driver_fallback": historical_names
        }).drop_duplicates("driverId")

        name_lookup = name_lookup.merge(
            historical_lookup,
            on="driverId",
            how="left"
        )

        name_lookup["driver"] = (
            name_lookup["driver"].replace("", pd.NA)
            .fillna(name_lookup["driver_fallback"])
        )

    name_lookup["driver"] = name_lookup["driver"].fillna(
        name_lookup["driverId"].map(lambda x: f"Unknown driver ({x})")
    )

    return name_lookup[["driverId", "driver"]]


MODEL_FILE = MODEL_DIR / "win_probability_model.cbm"

if not MODEL_FILE.exists():
    raise FileNotFoundError( "Win Prediction Model not found")

FEATURES = ["driverId", "constructorId", "circuitId", "year", "grid","driver_recent_form", "constructor_recent_form", "driver_recent_qualifying_form", "driver_circuit_form","constructor_circuit_form","driver_dnf_rate","driver_recent_3","driver_recent_10", "driver_recent_momentum", "constructor_recent_3", "constructor_recent_10","constructor_recent_momentum","driver_tracktype_form","constructor_tracktype_form","relative_grid","driver_vs_teammate"]
CATEGORICAL_FEATURES = ["driverId", "constructorId", "circuitId"]

TRACK_TYPE_MAP = {
    "monaco": "street",
    "baku": "street",
    "singapore": "street",
    "jeddah": "street",
    "monza": "high_speed",
    "spa": "high_speed",
    "silverstone": "high_speed",
    "hungaroring": "technical",
    "suzuka": "technical",
    "interlagos": "technical",
}

history = history.sort_values(["year", "round", "raceId"]).copy()

overall_finish = float(history["positionOrder"].mean())
overall_qualifying = float(history["qualifying_position"].mean())
overall_dnf_rate = float(history["is_dnf"].mean())

RECENT_WEIGHTS = np.array([0.4,0.25,0.15,0.1,0.1])

def weighted_recent(values, fallback):
    values = pd.Series(values).dropna().tail(5).to_numpy(dtype=float)
    if len(values) == 0:
        return float(fallback)
    weights = RECENT_WEIGHTS[-len(values):]
    weights = weights/ weights.sum()
    return float(np.sum(values*weights))

def recent_mean(values, window, fallback):
    values = pd.Series(values).dropna().tail(window)
    if values.empty:
        return float(fallback)
    return float(values.mean())

def mean_or_fallback(values, fallback):
    values = pd.Series(values).dropna()
    if values.empty:
        return float(fallback)
    return float(values.mean())

def get_track_type(circuit_ref):
    return TRACK_TYPE_MAP.get(str(circuit_ref).lower(), "technical")

def build_driver_features(circuit_id, circuit_ref):
    selected_track_type = get_track_type(circuit_ref)
    rows = []

    for _, entrant in drivers_2026.iterrows():
        driver_id = int(entrant["driverId"])
        constructor_id = int(entrant["constructorId"])
        driver_history = history[history["driverId"] == driver_id]
        constructor_history = history[history["constructorId"] == constructor_id]
        driver_positions = driver_history["positionOrder"]
        constructor_positions = constructor_history["positionOrder"]
        driver_recent_form = weighted_recent(driver_positions, overall_finish)
        driver_recent_3 = recent_mean(driver_positions, 3, overall_finish)
        driver_recent_10 = recent_mean(driver_positions, 10, overall_finish)
        driver_recent_momentum = driver_recent_10 - driver_recent_3
        constructor_recent_form = weighted_recent(constructor_positions, overall_finish)
        constructor_recent_3 = recent_mean(constructor_positions, 3, overall_finish)
        constructor_recent_10 = recent_mean(constructor_positions, 10, overall_finish)
        constructor_recent_momentum = (constructor_recent_10 - constructor_recent_3)
        driver_recent_qualifying = mean_or_fallback(driver_history["qualifying_position"].tail(5),overall_qualifying)
        driver_tracktype_form = mean_or_fallback(driver_history.loc[driver_history["track_type"] == selected_track_type, "positionOrder"],overall_finish)
        constructor_tracktype_form = mean_or_fallback(constructor_history.loc[constructor_history["track_type"] == selected_track_type, "positionOrder"], overall_finish)
        driver_circuit_form = mean_or_fallback(driver_history.loc[driver_history["circuitId"] == circuit_id, "positionOrder"], overall_finish)
        constructor_circuit_form = mean_or_fallback(constructor_history.loc[constructor_history["circuitId"] == circuit_id, "positionOrder"], overall_finish)
        driver_dnf_rate = mean_or_fallback(driver_history["is_dnf"], overall_dnf_rate)

        rows.append({"driverId": driver_id,
                     "constructorId": constructor_id,
                     "circuitId":int(circuit_id),
                     "year": 2026,
                     "grid": driver_recent_qualifying,
                     "driver_recent_form": driver_recent_form,
                     "constructor_recent_form": constructor_recent_form,
                     "driver_recent_qualifying_form" : driver_recent_qualifying,
                     "driver_circuit_form": driver_circuit_form,
                    "constructor_circuit_form": constructor_circuit_form,
                    "driver_dnf_rate": driver_dnf_rate,
                    "driver_recent_3": driver_recent_3,
                    "driver_recent_10": driver_recent_10,
                    "driver_recent_momentum": driver_recent_momentum,
                    "constructor_recent_3": constructor_recent_3,
                    "constructor_recent_10": constructor_recent_10,
                    "constructor_recent_momentum": constructor_recent_momentum,
                    "driver_tracktype_form": driver_tracktype_form,
                    "constructor_tracktype_form": constructor_tracktype_form,
                    "relative_grid": driver_recent_qualifying,
                    "historical_races": len(driver_history)})

    feature_df = pd.DataFrame(rows)
    max_grid = feature_df["grid"].max()
    if pd.isna(max_grid) or max_grid <= 0:
        feature_df["relative_grid"] = 0.5
    else:
        feature_df["relative_grid"] = feature_df["grid"] / max_grid

    team_averages = feature_df.groupby("constructorId")["driver_recent_form"].transform("mean")
    feature_df["driver_vs_teammate"] = (feature_df["driver_recent_form"] - team_averages)
    return feature_df

def choose_circuit():
    print("\n" + "=" *76)
    print("AVAILABLE CIRCUITS")
    print("=" * 76)
    display = circuits.sort_values(["country", "name"])

    for _, row in display.iterrows():
        print(f"{int(row['circuitId']):>3} | {row['name']} | {row['country']}")

    while True:
        choice = input("\nEnter Circuit ID: ").strip()
        if choice.isdigit():
            selected = circuits[circuits["circuitId"] == int(choice)]
            if not selected.empty:
                return selected.iloc[0]
        print("Invalid Circuit Id, please choose one from the list")

def main():
    selected_circuit = choose_circuit()
    circuit_id = int(selected_circuit["circuitId"])
    circuit_name = selected_circuit["name"]
    circuit_ref = selected_circuit["circuitRef"]
    print(f"\nBuilding 2026 predictions for {circuit_name}...")
    model = CatBoostClassifier()
    model.load_model(str(MODEL_FILE))
    prediction_features = build_driver_features(circuit_id, circuit_ref)
    X = prediction_features[FEATURES].copy()
    raw_probabilities = model.predict_proba(X)[:, 1]
    total = raw_probabilities.sum()
    if total <= 0:
        win_scores = np.full(len(raw_probabilities), 100 / len(raw_probabilities))
    else:
        win_scores = raw_probabilities / total*100

    predictions = prediction_features.copy()
    predictions["win_score_percent"] = win_scores

    predictions = predictions.merge(
    make_driver_name_lookup(),
    on="driverId",
    how="left")

    predictions = predictions.merge(
    constructors[["constructorId", "name"]]
    .drop_duplicates("constructorId")
    .rename(columns={"name": "constructor"}),
    on="constructorId",
    how="left")

    predictions["driver"] = predictions["driver"].fillna("Unknown")
    predictions["constructor"] = predictions["constructor"].fillna("Unknown constructor")

    predictions = predictions.sort_values("win_score_percent", ascending= False).reset_index(drop=True)
    predictions.insert(0, "rank", np.arange(1, len(predictions) +1))

    output_columns = ["rank", "driver", "constructor", "win_score_percent", "historical_races"]
    result = predictions[output_columns].copy()
    print(f"\nDrivers included: {len(result)} / {len(drivers_2026)}")

    print("\n" + "=" *76)
    print(f"2026 WIN-LIKELIHOOD RANKING: {circuit_name}")
    print("=" * 76)
    print(result.to_string(index=False, formatters={"win_score_percent": lambda value: f"{value:.2f}%"},))

    safe_circuit_name = "".join( character.lower() if character.isalnum() else "_" for character in str(circuit_ref)).strip("_")

    csv_path = DATA_DIR / f"2026_predictions_{safe_circuit_name}.csv"
    graph_path = GRAPH_DIR/ f"2026_win_ranking_{safe_circuit_name}.png"

    result.to_csv(csv_path, index= False)

    plt.figure(figsize=(12,9))
    plt.barh(predictions["driver"][::-1], predictions["win_score_percent"][::-1])
    plt.xlabel("Normalized model win score (%)")
    plt.ylabel("Driver")
    plt.title(f"2026 Win-Likelihood Ranking - {circuit_name}")
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(graph_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("\nSaved results:")
    print(csv_path)
    print(graph_path)

    print("\nIMPORTANT:")
    print("- Historical race data supplied ends in 2024.")
    print("- No 2025 or 2026 race results are used.")
    print("- Drivers with no historical entries receive fallback historical features.")
    print("- Scores are normalized model rankings, not calibrated betting odds.")
    print("- Grid is estimated from recent qualifying form until actual qualifying data is supplied.")


if __name__ == "__main__":
    main()