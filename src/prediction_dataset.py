from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"

drivers_2026 = pd.read_csv(DATA_DIR/"drivers_2026.csv")
master = pd.read_csv(DATA_DIR/"master_dataset_features.csv")
circuits = pd.read_csv(DATA_DIR / "circuits_2026.csv")

print("\n" + "=" * 50)
print("Available Circuits")
print("=" * 50)

for _, row in circuits.sort_values("circuitId").iterrows():
    print(f"{row['circuitId']:>2} {row['name']} {row['country']}")

circuit_id = int(input("\nEnter Circuit Id: "))
selected_circuit = circuits[
    circuits["circuitId"] == circuit_id
]

if selected_circuit.empty:
    print("Invalid Circuit ID")
    exit()

print("\nSelected Circuit:")
print(selected_circuit.iloc[0]["name"])