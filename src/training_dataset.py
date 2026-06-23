import pandas as pd
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR/"Data"
results = pd.read_csv(DATA_DIR/"results.csv")
races = pd.read_csv(DATA_DIR/"races.csv")
circuits = pd.read_csv(DATA_DIR/"circuits.csv")
drivers = pd.read_csv(DATA_DIR/"drivers.csv")
qualifying = pd.read_csv(DATA_DIR/"qualifying.csv")
status = pd.read_csv(DATA_DIR/"status.csv")

df = results[["raceId", "driverId", "constructorId", "grid", "positionOrder","statusId"]].copy()
df = df.merge(races[["raceId","year","circuitId","round"]], on="raceId")
df = df.merge(circuits[["circuitId","circuitRef","alt"]],on="circuitId",how="left")

TRACK_TYPE_MAP = {
    "monaco" : "street", "baku": "street", "singapore" : "street", "jeddah" : "street", "monza": "high_speed", "spa" : "high_speed", "silverstone" : "high_speed", "hungaroring" : "technical", "suzuka" : "technical", "interlagos" : "technical" }
df["track_type"] = (
    df["circuitRef"].str.lower().map(TRACK_TYPE_MAP).fillna("technical")
)
qualifying_small = qualifying[
    [
        "raceId","driverId","position"
    ]
].copy()
qualifying_small = qualifying_small.rename(columns={
    "position" : "qualifying_position"
})
df = df.merge(
    qualifying_small,on=[
        "raceId","driverId"
    ],how="left"
)

df = df.merge(
    status[[
        "statusId","status"
    ]],on="statusId",how="left"
)
df["is_dnf"] = (
    ~df["status"].str.contains("Finished|\\+",case=False,na=False)).astype(int)


df = df[["raceId","year","circuitId","driverId","constructorId","grid","positionOrder","statusId","round","track_type","alt","qualifying_position","is_dnf","circuitRef"]]
df = df.sort_values(
    [
        "year","round","raceId"
    ]
).reset_index(drop=True)
print(df)

df.to_csv(DATA_DIR/"master_dataset.csv", index = False)
print(df.isnull().sum())