import pandas as pd
results = pd.read_csv("results.csv")
races = pd.read_csv("races.csv")

df = results[["raceId", "driverId", "constructorId", "grid", "positionOrder"]].copy()
df = df.merge(races[["raceId","year","circuitId"]], on="raceId")
df = df[["raceId","year","circuitId","driverId","constructorId","grid","positionOrder"]]
print(df)
df.to_csv("master_dataset.csv", index = False)
print(df.isnull().sum())
