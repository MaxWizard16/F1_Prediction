import pandas as pd

df = pd.read_csv("master_dataset.csv")
df - df.sort_values(["year","raceId"])

df["driver_recent_form"] = (df.groupby("driverId")["positionOrder"].transform(lambda x: x.shift(1).rolling(5,min_periods=1).mean()))
# print(df[["driverId","raceId","positionOrder","driver_recent_form"]].head(20))
df["constructor_recent_form"] = (df.groupby("constructorId")["positionOrder"].transform(lambda x:x.shift(1).rolling(5,min_periods=1).mean()))
# driver_1 = df[df["driverId"] == 1]
# print(driver_1[["raceId","positionOrder","driver_recent_form"]].head(15))
# df["driver_circuit_form"] = (df.groupby(["driverId","circuitId"])["positionOrder"].transform(lambda x: x.shift(1).expanding().mean()))
qualifying = pd.read_csv("qualifying.csv")
qualifying_small = qualifying[["raceId","driverId","position"]].copy()
qualifying_small = qualifying_small.rename(columns={"position":"qualifying_position"})
df = df.merge(qualifying_small,on=["raceId","driverId"],how="left")
df["driver_recent_qualifying_form"] = (df.groupby("driverId")["qualifying_position"].transform(lambda x:x.shift(1).rolling(5,min_periods=1).mean()))

overall_average_finish = df["positionOrder"].mean()
overall_average_qualification = df["qualifying_position"].mean()
df["driver_recent_form"] = (df["driver_recent_form"].fillna(overall_average_finish))
df["constructor_recent_form"] = (df["constructor_recent_form"].fillna(overall_average_finish))
# df["driver_circuit_form"] = (df["driver_circuit_form"].fillna(overall_average_finish))
df["driver_recent_qualifying_form"] = (df["driver_recent_qualifying_form"].fillna(overall_average_qualification))

df.to_csv("master_dataset_features.csv",index = False)