import pandas as pd
import numpy as np
df = pd.read_csv("master_dataset.csv")
df = df.sort_values(["year","raceId"])

status = pd.read_csv("status.csv")
df = df.merge(status[["statusId","status"]],on="statusId",how="left")
df["is_dnf"] = (~df["status"].str.contains("Finished|\\+",case=False,na=False)).astype(int)

weights = np.array([0.4,0.25,0.15,0.1,0.1])
def weighted_average(x):
    x = np.asarray(x)
    current_weights = weights[-len(x):]
    current_weights = current_weights / current_weights.sum()
    return np.sum(x*current_weights)

df["driver_recent_form"] = (df.groupby("driverId")["positionOrder"].transform(lambda x: x.shift(1).rolling(5,min_periods=1).apply(weighted_average,raw=False)))
# print(df[["driverId","raceId","positionOrder","driver_recent_form"]].head(20))
df["constructor_recent_form"] = (df.groupby("constructorId")["positionOrder"].transform(lambda x:x.shift(1).rolling(5,min_periods=1).apply(weighted_average,raw=False)))
# driver_1 = df[df["driverId"] == 1]
# print(driver_1[["raceId","positionOrder","driver_recent_form"]].head(15))
df["driver_circuit_form"] = (df.groupby(["driverId","circuitId"])["positionOrder"].transform(lambda x: x.shift(1).expanding().mean()))
df["constructor_circuit_form"] = (df.groupby(["constructorId","circuitId"])["positionOrder"].transform(lambda x: x.shift(1).expanding().mean()))

df["driver_dnf_rate"] = (df.groupby("driverId")["is_dnf"].transform(lambda x: x.shift(1).expanding().mean()))
df["driver_dnf_rate"] = df["driver_dnf_rate"].fillna(0)

qualifying = pd.read_csv("qualifying.csv")
qualifying_small = qualifying[["raceId","driverId","position"]].copy()
qualifying_small = qualifying_small.rename(columns={"position":"qualifying_position"})
df = df.merge(qualifying_small,on=["raceId","driverId"],how="left")

df["driver_recent_qualifying_form"] = (df.groupby("driverId")["qualifying_position"].transform(lambda x:x.shift(1).rolling(5,min_periods=1).mean()))

overall_average_finish = df["positionOrder"].mean()
overall_average_qualification = df["qualifying_position"].mean()
df["driver_recent_form"] = (df["driver_recent_form"].fillna(overall_average_finish))
df["constructor_recent_form"] = (df["constructor_recent_form"].fillna(overall_average_finish))
df["driver_circuit_form"] = (df["driver_circuit_form"].fillna(overall_average_finish))
df["driver_recent_qualifying_form"] = (df["driver_recent_qualifying_form"].fillna(overall_average_qualification))
df["constructor_circuit_form"] = (df["constructor_circuit_form"]).fillna(overall_average_finish)
df.to_csv("master_dataset_features.csv",index = False)