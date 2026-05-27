import pandas as pd


races = pd.read_csv("races.csv")
constructors = pd.read_csv("constructors.csv")
drivers = pd.read_csv("drivers.csv")
qualifying = pd.read_csv("qualifying.csv")
results = pd.read_csv("results.csv")
drivers_2026 = pd.read_csv("drivers_2026.csv")
# print(races.head())
# print(races.columns)

monaco = races[races["circuit_id"]== "monaco"]
# print(monaco)
monaco_id=monaco["race_id"]
# print(monaco_id)
current_driver_id= drivers_2026["driver_id"]
monaco_driver_results = results[(results["race_id"].isin(monaco_id))&(results["driver_id"].isin(current_driver_id))]
print(monaco_driver_results)