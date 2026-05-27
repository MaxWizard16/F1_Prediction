import pandas as pd

drivers_2026 = pd.read_csv("drivers_2026.csv")
results = pd.read_csv("results.csv")
races = pd.read_csv("races.csv")
drivers = pd.read_csv("drivers.csv")

monaco_races = races[races["circuitId"] == 6]
# print(monaco_races)

monaco_race_id = monaco_races["raceId"]
# print(monaco_race_id)

monaco_driver_results = results[
    results["raceId"].isin(monaco_race_id) &
    results["driverId"].isin(drivers_2026["driverId"])
]
# print(monaco_driver_results)
driver_average_finish = (monaco_driver_results.groupby("driverId")["positionOrder"].mean().sort_values().reset_index())
driver_average_finish = driver_average_finish.merge(drivers[["driverId","forename","surname"]],on="driverId")
# print(driver_average_finish)
best_driver = driver_average_finish.sort_values("positionOrder").iloc[0]

print(
    f"The predicted winner of the next Monaco GP is "
    f"{best_driver['forename']} {best_driver['surname']} "
    f"with an average Monaco finish of "
    f"{best_driver['positionOrder']:.2f}"
)