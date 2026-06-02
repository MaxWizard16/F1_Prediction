import pandas as pd

drivers_2026 = pd.read_csv("drivers_2026.csv")
results = pd.read_csv("results.csv")
races = pd.read_csv("races.csv")
drivers = pd.read_csv("drivers.csv")

def build_prediction(circuitId):
    selected_races = races[races["circuitId"] == circuitId]
    #print(selected_races[["year","name"]])
    race_ids = selected_races["raceId"]
    # print(race_ids)
    race_driver_results = results[
    results["raceId"].isin(race_ids) &
    results["driverId"].isin(drivers_2026["driverId"])]
    # print(race_driver_results)
    driver_average_finish = (
    race_driver_results
    .groupby("driverId")["positionOrder"]
    .mean()
    .sort_values()
    .reset_index())
    driver_average_finish = driver_average_finish.merge(drivers[["driverId","forename","surname"]],on="driverId")
    # print(driver_average_finish)
    circuit_results = results[results["raceId"].isin(race_ids)]
    constructor_average_finish = (circuit_results.groupby("constructorId")["positionOrder"].mean().sort_values().reset_index())
    # print(constructor_average_finish)
    prediction_table = drivers_2026.copy()
    prediction_table = prediction_table.merge(driver_average_finish,on="driverId")
    prediction_table = prediction_table.merge(constructor_average_finish,on="constructorId")
    # print(prediction_table)
    prediction_table["prediction_score"] = (((prediction_table["positionOrder_x"])*0.7)+(prediction_table["positionOrder_y"])*0.3)
    prediction_table = prediction_table.sort_values("prediction_score")
    print(prediction_table)

build_prediction(9)

# monaco_races = races[races["circuitId"] == 6]
# # print(monaco_races)

# monaco_race_id = monaco_races["raceId"]
# # print(monaco_race_id)

# monaco_driver_results = results[
#     results["raceId"].isin(monaco_race_id) &
#     results["driverId"].isin(drivers_2026["driverId"])
# ]
# # print(monaco_driver_results)
# driver_average_finish = (monaco_driver_results.groupby("driverId")["positionOrder"].mean().sort_values().reset_index())
# driver_average_finish = driver_average_finish.merge(drivers[["driverId","forename","surname"]],on="driverId")
# # print(driver_average_finish)
# # best_driver = driver_average_finish.sort_values("positionOrder").iloc[0]

# # print(
# #     f"The predicted winner of the next Monaco GP is "
# #     f"{best_driver['forename']} {best_driver['surname']} "
# #     f"with an average Monaco finish of "
# #     f"{best_driver['positionOrder']:.2f}"
# # )

# constructor_average_finish = (monaco_driver_results.groupby("constructorId")["positionOrder"].mean().sort_values().reset_index())
# # print(constructor_average_finish)

# drivers_2026 = drivers_2026.merge(driver_average_finish,on="driverId")
# # print(drivers_2026.columns)

# drivers_2026 = drivers_2026.merge(constructor_average_finish,on="constructorId")
# drivers_2026["prediction_score"] = (drivers_2026["positionOrder_x"]*0.7) + (drivers_2026["positionOrder_y"]*0.3)
# drivers_2026 = drivers_2026.sort_values("prediction_score")
# print(drivers_2026[["forename","surname","prediction_score"]])