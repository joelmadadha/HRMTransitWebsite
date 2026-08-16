import pandas as pd
import numpy as np

# missing data flags
missing_values = ["", " ", "NA", "N/A", "M"]

# load datasets
data = pd.read_csv(r"halifax_transit_clean")
weather = pd.read_csv(
    r"hfx_weather_data.csv",
    na_values=missing_values,
    keep_default_na=True,
    encoding="latin1"
)
dates = pd.read_csv(r"calendar_dates.txt")

# extract date and hour from Start Time
data["Start Time"] = pd.to_datetime(data["Start Time"])
data["date"] = data["Start Time"].dt.strftime("%Y-%m-%d")
data["hour"] = data["Start Time"].dt.hour

# get holiday override dates
dates["date_clean"] = pd.to_datetime(dates["date"].astype(str), format="%Y%m%d")
holiday_dates = dates[dates["exception_type"] == 2]["date_clean"].dt.strftime("%Y-%m-%d").unique()
data["is_holiday_schedule"] = data["date"].isin(holiday_dates).astype(int)

# filter and clean weather
cols_to_keep = [
    "Date/Time (LST)", "Temp (°C)", "Rel Hum (%)",
    "Precip. Amount (mm)", "Wind Spd (km/h)"
]
clean_weather = weather[cols_to_keep].copy()
clean_weather.columns = ["datetime", "temp_c", "rel_hum", "precip_mm", "wind_speed_kmh"]

# parse weather dates
clean_weather["datetime"] = pd.to_datetime(clean_weather["datetime"])
clean_weather["date"] = clean_weather["datetime"].dt.strftime("%Y-%m-%d")
clean_weather["hour"] = clean_weather["datetime"].dt.hour

# fill missing weather values
clean_weather["precip_mm"] = clean_weather["precip_mm"].fillna(0.0)
num_weather_cols = ["temp_c", "rel_hum", "wind_speed_kmh"]
clean_weather[num_weather_cols] = clean_weather[num_weather_cols].ffill().bfill()

# merge weather onto main dataset
weather_merge_cols = ["date", "hour", "temp_c", "rel_hum", "precip_mm", "wind_speed_kmh"]
data = pd.merge(data, clean_weather[weather_merge_cols], on=["date", "hour"], how="left")

# load gtfs files
shapes = pd.read_csv(r"shapes.txt")
stop_times = pd.read_csv(r"stop_times.txt")
trips = pd.read_csv(r"trips.txt")
routes = pd.read_csv(r"routes.txt")

# check if shape crosses macdonald or mackay bridge
def check_bridge(group):
    mac = group["shape_pt_lat"].between(44.654, 44.662) & group["shape_pt_lon"].between(-63.585, -63.570)
    mck = group["shape_pt_lat"].between(44.675, 44.685) & group["shape_pt_lon"].between(-63.608, -63.590)
    return int((mac | mck).any())

# summarize shapes
shape_summary = shapes.groupby("shape_id").apply(lambda g: pd.Series({
    "route_length_km": g["shape_dist_traveled"].max(),
    "crosses_bridge": check_bridge(g)
})).reset_index()

# summarize stop counts per trip
stop_summary = stop_times.groupby("trip_id")["stop_sequence"].max().reset_index()
stop_summary.rename(columns={"stop_sequence": "num_stops"}, inplace=True)

# combine trip, shape, stop, and route info
trip_features = pd.merge(trips[["trip_id", "route_id", "shape_id"]], shape_summary, on="shape_id", how="left")
trip_features = pd.merge(trip_features, stop_summary, on="trip_id", how="left")
trip_features = pd.merge(trip_features, routes[["route_id", "route_type"]], on="route_id", how="left")

# merge gtfs features into main dataset
if "trip_id" in data.columns:
    data = pd.merge(data, trip_features[["trip_id", "route_length_km", "crosses_bridge", "num_stops", "route_type"]], on="trip_id", how="left")
elif "route_id" in data.columns or "Route" in data.columns:
    route_key = "route_id" if "route_id" in data.columns else "Route"
    route_summary = trip_features.groupby("route_id").agg({
        "route_length_km": "mean",
        "crosses_bridge": "max",
        "num_stops": "mean",
        "route_type": "first"
    }).reset_index()
    data = pd.merge(data, route_summary, left_on=route_key, right_on="route_id", how="left")

# fill missing gtfs values
data["crosses_bridge"] = data["crosses_bridge"].fillna(0).astype(int)
data["route_length_km"] = data["route_length_km"].fillna(data["route_length_km"].median())
data["num_stops"] = data["num_stops"].fillna(data["num_stops"].median())

# save dataset
data.to_csv(r"halifax_transit_clean.csv", index=False)
