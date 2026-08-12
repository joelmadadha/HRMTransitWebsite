#This file will house all functions related to creating the features for making predictions with my model. This will be used in website.py

import pandas as pd
import numpy as np

def build_feature_grid(date_list, start_hour, end_hour, expected_features, preset, gtfs_stats):
    """Constructs feature matrix matching the model's exact 24 numerical/GTFS features."""
    feature_rows = []
    raw_timestamps = []
    
    for dt in date_list:
        for h in range(start_hour, end_hour + 1):
            row_dict = {col: 0.0 for col in expected_features}
            
            month = dt.month
            day_of_week = dt.weekday() # 0 = Monday, 6 = Sunday
            
            # --- 1. Temporal & Calendar Features ---
            if "Service Day" in expected_features: row_dict["Service Day"] = float(day_of_week)
            if "hour" in expected_features: row_dict["hour"] = float(h)
            if "is_weekend" in expected_features: row_dict["is_weekend"] = 1.0 if day_of_week >= 5 else 0.0
            if "is_academic_year" in expected_features: row_dict["is_academic_year"] = 1.0 if month in [9, 10, 11, 12, 1, 2, 3, 4] else 0.0
            if "is_holiday_schedule" in expected_features: row_dict["is_holiday_schedule"] = 0.0
            
            # Cyclical Time Encoding
            if "hour_sin" in expected_features: row_dict["hour_sin"] = np.sin(2 * np.pi * h / 24)
            if "hour_cos" in expected_features: row_dict["hour_cos"] = np.cos(2 * np.pi * h / 24)
            if "month_sin" in expected_features: row_dict["month_sin"] = np.sin(2 * np.pi * month / 12)
            if "month_cos" in expected_features: row_dict["month_cos"] = np.cos(2 * np.pi * month / 12)
            if "day_sin" in expected_features: row_dict["day_sin"] = np.sin(2 * np.pi * day_of_week / 7)
            if "day_cos" in expected_features: row_dict["day_cos"] = np.cos(2 * np.pi * day_of_week / 7)

            # Rush Hour Flags
            if "is_morning_rush_hour" in expected_features: 
                row_dict["is_morning_rush_hour"] = 1.0 if 7 <= h <= 9 else 0.0
            if "is_evening_rush_hour" in expected_features: 
                row_dict["is_evening_rush_hour"] = 1.0 if 15 <= h <= 18 else 0.0

            # --- 2. Weather Features ---
            for w_col, w_val in preset.items():
                if w_col in expected_features:
                    row_dict[w_col] = float(w_val)

            # --- 3. GTFS Route Features (Differentiates Route 1 vs Route 10) ---
            for g_col, g_val in gtfs_stats.items():
                if g_col in expected_features:
                    row_dict[g_col] = float(g_val)

            feature_rows.append(row_dict)
            raw_timestamps.append((dt, h))

    return pd.DataFrame(feature_rows)[expected_features], raw_timestamps


WEATHER_PRESETS = {
        "Clear / Normal": {"temp_c": 15.0, "rel_hum": 60.0, "precip_mm": 0.0, "wind_speed_kmh": 10.0},
        "Light Rain / Fog": {"temp_c": 10.0, "rel_hum": 90.0, "precip_mm": 1.5, "wind_speed_kmh": 20.0},
        "Heavy Rain / Snowstorm": {"temp_c": -2.0, "rel_hum": 98.0, "precip_mm": 8.0, "wind_speed_kmh": 40.0}
    }

LABEL_MAP = {
    "hour": "Hour of Day", "hour_sin": "Daily Time Cycle", "hour_cos": "Daily Time Cycle", "direction_sin": "Travel Direction", "direction_cos": "Travel Direction",
    "month_sin": "Seasonal Cycle", "month_cos": "Seasonal Cycle",
    "day_sin": "Weekly Cycle", "day_cos": "Weekly Cycle",
    "is_morning_rush_hour": "Morning Rush Hour", "is_evening_rush_hour": "Evening Rush Hour",
    "temp_c": "Temperature", "rel_hum": "Humidity", "precip_mm": "Precipitation",
    "wind_speed_kmh": "Wind Speed", "route_length_km": "Route Length",
    "crosses_bridge": "Bridge Crossing", "num_stops": "Number of Stops"
}
