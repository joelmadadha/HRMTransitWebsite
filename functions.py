#This file will house all functions I will need to make my website. This will be used in website.py

import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
import json
import numpy as np
import duckdb


PARQUET_FILE = "halifax_transit_clean.parquet"
import os

# Temporarily remove @st.cache_data while debugging
@st.cache_data
def load_data(relative_filename):
    try:
        # Use absolute path resolving relative to this script file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(script_dir, relative_filename)
        
        # Check if file exists before attempting load
        if not os.path.exists(full_path):
            st.error(f"File not found at resolved path: {full_path}")
            return None

        # Load your data
        df = pd.read_parquet(full_path) # adjust reader as needed
        return df

    except Exception as e:
        st.error("An error occurred inside load_data():")
        st.exception(e)
        return None

@st.cache_resource
def load_model():
    model = XGBRegressor()
    model.load_model("halifax_transit_xgb.json")
    return model

@st.cache_data
def load_metrics():
    try:
        with open("model_metrics.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"mae_minutes": 3.23, "rmse_seconds": 333.23}


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


def style_delay_cell(val):
    #Applies high-contrast dark mode styling to delay numbers
    try:
        num = float(str(val).replace(" min", "").strip())
        if num < 3.0:
            # Deep Forest Green BG with Bright Emerald Text
            return 'background-color: #0e3a1e; color: #40c057; font-weight: bold;' 
        elif num < 6.0:
            # Deep Amber BG with Bright Yellow/Gold Text
            return 'background-color: #3f2e00; color: #ffd43b; font-weight: bold;' 
        else:
            # Deep Crimson BG with Bright Red/Coral Text
            return 'background-color: #4a1215; color: #ff6b6b; font-weight: bold;' 
    except:
        return ''





@st.cache_data
def get_route_list():
    """Fetches unique routes instantly from Parquet without loading the dataset into memory."""
    conn = duckdb.connect()
    query = f"""
        SELECT DISTINCT Route 
        FROM '{PARQUET_FILE}'
        WHERE TRY_CAST("Start Time" AS TIMESTAMP) <= '2026-12-31 23:59:59'
        ORDER BY Route
    """
    routes = conn.execute(query).fetchall()
    return [r[0] for r in routes if r[0] is not None]


@st.cache_data
def load_gtfs_lookup():
    """Generates GTFS branch aggregate metrics directly in DuckDB SQL."""
    conn = duckdb.connect()
    query = f"""
        SELECT 
            route_branch,
            MAX(crosses_bridge) AS crosses_bridge,
            AVG(route_length_km) AS route_length_km,
            AVG(num_stops) AS num_stops,
            AVG(direction_sin) AS direction_sin,
            AVG(direction_cos) AS direction_cos
        FROM '{PARQUET_FILE}'
        WHERE TRY_CAST("Start Time" AS TIMESTAMP) <= '2026-12-31 23:59:59'
        GROUP BY route_branch
    """
    return conn.execute(query).df()


@st.cache_data
def load_filtered_data(selected_route=None):
    """Queries and loads into RAM only the rows matching the selected route and valid dates (<= 2026)."""
    try:
        conn = duckdb.connect()

        # Build SQL query safely with explicit double quotes for column names with spaces
        query = f"""
            SELECT * 
            FROM '{PARQUET_FILE}'
            WHERE TRY_CAST("Start Time" AS TIMESTAMP) <= '2026-12-31 23:59:59'
        """

        if selected_route and selected_route != "All":
            # Sanitize single quotes in route string
            safe_route = str(selected_route).replace("'", "''")
            query += f" AND Route = '{safe_route}'"

        df = conn.execute(query).df()

        if df.empty:
            st.warning(f"No records found for route: {selected_route}")
            return pd.DataFrame(), "Route", "route_branch"

        # Datetime conversion and temporal features
        df["Start Time"] = pd.to_datetime(df["Start Time"])
        df["Hour"] = df["Start Time"].dt.hour.astype("int8")
        df["Month"] = df["Start Time"].dt.month_name().astype("category")
        df["Day of the Week"] = df["Start Time"].dt.day_name().astype("category")

        # Column mappings
        branch_col = "route_branch" if "route_branch" in df.columns else ("Branch" if "Branch" in df.columns else "Route")
        route_col = "Route" if "Route" in df.columns else branch_col

        branch_str = df[branch_col].astype(str)
        has_underscore = branch_str.str.contains("_", regex=False)
        df["Branch_Clean"] = branch_str.where(~has_underscore, branch_str.str.split("_").str[1]).astype("category")

        return df, route_col, branch_col

    except Exception as e:
        st.error("❌ Error executing load_filtered_data():")
        st.exception(e)
        return pd.DataFrame(), "Route", "route_branch"