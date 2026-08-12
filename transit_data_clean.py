#This file is for the cleaning and modification of the halifax_transit_data.csv dataset so it is easy to feed it to a machine learning algorithm

import pandas as pd
import numpy as np
import re

data = pd.read_csv(
    r"C:\Users\joel regular\Downloads\Git\HRMTransitWebsite\halifax_transit_data.csv",
    low_memory=False,
    on_bad_lines="skip",
).copy()


def mm_ss_to_seconds(time_str):
#Converts a 'MM:SS' or '-MM:SS' string into total integer seconds.
    if pd.isna(time_str) or not isinstance(time_str, str) or ':' not in time_str:
        return 0

    is_negative = time_str.startswith('-')
    clean_str = time_str.lstrip('-')

    parts = clean_str.split(':')
    minutes = int(parts[0])
    seconds = int(parts[1])

    total_seconds = (minutes * 60) + seconds
    return -total_seconds if is_negative else total_seconds

def seconds_to_mm_ss(total_seconds):
#Converts integer seconds back into a clean 'MM:SS' string.
    sign = "-" if total_seconds < 0 else ""
    total_seconds = abs(int(total_seconds))
    mins = total_seconds // 60
    secs = total_seconds % 60
    return f"{sign}{mins}:{secs:02d}"
#There appears to be a row 271454 where all values are "Asc\xa0/Desc". This row must be dropped
data = data[~data['Start Time'].astype(str).str.contains('Asc', case=False, na=False)].reset_index(drop=True)

def is_gibberish(branch_val):
    if pd.isna(branch_val):
        return False
    return bool(re.match(r'^[01]_[a-f0-9]+', str(branch_val).strip()))

# 2. Build lookup dictionary of (Route, Direction) -> Most Common Clean Branch Name
clean_rows = data[~data['Branch'].apply(is_gibberish) & data['Branch'].notna()]

direction_to_branch_map = (
    clean_rows.groupby(['Route', 'Direction'])['Branch']
    .agg(lambda x: x.mode()[0] if not x.empty else None)
    .to_dict()
)

# 3. Define the replacement function
def clean_branch_name(row):
    current_branch = str(row['Branch'])
    route = row['Route']
    direction = row['Direction']
    
    if is_gibberish(current_branch):
        lookup_key = (route, direction)
        if lookup_key in direction_to_branch_map and direction_to_branch_map[lookup_key]:
            return direction_to_branch_map[lookup_key]
        else:
            return f"{route} ({direction})"
            
    return current_branch

# 4. Apply the cleanup directly to the Branch column
data['Branch'] = data.apply(clean_branch_name, axis=1)


# Convert the Hour column to a numeric type
data['Hour'] = pd.to_numeric(data['Hour'], errors='coerce')
data = data.dropna(subset=['Hour']).reset_index(drop=True)

#removing (trip path) from start time column
data["Start Time"] = data["Start Time"].astype(str).str.removesuffix("(trip path)").str.strip()

# Replace multiple consecutive spaces with a single space and strip edges
data['Start Time'] = data['Start Time'].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()

# Now parse with format='mixed' or the exact format
data["Start Time"] = pd.to_datetime(data["Start Time"], format="mixed", errors="coerce")

#Sorting datetimes chronologically
data = data.sort_values(by=['Route', 'Direction', 'Start Time']).reset_index(drop=True)
data["date"] = data["Start Time"].dt.date
data["time"] = data["Start Time"].dt.time

#Converting WeekDay into 0 through 6 for Sunday through Monday
data["WeekDay"] = data["WeekDay"].map({"Monday":0, "Tuesday":1, "Wednesday":2, "Thursday":3, "Friday":4, "Saturday":5, "Sunday":6})

#Converting WeekDay to fit XGBoost algorithm. Using cyclical encoding with sin and cos to represent how the days loop
data['day_sin'] = np.sin(2 * np.pi * data['WeekDay'] / 7)
data['day_cos'] = np.cos(2 * np.pi * data['WeekDay'] / 7)

#Creating a binary is_weekend column
data['is_weekend'] = data['WeekDay'].isin([5, 6]).astype(int)

#Converting Service Day into 1's and 0's (1 for Full service, 0 for Reduced service)
data["Service Day"]= data["Service Day"].map({"Full service":1, "Reduced service":0})

#Converting Schedule Adherance and Dwell Time at Stops from mm:ss to seconds using functions defined above
data["Schedule Adherance"] = data["Schedule Adherance"].apply(mm_ss_to_seconds)
data["Dwell Time at Stops"] = data["Dwell Time at Stops"].apply(mm_ss_to_seconds)
#Note: Schedule Adherence is misspelled in the original dataset. Fixing this,
data = data.rename(columns={"Schedule Adherance":"Schedule Adherence"})

#Encoding Direction using sin and cos
#First, we must map North, East, South and West to radians
data["direction_idx"] = data["Direction"].map({"North":0, "East":np.pi/2, "South":np.pi, "West":3*np.pi/2})
data["direction_sin"] = np.sin(data["direction_idx"])
data["direction_cos"] = np.cos(data["direction_idx"])

#Encoding month using sin and cos
data["month_idx"] = data["Month"].map({"January":0, "February":1, "March":2, "April":3, "May":4, "June":5, "July":6, "August":7, "September":8, "October":9, "November":10, "December":11})
data["month_sin"] = np.sin(2 * np.pi * data['month_idx'] / 12)
data["month_cos"] = np.cos(2 * np.pi * data['month_idx'] / 12)

#Encoding Hour with sin and cos
data["hour_sin"] = np.sin(2 * np.pi * data['Hour'] / 24)
data["hour_cos"] = np.cos(2 * np.pi * data['Hour'] / 24)

#making is_morning_rush_hour column and an is_evening_rush_hour column
data["is_morning_rush_hour"] = data["Hour"].between(7, 9).astype(int)
data["is_evening_rush_hour"] = data["Hour"].between(16, 18).astype(int)

#making an is_academic_year column
data["is_academic_year"] = data["month_idx"].isin([0,1,2,3,4,5,8,9,10,11]).astype(int)

# 1. Create combined route_branch identifier
data['route_branch'] = data['Route'].astype(str) + "_" + data['Branch'].astype(str)

# 2. Complete list of columns to drop (Leakage + Temporary intermediate columns)
columns_to_drop = [
    # Data Leakage / Post-trip metrics
    'Average Headway', 'Actual Gap', 'Additional Time due to Missing Service',
    '% Additional Time due to Missing Service', 'Waiting Time as Expirenced',
    'Additional Time due to Bunching', '% Additional Time due to Bunching',
    'Dwell Time at Stops', 'Stops Served',

    # Raw Strings & Vehicle metadata
    'Vehicle', 'Year', 'Time Period', 'date', 'time',

    # Temporary mapping indices and redundant unencoded time features
    'month_idx', 'direction_idx', 'WeekDay', 'Month', 'Hour'
]

# Drop columns cleanly
clean_data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

# Save clean dataset for training
clean_data.to_csv("halifax_transit_clean.csv", index=False)