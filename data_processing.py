import pandas as pd

def mm_ss_to_seconds(time_str):
    """Converts a 'MM:SS' or '-MM:SS' string into total integer seconds."""
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
    """Converts integer seconds back into a clean 'MM:SS' string."""
    sign = "-" if total_seconds < 0 else ""
    total_seconds = abs(int(total_seconds))
    mins = total_seconds // 60
    secs = total_seconds % 60
    return f"{sign}{mins}:{secs:02d}"

data = pd.read_csv(r"C:\Users\joel regular\Downloads\Git\HRMTransitWebsite-1\halifax_transit_simulated.csv")

#turning % cols into decimals
data["% Additional Time due to Missing Service"] = data["% Additional Time due to Missing Service"].apply(lambda x: float(x.strip("%"))/100 if "%" in x else x)
data["% Additional Time due to Bunching"] = data["% Additional Time due to Bunching"].apply(lambda x: float(x.strip("%"))/100 if "%" in x else x)

#turning cols from mm:ss format to seconds
cols_with_colon = [
    col for col in data.columns 
    if data[col].astype(str).str.contains(":").any()
]
data[cols_with_colon]= data[cols_with_colon].map(mm_ss_to_seconds)

#Feature Engineering
data["Total Dwell Time"] = data["Stops Served"] * data["Dwell Time at Stops"]
data["Branch_Direction"] = data["Branch"] + "_" + data["Direction"]
data["Weekday_Hour"] = data["Weekday"] + "_" + data["Hour"].astype(str)
data['previous_bus_delay'] = data.groupby(['Branch', 'Direction'])['Schedule Adherence'].shift(1)
data.to_csv("polished_data.csv", index=False)