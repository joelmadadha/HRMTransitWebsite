import pandas as pd

wdata = pd.read_csv(r"hfx_weather_data.csv", encoding = "cp1252")

#Dropping columns that have few unique values as they will have no purpose
wdata = wdata.drop(columns = [col for col in wdata.columns.tolist() if wdata[col].nunique()<2])


