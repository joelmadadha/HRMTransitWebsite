import streamlit as st
import pandas as pd
from data_processing import mm_ss_to_seconds, seconds_to_mm_ss

#load data. Use @st.cache_data() to make sure it only needs to be loaded once
@st.cache_data()
def load_data(file):
    return pd.read_csv(file)



data = load_data(r"C:\Users\joel regular\Downloads\Git\HRMTransitWebsite-1\halifax_transit_simulated.csv")
st.title("Welcome to The Halifax Transit Data Website (W.I.P)")
select = st.selectbox("Select Your Route",options=data["Route"].unique(),index=0,placeholder= "Select Your Route", label_visibility="visible")
if select:
    st.markdown(f"### How late is route {select} each day? Let's find out.")
    
    # 1. Filter data down to just the selected route
    route_df = data[data["Route"] == select].copy()
    
    # 2. Convert the 'Schedule Adherence' column from string text to raw numeric seconds
    route_df["Adherence_Seconds"] = route_df["Schedule Adherence"].apply(mm_ss_to_seconds)
    
    # 3. Group by Weekday and calculate the true numerical mean
    summary_df = route_df.groupby("Weekday")["Adherence_Seconds"].mean().reset_index()
    
    # 4. Convert that mean back into a readable 'MM:SS' string for display
    summary_df["Average Delay"] = summary_df["Adherence_Seconds"].apply(seconds_to_mm_ss)
    
    # 5. Clean up the dataframe columns so it looks pristine for the user
    final_display = summary_df[["Weekday", "Average Delay"]].set_index("Weekday")

    route_df["Average Delay (Minutes)"] = route_df["Adherence_Seconds"] / 60.0
    chart_data = route_df.groupby(["Weekday", "Hour"])["Average Delay (Minutes)"].mean().reset_index()
    
    # 5. Pivot the data: Hours become rows, Days become columns!
    # This is exactly how Streamlit needs the dataframe structured to draw multiple lines.
    pivot_df = chart_data.pivot(index="Hour", columns="Weekday", values="Average Delay (Minutes)")
    
   # 6. Clean up day sorting so Monday to Sunday appear in calendar order
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    existing_days = [day for day in day_order if day in pivot_df.columns]
    pivot_df = pivot_df[existing_days]
    
    # Create a multiselect box populated with the days, defaulting to showing all of them initially
    selected_days = st.multiselect(
        "Filter Days to Display:",
        options=existing_days,
        default=existing_days
    )
    
    # If the user clears all checkboxes, show a warning instead of crashing the chart
    if not selected_days:
        st.warning("Please select at least one day to view the trend chart.")
    else:
        # Filter the columns of your pivot table based on the user's selection
        filtered_pivot = pivot_df[selected_days]
        
        # 7. Render the line chart with ONLY the chosen days
        st.line_chart(filtered_pivot)
    
    # 8. Show the raw table below it just in case they want to check the exact numbers
    with st.expander("Show Data For The Line Chart"):
        st.dataframe(pivot_df)
    
    with st.expander("Show Average Delay Per Day"):
        st.dataframe(final_display)

