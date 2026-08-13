import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime

st.set_page_config(
    page_title="Halifax Transit Predictor", 
    layout="wide"
)

#importing functions from my own files in the repository
from functions import load_data, load_metrics, load_model, build_feature_grid, style_delay_cell, WEATHER_PRESETS, LABEL_MAP




# Initialize Session State for Dynamic Route Counters
if "num_routes" not in st.session_state:
    st.session_state.num_routes = 1

def add_route():
    if st.session_state.num_routes < 4:
        st.session_state.num_routes += 1

def remove_route():
    if st.session_state.num_routes > 1:
        st.session_state.num_routes -= 1

df, branch_lookup, route_col, branch_col = load_data()
model = load_model()
metrics = load_metrics()

st.title("Halifax Transit Delay & Analytics Dashboard")

# Tab Layout
tab1, tab2 = st.tabs(["Delay Prediction & Analytics", "Historical Data Exploration"])


# TAB 1: BATCH PREDICTION & MULTI-ROUTE ANALYTICS
# ===================================================
with tab1:
    st.header("Query Delay Estimates")
    st.write("This tab is for predicting future delays")
    
    # 1. Dynamic Bus Route & Branch Selectors
    st.subheader("1. Select Route & Branch")
    
    selected_combos = []

    for idx in range(st.session_state.num_routes):
        col_r, col_b = st.columns(2)
        with col_r:
            r = st.selectbox(
                f"Bus Route #{idx + 1}", 
                sorted(df[route_col].dropna().unique()), 
                key=f"route_select_{idx}"
            )
        
        available_branches = sorted(df[df[route_col] == r]["Branch_Clean"].dropna().unique())
        with col_b:
            b_clean = st.selectbox(
                f"Branch #{idx + 1}", 
                available_branches, 
                key=f"branch_select_{idx}"
            )

        matching_rows = df[(df[route_col] == r) & (df["Branch_Clean"] == b_clean)]
        raw_b = matching_rows[branch_col].iloc[0] if not matching_rows.empty else b_clean

        selected_combos.append({
            "route": r,
            "branch_clean": b_clean,
            "raw_branch": raw_b,
            "label": f"Route {r} ({b_clean})"
        })

    # Add / Remove Route Buttons
    col_btn1, col_btn2, _ = st.columns([1, 1, 3])
    with col_btn1:
        st.button("➕ Add Route", on_click=add_route, disabled=(st.session_state.num_routes >= 4))
    with col_btn2:
        st.button("🗑️ Remove Route", on_click=remove_route, disabled=(st.session_state.num_routes <= 1))

    st.markdown("---")

    # 2. Date & Hour Range Inputs
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        start_date = st.date_input("From Date", datetime.today())
    with col_d2:
        end_date = st.date_input("To Date", datetime.today())
        
    start_hour, end_hour = st.slider(
        "Select Hour Range (24-Hour Clock)", 
        min_value=0, 
        max_value=23, 
        value=(7, 19)
    )

    # 3. Weather Preset Toggle
    weather_condition = st.radio(
        "Weather Condition",
        list(WEATHER_PRESETS.keys()),
        horizontal=True
    )

    if st.button("Query"):
        if start_date > end_date:
            st.error("Error: 'From Date' must be before or equal to 'To Date'.")
        else:
            date_list = pd.date_range(start_date, end_date, freq="D")
            expected_features = model.get_booster().feature_names
            preset = WEATHER_PRESETS[weather_condition]
            
            chart_records = []
            display_data = []

            # --- Loop over all active routes selected ---
            for combo in selected_combos:
                r = combo["route"]
                b_clean = combo["branch_clean"]
                raw_b = combo["raw_branch"]
                combo_label = combo["label"]

                gtfs_stats = {}
                if not branch_lookup.empty and raw_b in branch_lookup[branch_col].values:
                    gtfs_stats = branch_lookup[branch_lookup[branch_col] == raw_b].iloc[0].to_dict()

                X_batch, raw_timestamps = build_feature_grid(
                    date_list, start_hour, end_hour, expected_features, preset, gtfs_stats
                )
                pred_sec = model.predict(X_batch)
                pred_min = pred_sec / 60.0

                mae_min = metrics.get("mae_minutes", 3.23)
                margin = 1.28 * mae_min

                for (dt, h), p_min in zip(raw_timestamps, pred_min):
                    ci_lower = max(0.0, p_min - margin)
                    ci_upper = p_min + margin
                    ts_str = f"{dt.strftime('%Y-%m-%d')} {h:02d}:00"
                    date_str = dt.strftime('%Y-%m-%d')
                    
                    chart_records.append({
                        "Timestamp": ts_str,
                        "Date": date_str,
                        "Hour": h,
                        "Hour_Label": f"{h:02d}:00",
                        "Estimated Delay (min)": round(p_min, 1),
                        "Route Option": combo_label
                    })

                    display_data.append({
                        "Date": ts_str,
                        "Route": r,
                        "Branch": b_clean,
                        "Estimated Delay": f"{p_min:.1f} min",
                        "80% confidence range": f"{ci_lower:.1f} - {ci_upper:.1f} min"
                    })

            chart_df = pd.DataFrame(chart_records)
            results_df = pd.DataFrame(display_data)

            # --- Results Summary KPI Cards (Per Route Container) ---
            st.markdown("---")
            st.subheader(f"Results Summary ({len(results_df)} Time Slots Queried)")

            unique_route_labels = [c["label"] for c in selected_combos]

            if len(unique_route_labels) > 1:
                summary_cols = st.columns(len(unique_route_labels))
                for i, r_label in enumerate(unique_route_labels):
                    with summary_cols[i]:
                        with st.container(border=True):
                            sub_df = chart_df[chart_df["Route Option"] == r_label]
                            avg_delay = sub_df["Estimated Delay (min)"].mean()
                            max_row = sub_df.loc[sub_df["Estimated Delay (min)"].idxmax()]

                            st.markdown(f"#### {r_label}")
                            st.divider()
                            st.metric("Avg Delay", f"{avg_delay:.1f} min")
                            st.metric("Peak Time", f"{max_row['Timestamp']}")
                            st.metric("Peak Delay", f"{max_row['Estimated Delay (min)']:.1f} min")
            else:
                with st.container(border=True):
                    r_label = unique_route_labels[0]
                    sub_df = chart_df[chart_df["Route Option"] == r_label]
                    avg_delay = sub_df["Estimated Delay (min)"].mean()
                    max_row = sub_df.loc[sub_df["Estimated Delay (min)"].idxmax()]

                    st.markdown(f"#### {r_label}")
                    st.divider()

                    kpi1, kpi2, kpi3 = st.columns(3)
                    kpi1.metric("Avg Delay", f"{avg_delay:.1f} min")
                    kpi2.metric("Peak Delay Time Slot", f"{max_row['Timestamp']}")
                    kpi3.metric("Peak Expected Delay", f"{max_row['Estimated Delay (min)']:.1f} min")

            # VISUALIZATIONS & BENCHMARK
            st.subheader("Interactive Delay Trend")
            
            # Line Chart handles 1 to 4 routes automatically
            fig_line = px.line(
                chart_df, 
                x="Timestamp", 
                y="Estimated Delay (min)",
                color="Route Option",
                markers=True,
                title="Predicted Delay Trend Comparison",
                color_discrete_sequence=["#FF4B4B", "#1F77B4", "#2CA02C", "#FF7F0E"]
            )
            fig_line.update_layout(xaxis_title="Time Slot", yaxis_title="Predicted Delay (Minutes)")
            st.plotly_chart(fig_line, use_container_width=True)

            # --- Heatmap / Bar Chart Section ---
            st.subheader("Hourly Delay Patterns")

            if len(date_list) > 1:
                min_delay = chart_df["Estimated Delay (min)"].min()
                max_delay = chart_df["Estimated Delay (min)"].max()

                # Case 1: Single Route
                if len(unique_route_labels) == 1:
                    r_label = unique_route_labels[0]
                    sub_df = chart_df[chart_df["Route Option"] == r_label]
                    pivot_df = sub_df.pivot(index="Hour_Label", columns="Date", values="Estimated Delay (min)")
                    fig_heat = px.imshow(
                        pivot_df,
                        labels=dict(x="Date", y="Hour of Day", color="Delay (min)"),
                        x=pivot_df.columns,
                        y=pivot_df.index,
                        color_continuous_scale="Reds",
                        title=f"Hourly Delay Heatmap: {r_label}",
                        aspect="auto"
                    )
                    fig_heat.update_xaxes(type="category")
                    st.plotly_chart(fig_heat, use_container_width=True)

                # Case 2: Exactly 2 Routes (Side-by-Side Columns)
                elif len(unique_route_labels) == 2:
                    col_h1, col_h2 = st.columns(2)
                    for col, r_label in zip([col_h1, col_h2], unique_route_labels):
                        with col:
                            sub_df = chart_df[chart_df["Route Option"] == r_label]
                            pivot_df = sub_df.pivot(index="Hour_Label", columns="Date", values="Estimated Delay (min)")
                            fig_heat = px.imshow(
                                pivot_df,
                                labels=dict(x="Date", y="Hour of Day", color="Delay (min)"),
                                x=pivot_df.columns,
                                y=pivot_df.index,
                                color_continuous_scale="Reds",
                                range_color=[min_delay, max_delay],
                                title=r_label,
                                aspect="auto"
                            )
                            fig_heat.update_xaxes(type="category")
                            st.plotly_chart(fig_heat, use_container_width=True)

                # Case 3: 3 or 4 Routes (Tabbed View to Avoid Squeezing)
                else:
                    heat_tabs = st.tabs([f"{lbl}" for lbl in unique_route_labels])
                    for tab, r_label in zip(heat_tabs, unique_route_labels):
                        with tab:
                            sub_df = chart_df[chart_df["Route Option"] == r_label]
                            pivot_df = sub_df.pivot(index="Hour_Label", columns="Date", values="Estimated Delay (min)")
                            fig_heat = px.imshow(
                                pivot_df,
                                labels=dict(x="Date", y="Hour of Day", color="Delay (min)"),
                                x=pivot_df.columns,
                                y=pivot_df.index,
                                color_continuous_scale="Reds",
                                range_color=[min_delay, max_delay],
                                title=f"Hourly Delay Heatmap: {r_label}",
                                aspect="auto"
                            )
                            fig_heat.update_xaxes(type="category")
                            st.plotly_chart(fig_heat, use_container_width=True)

            else:
                # Single Day Hourly Bar Charts
                if len(unique_route_labels) <= 2:
                    cols = st.columns(len(unique_route_labels))
                    for col, r_label in zip(cols, unique_route_labels):
                        with col:
                            sub_df = chart_df[chart_df["Route Option"] == r_label]
                            fig_bar = px.bar(
                                sub_df,
                                x="Hour_Label",
                                y="Estimated Delay (min)",
                                title=r_label,
                                color="Estimated Delay (min)",
                                color_continuous_scale="Reds"
                            )
                            fig_bar.update_layout(xaxis_title="Hour of Day", yaxis_title="Predicted Delay (Minutes)")
                            st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    bar_tabs = st.tabs([f"{lbl}" for lbl in unique_route_labels])
                    for tab, r_label in zip(bar_tabs, unique_route_labels):
                        with tab:
                            sub_df = chart_df[chart_df["Route Option"] == r_label]
                            fig_bar = px.bar(
                                sub_df,
                                x="Hour_Label",
                                y="Estimated Delay (min)",
                                title=r_label,
                                color="Estimated Delay (min)",
                                color_continuous_scale="Reds"
                            )
                            fig_bar.update_layout(xaxis_title="Hour of Day", yaxis_title="Predicted Delay (Minutes)")
                            st.plotly_chart(fig_bar, use_container_width=True)

            # --- WEATHER SIMULATOR (Driven by Primary Route) ---
            with st.expander("Weather 'What-If' Sensitivity Simulator", expanded=False):
                primary_combo = selected_combos[0]
                st.write(f"Quantifies weather impact for baseline selection: **{primary_combo['label']}**.")
                
                gtfs_stats_p = {}
                if not branch_lookup.empty and primary_combo["raw_branch"] in branch_lookup[branch_col].values:
                    gtfs_stats_p = branch_lookup[branch_lookup[branch_col] == primary_combo["raw_branch"]].iloc[0].to_dict()

                weather_sim_records = []
                for w_name, w_preset in WEATHER_PRESETS.items():
                    X_sim, _ = build_feature_grid(date_list, start_hour, end_hour, expected_features, w_preset, gtfs_stats_p)
                    sim_preds_sec = model.predict(X_sim)
                    sim_preds_min = sim_preds_sec / 60.0
                    
                    weather_sim_records.append({
                        "Weather Condition": w_name,
                        "Average Delay (min)": round(sim_preds_min.mean(), 1),
                        "Peak Delay (min)": round(sim_preds_min.max(), 1)
                    })
                
                sim_df = pd.DataFrame(weather_sim_records)
                fig_weather = px.bar(
                    sim_df,
                    x="Weather Condition",
                    y="Average Delay (min)",
                    color="Weather Condition",
                    text_auto=True,
                    title="Average Delay Delta Across Weather Conditions",
                    color_discrete_sequence=["#2CA02C", "#FF7F0E", "#D62728"]
                )
                st.plotly_chart(fig_weather, use_container_width=True)

            # --- MODEL EXPLAINABILITY ENGINE (Tree SHAP) ---
            with st.expander("Feature Driver Explainer ('Why is this trip delayed?')", expanded=False):
                st.write(f"Calculates feature contributions (Tree SHAP) for baseline selection: **{primary_combo['label']}**.")
                
                # Fetch baseline feature grid
                X_shap, _ = build_feature_grid(date_list, start_hour, end_hour, expected_features, preset, gtfs_stats_p)
                dmatrix = xgb.DMatrix(X_shap)
                contribs = model.get_booster().predict(dmatrix, pred_contribs=True)
                
                feat_names = X_shap.columns.tolist()
                contrib_df = pd.DataFrame(contribs[:, :-1], columns=feat_names)
                mean_impact = contrib_df.abs().mean().sort_values(ascending=True)
                
                clean_impact = (
                    mean_impact
                    .rename(index=lambda x: LABEL_MAP.get(x, x))
                    .groupby(level=0)
                    .mean()
                    .sort_values(ascending=True)
                    .tail(8)
                )
                
                fig_shap = px.bar(
                    x=clean_impact.values,
                    y=clean_impact.index,
                    orientation='h',
                    title="Top Model Feature Contributions (Average Driver Impact)",
                    color_discrete_sequence=["#3366CC"]
                )
                fig_shap.update_layout(xaxis_title="Impact Magnitude (Seconds)", yaxis_title="Feature Driver")
                st.plotly_chart(fig_shap, use_container_width=True)

            # --- CONDITIONAL HIGHLIGHTED DATA TABLE ---
            st.subheader("📋 Query Data Table")
            try:
                styled_df = results_df.style.map(style_delay_cell, subset=["Estimated Delay"])
            except AttributeError:
                styled_df = results_df.style.applymap(style_delay_cell, subset=["Estimated Delay"])
                
            st.dataframe(styled_df, use_container_width=True, hide_index=True)


# TAB 2: HISTORICAL DATA EXPLORATION
# ====================================
with tab2:
    st.header("📊 Historical Transit Performance & Analytics")
    st.write("This tab is for exploring historical performace of bus routes in Halifax")

    # --- 1. Route & Branch Selection ---
    st.subheader("1. Select Route & Branch")
    col_e1, col_e2 = st.columns(2)

    with col_e1:
        explore_route = st.selectbox(
            "Select Bus Route",
            sorted(df[route_col].dropna().unique()),
            key="explore_route_select"
        )

    avail_explore_branches = sorted(df[df[route_col] == explore_route]["Branch_Clean"].dropna().unique())
    with col_e2:
        explore_branch = st.selectbox(
            "Select Branch",
            avail_explore_branches,
            key="explore_branch_select"
        )

    # Filter raw historical dataframe
    historical_subset = df[
        (df[route_col] == explore_route) & (df["Branch_Clean"] == explore_branch)
    ].copy()

    if not historical_subset.empty and "Schedule Adherence" in historical_subset.columns:
        # Convert schedule adherence (seconds) to minutes
        if "Delay (Minutes)" not in historical_subset.columns:
            historical_subset["Delay (Minutes)"] = historical_subset["Schedule Adherence"] / 60.0

        # --- 2. Historical KPI Summary Cards ---
        st.markdown("---")
        st.subheader(f"Historical Summary: Route {explore_route} ({explore_branch})")

        avg_hist_delay = historical_subset["Delay (Minutes)"].mean()
        max_hist_delay = historical_subset["Delay (Minutes)"].max()
        total_trips = len(historical_subset)
        
        # Calculate On-Time Performance (trips delayed by <= 3.0 mins)
        on_time_pct = (historical_subset["Delay (Minutes)"] <= 3.0).mean() * 100.0

        with st.container(border=True):
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Historical Avg Delay", f"{avg_hist_delay:.1f} min")
            kpi2.metric("On-Time Performance (≤3m)", f"{on_time_pct:.1f}%")
            kpi3.metric("Max Recorded Delay", f"{max_hist_delay:.1f} min")
            kpi4.metric("Total Trips Recorded", f"{total_trips:,}")

        st.markdown("---")

        # --- 3. Interactive Historical Breakdown ---
        st.subheader("📈 Historical Delay Patterns & Variance")

        col_metric, col_chart_type = st.columns([2, 1])

        metric_mapping = {
            "Hour of Day": "Hour",
            "Day of Week": "Day of the Week",
            "Month of Year": "Month"
        }
        if "precip_mm" in historical_subset.columns:
            metric_mapping["Precipitation (mm)"] = "precip_mm"

        with col_metric:
            selected_metric_name = st.selectbox(
                "Breakdown X-Axis Metric",
                list(metric_mapping.keys()),
                key="hist_metric_select"
            )
            selected_metric_col = metric_mapping[selected_metric_name]

        with col_chart_type:
            chart_style = st.radio(
                "Visualization Style",
                ["Box Plot (Variability)", "Bar Chart (Average)"],
                horizontal=True,
                key="hist_chart_type"
            )

        plot_df = historical_subset.dropna(subset=[selected_metric_col, "Delay (Minutes)"]).copy()

        # Render chosen plot type
        if chart_style == "Box Plot (Variability)":
            fig_hist = px.box(
                plot_df,
                x=selected_metric_col,
                y="Delay (Minutes)",
                title=f"Delay Variability by {selected_metric_name} for Route {explore_route} ({explore_branch})",
                color_discrete_sequence=["#FF4B4B"],
                points="outliers"
            )
        else:
            avg_grouped = plot_df.groupby(selected_metric_col, as_index=False)["Delay (Minutes)"].mean()
            fig_hist = px.bar(
                avg_grouped,
                x=selected_metric_col,
                y="Delay (Minutes)",
                title=f"Average Delay by {selected_metric_name} for Route {explore_route} ({explore_branch})",
                color="Delay (Minutes)",
                color_continuous_scale="Reds",
                text_auto=".1f"
            )

        # Apply calendar sorting where appropriate
        if selected_metric_name == "Day of Week":
            fig_hist.update_xaxes(
                categoryorder="array",
                categoryarray=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            )
        elif selected_metric_name == "Month of Year":
            fig_hist.update_xaxes(
                categoryorder="array",
                categoryarray=[
                    "January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December"
                ]
            )

        fig_hist.update_layout(xaxis_title=selected_metric_name, yaxis_title="Delay (Minutes)")
        st.plotly_chart(fig_hist, use_container_width=True)

    else:
        st.warning("No historical data available for this specific route and branch selection.")
