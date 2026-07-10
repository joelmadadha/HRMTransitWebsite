import pandas as pd
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
from sklearn.impute import SimpleImputer
# 1. Load your clean data
data = pd.read_csv(r"C:\Users\joel regular\Downloads\Git\HRMTransitWebsite-1\polished_data.csv")


# 1. Create the lag feature in your dataframe first so it actually exists!
data['previous_bus_delay'] = data.groupby(['Route', 'Direction'])['Schedule Adherence'].shift(1)

# 2. Define X - making sure EVERY single column used below is included here
X = data[[
    "Route", 
    "Branch_Direction", 
    "Weekday_Hour", 
    "Service Day", 
    "Month", 
    "Hour", 
    "Stops Served", 
    "Dwell Time at Stops", 
    "Total Dwell Time",
    "previous_bus_delay"
]]
y = data["Schedule Adherence"]

# 3. Explicit alignment for your transformers
month_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

# Grouping the numeric columns that exist in X
numeric_features = ["Hour", "Stops Served", "Dwell Time at Stops", "Total Dwell Time", "previous_bus_delay"]

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0))
])

# 4. The Master Transformer (All columns in X are accounted for exactly once)
preprocessor = ColumnTransformer(
    transformers=[
        ('high_cardinality_target', TargetEncoder(smooth="auto", cv=5, target_type="continuous"), ['Route', 'Branch_Direction', 'Weekday_Hour']),
        ('binary_onehot', OneHotEncoder(handle_unknown='ignore'), ['Service Day']),
        ('time_ordinal', OrdinalEncoder(categories=[month_order]), ['Month']),
        ('numeric_clean', numeric_transformer, numeric_features)
    ]
)

# 5. The XGBoost Pipeline stays the same
xgboost_pipeline = Pipeline(steps=[
    ('preprocessing_engine', preprocessor),
    ('xgb_regressor', XGBRegressor(
        n_estimators=150, 
        learning_rate=0.05, 
        max_depth=4, 
        subsample=0.8, 
        colsample_bytree=0.8, 
        random_state=42,
        n_jobs=-2
    ))
])

# 6. Validate using Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
print("Evaluating your multi-encoder pipeline...")

cv_scores = cross_val_score(xgboost_pipeline, X, y, cv=kf, scoring="neg_root_mean_squared_error")
rmse_scores = -cv_scores

print(f"Overall Performance: {np.mean(rmse_scores):.2f} ± {np.std(rmse_scores):.2f} seconds of error")

# 7. Train the final model on everything
print("Training final model...")
xgboost_pipeline.fit(X, y)