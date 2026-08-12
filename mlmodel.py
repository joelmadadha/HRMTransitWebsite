#This is the file for making the machine learning algorithm. The model will be an XGBRegressor. I am using an 80/10/10 split for training, validation, and test data respectively. 
# I will also be outputting a list of the top 10 most important features so I can take away more information on how to proceed
#The metrics measured will be MAE, RMSE and R2

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json

# load dataset
df = pd.read_csv("halifax_transit_clean.csv")

# parse start time for accurate chronological sorting
df["Start Time"] = pd.to_datetime(df["Start Time"])
df = df.sort_values("Start Time").reset_index(drop=True)

# define target and drop non-predictive identifiers/merge keys
target = "Schedule Adherence"
drop_cols = ["Start Time", "date", "trip_id", "shape_id"]
drop_cols = [col for col in drop_cols if col in df.columns]

X = df.drop(columns=[target] + drop_cols)
y = df[target]

# train, val, test split (80/10/10)
n = len(df)
train_end = int(n * 0.80)
val_end = int(n * 0.90)

X_train, y_train = X.iloc[:train_end].copy(), y.iloc[:train_end].copy()
X_val, y_val = X.iloc[train_end:val_end].copy(), y.iloc[train_end:val_end].copy()
X_test, y_test = X.iloc[val_end:].copy(), y.iloc[val_end:].copy()

# target encode string route identifiers if present
cat_cols = [col for col in ["route_branch", "Route", "route_id"] if col in X_train.columns]
for col in cat_cols:
    if X_train[col].dtype == "object":
        encoder = TargetEncoder(target_type="continuous", smooth="auto", cv=5)
        X_train[f"{col}_enc"] = encoder.fit_transform(X_train[[col]], y_train)
        X_val[f"{col}_enc"] = encoder.transform(X_val[[col]])
        X_test[f"{col}_enc"] = encoder.transform(X_test[[col]])

        X_train = X_train.drop(columns=[col])
        X_val = X_val.drop(columns=[col])
        X_test = X_test.drop(columns=[col])

# drop remaining string columns
string_cols = X_train.select_dtypes(include=["object", "string"]).columns
if len(string_cols) > 0:
    X_train = X_train.drop(columns=string_cols)
    X_val = X_val.drop(columns=string_cols)
    X_test = X_test.drop(columns=string_cols)

# initialize xgboost regressor
model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=42,
    early_stopping_rounds=50,
    eval_metric="rmse"
)

# train model
model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=100
)

# predict on test set
y_pred = model.predict(X_test)

# calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n--- Test Set Evaluation ---")
print(f"MAE:  {mae:.2f} seconds ({mae/60:.2f} minutes)")
print(f"RMSE: {rmse:.2f} seconds ({rmse/60:.2f} minutes)")
print(f"R2:   {r2:.4f}")

# display top 10 feature importances
importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

print(f"\n--- Top 10 Features ---")
print(importance.head(10).to_string(index=False))

metrics = {
    "mae_seconds": float(mae),
    "mae_minutes": float(mae/ 60.0),
    "rmse_seconds": float(rmse)
}

# 2. Save dictionary to disk as JSON
with open("model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# save trained model
model.save_model("halifax_transit_xgb.json")