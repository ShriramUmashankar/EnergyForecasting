# ==========================================================
# XGBOOST HOURLY POWER FORECASTING
# Train: 2006-2009
# Test : 2010
# Future Forecast: Rolling 7 Days (168 Hours)
# Prints:
#   - Test metrics
#   - Future 7-day rolling forecast
#   - Error by Day 1 to Day 7 on 2010 holdout slice
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv(
    "data/raw_data/household_power_consumption.txt",
    sep=";",
    low_memory=False,
    na_values=["nan", "?"]
)

df["dt"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    format="%d/%m/%Y %H:%M:%S"
)

df.drop(columns=["Date", "Time"], inplace=True)
df.set_index("dt", inplace=True)

df = df.apply(pd.to_numeric, errors="coerce")
df = df.interpolate()

# hourly data
df = df.resample("h").mean()

# keep target only
data = df["Global_active_power"].copy()

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

df_xgb = data.to_frame()

df_xgb["hour"] = df_xgb.index.hour
df_xgb["day"] = df_xgb.index.day
df_xgb["weekday"] = df_xgb.index.weekday
df_xgb["month"] = df_xgb.index.month

# lag 1 to 24
for i in range(1, 25):
    df_xgb[f"lag_{i}"] = df_xgb["Global_active_power"].shift(i)

df_xgb.dropna(inplace=True)

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X = df_xgb.drop("Global_active_power", axis=1)
y = df_xgb["Global_active_power"]

X_train = X[:'2009']
y_train = y[:'2009']

X_test = X['2010':]
y_test = y['2010':]

print("Train size:", X_train.shape)
print("Test size :", X_test.shape)

# ==========================================================
# TRAIN MODEL
# ==========================================================

xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)

xgb_model.fit(X_train, y_train)

# ==========================================================
# TEST PREDICTION
# ==========================================================

y_pred_xgb = xgb_model.predict(X_test)

mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)

avg_load = y_test.mean()

print("\n===== TEST METRICS (2010) =====")
print(f"Average Load : {avg_load:.4f} kW")
print(f"MAE          : {mae_xgb:.4f} kW")
print(f"RMSE         : {rmse_xgb:.4f} kW")
print(f"R2 Score     : {r2_xgb:.4f}")
print(f"MAE % Avg    : {100*mae_xgb/avg_load:.2f}%")
print(f"RMSE % Avg   : {100*rmse_xgb/avg_load:.2f}%")

# ==========================================================
# PLOT TEST SAMPLE
# ==========================================================

plt.figure(figsize=(16,6))
plt.plot(y_test.values[:300], label="True")
plt.plot(y_pred_xgb[:300], label="Predicted")
plt.title("XGBoost Test Forecast (2010 Sample)")
plt.xlabel("Hour")
plt.ylabel("kW")
plt.legend()
plt.show()

# ==========================================================
# FUTURE FORECAST: 7 DAYS (168 HOURS)
# Autoregressive rolling prediction
# ==========================================================

future_hours = 168

last_timestamp = df_xgb.index[-1]

future_index = pd.date_range(
    start=last_timestamp + pd.Timedelta(hours=1),
    periods=future_hours,
    freq="h"
)

future_df = pd.DataFrame(index=future_index)

future_df["hour"] = future_df.index.hour
future_df["day"] = future_df.index.day
future_df["weekday"] = future_df.index.weekday
future_df["month"] = future_df.index.month

future_predictions = []

last_known = df_xgb.copy()

for i in range(future_hours):

    # create lag features from rolling history
    for lag in range(1, 25):
        future_df.loc[future_index[i], f"lag_{lag}"] = \
            last_known["Global_active_power"].iloc[-lag]

    X_future = future_df.iloc[[i]]

    pred = xgb_model.predict(X_future)[0]

    future_predictions.append(pred)

    new_entry = pd.DataFrame(
        {
            "Global_active_power": [pred],
            "hour": [X_future["hour"].values[0]],
            "day": [X_future["day"].values[0]],
            "weekday": [X_future["weekday"].values[0]],
            "month": [X_future["month"].values[0]]
        },
        index=[future_index[i]]
    )

    last_known = pd.concat([last_known, new_entry])

future_forecast = pd.Series(future_predictions, index=future_index)

# ==========================================================
# PLOT 7 DAY FUTURE FORECAST
# ==========================================================

plt.figure(figsize=(16,6))
plt.plot(future_forecast.values, marker='o', linewidth=1)
plt.title("7-Day Future Forecast (168 Hours)")
plt.xlabel("Future Hour")
plt.ylabel("Predicted kW")
plt.show()

# ==========================================================
# DAY-WISE ERROR ON TEST SET
# Use first 7 days of 2010 test set
# Predict rolling same style and compare day1...day7
# ==========================================================

print("\n===== DAY-WISE FORECAST ERROR (FIRST 7 TEST DAYS OF 2010) =====")

rolling_history = df_xgb[:'2009'].copy()

day_preds = []
day_truth = []

test_hours = 168
test_index = X_test.index[:test_hours]

for i in range(test_hours):

    ts = test_index[i]

    row = pd.DataFrame(index=[ts])
    row["hour"] = ts.hour
    row["day"] = ts.day
    row["weekday"] = ts.weekday()
    row["month"] = ts.month

    for lag in range(1, 25):
        row[f"lag_{lag}"] = rolling_history["Global_active_power"].iloc[-lag]

    pred = xgb_model.predict(row)[0]
    true_val = y_test.loc[ts]

    day_preds.append(pred)
    day_truth.append(true_val)

    # append true value to history for realistic one-step walk-forward
    new_row = pd.DataFrame(
        {"Global_active_power": [true_val]},
        index=[ts]
    )
    rolling_history = pd.concat([rolling_history, new_row])

day_preds = np.array(day_preds)
day_truth = np.array(day_truth)

# ==========================================================
# PER DAY METRICS
# ==========================================================

for d in range(7):

    start = d * 24
    end = start + 24

    yt = day_truth[start:end]
    yp = day_preds[start:end]

    mae = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    avg = yt.mean()

    mae_pct = 100 * mae / avg
    rmse_pct = 100 * rmse / avg

    print(f"Day {d+1}:")
    print(f"   Avg Load : {avg:.4f} kW")
    print(f"   MAE      : {mae:.4f} kW")
    print(f"   RMSE     : {rmse:.4f} kW")
    print(f"   MAE %    : {mae_pct:.2f}%")
    print(f"   RMSE %   : {rmse_pct:.2f}%")

# ==========================================================
# DAYWISE PLOT
# ==========================================================

plt.figure(figsize=(16,6))
plt.plot(day_truth, label="True", linewidth=2)
plt.plot(day_preds, label="Predicted", linewidth=2)
plt.title("First 7 Days of 2010: Rolling Forecast")
plt.xlabel("Hour")
plt.ylabel("kW")
plt.legend()
plt.show()