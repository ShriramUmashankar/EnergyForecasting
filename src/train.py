# src/train.py
import os
import json
import yaml
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import mlflow
import mlflow.xgboost

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================================
# FIXED PATHS
# ==========================================================

TRAIN_PATH = "data/processed/train.csv"
PARAMS_PATH = "params.yaml"

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

TARGET_COL = "target_next_hour"
RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================================
# MLflow CONFIG
# ==========================================================

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Energy_XGBoost")

# ==========================================================
# LOAD CONFIG
# ==========================================================

with open(PARAMS_PATH, "r") as f:
    config = yaml.safe_load(f)

model_cfg = config["model"]
feat_cfg = config["features"]

LAG_START = feat_cfg["lag_start"]
LAG_END = feat_cfg["lag_end"]

# ==========================================================
# LOAD DATA
# ==========================================================

train = pd.read_csv(TRAIN_PATH, parse_dates=["timestamp"])

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

def create_features(df):
    df = df.copy()

    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["weekday"] = df["timestamp"].dt.weekday
    df["month"] = df["timestamp"].dt.month

    for lag in range(LAG_START, LAG_END + 1):
        df[f"lag_{lag}"] = df["Global_active_power"].shift(lag)

    return df

train = create_features(train)

train.dropna(inplace=True)


# ==========================================================
# SPLIT TRAIN -> TRAIN / VAL
# last month of training becomes validation
# ==========================================================

val_start_ts = train["timestamp"].max() - pd.DateOffset(months=1)

train_fit = train[train["timestamp"] < val_start_ts].copy()
val_df = train[train["timestamp"] >= val_start_ts].copy()

drop_cols = ["timestamp", TARGET_COL]

X_train = train_fit.drop(columns=drop_cols)
y_train = train_fit[TARGET_COL]

X_val = val_df.drop(columns=drop_cols)
y_val = val_df[TARGET_COL]


print("Train shape:", X_train.shape)
print("Val shape  :", X_val.shape)


feature_cols = X_train.columns.tolist()

# ==========================================================
# MODEL
# ==========================================================

model = xgb.XGBRegressor(
    n_estimators=model_cfg["n_estimators"],
    max_depth=model_cfg["max_depth"],
    learning_rate=model_cfg["learning_rate"],
    subsample=model_cfg["subsample"],
    colsample_bytree=model_cfg["colsample_bytree"],
    min_child_weight=model_cfg["min_child_weight"],
    gamma=model_cfg["gamma"],
    reg_alpha=model_cfg["reg_alpha"],
    reg_lambda=model_cfg["reg_lambda"],
    objective="reg:squarederror",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# ==========================================================
# TRAIN + LOG TO MLFLOW
# ==========================================================

with mlflow.start_run() as run:

    # Track Run ID
    run_id = run.info.run_id

    print(f"MLflow Run ID: {run_id}")
    
    # log params
    mlflow.log_param("random_state", RANDOM_STATE)
    for k, v in model_cfg.items():
        mlflow.log_param(f"model.{k}", v)
    for k, v in feat_cfg.items():
        mlflow.log_param(f"features.{k}", v)

    print("\nTraining model...")
    model.fit(X_train, y_train)

    # save local model
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH}")

    # also log model to MLflow
    mlflow.xgboost.log_model(model, name="xgb_model")

    # ======================================================
    # PREDICTIONS
    # ======================================================

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    # ======================================================
    # METRICS
    # ======================================================

    train_mae = mean_absolute_error(y_train, train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_r2 = r2_score(y_train, train_pred)

    val_mae = mean_absolute_error(y_val, val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_r2 = r2_score(y_val, val_pred)


    print("\n===== TRAIN RESULTS =====")
    print(f"RMSE : {train_rmse:.4f} kW")
    print(f"MAE  : {train_mae:.4f} kW")
    print(f"R2   : {train_r2:.4f}")

    print("\n===== VALIDATION RESULTS =====")
    print(f"RMSE : {val_rmse:.4f} kW")
    print(f"MAE  : {val_mae:.4f} kW")
    print(f"R2   : {val_r2:.4f}")


    # log metrics to MLflow
    mlflow.log_metrics(
        {
            "train_mae": train_mae,
            "train_rmse": train_rmse,
            "train_r2": train_r2,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "val_r2": val_r2
        }
    )

    # write DVC-friendly metrics file
    metrics_payload = {
        "train": {
            "mae": float(train_mae),
            "rmse": float(train_rmse),
            "r2": float(train_r2),
        },
        "val": {
            "mae": float(val_mae),
            "rmse": float(val_rmse),
            "r2": float(val_r2),
        },
        "run_id": str(run_id)
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    # ======================================================
    # PLOT 1: VALIDATION ACTUAL VS PREDICTED
    # ======================================================

    fig1 = plt.figure(figsize=(16, 6))
    plt.plot(y_val.values[:200], label="Actual")
    plt.plot(val_pred[:200], label="Predicted")
    plt.title("Validation: Actual vs Predicted (First 200 Hours)")
    plt.xlabel("Hour")
    plt.ylabel("kW")
    plt.legend()
    plt.tight_layout()
    #plt.show()

    mlflow.log_figure(fig1, "plots/validation_actual_vs_predicted.png")
    plt.close(fig1)


    # ======================================================
    # PLOT 2: VALIDATION AUTOREGRESSIVE 168-HOUR FORECAST
    # ======================================================

    future_hours = min(168, len(val_df))
    history = train_fit["Global_active_power"].iloc[-24:].tolist()

    preds_168 = []
    truth_168 = val_df["Global_active_power"].iloc[:future_hours].values
    timestamps_168 = val_df["timestamp"].iloc[:future_hours].values

    for i in range(future_hours):
        ts = val_df["timestamp"].iloc[i]

        if i == 0:
            base_row = train_fit.iloc[-1].copy()
        else:
            base_row = val_df.iloc[i - 1].copy()

        row = {}

        row["Global_active_power"] = history[-1]
        row["Global_reactive_power"] = base_row["Global_reactive_power"]
        row["Voltage"] = base_row["Voltage"]
        row["Global_intensity"] = base_row["Global_intensity"]
        row["Sub_metering_1"] = base_row["Sub_metering_1"]
        row["Sub_metering_2"] = base_row["Sub_metering_2"]
        row["Sub_metering_3"] = base_row["Sub_metering_3"]

        row["hour"] = ts.hour
        row["day"] = ts.day
        row["weekday"] = ts.weekday()
        row["month"] = ts.month

        for lag in range(LAG_START, LAG_END + 1):
            row[f"lag_{lag}"] = history[-lag]

        X_future = pd.DataFrame([row])[feature_cols]
        pred_val = model.predict(X_future)[0]

        preds_168.append(pred_val)
        history.append(pred_val)
        history = history[-24:]

    val_168_mae = mean_absolute_error(truth_168, preds_168)
    val_168_rmse = np.sqrt(mean_squared_error(truth_168, preds_168))

    print("\n===== VALIDATION AUTOREGRESSIVE 168H WINDOW =====")
    print(f"MAE  : {val_168_mae:.4f}")
    print(f"RMSE : {val_168_rmse:.4f}")

    mlflow.log_metrics(
        {
            "val_168_mae": val_168_mae,
            "val_168_rmse": val_168_rmse,
        }
    )

    fig3 = plt.figure(figsize=(18, 6))
    plt.plot(
        timestamps_168,
        truth_168,
        linewidth=2,
        label="Ground Truth"
    )
    plt.plot(
        timestamps_168,
        preds_168,
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="168h Recursive Forecast"
    )
    plt.title("Validation: Autoregressive 7-Day Forecast vs Ground Truth")
    plt.xlabel("Timestamp")
    plt.ylabel("kW")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    #plt.show()

    mlflow.log_figure(fig3, "plots/validation_recursive_168h.png")
    plt.close(fig3)

print("\nTraining complete. MLflow run logged.")
print(f"DVC metrics file written to: {METRICS_PATH}")