import io
import time
from collections import deque
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import mlflow
import mlflow.pyfunc
from pydantic import BaseModel

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge, Counter, Histogram

class PredictRequest(BaseModel):
    timestamp: str
    Global_active_power: float
    Global_reactive_power: float
    Voltage: float
    Global_intensity: float
    Sub_metering_1: float
    Sub_metering_2: float
    Sub_metering_3: float

# ======================================================
# CONFIG
# ======================================================

mlflow.set_tracking_uri("http://mlflow:5000")

MODEL_URI = "models:/EnergyForecastModel/Production"
TRAIN_PATH = "data/processed/train.csv"
LIVE_PATH = "data/processed/live_data.csv"
TARGET_COL = "Global_active_power"

RAW_FEATURES = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]


app = FastAPI(title="Energy Forecast API")

# Prometheus Metrics Definitions
PROMETHEUS_RMSE = Gauge("model_rmse", "Root Mean Square Error of the model")
PROMETHEUS_MAE = Gauge("model_mae", "Mean Absolute Error of the model")
PROMETHEUS_PREDS = Counter("model_predictions_total", "Total predictions made")
PROMETHEUS_INFERENCE_TIME = Histogram("model_inference_time_seconds", "Time taken for a single next-hour prediction")
PROMETHEUS_FORECAST_TIME = Histogram("model_forecast_time_seconds", "Time taken for the 168-hour recursive forecast")

# Automatically exposes /metrics endpoint and tracks API latency/throughput
Instrumentator().instrument(app).expose(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
history_buffer = deque(maxlen=24)
pending_preds = deque()

error_count = 0
sse = 0.0
sae = 0.0

live_points = deque(maxlen=500)

forecast_cache = {
    "timestamps": [],
    "values": []
}



def load_model():
    global model
    model = mlflow.pyfunc.load_model(MODEL_URI)
    run_id = model.metadata.run_id if hasattr(model, 'metadata') else "Unknown"
    print(f" Model loaded successfully! (Run ID: {run_id})")

def bootstrap_history():
    df_train = pd.read_csv(TRAIN_PATH)
    
    if os.path.exists(LIVE_PATH) and os.path.getsize(LIVE_PATH) > 0:
        df_live = pd.read_csv(LIVE_PATH)
        df_combined = pd.concat([df_train, df_live], ignore_index=True)
    else:
        df_combined = df_train
    vals = df_combined[TARGET_COL].tail(24).tolist()
    
    for v in vals:
        history_buffer.append(float(v))

    # Grab the very last row to print the most recent state
    last_row = df_combined.iloc[-1]
    last_time = last_row.get("timestamp", "Unknown timestamp")
    last_val = last_row.get(TARGET_COL, "Unknown value")
    
    print(f" Bootstrapping complete! Most recent historical point -> Time: {last_time} | {TARGET_COL}: {last_val}")    
        
# Features

def build_feature_row(row):
    ts = pd.to_datetime(row["timestamp"])
    feat = {}

    for c in RAW_FEATURES:
        feat[c] = float(row[c])

    feat["hour"] = ts.hour
    feat["day"] = ts.day
    feat["weekday"] = ts.weekday()
    feat["month"] = ts.month

    hist = list(history_buffer)
    for i in range(1, 25):
        feat[f"lag_{i}"] = float(hist[-i])

    return pd.DataFrame([feat])

# FORECAST

def recursive_168_forecast(row):
    start_time = time.time()
    
    local_hist = deque(list(history_buffer), maxlen=24)
    ts = pd.to_datetime(row["timestamp"])
    preds = []

    for step in range(168):
        future_ts = ts + pd.Timedelta(hours=step + 1)
        feat = {}

        for c in RAW_FEATURES:
            feat[c] = float(row[c])

        feat["hour"] = future_ts.hour
        feat["day"] = future_ts.day
        feat["weekday"] = future_ts.weekday()
        feat["month"] = future_ts.month

        for i in range(1, 25):
            feat[f"lag_{i}"] = float(local_hist[-i])

        X = pd.DataFrame([feat])
        pred = float(model.predict(X)[0])

        preds.append(pred)
        local_hist.append(pred)

    idx = pd.date_range(
        start=ts + pd.Timedelta(hours=1),
        periods=168,
        freq="h"
    )

    PROMETHEUS_FORECAST_TIME.observe(time.time() - start_time)
    
    return pd.Series(preds, index=idx)

# METRICS

def update_metrics(actual, ts):
    global error_count, sse, sae

    rmse = 0.0
    mae = 0.0
    matched_pred = None

    for item in live_points:
        if item["timestamp"] == str(ts) and item["actual"] is None:
            item["actual"] = float(actual)
            matched_pred = float(item["predicted"])
            break

    if matched_pred is not None:
        err = matched_pred - actual
        error_count += 1
        sse += err ** 2
        sae += abs(err)

        rmse = float(np.sqrt(sse / error_count))
        mae = float(sae / error_count)

    return rmse, mae

# ENDPOINTS

@app.get("/health")
def health():
    return {"status": "up"}

@app.get("/stats")
def stats():
    rmse = np.sqrt(sse / error_count) if error_count else 0
    mae = sae / error_count if error_count else 0

    return {
        "status": "up",
        "samples": error_count,
        "rmse": rmse,
        "mae": mae,
        "live_points": list(live_points),
        "forecast_168h": forecast_cache
    }

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    if len(df) < 25:
        return {"error": "Need at least 25 rows"}

    local_hist = deque(maxlen=24)

    for _, r in df.iloc[:24].iterrows():
        local_hist.append(float(r[TARGET_COL]))

    for _, r in df.iloc[24:].iterrows():
        local_hist.append(float(r[TARGET_COL]))
        last_row = r.to_dict()

    old_hist = deque(history_buffer, maxlen=24)
    history_buffer.clear()
    for x in local_hist:
        history_buffer.append(x)

    fc = recursive_168_forecast(last_row)

    forecast_cache["timestamps"] = [str(x) for x in fc.index]
    forecast_cache["values"] = [float(v) for v in fc.values]

    history_buffer.clear()
    for x in old_hist:
        history_buffer.append(x)

    return {
        "message": "Forecast created",
        "plot_url": "/plot/week-forecast"
    }

@app.post("/predict")
async def predict(payload: PredictRequest):
    row = payload.dict()
    ts = pd.to_datetime(row["timestamp"])
    actual = float(row["Global_active_power"])

    rmse, mae = update_metrics(actual, ts)

    PROMETHEUS_RMSE.set(rmse)
    PROMETHEUS_MAE.set(mae)
    PROMETHEUS_PREDS.inc()

    X = build_feature_row(row)

    start_time = time.time()
    pred_next = float(model.predict(X)[0])
    PROMETHEUS_INFERENCE_TIME.observe(time.time() - start_time)

    future_ts = ts + pd.Timedelta(hours=1)

    live_points.append({
        "timestamp": str(future_ts),
        "predicted": float(pred_next),
        "actual": None
    })

    history_buffer.append(actual)

    fc = recursive_168_forecast(row)

    forecast_cache["timestamps"] = [str(x) for x in fc.index]
    forecast_cache["values"] = [float(v) for v in fc.values]

    return {
        "timestamp": str(ts),
        "actual": actual,
        "next_hour_prediction": pred_next,
        "overall_rmse": rmse,
        "overall_mae": mae,
        "plot_url": "/plot/live",
        "forecast_plot_url": "/plot/week-forecast"
    }

@app.get("/plot/week-forecast", response_class=HTMLResponse)
def plot_upload():
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=forecast_cache["timestamps"],
        y=forecast_cache["values"],
        mode="lines+markers",
        name="7 Day Forecast"
    ))

    fig.update_layout(
        title="168 Hour Forecast",
        xaxis_title="Timestamp",
        yaxis_title="kW"
    )

    return fig.to_html(full_html=True)

@app.get("/plot/live", response_class=HTMLResponse)
def plot_live():
    if len(live_points) == 0:
        return "<h2>No prediction data yet</h2>"

    df = pd.DataFrame(list(live_points))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["predicted"],
        mode="lines+markers",
        name="Predicted"
    ))

    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["actual"],
        mode="lines+markers",
        name="Actual"
    ))

    fig.update_layout(
        title="Next Hour Prediction vs Actual",
        xaxis_title="Timestamp",
        yaxis_title="kW"
    )

    return fig.to_html(full_html=True)

@app.post("/reload")
def reload_system():
    try:
        load_model()
        bootstrap_history()
        return {"message": "Success: Backend reloaded with latest model and data."}
    except Exception as e:
        return {"error": str(e)}

# STARTUP

@app.on_event("startup")
def startup():
    load_model()
    bootstrap_history()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 