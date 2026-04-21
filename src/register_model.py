import subprocess
import csv
import io
import json
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "EnergyForecastModel"

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

# ==========================================================
# STEP 1: Fetch DVC experiments in CSV format
# ==========================================================
print("Fetching DVC experiments via CSV...")
result = subprocess.run(
    ["dvc", "exp", "show", "--csv"],
    capture_output=True,
    text=True,
    check=True
)

# Parse the CSV output just like a standard spreadsheet
reader = csv.DictReader(io.StringIO(result.stdout))

# ==========================================================
# STEP 2: Find best globally based on table columns
# ==========================================================
best_exp = None
best_rmse = float("inf")

for row in reader:
    # Get the experiment name (e.g., 'sharp-oast') or fallback to 'rev' hash
    exp_name = row.get("Experiment")
    rev = row.get("rev", "")
    
    # Skip the workspace (we can't 'dvc exp apply' the workspace to itself)
    if exp_name == "workspace" or rev == "workspace":
        continue

    # Search the row keys for our target metric. 
    # Depending on the DVC version, the column header might be 'val.rmse' 
    # or 'model/metrics.json:val.rmse'. This safely catches both.
    rmse = None
    for key, value in row.items():
        if key and "val.rmse" in key and value.strip() != "":
            try:
                rmse = float(value)
                break
            except ValueError:
                continue
                
    # Evaluate if this is the best one
    if rmse is not None and rmse < best_rmse:
        best_rmse = rmse
        best_exp = exp_name if exp_name else rev

if best_exp is None:
    raise Exception("No valid experiment with a 'val.rmse' value found in the table.")

print(f"Best Experiment : {best_exp}")
print(f"Best Val RMSE   : {best_rmse}")

# ==========================================================
# STEP 3: Apply best params locally
# ==========================================================
print(f"Applying best experiment: {best_exp}...")
subprocess.run(["dvc", "exp", "apply", best_exp], check=True)

with open("model/metrics.json", "r") as f:
    payload = json.load(f)

run_id = payload.get("run_id")
if not run_id:
    raise Exception("Applied experiment does not contain an mlflow run_id.")

print(f"Winning MLflow Run ID: {run_id}")


model_uri = f"runs:/{run_id}/xgb_model"

registered = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME
)

version = registered.version
print(f"Registered Version: {version}")

# ==========================================================
# STEP 6: Promote to Production
# ==========================================================
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=version,
    stage="Production",
    archive_existing_versions=True
)

print(f"Model '{MODEL_NAME}' version {version} promoted to Production.")