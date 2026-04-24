import os
import pandas as pd
import requests
from airflow import DAG
from datetime import datetime, timedelta

# AIRFLOW 3.0 UPDATED IMPORTS
from airflow.providers.standard.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.models import Variable  # Airflow 3 SDK for task-safe variables

PROJECT_ROOT = "/opt/airflow/project"
API_PREDICT_URL = "http://fastapi-backend:8000/predict"
API_METRICS_URL = "http://fastapi-backend:8000/metrics"

# Set threshold to 4 for your testing
RMSE_THRESHOLD = 0.5
ROW_DELTA_THRESHOLD = 3

default_args = {
    'owner': 'admin',
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

def pop_and_predict():
    # Paths based on your setup
    test_path = os.path.join(PROJECT_ROOT, "data/processed/test.csv")
    live_path = os.path.join(PROJECT_ROOT, "data/processed/live_data.csv")

    # 1. Read test data
    df_test = pd.read_csv(test_path)
    if len(df_test) == 0:
        raise ValueError("test.csv is empty! Simulation complete.")

    # 2. Extract the first row
    top_row = df_test.iloc[[0]]
    df_remaining = df_test.iloc[1:]

    # 3. Physically update the files (The 'Pop' logic)
    df_remaining.to_csv(test_path, index=False)
    
    # Append to live_data (handles headers automatically)
    file_exists = os.path.exists(live_path) and os.path.getsize(live_path) > 0
    top_row.to_csv(live_path, mode='a', header=not file_exists, index=False)

    # 4. Trigger Prediction API
    payload = top_row.to_dict(orient='records')[0]
    payload['timestamp'] = str(payload['timestamp']) 
    
    headers = {'Content-Type': 'application/json'}
    response = requests.post(API_PREDICT_URL, json=payload, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"API Error: {response.text}")

def check_retrain_conditions(**kwargs):
    train_path = os.path.join(PROJECT_ROOT, "data/processed/live_data.csv")
    
    with open(train_path, 'r') as f:
        current_count = sum(1 for row in f) - 1 

    last_count = int(Variable.get("last_trained_row_count", default_var=current_count))
    action = Variable.get("retrain_action", default_var="ready").strip().lower()

    try:
        res = requests.get(API_METRICS_URL).json()
        current_rmse = float(res.get("rmse", 0.0))
    except Exception:
        current_rmse = 0.0

    print(f"Current RMSE: {current_rmse} | Rows: {current_count} | Last Trained: {last_count}")

    if current_rmse > RMSE_THRESHOLD or (current_count - last_count) >= ROW_DELTA_THRESHOLD:
        
        if action == "rejected":
            print("Retrain is currently manually rejected by human. Ignoring triggers.")
            return 'skip_retrain'

        # Instead of querying the database for active runs, we check the variable!
        if action == "pending":
            print("Retrain pipeline is already active (pending human approval). Skipping.")
            return 'skip_retrain'

        Variable.set("retrain_action", "pending")
        Variable.set("last_trained_row_count", str(current_count))
        return 'trigger_retrain'
    
    return 'skip_retrain'

with DAG(
    dag_id='hourly_ingestion_and_monitor',
    default_args=default_args,
    schedule= None,
    start_date=datetime(2023, 1, 1),
    catchup=False
) as dag:

    task_pop_predict = PythonOperator(
        task_id='pop_and_predict',
        python_callable=pop_and_predict
    )

    task_check_conditions = BranchPythonOperator(
        task_id='check_retrain_conditions',
        python_callable=check_retrain_conditions
    )

    task_trigger_retrain = TriggerDagRunOperator(
        task_id='trigger_retrain',
        trigger_dag_id='model_retrain_pipeline' 
    )

    task_skip = EmptyOperator(task_id='skip_retrain')

    task_pop_predict >> task_check_conditions
    task_check_conditions >> [task_trigger_retrain, task_skip]