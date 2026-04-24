from airflow import DAG


# AIRFLOW 3.0 UPDATED IMPORTS
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.sensors.python import PythonSensor
from airflow.exceptions import AirflowSkipException
from airflow.models import Variable
from datetime import datetime, timedelta
import requests


PROJECT_ROOT = "/opt/airflow/project"

default_args = {
    'owner': 'admin',
    'retries': 0,
}

MAIN_FOLDER_TRAIN_SCRIPT = f"""
set -e # This tells bash to stop immediately if any command fails
cd {PROJECT_ROOT}

echo "setting git config mail"
git config --global user.email 'mlops-bot@example.com'
git config --global user.name 'MLOps Bot'

echo "Updating DVC hashes for current data state..."
dvc add data/processed/live_data.csv data/processed/test.csv

echo "Stage the .dvc files so the Hash Filter can read the new MD5..."
git add data/processed/live_data.csv.dvc data/processed/test.csv.dvc    

echo " Running Hyperparameter Sweep..."
export MLFLOW_TRACKING_URI=http://mlflow:5000
python3 -u src/hyperparameter_sweep.py

echo " Promoting Best Model..."
python3 -u src/register_model.py
"""

def ping_backend():
    # Since they are in the same docker-compose network, Airflow can just call it by name!
    response = requests.post("http://fastapi-backend:8000/reload")
    response.raise_for_status()
    print("Backend reloaded successfully!")
 
def await_human_approval():
    action = Variable.get("retrain_action", default_var="ready").strip().lower()
    
    if action == "approved":
        return True  # Sensor succeeds, moves to next task
    elif action == "rejected":
        # Gracefully skips the rest of the DAG
        raise AirflowSkipException("Human explicitly rejected the retraining.")
    
    return False # Keeps polling if "pending"

def reset_approval_state():
    # Resets to ready so the next hourly trigger isn't auto-approved
    Variable.set("retrain_action", "ready")
    print("Action variable reset to 'ready'")

with DAG(
    dag_id='model_retrain_pipeline',
    default_args=default_args,
    schedule=None, 
    start_date=datetime(2023, 1, 1),
    catchup=False
) as dag:

    # 1. UI-Based Human Sensor
    wait_for_human = PythonSensor(
        task_id='wait_for_human_approval',
        python_callable=await_human_approval,
        poke_interval=30, 
        timeout=60 * 60 * 24 * 3, 
        mode='reschedule' 
    )

    # 2. Reset the Variable immediately upon approval
    reset_state = PythonOperator(
        task_id='reset_state_to_ready',
        python_callable=reset_approval_state
    )

    # 3. Isolated Training Pipeline
    run_isolated_pipeline = BashOperator(
        task_id='run_dvc_pipeline_and_promote',
        bash_command=MAIN_FOLDER_TRAIN_SCRIPT
    )

    reload_backend = PythonOperator(
        task_id='reload_fastapi_backend',
        python_callable=ping_backend,
    )

    wait_for_human >> reset_state >> run_isolated_pipeline >> reload_backend