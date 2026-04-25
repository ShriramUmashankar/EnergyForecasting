#!/bin/bash

# 1. Generate a permanent Fernet Key if one doesn't exist
if [ ! -f .env ] || ! grep -q "AIRFLOW__CORE__FERNET_KEY" .env; then
    echo "🔑 Generating secure Fernet Key..."
    # We use standard Python libraries so it works on any machine without installing extra packages
    FERNET_KEY=$(python3 -c "import base64, os; print(base64.urlsafe_b64encode(os.urandom(32)).decode())")
    
    # Save it to the .env file
    echo "AIRFLOW__CORE__FERNET_KEY=$FERNET_KEY" >> .env
    echo "✅ Fernet key saved to .env"
else
    echo "✅ Fernet key already exists."
fi

if ! grep -q "HOST_UID" .env; then
    echo "👤 Detecting Host User ID..."
    # 'id -u' dynamically grabs the ID of the person running the script
    CURRENT_UID=$(id -u)
    echo "HOST_UID=$CURRENT_UID" >> .env
    echo "✅ Host UID ($CURRENT_UID) saved to .env"
fi

echo "🚀 Starting Energy Forecasting MLOps Setup..."

# 1. Create the necessary folders safely
echo "📁 Ensuring volume directories exist..."
mkdir -p mlartifacts
mkdir -p data/processed
mkdir -p model

# echo "🔓 Setting permissions so Docker can write to your files..."

# sudo chmod -R 777 mlartifacts
# sudo chmod -R 777 data
# sudo chmod -R 777 model
# sudo chmod -R 777 .dvc
# sudo chmod -R 777 dags

# # Only chmod the lock file IF it already exists (do not touch/create it!)
# if [ -f dvc.lock ]; then
#     sudo chmod 777 dvc.lock
# fi

# 2. Build the isolated images
echo "🐳 Building Docker images..."
docker compose build

# 3. Start ONLY the Database and MLflow first
echo "🗄️ Starting Postgres and MLflow..."
docker compose up -d postgres mlflow

# Wait for Postgres to initialize the airflow and mlflow databases
echo "⏳ Waiting 10 seconds for the database to initialize..."
sleep 10

# 4. Run the initial training pipeline INSIDE the Airflow container
echo "🧠 Running initial training pipeline (DVC + MLflow)..."
docker compose run --rm airflow bash -c "
    cd /opt/airflow/project && \
    # --- ADD THESE TWO LINES ---
    git config --global user.email 'mlops-bot@example.com' && \
    git config --global user.name 'MLOps Bot' && \
    # ---------------------------
    python3 -m dvc repro -f && \
    export MLFLOW_TRACKING_URI=http://mlflow:5000 && \
    python3 src/hyperparameter_sweep.py && \
    python3 src/register_model.py
"


# 5. Start the rest of the stack (FastAPI, Streamlit, Airflow UI)
echo "🌐 Starting the full application stack..."
docker compose up -d

echo "✅ Setup Complete! The system is fully online."
echo "-------------------------------------------------"
echo "📊 Streamlit Frontend: http://localhost:8501"
echo "⚙️  Airflow UI:       http://localhost:8080 (admin/admin)"
echo "🧪 MLflow Tracking:  http://localhost:5000"
echo "-------------------------------------------------"