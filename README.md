# Energy Consumption Forecasting System

This project provides an end-to-end Machine Learning Operations (MLOps) pipeline for forecasting household power consumption. It integrates automated data versioning, model training, and hyperparameter tuning alongside a scalable model serving API and frontend. The entire ecosystem is fully containerized and includes a comprehensive monitoring stack to track infrastructure health, prediction latency, and model drift in real-time.

## Prerequisites

Before setting up the project, ensure you have the following installed on your host system:
* Docker and Docker Compose
* DVC (Data Version Control)

## Setup Instructions

### 1. Data Preparation
You need to download the raw dataset and place it in the correct directory before initializing the pipeline.
1. Download the dataset from here: [https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption]
2. Extract the file and place `household_power_consumption.txt` inside the `data/raw_data/` folder.

### 2. Configure Alerts
The monitoring stack includes an automated alerting system. To receive email notifications for system downtime or high model error rates (RMSE > 0.5) or high CPU usage etc:
* Open `monitoring/alertmanager.yml`.
* Update the SMTP configuration section with your mail server username and password.

### 3. Port Allocation
Ensure no other Docker containers are running on your system to prevent network conflicts. The application stack requires the following ports to be free. 

| Service | Port | Localhost Link |
| :--- | :--- | :--- |
| Apache Airflow (DAGs) | 8080 | [http://localhost:8080](http://localhost:8080) |
| FastAPI Backend | 8000 | [http://localhost:8000](http://localhost:8000) |
| Streamlit Frontend | 8501 | [http://localhost:8501](http://localhost:8501) |
| MLflow (Tracking) | 5000 | [http://localhost:5000](http://localhost:5000) |
| Grafana (Dashboards) | 3000 | [http://localhost:3000](http://localhost:3000) |
| Prometheus (Metrics) | 9090 | [http://localhost:9090](http://localhost:9090) |
| Alert Manager | 9093 | [http://localhost:9093](http://localhost:9093) |
| Node Exporter | 9100 | [http://localhost:9100](http://localhost:9100) |

**Handling Port Conflicts:**
If any of these ports are already in use on your host machine, you can change the routing in the `docker-compose.yml` file. Look for the `ports` section under the conflicting service, which follows the format `xxxx:yyyy` (External Host Port : Internal Container Port). 
Change the `xxxx` value to a free port on your machine (e.g., change `"8000:8000"` to `"8081:8000"`).

## Running the Project

Once the data is in place and configuration is set, start the pipeline by running the bootstrap script:

```bash
bash bootstrap.sh
```

This script will automatically:
1. Fix local directory permissions to ensure smooth Docker volume mounting.
2. Build all required isolated Docker images.
3. Trigger the initial DVC data pipeline and MLflow model training.
4. Start the full application and monitoring stack.
5. Generate a `credentials_airflow.json` file in your root directory containing the secure login credentials for the Airflow UI.

## Resetting the Environment

If you need to completely tear down the application, wipe all model history, clear database volumes, and start from a clean slate, run the reset script:

```bash
bash reset.sh
```
*Note: This will delete all tracked experiments in MLflow, historical data in Prometheus, and custom Grafana configurations not saved to the provisioning folder.*