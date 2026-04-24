#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "INITIATING HARD RESET..."

echo "Stopping Docker containers and wiping volumes..."
docker compose down -v

echo " Deleting DVC cache and lock files..."
# Wipes DVC's memory and stale locks
rm -rf .dvc/cache
rm -rf .dvc/tmp
rm -f dvc.lock

echo " Deleting airflow related files"
rm -rf .env


echo " Deleting MLflow artifacts..."
# Wipes the physical models saved by MLflow
rm -rf mlartifacts

echo " Clean up successful!"
echo " Run './bootstrap.sh' to rebuild the environment from scratch."