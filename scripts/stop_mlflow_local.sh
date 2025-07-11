#!/bin/bash

echo "Stopping MLflow Server (Local Setup)..."
docker-compose -f docker-compose.local.yml down

echo "MLflow Server stopped."
echo "Note: Your data is preserved in:"
echo "  - MySQL data: Docker volume 'mysql_data_local'"
echo "  - Artifacts: ./mlflow-artifacts directory"
echo "  - Runs: ./mlruns directory"
