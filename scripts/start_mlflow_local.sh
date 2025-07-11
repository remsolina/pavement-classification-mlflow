#!/bin/bash

# Create local directories for artifact storage
mkdir -p ./mlflow-artifacts
mkdir -p ./mlruns

echo "Starting MLflow Server (Local Setup)..."
docker-compose -f docker-compose.local.yml up -d

echo "MLflow Server is running at: http://localhost:5000"
echo "Artifacts will be stored in: ./mlflow-artifacts"
echo "Additional runs data in: ./mlruns"

# Wait a moment for services to start
sleep 5

# Check if services are running
echo "Checking service status..."
docker-compose -f docker-compose.local.yml ps
