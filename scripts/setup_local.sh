#!/bin/bash

echo "Setting up MLflow for local development..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is not installed. Please install Docker Compose first."
    echo "Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

# Create necessary directories
echo "Creating local directories..."
mkdir -p ./mlflow-artifacts
mkdir -p ./mlruns

# Make scripts executable
chmod +x scripts/start_mlflow_local.sh
chmod +x scripts/stop_mlflow_local.sh

echo "✅ Local setup complete!"
echo "Run './scripts/start_mlflow_local.sh' to start MLflow server"
