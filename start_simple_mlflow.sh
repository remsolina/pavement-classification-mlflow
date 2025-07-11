#!/bin/bash

echo "🧹 Cleaning up directories..."
rm -rf ./mlflow-artifacts ./mlruns

echo "📁 Creating fresh directories..."
mkdir -p ./mlflow-artifacts
mkdir -p ./mlruns

echo "🚀 Starting simple MLflow server locally..."
echo "This will run MLflow directly on your machine (no Docker)"

# Start MLflow server with file-based backend and local artifacts
mlflow server \
    --backend-store-uri ./mlruns \
    --default-artifact-root ./mlflow-artifacts \
    --host 127.0.0.1 \
    --port 5005 &

# Store the process ID
MLFLOW_PID=$!
echo $MLFLOW_PID > mlflow.pid

echo "✅ MLflow server started with PID: $MLFLOW_PID"
echo "🌐 Access MLflow UI at: http://localhost:5005"
echo "📦 Artifacts stored in: ./mlflow-artifacts"
echo "📊 Runs stored in: ./mlruns"
echo ""
echo "To stop the server, run: kill $MLFLOW_PID"
echo "Or use: ./stop_simple_mlflow.sh"

# Wait a moment for server to start
sleep 3

# Test if server is running
if curl -s http://localhost:5005/health > /dev/null; then
    echo "🎉 MLflow server is healthy and ready!"
else
    echo "⚠️  MLflow server may still be starting..."
fi
