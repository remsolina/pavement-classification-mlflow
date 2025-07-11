#!/bin/bash

echo "🛑 Stopping simple MLflow server..."

if [ -f mlflow.pid ]; then
    PID=$(cat mlflow.pid)
    if kill $PID 2>/dev/null; then
        echo "✅ MLflow server (PID: $PID) stopped successfully"
    else
        echo "⚠️  Process $PID not found or already stopped"
    fi
    rm -f mlflow.pid
else
    echo "⚠️  No PID file found. Trying to find and kill MLflow processes..."
    pkill -f "mlflow server"
fi

echo "🧹 MLflow server stopped"
