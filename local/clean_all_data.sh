#!/bin/bash

echo "🗑️  COMPLETE DATA CLEANUP"
echo "========================"
echo "⚠️  WARNING: This will permanently delete ALL MLflow data!"
echo "   - All experiments and runs"
echo "   - All model artifacts"
echo "   - All training history"
echo "   - MySQL database data"
echo ""

read -p "Are you sure you want to delete ALL data? (type 'yes' to confirm): " confirm

if [ "$confirm" != "yes" ]; then
    echo "❌ Cleanup cancelled"
    exit 0
fi

echo "🛑 Stopping containers..."
docker-compose -f docker-compose.local.yml down

echo "🗑️  Removing Docker volumes (database data)..."
docker-compose -f docker-compose.local.yml down -v

echo "🗑️  Removing local directories..."
rm -rf ../mlflow-artifacts
rm -rf ../mlruns

echo "🧹 Removing any orphaned containers..."
docker system prune -f

echo "✅ Complete cleanup finished!"
echo ""
echo "📋 To start fresh:"
echo "  ./start_mlflow_local.sh"
echo "  python model_7_local.py"
