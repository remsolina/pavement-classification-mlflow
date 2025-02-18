#!/bin/bash

echo "Starting MLflow Server..."
docker-compose up -d

echo "MLflow Server is running! Access it at: http://your-server-ip:5000"
