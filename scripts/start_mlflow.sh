#!/bin/bash

# Fetch EC2 instance's public IP
SERVER_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)

echo "Starting MLflow Server..."
docker-compose up -d

echo "MLflow Server is running at: http://$SERVER_IP:5000"