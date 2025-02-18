#!/bin/bash

# Fetch EC2 instance's public IP
SERVER_IP=$(curl -s ifconfig.me)

echo "Starting MLflow Server..."
sudo docker-compose up -d

echo "MLflow Server is running at: http://$SERVER_IP:5000"