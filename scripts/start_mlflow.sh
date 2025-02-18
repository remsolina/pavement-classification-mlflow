#!/bin/bash

# Fetch the public IP of the server
SERVER_IP=$(hostname -I | awk '{print $1}')

echo "Starting MLflow Server..."
sudo docker-compose up -d

echo "MLflow Server is running at: http://$SERVER_IP:5000"
