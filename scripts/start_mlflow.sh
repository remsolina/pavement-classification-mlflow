#!/bin/bash

# Fetch EC2 instance's public IP
SERVER_IP=$(curl -s ifconfig.me)

echo "Starting MLflow Server..."
sudo docker-compose up -d

echo "MLflow Server is running at: http://$SERVER_IP:5000"

# Ensure firewall allow port 5000
#sudo ufw allow 5000/tcp
#sudo ufw enable
#sudo ufw status


