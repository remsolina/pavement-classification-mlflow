#!/bin/bash

echo "Updating system..."
sudo yum update && sudo yum upgrade -y

echo "Installing Docker..."
sudo yum install docker.io -y
sudo systemctl enable docker
sudo systemctl start docker

echo "Installing AWS CLI..."
sudo yum install awscli -y

echo "Installing Docker Compose..."
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

echo "Setup complete!"
