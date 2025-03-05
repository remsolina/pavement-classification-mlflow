#!/bin/bash

echo "Updating system..."
sudo yum update -y

echo "Installing Docker..."
sudo amazon-linux-extras install docker -y
sudo systemctl enable docker
sudo systemctl start docker

# (Optional) Add the ec2-user to the docker group to run docker commands without sudo.
sudo usermod -a -G docker ec2-user

echo "Installing AWS CLI..."
sudo yum install awscli -y

echo "Installing Docker Compose..."
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

echo "Setup complete!"
