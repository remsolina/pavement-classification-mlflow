#!/usr/bin/env python3
"""
🐳 Containerized MLflow Training Runner

This script manages containerized training workflows with MLflow integration.
It provides a clean Python interface for building, running, and managing
training containers with proper MLflow server connectivity.

Usage:
    python run_training.py train [script_name]
    python run_training.py build
    python run_training.py cleanup
    python run_training.py --help

Examples:
    python run_training.py train                    # Run default training script
    python run_training.py train model_7_local.py   # Run specific training script
    python run_training.py build                    # Just build the image
    python run_training.py cleanup                  # Clean up containers/images
"""

import argparse
import subprocess
import sys
import os
import requests
from pathlib import Path
from typing import Optional, List
import time


class Colors:
    """ANSI color codes for terminal output."""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    BOLD = '\033[1m'
    NC = '\033[0m'  # No Color


class TrainingRunner:
    """Manages containerized MLflow training workflows."""
    
    def __init__(self):
        self.mlflow_url = "http://localhost:5005"
        self.compose_file = "docker-compose.local.yml"
        self.default_script = "model_7_local.py"
        
    def print_status(self, message: str) -> None:
        """Print status message in blue."""
        print(f"{Colors.BLUE}ℹ️  {message}{Colors.NC}")
        
    def print_success(self, message: str) -> None:
        """Print success message in green."""
        print(f"{Colors.GREEN}✅ {message}{Colors.NC}")
        
    def print_warning(self, message: str) -> None:
        """Print warning message in yellow."""
        print(f"{Colors.YELLOW}⚠️  {message}{Colors.NC}")
        
    def print_error(self, message: str) -> None:
        """Print error message in red."""
        print(f"{Colors.RED}❌ {message}{Colors.NC}")
        
    def print_header(self) -> None:
        """Print the application header."""
        print(f"{Colors.BOLD}🐳 MLflow Containerized Training{Colors.NC}")
        print("=" * 40)
        
    def check_mlflow_server(self) -> bool:
        """
        Check if MLflow server is running and accessible.
        
        Returns:
            bool: True if server is running, False otherwise
        """
        self.print_status("Checking if MLflow server is running...")
        
        try:
            response = requests.get(self.mlflow_url, timeout=5)
            if response.status_code == 200:
                self.print_success(f"MLflow server is running at {self.mlflow_url}")
                return True
        except requests.exceptions.RequestException:
            pass
            
        self.print_error("MLflow server is not running!")
        print("Please start the MLflow server first:")
        print("  ./start_mlflow_local.sh")
        return False
        
    def run_command(self, command: List[str], description: str) -> bool:
        """
        Run a shell command and handle errors.
        
        Args:
            command: List of command parts
            description: Description for error messages
            
        Returns:
            bool: True if command succeeded, False otherwise
        """
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=False,
                text=True
            )
            return True
        except subprocess.CalledProcessError as e:
            self.print_error(f"Failed to {description}")
            return False
        except FileNotFoundError:
            self.print_error(f"Command not found: {command[0]}")
            self.print_error("Please ensure Docker and docker-compose are installed")
            return False
            
    def build_training_image(self) -> bool:
        """
        Build the training container image.
        
        Returns:
            bool: True if build succeeded, False otherwise
        """
        self.print_status("Building training container image...")
        
        command = ["docker-compose", "-f", self.compose_file, "build", "training"]
        
        if self.run_command(command, "build training image"):
            self.print_success("Training image built successfully")
            return True
        return False
        
    def create_directories(self) -> None:
        """Create necessary shared directories."""
        directories = ["../data", "../mlflow-artifacts"]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
    def run_training(self, script_name: Optional[str] = None) -> bool:
        """
        Run training in a container.
        
        Args:
            script_name: Name of the training script to run
            
        Returns:
            bool: True if training succeeded, False otherwise
        """
        script_name = script_name or self.default_script
        
        self.print_status(f"Starting containerized training with script: {script_name}")
        self.print_status("This will run in the same Docker network as MLflow server")
        
        # Create shared directories
        self.create_directories()
        
        # Run training container
        command = [
            "docker-compose", "-f", self.compose_file,
            "run", "--rm", "training",
            "python", script_name
        ]
        
        if self.run_command(command, "run training"):
            self.print_success("Training completed successfully!")
            self.print_success(f"View results at: {self.mlflow_url}")
            return True
        return False
        
    def cleanup(self) -> bool:
        """
        Clean up training containers and images.
        
        Returns:
            bool: True if cleanup succeeded, False otherwise
        """
        self.print_status("Cleaning up training containers and images...")
        
        # Remove training containers
        subprocess.run(
            ["docker-compose", "-f", self.compose_file, "down", "training"],
            capture_output=True
        )
        
        # Remove training image
        subprocess.run(
            ["docker", "rmi", "local_training"],
            capture_output=True
        )
        
        self.print_success("Cleanup completed")
        return True
        
    def train_workflow(self, script_name: Optional[str] = None) -> bool:
        """
        Complete training workflow: check server, build, and run.
        
        Args:
            script_name: Name of the training script to run
            
        Returns:
            bool: True if entire workflow succeeded, False otherwise
        """
        if not self.check_mlflow_server():
            return False
            
        if not self.build_training_image():
            return False
            
        return self.run_training(script_name)


def main():
    """Main entry point for the training runner."""
    parser = argparse.ArgumentParser(
        description="🐳 Containerized MLflow Training Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_training.py train                    # Run default training script
  python run_training.py train model_7_local.py   # Run specific training script
  python run_training.py build                    # Just build the image
  python run_training.py cleanup                  # Clean up containers/images

Prerequisites:
  - MLflow server must be running (./start_mlflow_local.sh)
  - Docker and docker-compose must be installed
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Run training workflow')
    train_parser.add_argument(
        'script',
        nargs='?',
        default='model_7_local.py',
        help='Training script to run (default: model_7_local.py)'
    )
    
    # Build command
    subparsers.add_parser('build', help='Build training container image')
    
    # Cleanup command
    subparsers.add_parser('cleanup', help='Clean up containers and images')
    
    args = parser.parse_args()
    
    # Create runner instance
    runner = TrainingRunner()
    runner.print_header()
    
    # Handle commands
    success = False
    
    if args.command == 'train':
        success = runner.train_workflow(args.script)
    elif args.command == 'build':
        success = runner.build_training_image()
    elif args.command == 'cleanup':
        success = runner.cleanup()
    else:
        parser.print_help()
        return 0
        
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
