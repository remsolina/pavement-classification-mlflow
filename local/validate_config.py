#!/usr/bin/env python3
"""
🔧 Configuration Validator

This script validates your Docker Compose and training configuration
to ensure everything is set up correctly before running training.

Usage:
    python validate_config.py
"""

import os
import sys
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output."""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    BOLD = '\033[1m'
    NC = '\033[0m'  # No Color


def print_header():
    """Print validation header."""
    print(f"{Colors.BOLD}🔧 Configuration Validation{Colors.NC}")
    print("=" * 50)


def print_success(message: str):
    """Print success message."""
    print(f"{Colors.GREEN}✅ {message}{Colors.NC}")


def print_warning(message: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.NC}")


def print_error(message: str):
    """Print error message."""
    print(f"{Colors.RED}❌ {message}{Colors.NC}")


def print_info(message: str):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ️  {message}{Colors.NC}")


def load_env_file(env_file_path: str) -> dict:
    """Load environment variables from .env file."""
    env_vars = {}
    if not os.path.exists(env_file_path):
        print_error(f"Environment file not found: {env_file_path}")
        return env_vars
    
    try:
        with open(env_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
        print_success(f"Loaded environment file: {env_file_path}")
    except Exception as e:
        print_error(f"Error loading environment file: {e}")
    
    return env_vars


def validate_docker_config():
    """Validate Docker Compose configuration."""
    print_info("Validating Docker Compose configuration...")
    
    # Load docker-config.env
    env_vars = load_env_file("docker-config.env")
    if not env_vars:
        print_error("Could not load docker-config.env")
        return False
    
    # Check required variables
    required_vars = [
        'MYSQL_ROOT_PASSWORD',
        'MYSQL_USER', 
        'MYSQL_PASSWORD',
        'TRAINING_DATA_PATH',
        'MLFLOW_HOST_PORT',
        'MYSQL_HOST_PORT'
    ]
    
    missing_vars = []
    for var in required_vars:
        if var not in env_vars:
            missing_vars.append(var)
    
    if missing_vars:
        print_error(f"Missing required variables in docker-config.env: {missing_vars}")
        return False
    
    # Validate training data path
    training_data_path = env_vars.get('TRAINING_DATA_PATH')
    if not os.path.exists(training_data_path):
        print_error(f"Training data path does not exist: {training_data_path}")
        print_info("Update TRAINING_DATA_PATH in docker-config.env to point to your data")
        return False
    
    print_success(f"Training data path exists: {training_data_path}")
    
    # Check data structure
    data_path = Path(training_data_path)
    subdirs = [d for d in data_path.iterdir() if d.is_dir()]
    if len(subdirs) == 0:
        print_warning(f"No subdirectories found in {training_data_path}")
        print_info("Expected structure: training_data_path/class_name/images.jpg")
    else:
        print_success(f"Found {len(subdirs)} class directories: {[d.name for d in subdirs]}")
    
    # Validate ports
    try:
        mlflow_port = int(env_vars.get('MLFLOW_HOST_PORT', 5005))
        mysql_port = int(env_vars.get('MYSQL_HOST_PORT', 3308))
        
        if mlflow_port == mysql_port:
            print_error("MLFLOW_HOST_PORT and MYSQL_HOST_PORT cannot be the same")
            return False
        
        print_success(f"Port configuration valid - MLflow: {mlflow_port}, MySQL: {mysql_port}")
    except ValueError:
        print_error("Invalid port numbers in docker-config.env")
        return False
    
    # Check passwords
    if env_vars.get('MYSQL_ROOT_PASSWORD') == 'root':
        print_warning("Using default MySQL root password - change for production!")
    
    if env_vars.get('MYSQL_PASSWORD') == 'mlflow_pass':
        print_warning("Using default MySQL user password - change for production!")
    
    return True


def validate_training_config():
    """Validate training configuration."""
    print_info("Validating training configuration...")
    
    # Check if config_local.py exists
    if not os.path.exists("config_local.py"):
        print_error("config_local.py not found")
        return False
    
    # Try to import and validate config
    try:
        sys.path.insert(0, '.')
        import config_local
        
        # Run config validation
        config_local.validate_config()
        print_success("Training configuration is valid")
        return True
    except Exception as e:
        print_error(f"Training configuration error: {e}")
        return False


def validate_docker_files():
    """Validate Docker-related files."""
    print_info("Validating Docker files...")
    
    required_files = [
        "docker-compose.local.yml",
        "Dockerfile.training",
        "requirements.txt"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            print_success(f"Found: {file}")
    
    if missing_files:
        print_error(f"Missing required files: {missing_files}")
        return False
    
    return True


def validate_python_runner():
    """Validate Python training runner."""
    print_info("Validating Python training runner...")
    
    if not os.path.exists("run_training.py"):
        print_error("run_training.py not found")
        return False
    
    print_success("Found: run_training.py")
    
    # Check if it's executable
    if not os.access("run_training.py", os.X_OK):
        print_warning("run_training.py is not executable")
        print_info("Run: chmod +x run_training.py")
    
    return True


def main():
    """Main validation function."""
    print_header()
    
    all_valid = True
    
    # Validate each component
    validations = [
        ("Docker files", validate_docker_files),
        ("Docker configuration", validate_docker_config),
        ("Training configuration", validate_training_config),
        ("Python runner", validate_python_runner)
    ]
    
    for name, validator in validations:
        print(f"\n{Colors.BOLD}📋 {name}:{Colors.NC}")
        if not validator():
            all_valid = False
    
    # Final result
    print(f"\n{Colors.BOLD}🎯 Validation Summary:{Colors.NC}")
    if all_valid:
        print_success("All configurations are valid!")
        print_info("You can now run: python run_training.py train")
    else:
        print_error("Some configurations need attention")
        print_info("Please fix the issues above before running training")
    
    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
