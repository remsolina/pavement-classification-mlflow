#!/usr/bin/env python3
"""
🧪 Setup Testing Script

This script validates that your local development environment is properly
configured and ready for MLflow training. It performs comprehensive checks
of Python environment, dependencies, configuration, and connectivity.

Usage:
    python test_setup.py
    python test_setup.py --verbose
    python test_setup.py --quick
"""

import sys
import os
import subprocess
import importlib
import argparse
from pathlib import Path
import requests
import time


class Colors:
    """ANSI color codes for terminal output."""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    BOLD = '\033[1m'
    NC = '\033[0m'  # No Color


class SetupTester:
    """Comprehensive setup testing for local MLflow environment."""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.passed_tests = 0
        self.failed_tests = 0
        self.warnings = 0
        
    def print_header(self):
        """Print test header."""
        print(f"{Colors.BOLD}🧪 MLflow Setup Testing{Colors.NC}")
        print("=" * 50)
        
    def print_success(self, message: str):
        """Print success message."""
        print(f"{Colors.GREEN}✅ {message}{Colors.NC}")
        self.passed_tests += 1
        
    def print_error(self, message: str):
        """Print error message."""
        print(f"{Colors.RED}❌ {message}{Colors.NC}")
        self.failed_tests += 1
        
    def print_warning(self, message: str):
        """Print warning message."""
        print(f"{Colors.YELLOW}⚠️  {message}{Colors.NC}")
        self.warnings += 1
        
    def print_info(self, message: str):
        """Print info message."""
        if self.verbose:
            print(f"{Colors.BLUE}ℹ️  {message}{Colors.NC}")
    
    def test_python_environment(self):
        """Test Python environment and version."""
        print(f"\n{Colors.BOLD}🐍 Python Environment{Colors.NC}")
        
        # Check Python version
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            self.print_success(f"Python {version.major}.{version.minor}.{version.micro}")
        else:
            self.print_error(f"Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
            
        # Check virtual environment
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            self.print_success("Virtual environment detected")
        else:
            self.print_warning("No virtual environment detected - recommended for isolation")
            
        self.print_info(f"Python executable: {sys.executable}")
        
    def test_required_packages(self):
        """Test required Python packages."""
        print(f"\n{Colors.BOLD}📦 Required Packages{Colors.NC}")
        
        required_packages = {
            'torch': '2.0.0',
            'torchvision': '0.15.0',
            'numpy': '1.21.0',
            'mlflow': '2.8.0',
            'PIL': '9.0.0',  # Pillow
            'matplotlib': '3.5.0',
            'seaborn': '0.11.0',
            'sklearn': '1.0.0'  # scikit-learn
        }
        
        for package, min_version in required_packages.items():
            try:
                if package == 'PIL':
                    module = importlib.import_module('PIL')
                    # Pillow version check
                    import PIL
                    version = PIL.__version__
                elif package == 'sklearn':
                    module = importlib.import_module('sklearn')
                    version = module.__version__
                else:
                    module = importlib.import_module(package)
                    version = module.__version__
                
                self.print_success(f"{package} {version}")
                self.print_info(f"Required: {min_version}+")
                
            except ImportError:
                self.print_error(f"{package} not installed")
            except AttributeError:
                self.print_warning(f"{package} installed but version unknown")
                
    def test_configuration_files(self):
        """Test configuration files."""
        print(f"\n{Colors.BOLD}⚙️ Configuration Files{Colors.NC}")
        
        required_files = {
            'config_local.py': 'Training configuration',
            'docker-config.env': 'Docker configuration',
            'requirements.txt': 'Python dependencies',
            'docker-compose.local.yml': 'Docker Compose setup',
            'Dockerfile.training': 'Training container definition'
        }
        
        for file, description in required_files.items():
            if os.path.exists(file):
                self.print_success(f"{file} - {description}")
            else:
                self.print_error(f"{file} missing - {description}")
                
        # Test config_local.py import
        try:
            sys.path.insert(0, '.')
            import config_local
            self.print_success("config_local.py imports successfully")
            
            # Test config validation
            config_local.validate_config()
            self.print_success("Configuration validation passed")
            
        except Exception as e:
            self.print_error(f"Configuration error: {e}")
            
    def test_data_path(self):
        """Test data path configuration."""
        print(f"\n{Colors.BOLD}📁 Data Path Configuration{Colors.NC}")
        
        try:
            import config_local
            data_path = config_local.LOCAL_DATA_PATH
            
            if os.path.exists(data_path):
                self.print_success(f"Data path exists: {data_path}")
                
                # Check for class directories
                subdirs = [d for d in Path(data_path).iterdir() if d.is_dir()]
                if len(subdirs) > 0:
                    self.print_success(f"Found {len(subdirs)} class directories: {[d.name for d in subdirs]}")
                    
                    # Count images
                    total_images = 0
                    for subdir in subdirs:
                        images = list(subdir.glob('*.jpg')) + list(subdir.glob('*.jpeg')) + list(subdir.glob('*.png'))
                        total_images += len(images)
                        self.print_info(f"{subdir.name}: {len(images)} images")
                    
                    if total_images > 0:
                        self.print_success(f"Total images found: {total_images}")
                    else:
                        self.print_warning("No images found in class directories")
                else:
                    self.print_warning("No class directories found in data path")
            else:
                self.print_error(f"Data path does not exist: {data_path}")
                self.print_info("Update LOCAL_DATA_PATH in config_local.py")
                
        except Exception as e:
            self.print_error(f"Data path test failed: {e}")
            
    def test_docker_setup(self):
        """Test Docker setup."""
        print(f"\n{Colors.BOLD}🐳 Docker Setup{Colors.NC}")
        
        # Test Docker installation
        try:
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                version = result.stdout.strip()
                self.print_success(f"Docker installed: {version}")
            else:
                self.print_error("Docker not working properly")
        except FileNotFoundError:
            self.print_error("Docker not installed")
            
        # Test Docker Compose
        try:
            result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                version = result.stdout.strip()
                self.print_success(f"Docker Compose installed: {version}")
            else:
                self.print_error("Docker Compose not working properly")
        except FileNotFoundError:
            self.print_error("Docker Compose not installed")
            
        # Test Docker Compose file validation
        try:
            result = subprocess.run(['docker-compose', '-f', 'docker-compose.local.yml', 'config'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                self.print_success("Docker Compose configuration valid")
            else:
                self.print_error(f"Docker Compose configuration invalid: {result.stderr}")
        except Exception as e:
            self.print_error(f"Docker Compose validation failed: {e}")
            
    def test_mlflow_connectivity(self, quick=False):
        """Test MLflow server connectivity."""
        print(f"\n{Colors.BOLD}🔗 MLflow Connectivity{Colors.NC}")
        
        if quick:
            self.print_info("Skipping MLflow connectivity test in quick mode")
            return
            
        try:
            import config_local
            mlflow_uri = config_local.MLFLOW_TRACKING_URI
            
            self.print_info(f"Testing connection to: {mlflow_uri}")
            
            # Test connection
            try:
                response = requests.get(mlflow_uri, timeout=5)
                if response.status_code == 200:
                    self.print_success("MLflow server is accessible")
                else:
                    self.print_warning(f"MLflow server responded with status {response.status_code}")
            except requests.exceptions.ConnectionError:
                self.print_warning("MLflow server not running - start with ./start_mlflow_local.sh")
            except requests.exceptions.Timeout:
                self.print_warning("MLflow server connection timeout")
                
        except Exception as e:
            self.print_error(f"MLflow connectivity test failed: {e}")
            
    def test_training_runner(self):
        """Test Python training runner."""
        print(f"\n{Colors.BOLD}🚀 Training Runner{Colors.NC}")
        
        if os.path.exists('run_training.py'):
            self.print_success("run_training.py exists")
            
            # Test help command
            try:
                result = subprocess.run([sys.executable, 'run_training.py', '--help'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    self.print_success("Training runner help command works")
                else:
                    self.print_error("Training runner help command failed")
            except Exception as e:
                self.print_error(f"Training runner test failed: {e}")
        else:
            self.print_error("run_training.py not found")
            
    def run_all_tests(self, quick=False):
        """Run all setup tests."""
        self.print_header()
        
        # Run all test categories
        self.test_python_environment()
        self.test_required_packages()
        self.test_configuration_files()
        self.test_data_path()
        self.test_docker_setup()
        self.test_mlflow_connectivity(quick=quick)
        self.test_training_runner()
        
        # Print summary
        self.print_summary()
        
    def print_summary(self):
        """Print test summary."""
        print(f"\n{Colors.BOLD}📊 Test Summary{Colors.NC}")
        print("=" * 30)
        
        total_tests = self.passed_tests + self.failed_tests
        
        if self.failed_tests == 0:
            print(f"{Colors.GREEN}🎉 All tests passed! ({self.passed_tests}/{total_tests}){Colors.NC}")
            if self.warnings > 0:
                print(f"{Colors.YELLOW}⚠️  {self.warnings} warnings - check above for details{Colors.NC}")
            print(f"\n{Colors.BOLD}✅ Your setup is ready for MLflow training!{Colors.NC}")
            print("\nNext steps:")
            print("1. Start MLflow server: ./start_mlflow_local.sh")
            print("2. Run training: python run_training.py train")
        else:
            print(f"{Colors.RED}❌ {self.failed_tests} tests failed{Colors.NC}")
            print(f"{Colors.GREEN}✅ {self.passed_tests} tests passed{Colors.NC}")
            if self.warnings > 0:
                print(f"{Colors.YELLOW}⚠️  {self.warnings} warnings{Colors.NC}")
            print(f"\n{Colors.BOLD}🔧 Please fix the issues above before training{Colors.NC}")
            
        return self.failed_tests == 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Test MLflow setup')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--quick', '-q', action='store_true', help='Quick test (skip connectivity)')
    
    args = parser.parse_args()
    
    tester = SetupTester(verbose=args.verbose)
    success = tester.run_all_tests(quick=args.quick)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
