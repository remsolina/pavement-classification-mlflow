# 🚀 Local CI/CD Pipeline Documentation

## Overview

This project now includes a comprehensive **local CI/CD pipeline** that runs GitHub Actions workflows locally using **Act**. This allows you to test your entire pipeline without pushing to GitHub, ensuring faster development cycles and catching issues early.

## 🏗️ **What We've Built**

### ✅ **Complete Testing Framework**
- **Unit Tests**: Test individual components (config, validation, etc.)
- **Integration Tests**: Test end-to-end workflows
- **Setup Validation**: Comprehensive environment checking
- **Configuration Validation**: Automated config verification

### ✅ **Local CI/CD Pipeline**
- **GitHub Actions Workflow**: `.github/workflows/local-ci.yml`
- **Local Runner**: `run_local_ci.py` (Python-based Act runner)
- **Makefile**: Convenient commands for all operations
- **Automated Testing**: Unit, integration, Docker, and training tests

### ✅ **Testing Infrastructure**
```
local/
├── tests/
│   ├── unit/
│   │   └── test_config.py           # Unit tests for configuration
│   └── integration/
│       └── test_training_pipeline.py # End-to-end pipeline tests
├── test_setup.py                    # Setup validation (fixes QUICK_START.md)
├── pytest.ini                      # Pytest configuration
├── run_local_ci.py                  # Local CI/CD runner
├── Makefile                         # Convenient commands
└── README_CI_CD.md                  # This documentation
```

## 🚀 **Quick Start**

### 1. **Run Setup Validation** (fixes missing test_setup.py)
```bash
cd local
python test_setup.py --quick
```

### 2. **Run Unit Tests**
```bash
make unit-test
# or
python -m pytest tests/unit/ -v
```

### 3. **Run All Tests**
```bash
make test
# or
python -m pytest tests/ -v --cov=.
```

### 4. **Check Project Status**
```bash
make status
```

## 🧪 **Testing Commands**

### **Using Makefile (Recommended)**
```bash
# Quick tests
make quick-test              # Unit tests only, fast
make unit-test              # All unit tests
make integration-test       # All integration tests
make test                   # All tests (unit + integration)

# With coverage
make test-coverage          # Tests with HTML coverage report
make report                 # Generate comprehensive reports

# Setup and validation
make setup-test             # Run setup validation
make validate               # Validate configuration
```

### **Using Pytest Directly**
```bash
# Run specific test files
python -m pytest tests/unit/test_config.py -v
python -m pytest tests/integration/ -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html

# Run specific test methods
python -m pytest tests/unit/test_config.py::TestConfigLocal::test_default_values -v
```

## 🚀 **CI/CD Pipeline**

### **Local GitHub Actions with Act**

#### **Prerequisites**
```bash
# Install Act (GitHub Actions runner)
# macOS
brew install act

# Linux
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Windows
choco install act-cli
```

#### **CI/CD Commands**
```bash
# Setup CI/CD environment
make setup
python run_local_ci.py --setup

# Run full pipeline
make ci
python run_local_ci.py

# Run specific jobs
make ci-job JOB=unit-tests
python run_local_ci.py --job unit-tests

# List available jobs
make ci-list
python run_local_ci.py --list

# Quick CI (tests only)
make ci-tests
python run_local_ci.py --tests-only

# Dry run (no execution)
make ci-dry-run
python run_local_ci.py --dry-run
```

### **CI/CD Pipeline Jobs**

The pipeline includes these jobs:

1. **🔧 setup-validation**: Validates environment and dependencies
2. **🧪 unit-tests**: Runs unit tests with coverage
3. **🔗 integration-tests**: Runs integration tests with test data
4. **🐳 docker-build**: Builds and tests Docker images
5. **🐳 docker-compose-test**: Tests Docker Compose setup
6. **🎯 training-test**: Tests complete training pipeline
7. **📊 generate-report**: Creates CI/CD reports

## 🐳 **Docker Integration**

### **Docker Commands**
```bash
# Build training image
make docker-build
docker build -f Dockerfile.training -t mlflow-training-local .

# Test Docker setup
make docker-test
docker-compose -f docker-compose.local.yml config

# Start/stop services
make docker-up          # Start MLflow services
make docker-down        # Stop MLflow services
make docker-logs        # View logs

# Clean up
make clean-docker       # Remove containers and images
```

## 📊 **Test Coverage and Reporting**

### **Generate Reports**
```bash
# HTML coverage report
make test-coverage
open htmlcov/index.html

# Comprehensive reports
make report
```

### **Coverage Files**
- `htmlcov/index.html`: Interactive HTML coverage report
- `coverage.xml`: XML coverage for CI/CD integration
- `.coverage`: Coverage database

## 🔧 **Development Workflow**

### **Recommended Development Cycle**
```bash
# 1. Validate setup
make setup-test

# 2. Run quick tests during development
make quick-test

# 3. Run full tests before committing
make test

# 4. Run local CI/CD before pushing
make ci-tests

# 5. Clean up when done
make clean
```

### **Adding New Tests**

#### **Unit Tests**
```python
# tests/unit/test_new_feature.py
import pytest
import sys
from pathlib import Path

# Add local directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class TestNewFeature:
    def test_feature_functionality(self):
        # Your test here
        assert True
```

#### **Integration Tests**
```python
# tests/integration/test_new_integration.py
import pytest
import tempfile
from pathlib import Path

class TestNewIntegration:
    @pytest.fixture
    def test_environment(self):
        # Setup test environment
        yield
        # Cleanup
        
    def test_end_to_end_workflow(self, test_environment):
        # Your integration test here
        assert True
```

## 🎯 **Training Integration**

### **Training Commands**
```bash
# Build training container
make train-build
python run_training.py build

# Run training
make train
python run_training.py train

# Test training pipeline
python run_local_ci.py --job training-test
```

## 🧹 **Cleanup Commands**

```bash
# Clean temporary files
make clean

# Clean Docker resources
make clean-docker

# Complete cleanup
make clean && make clean-docker
```

## 📋 **Troubleshooting**

### **Common Issues**

#### **Tests Failing**
```bash
# Check setup first
python test_setup.py

# Validate configuration
python validate_config.py

# Check specific test
python -m pytest tests/unit/test_config.py::TestConfigLocal::test_default_values -v -s
```

#### **CI/CD Issues**
```bash
# Check Act installation
act --version

# Check Docker
docker --version

# List available jobs
python run_local_ci.py --list

# Run in dry-run mode
python run_local_ci.py --dry-run
```

#### **Docker Issues**
```bash
# Test Docker setup
make docker-test

# Check configuration
docker-compose -f docker-compose.local.yml config

# View logs
make docker-logs
```

## 🎉 **Success Indicators**

Your CI/CD setup is working when:

1. ✅ `make status` shows all components as working
2. ✅ `make test` passes all tests
3. ✅ `python test_setup.py` validates environment
4. ✅ `make ci-tests` completes successfully
5. ✅ Docker builds and tests pass

## 🔗 **Integration with Development**

This CI/CD pipeline integrates seamlessly with your existing MLflow setup:

- **Validates configuration** before training
- **Tests Docker setup** before containerized training
- **Ensures code quality** through comprehensive testing
- **Provides fast feedback** through local execution
- **Maintains compatibility** with existing workflows

## 📚 **Next Steps**

1. **Customize tests** for your specific use cases
2. **Add more integration tests** for edge cases
3. **Extend CI/CD pipeline** with additional jobs
4. **Integrate with IDE** for automated testing
5. **Set up pre-commit hooks** for automatic validation

Happy testing! 🚀
