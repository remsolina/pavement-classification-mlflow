# 🚀 MLflow Local CI/CD Makefile
# 
# This Makefile provides convenient commands for running tests,
# CI/CD pipeline, and development tasks locally.
#
# Usage:
#   make help          # Show this help
#   make test          # Run all tests
#   make ci            # Run full CI/CD pipeline
#   make setup         # Setup development environment

.PHONY: help test unit-test integration-test ci setup clean lint format docker-test

# Default target
.DEFAULT_GOAL := help

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
BOLD := \033[1m
NC := \033[0m # No Color

help: ## 📋 Show this help message
	@echo "$(BOLD)🚀 MLflow Local CI/CD Commands$(NC)"
	@echo "=================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(BLUE)%-20s$(NC) %s\n", $$1, $$2}'

# ===================================================================
# 🧪 TESTING COMMANDS
# ===================================================================

test: ## 🧪 Run all tests (unit + integration)
	@echo "$(BOLD)🧪 Running all tests...$(NC)"
	python -m pytest tests/ -v

unit-test: ## 🔬 Run unit tests only
	@echo "$(BOLD)🔬 Running unit tests...$(NC)"
	python -m pytest tests/unit/ -v

integration-test: ## 🔗 Run integration tests only
	@echo "$(BOLD)🔗 Running integration tests...$(NC)"
	python -m pytest tests/integration/ -v

test-coverage: ## 📊 Run tests with coverage report
	@echo "$(BOLD)📊 Running tests with coverage...$(NC)"
	python -m pytest tests/ -v --cov=. --cov-report=html --cov-report=term

# ===================================================================
# 🚀 CI/CD COMMANDS
# ===================================================================

ci: ## 🚀 Run full CI/CD pipeline locally
	@echo "$(BOLD)🚀 Running full CI/CD pipeline...$(NC)"
	python run_local_ci.py

ci-tests: ## 🧪 Run only CI/CD test jobs
	@echo "$(BOLD)🧪 Running CI/CD test jobs...$(NC)"
	python run_local_ci.py --tests-only

ci-job: ## 🎯 Run specific CI/CD job (usage: make ci-job JOB=job-name)
	@echo "$(BOLD)🎯 Running CI/CD job: $(JOB)$(NC)"
	python run_local_ci.py --job $(JOB)

ci-list: ## 📋 List available CI/CD jobs
	@echo "$(BOLD)📋 Available CI/CD jobs:$(NC)"
	python run_local_ci.py --list

ci-dry-run: ## 🔍 Run CI/CD pipeline in dry-run mode
	@echo "$(BOLD)🔍 Running CI/CD pipeline (dry-run)...$(NC)"
	python run_local_ci.py --dry-run

# ===================================================================
# 🔧 SETUP AND ENVIRONMENT
# ===================================================================

setup: ## 🔧 Setup development environment
	@echo "$(BOLD)🔧 Setting up development environment...$(NC)"
	python run_local_ci.py --setup
	@echo "$(GREEN)✅ Setup complete!$(NC)"

setup-test: ## 🧪 Run setup validation tests
	@echo "$(BOLD)🧪 Running setup validation...$(NC)"
	python test_setup.py

validate: ## ✅ Validate configuration
	@echo "$(BOLD)✅ Validating configuration...$(NC)"
	python validate_config.py

# ===================================================================
# 🐳 DOCKER COMMANDS
# ===================================================================

docker-build: ## 🐳 Build Docker training image
	@echo "$(BOLD)🐳 Building Docker training image...$(NC)"
	docker build -f Dockerfile.training -t mlflow-training-local .

docker-test: ## 🧪 Test Docker setup
	@echo "$(BOLD)🧪 Testing Docker setup...$(NC)"
	docker-compose -f docker-compose.local.yml config
	@echo "$(GREEN)✅ Docker configuration valid$(NC)"

docker-up: ## 🚀 Start MLflow services
	@echo "$(BOLD)🚀 Starting MLflow services...$(NC)"
	docker-compose -f docker-compose.local.yml up -d

docker-down: ## 🛑 Stop MLflow services
	@echo "$(BOLD)🛑 Stopping MLflow services...$(NC)"
	docker-compose -f docker-compose.local.yml down

docker-logs: ## 📋 Show Docker logs
	@echo "$(BOLD)📋 Docker logs:$(NC)"
	docker-compose -f docker-compose.local.yml logs

# ===================================================================
# 🧹 CLEANUP COMMANDS
# ===================================================================

clean: ## 🧹 Clean up temporary files and artifacts
	@echo "$(BOLD)🧹 Cleaning up...$(NC)"
	python run_local_ci.py --clean
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	@echo "$(GREEN)✅ Cleanup complete!$(NC)"

clean-docker: ## 🐳 Clean up Docker containers and images
	@echo "$(BOLD)🐳 Cleaning up Docker...$(NC)"
	docker-compose -f docker-compose.local.yml down -v
	docker system prune -f
	@echo "$(GREEN)✅ Docker cleanup complete!$(NC)"

# ===================================================================
# 🎯 TRAINING COMMANDS
# ===================================================================

train: ## 🎯 Run training pipeline
	@echo "$(BOLD)🎯 Running training pipeline...$(NC)"
	python run_training.py train

train-build: ## 🔨 Build training container
	@echo "$(BOLD)🔨 Building training container...$(NC)"
	python run_training.py build

# ===================================================================
# 📊 REPORTING COMMANDS
# ===================================================================

report: ## 📊 Generate test and coverage reports
	@echo "$(BOLD)📊 Generating reports...$(NC)"
	python -m pytest tests/ --cov=. --cov-report=html --cov-report=xml
	@echo "$(GREEN)✅ Reports generated:$(NC)"
	@echo "  - HTML Coverage: htmlcov/index.html"
	@echo "  - XML Coverage: coverage.xml"

# ===================================================================
# 🚀 QUICK COMMANDS
# ===================================================================

quick-test: ## ⚡ Quick test (unit tests only, no coverage)
	@echo "$(BOLD)⚡ Running quick tests...$(NC)"
	python -m pytest tests/unit/ -v --tb=short

quick-ci: ## ⚡ Quick CI (tests only, no Docker)
	@echo "$(BOLD)⚡ Running quick CI...$(NC)"
	python run_local_ci.py --tests-only

all: clean setup test ci ## 🎯 Run complete workflow (clean, setup, test, ci)
	@echo "$(BOLD)$(GREEN)🎉 Complete workflow finished!$(NC)"

# ===================================================================
# 📝 DEVELOPMENT HELPERS
# ===================================================================

install-act: ## 📦 Install Act (GitHub Actions runner)
	@echo "$(BOLD)📦 Installing Act...$(NC)"
	@echo "Please install Act manually from: https://github.com/nektos/act#installation"
	@echo "Or use one of these methods:"
	@echo "  - macOS: brew install act"
	@echo "  - Linux: curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash"
	@echo "  - Windows: choco install act-cli"

status: ## 📊 Show project status
	@echo "$(BOLD)📊 Project Status$(NC)"
	@echo "=================="
	@echo "$(BLUE)Python Version:$(NC) $$(python --version)"
	@echo "$(BLUE)Docker Status:$(NC) $$(docker --version 2>/dev/null || echo 'Not installed')"
	@echo "$(BLUE)Act Status:$(NC) $$(act --version 2>/dev/null || echo 'Not installed')"
	@echo "$(BLUE)MLflow Status:$(NC) $$(curl -s http://localhost:5005 >/dev/null && echo 'Running' || echo 'Not running')"
	@echo "$(BLUE)Test Files:$(NC) $$(find tests/ -name '*.py' | wc -l) files"
	@echo "$(BLUE)Config Valid:$(NC) $$(python validate_config.py >/dev/null 2>&1 && echo 'Yes' || echo 'No')"
