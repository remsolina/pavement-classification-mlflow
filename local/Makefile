# 🚀 MLflow Local CI/CD Makefile
# 
# This Makefile provides convenient commands for running tests,
# CI/CD pipeline, and development tasks locally.
#
# Usage:
#   make help          # Show this help
#   make test          # Run all tests
#   make train         # Run training
#   make setup         # Setup development environment

.PHONY: help test unit-test integration-test train setup clean lint format docker-test build-training build-training-in-prefect retag-training-image

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
# 🎯 TRAINING COMMANDS
# ===================================================================

train: ## 🎯 Run training pipeline
	@echo "$(BOLD)🎯 Running training pipeline...$(NC)"
	python scripts/run_training.py train

train-build: ## 🔨 Build training container
	@echo "$(BOLD)🔨 Building training container...$(NC)"
	python scripts/run_training.py build

train-cleanup: ## 🧹 Clean up training containers
	@echo "$(BOLD)🧹 Cleaning up training containers...$(NC)"
	python scripts/run_training.py cleanup

# ===================================================================
# 🐳 DOCKER COMMANDS
# ===================================================================

docker-build: ## 🐳 Build Docker training image
	@echo "$(BOLD)🐳 Building Docker training image...$(NC)"
	docker-compose -f config/docker-compose.local.yml build training

docker-test: ## 🧪 Test Docker setup
	@echo "$(BOLD)🧪 Testing Docker setup...$(NC)"
	docker-compose -f config/docker-compose.local.yml config
	@echo "$(GREEN)✅ Docker configuration valid$(NC)"

docker-up: ## 🚀 Start MLflow services
	@echo "$(BOLD)🚀 Starting MLflow services...$(NC)"
	docker-compose -f config/docker-compose.local.yml up -d mlflow-mysql mlflow

docker-down: ## 🛑 Stop MLflow services
	@echo "$(BOLD)🛑 Stopping MLflow services...$(NC)"
	docker-compose -f config/docker-compose.local.yml down

docker-logs: ## 📋 Show Docker logs
	@echo "$(BOLD)📋 Docker logs:$(NC)"
	docker-compose -f config/docker-compose.local.yml logs

# ===================================================================
# 🧹 CLEANUP COMMANDS
# ===================================================================

clean: ## 🧹 Clean up temporary files and artifacts
	@echo "$(BOLD)🧹 Cleaning up...$(NC)"
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	@echo "$(GREEN)✅ Cleanup complete!$(NC)"

clean-docker: ## 🐳 Clean up Docker containers and images
	@echo "$(BOLD)🧹 Cleaning up Docker...$(NC)"
	docker-compose -f config/docker-compose.local.yml down -v
	docker system prune -f
	@echo "$(GREEN)✅ Docker cleanup complete!$(NC)"

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

health-check: ## 🔍 Run health checks
	@echo "$(BOLD)🔍 Running health checks...$(NC)"
	python scripts/check_health.py

all: clean test train ## 🎯 Run complete workflow (clean, test, train)
	@echo "$(BOLD)$(GREEN)🎉 Complete workflow finished!$(NC)"

# ===================================================================
# 📝 DEVELOPMENT HELPERS
# ===================================================================

status: ## 📊 Show project status
	@echo "$(BOLD)📊 Project Status$(NC)"
	@echo "=================="
	@echo "$(BLUE)Python Version:$(NC) $$(python --version)"
	@echo "$(BLUE)Docker Status:$(NC) $$(docker --version 2>/dev/null || echo 'Not installed')"
	@echo "$(BLUE)MLflow Status:$(NC) $$(curl -s http://localhost:5005 >/dev/null && echo 'Running' || echo 'Not running')"
	@echo "$(BLUE)Test Files:$(NC) $$(find tests/ -name '*.py' | wc -l) files"
	@echo "$(BLUE)Training Script:$(NC) src/model_7_local.py"

# Build the training image from the host (recommended for Prefect)
build-training:
	docker compose -f local/config/docker-compose.local.yml build training

# Build the training image inside the Prefect container (advanced/debug)
build-training-in-prefect:
	docker exec -it prefect bash -c 'cd /workspace && docker build -f local/config/Dockerfile.training -t config-training:latest .'

# Retag a custom image as config-training:latest
retag-training-image:
	docker tag test-training config-training:latest
