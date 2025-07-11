# 🚀 MLflow Local Pipeline Makefile
# 
# This Makefile provides convenient commands for running the Prefect-driven
# MLflow pipeline, testing, and development tasks locally.
#
# Usage:
#   make help          # Show this help
#   make test          # Run all tests
#   make train         # Run complete Prefect pipeline
#   make setup         # Setup development environment

.PHONY: help test unit-test integration-test train setup clean lint format docker-test build-training docker-up docker-down docker-logs clean-docker status

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
	@echo "$(BOLD)🚀 MLflow Local Pipeline Commands$(NC)"
	@echo "====================================="
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
# 🎯 TRAINING COMMANDS (PREFECT)
# ===================================================================

train: ## 🎯 Run complete Prefect pipeline (build, test, train)
	@echo "$(BOLD)🎯 Running complete Prefect pipeline...$(NC)"
	cd .. && python local/scripts/prefect_flow.py

train-with-cleanup: ## 🎯 Run Prefect pipeline with cleanup
	@echo "$(BOLD)🎯 Running Prefect pipeline with cleanup...$(NC)"
	cd .. && python local/scripts/prefect_flow.py --do-cleanup

# ===================================================================
# 🐳 DOCKER COMMANDS
# ===================================================================

docker-build: ## 🐳 Build Docker training image
	@echo "$(BOLD)🐳 Building Docker training image...$(NC)"
	docker compose --env-file config/docker-config.env -f config/docker-compose.local.yml build training

docker-test: ## 🧪 Test Docker setup
	@echo "$(BOLD)🧪 Testing Docker setup...$(NC)"
	docker compose --env-file config/docker-config.env -f config/docker-compose.local.yml config
	@echo "$(GREEN)✅ Docker configuration valid$(NC)"

docker-up: ## 🚀 Start MLflow services
	@echo "$(BOLD)🚀 Starting MLflow services...$(NC)"
	docker compose --env-file config/docker-config.env -f config/docker-compose.local.yml up -d mlflow-mysql mlflow

docker-down: ## 🛑 Stop MLflow services
	@echo "$(BOLD)🛑 Stopping MLflow services...$(NC)"
	docker compose --env-file config/docker-config.env -f config/docker-compose.local.yml down

docker-logs: ## 📋 Show Docker logs
	@echo "$(BOLD)📋 Docker logs:$(NC)"
	docker compose --env-file config/docker-config.env -f config/docker-compose.local.yml logs

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
	docker compose --env-file config/docker-config.env -f config/docker-compose.local.yml down -v
	docker system prune -f
	@echo "$(GREEN)✅ Docker cleanup complete!$(NC)"

clean-all: ## 🧹 Clean everything (Docker + local)
	@echo "$(BOLD)🧹 Cleaning everything...$(NC)"
	$(MAKE) clean
	$(MAKE) clean-docker
	@echo "$(GREEN)✅ Complete cleanup finished!$(NC)"

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
	@echo "$(BLUE)Checking MLflow server...$(NC)"
	@curl -s http://localhost:5005 >/dev/null && echo "$(GREEN)✅ MLflow server running$(NC)" || echo "$(RED)❌ MLflow server not running$(NC)"
	@echo "$(BLUE)Checking MySQL...$(NC)"
	@docker compose --env-file config/docker-config.env -f config/docker-compose.local.yml ps | grep -q "Up" && echo "$(GREEN)✅ MySQL running$(NC)" || echo "$(RED)❌ MySQL not running$(NC)"

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
	@echo "$(BLUE)Prefect Status:$(NC) $$(prefect version 2>/dev/null || echo 'Not installed')"
	@echo "$(BLUE)MLflow Status:$(NC) $$(curl -s http://localhost:5005 >/dev/null && echo 'Running' || echo 'Not running')"
	@echo "$(BLUE)Test Files:$(NC) $$(find tests/ -name '*.py' | wc -l) files"
	@echo "$(BLUE)Training Script:$(NC) src/model_7_local.py"
	@echo "$(BLUE)Prefect Flow:$(NC) scripts/prefect_flow.py"

# ===================================================================
# 🔧 SETUP COMMANDS
# ===================================================================

setup: ## 🔧 Setup development environment
	@echo "$(BOLD)🔧 Setting up development environment...$(NC)"
	@echo "$(BLUE)Installing Python dependencies...$(NC)"
	pip install -r src/requirements.txt
	@echo "$(BLUE)Installing Prefect...$(NC)"
	pip install prefect
	@echo "$(GREEN)✅ Setup complete!$(NC)"
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  1. Start Prefect server: prefect server start"
	@echo "  2. Run the pipeline: make train"

# ===================================================================
# 🐳 TRAINING IMAGE COMMANDS
# ===================================================================

build-training: ## 🐳 Build training image (alias for docker-build)
	@echo "$(BOLD)🐳 Building training image...$(NC)"
	$(MAKE) docker-build
