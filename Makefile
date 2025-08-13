# Multi-omics Biomarker Discovery Makefile
# Provides convenient commands for development, testing, and deployment

.PHONY: help install install-dev setup clean test lint format docs docker-build docker-run pipeline

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := pip3
CONDA := conda
DOCKER := docker
DOCKER_COMPOSE := docker-compose
PROJECT_NAME := multiomics-biomarker-discovery
IMAGE_NAME := $(PROJECT_NAME)
CONTAINER_NAME := $(PROJECT_NAME)-container

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# =============================================================================
# HELP
# =============================================================================
help: ## Show this help message
	@echo "$(BLUE)Multi-omics Biomarker Discovery - Available Commands$(NC)"
	@echo "=================================================="
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================
install: ## Install production dependencies
	@echo "$(YELLOW)Installing production dependencies...$(NC)"
	$(PIP) install -r requirements.txt

install-dev: ## Install development dependencies
	@echo "$(YELLOW)Installing development dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov black flake8 mypy pre-commit
	pre-commit install

install-conda: ## Install dependencies using conda
	@echo "$(YELLOW)Creating conda environment...$(NC)"
	$(CONDA) env create -f environment.yml
	@echo "$(GREEN)Activate with: conda activate $(PROJECT_NAME)$(NC)"

setup: ## Initial project setup
	@echo "$(YELLOW)Setting up project...$(NC)"
	cp .env.template .env
	mkdir -p data/raw data/processed data/results logs models artifacts
	mkdir -p notebooks/01_data_exploration notebooks/02_preprocessing
	mkdir -p notebooks/03_integration notebooks/04_modeling
	mkdir -p notebooks/05_pathway_analysis notebooks/06_biomarker_discovery
	@echo "$(GREEN)Project setup complete!$(NC)"
	@echo "$(BLUE)Next steps:$(NC)"
	@echo "1. Edit .env file with your configuration"
	@echo "2. Run 'make install' or 'make install-conda'"
	@echo "3. Run 'make test' to verify installation"

# =============================================================================
# DEVELOPMENT
# =============================================================================
clean: ## Clean up temporary files and caches
	@echo "$(YELLOW)Cleaning up...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	@echo "$(GREEN)Cleanup complete!$(NC)"

format: ## Format code with black and isort
	@echo "$(YELLOW)Formatting code...$(NC)"
	black src/ tests/ scripts/
	isort src/ tests/ scripts/
	@echo "$(GREEN)Code formatting complete!$(NC)"

lint: ## Run linting checks
	@echo "$(YELLOW)Running linting checks...$(NC)"
	flake8 src/ tests/ scripts/
	mypy src/
	@echo "$(GREEN)Linting complete!$(NC)"

type-check: ## Run type checking
	@echo "$(YELLOW)Running type checks...$(NC)"
	mypy src/ --strict
	@echo "$(GREEN)Type checking complete!$(NC)"

# =============================================================================
# TESTING
# =============================================================================
test: ## Run all tests
	@echo "$(YELLOW)Running tests...$(NC)"
	pytest tests/ -v

test-cov: ## Run tests with coverage
	@echo "$(YELLOW)Running tests with coverage...$(NC)"
	pytest tests/ --cov=src --cov-report=html --cov-report=term
	@echo "$(GREEN)Coverage report generated in htmlcov/$(NC)"

test-fast: ## Run fast tests only
	@echo "$(YELLOW)Running fast tests...$(NC)"
	pytest tests/ -v -m "not slow"

test-integration: ## Run integration tests
	@echo "$(YELLOW)Running integration tests...$(NC)"
	pytest tests/integration/ -v

test-unit: ## Run unit tests
	@echo "$(YELLOW)Running unit tests...$(NC)"
	pytest tests/unit/ -v

# =============================================================================
# DATA AND PIPELINE
# =============================================================================
download-data: ## Download sample data
	@echo "$(YELLOW)Downloading sample data...$(NC)"
	$(PYTHON) scripts/download_data.py --cancer-types BRCA LUAD --drugs Paclitaxel Cisplatin
	@echo "$(GREEN)Data download complete!$(NC)"

preprocess: ## Run data preprocessing
	@echo "$(YELLOW)Running data preprocessing...$(NC)"
	$(PYTHON) scripts/run_preprocessing.py
	@echo "$(GREEN)Preprocessing complete!$(NC)"

integrate: ## Run data integration
	@echo "$(YELLOW)Running data integration...$(NC)"
	$(PYTHON) scripts/run_integration.py
	@echo "$(GREEN)Integration complete!$(NC)"

model: ## Run model training
	@echo "$(YELLOW)Running model training...$(NC)"
	$(PYTHON) scripts/run_modeling.py
	@echo "$(GREEN)Model training complete!$(NC)"

pathway-analysis: ## Run pathway analysis
	@echo "$(YELLOW)Running pathway analysis...$(NC)"
	$(PYTHON) scripts/run_pathway_analysis.py
	@echo "$(GREEN)Pathway analysis complete!$(NC)"

pipeline: ## Run complete analysis pipeline
	@echo "$(YELLOW)Running complete pipeline...$(NC)"
	$(MAKE) download-data
	$(MAKE) preprocess
	$(MAKE) integrate
	$(MAKE) model
	$(MAKE) pathway-analysis
	$(PYTHON) scripts/generate_report.py
	@echo "$(GREEN)Pipeline complete! Check results/ directory$(NC)"

# =============================================================================
# DOCKER
# =============================================================================
docker-build: ## Build Docker image
	@echo "$(YELLOW)Building Docker image...$(NC)"
	$(DOCKER) build -t $(IMAGE_NAME):latest .
	@echo "$(GREEN)Docker image built successfully!$(NC)"

docker-build-dev: ## Build development Docker image
	@echo "$(YELLOW)Building development Docker image...$(NC)"
	$(DOCKER) build --target development -t $(IMAGE_NAME):dev .
	@echo "$(GREEN)Development Docker image built successfully!$(NC)"

docker-run: ## Run Docker container
	@echo "$(YELLOW)Running Docker container...$(NC)"
	$(DOCKER) run -it --rm \
		-p 8888:8888 -p 8050:8050 \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/results:/app/results \
		--name $(CONTAINER_NAME) \
		$(IMAGE_NAME):latest

docker-run-dev: ## Run development Docker container
	@echo "$(YELLOW)Running development Docker container...$(NC)"
	$(DOCKER) run -it --rm \
		-p 8888:8888 -p 8050:8050 \
		-v $(PWD):/app \
		--name $(CONTAINER_NAME)-dev \
		$(IMAGE_NAME):dev

docker-compose-up: ## Start all services with docker-compose
	@echo "$(YELLOW)Starting services with docker-compose...$(NC)"
	$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)Services started!$(NC)"
	@echo "$(BLUE)Access points:$(NC)"
	@echo "- Jupyter Lab: http://localhost:8888"
	@echo "- Dash App: http://localhost:8050"
	@echo "- MLflow: http://localhost:5000"

docker-compose-down: ## Stop all services
	@echo "$(YELLOW)Stopping services...$(NC)"
	$(DOCKER_COMPOSE) down
	@echo "$(GREEN)Services stopped!$(NC)"

docker-compose-logs: ## View service logs
	$(DOCKER_COMPOSE) logs -f

docker-clean: ## Clean up Docker images and containers
	@echo "$(YELLOW)Cleaning up Docker resources...$(NC)"
	$(DOCKER) system prune -f
	$(DOCKER) image prune -f
	@echo "$(GREEN)Docker cleanup complete!$(NC)"

# =============================================================================
# JUPYTER AND NOTEBOOKS
# =============================================================================
jupyter: ## Start Jupyter Lab
	@echo "$(YELLOW)Starting Jupyter Lab...$(NC)"
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
	@echo "$(GREEN)Jupyter Lab started at http://localhost:8888$(NC)"

notebook-clean: ## Clean notebook outputs
	@echo "$(YELLOW)Cleaning notebook outputs...$(NC)"
	jupyter nbconvert --clear-output --inplace notebooks/**/*.ipynb
	@echo "$(GREEN)Notebook outputs cleaned!$(NC)"

notebook-run-all: ## Run all notebooks
	@echo "$(YELLOW)Running all notebooks...$(NC)"
	find notebooks/ -name "*.ipynb" -exec jupyter nbconvert --execute --inplace {} \;
	@echo "$(GREEN)All notebooks executed!$(NC)"

# =============================================================================
# DOCUMENTATION
# =============================================================================
docs: ## Build documentation
	@echo "$(YELLOW)Building documentation...$(NC)"
	cd docs && make html
	@echo "$(GREEN)Documentation built! Open docs/_build/html/index.html$(NC)"

docs-serve: ## Serve documentation locally
	@echo "$(YELLOW)Serving documentation...$(NC)"
	cd docs/_build/html && $(PYTHON) -m http.server 8000
	@echo "$(GREEN)Documentation served at http://localhost:8000$(NC)"

docs-clean: ## Clean documentation build
	@echo "$(YELLOW)Cleaning documentation...$(NC)"
	cd docs && make clean
	@echo "$(GREEN)Documentation cleaned!$(NC)"

# =============================================================================
# MONITORING AND PROFILING
# =============================================================================
profile: ## Profile the main pipeline
	@echo "$(YELLOW)Profiling pipeline...$(NC)"
	$(PYTHON) -m cProfile -o profile_results.prof scripts/run_modeling.py
	@echo "$(GREEN)Profiling complete! Results in profile_results.prof$(NC)"

memory-profile: ## Profile memory usage
	@echo "$(YELLOW)Profiling memory usage...$(NC)"
	mprof run scripts/run_modeling.py
	mprof plot
	@echo "$(GREEN)Memory profiling complete!$(NC)"

# =============================================================================
# DEPLOYMENT
# =============================================================================
deploy-local: ## Deploy locally
	@echo "$(YELLOW)Deploying locally...$(NC)"
	$(MAKE) docker-compose-up
	@echo "$(GREEN)Local deployment complete!$(NC)"

deploy-prod: ## Deploy to production (placeholder)
	@echo "$(RED)Production deployment not implemented yet$(NC)"
	@echo "$(BLUE)TODO: Implement production deployment$(NC)"

# =============================================================================
# DATABASE
# =============================================================================
db-init: ## Initialize database
	@echo "$(YELLOW)Initializing database...$(NC)"
	$(PYTHON) scripts/init_database.py
	@echo "$(GREEN)Database initialized!$(NC)"

db-migrate: ## Run database migrations
	@echo "$(YELLOW)Running database migrations...$(NC)"
	$(PYTHON) scripts/migrate_database.py
	@echo "$(GREEN)Database migrations complete!$(NC)"

db-reset: ## Reset database
	@echo "$(YELLOW)Resetting database...$(NC)"
	$(DOCKER_COMPOSE) down -v
	$(DOCKER_COMPOSE) up -d postgres
	sleep 5
	$(MAKE) db-init
	@echo "$(GREEN)Database reset complete!$(NC)"

# =============================================================================
# UTILITIES
# =============================================================================
check: ## Run all checks (lint, type-check, test)
	@echo "$(YELLOW)Running all checks...$(NC)"
	$(MAKE) lint
	$(MAKE) type-check
	$(MAKE) test
	@echo "$(GREEN)All checks passed!$(NC)"

install-hooks: ## Install git hooks
	@echo "$(YELLOW)Installing git hooks...$(NC)"
	pre-commit install
	@echo "$(GREEN)Git hooks installed!$(NC)"

update-deps: ## Update dependencies
	@echo "$(YELLOW)Updating dependencies...$(NC)"
	$(PIP) list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 $(PIP) install -U
	@echo "$(GREEN)Dependencies updated!$(NC)"

security-check: ## Run security checks
	@echo "$(YELLOW)Running security checks...$(NC)"
	safety check
	bandit -r src/
	@echo "$(GREEN)Security checks complete!$(NC)"

# =============================================================================
# QUICK COMMANDS
# =============================================================================
dev: ## Quick development setup
	$(MAKE) setup
	$(MAKE) install-dev
	$(MAKE) install-hooks

start: ## Quick start (docker-compose up)
	$(MAKE) docker-compose-up

stop: ## Quick stop (docker-compose down)
	$(MAKE) docker-compose-down

restart: ## Quick restart
	$(MAKE) stop
	$(MAKE) start

status: ## Show service status
	$(DOCKER_COMPOSE) ps

# =============================================================================
# EXAMPLES
# =============================================================================
example-basic: ## Run basic example
	@echo "$(YELLOW)Running basic example...$(NC)"
	$(PYTHON) examples/basic_analysis.py
	@echo "$(GREEN)Basic example complete!$(NC)"

example-advanced: ## Run advanced example
	@echo "$(YELLOW)Running advanced example...$(NC)"
	$(PYTHON) examples/advanced_analysis.py
	@echo "$(GREEN)Advanced example complete!$(NC)"