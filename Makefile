.PHONY: help install install-dev venv
.PHONY: format lint typecheck check ruff-fix ruff-all
.PHONY: test test-go test-all
.PHONY: clean clean-venv clean-all
.PHONY: server dev server-test example-client
.PHONY: list show shell jupyter info quickstart

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

##@ General

help: ## Display this help message
	@echo "$(BLUE)Yaak PII Detection - Development Commands$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make $(GREEN)<target>$(NC)\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BLUE)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

info: ## Show project info
	@echo "$(BLUE)Project Information$(NC)"
	@echo "Name:    $(GREEN)yaak-pii-detection$(NC)"
	@echo "Version: $(GREEN)0.1.0$(NC)"
	@echo "Python:  $(GREEN)$(shell python --version 2>&1)$(NC)"
	@echo "UV:      $(GREEN)$(shell uv --version 2>&1)$(NC)"
	@echo ""
	@echo "$(BLUE)Virtual Environment$(NC)"
	@if [ -d ".venv" ]; then \
		echo "Status:  $(GREEN)Created$(NC)"; \
		echo "Path:    $(GREEN).venv$(NC)"; \
	else \
		echo "Status:  $(YELLOW)Not created (run 'make venv')$(NC)"; \
	fi
	@echo ""
	@echo "$(BLUE)Quick Commands$(NC)"
	@echo "  make install     - Install dependencies"
	@echo "  make dev         - Start dev server"
	@echo "  make test        - Run tests"
	@echo "  make help        - Show all commands"

##@ Setup & Installation

venv: ## Create virtual environment with uv
	@echo "$(BLUE)Creating virtual environment...$(NC)"
	uv venv
	@echo "$(GREEN)✅ Virtual environment created at .venv$(NC)"
	@echo "$(YELLOW)Activate with: source .venv/bin/activate$(NC)"

install: venv ## Install project with all dependencies
	@echo "$(BLUE)Installing project dependencies...$(NC)"
	uv pip install -e .
	@echo "$(GREEN)✅ Installation complete$(NC)"

install-dev: venv ## Install with development dependencies
	@echo "$(BLUE)Installing with dev dependencies...$(NC)"
	uv pip install -e ".[dev]"
	@echo "$(GREEN)✅ Dev installation complete$(NC)"

##@ FastAPI Server

server: ## Start FastAPI server (production mode)
	@echo "$(BLUE)Starting FastAPI server...$(NC)"
	cd model_server && ./start_server.sh

dev: ## Start FastAPI server (development mode with auto-reload)
	@echo "$(BLUE)Starting server in development mode...$(NC)"
	cd model_server && ./start_server.sh --reload

server-test: ## Test the FastAPI server
	@echo "$(BLUE)Testing FastAPI server...$(NC)"
	uv run python model_server/test_server.py

example-client: ## Run example client
	@echo "$(BLUE)Running example client...$(NC)"
	uv run python model_server/example_client.py

##@ Code Quality

format: ## Format code with ruff
	@echo "$(BLUE)Formatting code...$(NC)"
	uv run ruff format .
	@echo "$(GREEN)✅ Code formatted$(NC)"

lint: ## Run linters with ruff
	@echo "$(BLUE)Running linters...$(NC)"
	uv run ruff check pii/ model/ model_server/
	@echo "$(GREEN)✅ Linting complete$(NC)"

typecheck: ## Run type checker with ruff
	@echo "$(BLUE)Running type checker...$(NC)"
	uv run ruff check pii/ model/ model_server/ --select TYP
	@echo "$(GREEN)✅ Type checking complete$(NC)"

check: format lint typecheck ## Run all code quality checks

ruff-fix: ## Auto-fix ruff issues
	@echo "$(BLUE)Auto-fixing ruff issues...$(NC)"
	uv run ruff check pii/ model/ model_server/ --fix
	@echo "$(GREEN)✅ Auto-fix complete$(NC)"

ruff-all: ## Run all ruff checks (lint + format + typecheck)
	@echo "$(BLUE)Running all ruff checks...$(NC)"
	uv run ruff check pii/ model/ model_server/ --fix
	uv run ruff format .
	@echo "$(GREEN)✅ All ruff checks complete$(NC)"

##@ Testing

test: ## Run Python tests
	@echo "$(BLUE)Running Python tests...$(NC)"
	uv run pytest tests/ -v
	@echo "$(GREEN)✅ Tests complete$(NC)"

test-go: ## Run Go tests
	@echo "$(BLUE)Running Go tests...$(NC)"
	go test ./... -v
	@echo "$(GREEN)✅ Go tests complete$(NC)"

test-all: test test-go server-test ## Run all tests (Python, Go, and server)
	@echo "$(GREEN)✅ All tests complete$(NC)"

##@ Cleanup

clean: ## Remove build artifacts and cache
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	find . -type d \( -name "__pycache__" -o -name "*.egg-info" -o -name ".pytest_cache" -o -name ".mypy_cache" \) -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info
	@echo "$(GREEN)✅ Cleanup complete$(NC)"

clean-venv: ## Remove virtual environment
	@echo "$(BLUE)Removing virtual environment...$(NC)"
	rm -rf .venv
	@echo "$(GREEN)✅ Virtual environment removed$(NC)"

clean-all: clean clean-venv ## Remove everything (artifacts, cache, and venv)
	@echo "$(GREEN)✅ Full cleanup complete$(NC)"

