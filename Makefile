.PHONY: help install install-dev venv
.PHONY: format lint lint-go lint-frontend lint-frontend-fix lint-all typecheck typecheck-frontend check check-all ruff-fix ruff-all
.PHONY: test test-go test-all
.PHONY: clean clean-venv clean-all
.PHONY: build-dmg
.PHONY: electron-build electron-run electron electron-dev electron-install
.PHONY: list show shell jupyter info quickstart

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

# Version from package.json
VERSION := $(shell cd src/frontend && node -p "require('./package.json').version" 2>/dev/null || echo "0.0.0")

##@ General

help: ## Display this help message
	@echo "$(BLUE)Yaak PII Detection - Development Commands$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make $(GREEN)<target>$(NC)\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BLUE)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

info: ## Show project info
	@echo "$(BLUE)Project Information$(NC)"
	@echo "Name:    $(GREEN)yaak-pii-detection$(NC)"
	@echo -n "Version: $(GREEN)"
	@cd src/frontend && node -p "require('./package.json').version" 2>/dev/null || echo "unknown"
	@echo "$(NC)"
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

##@ Code Quality

format: ## Format code with ruff
	@echo "$(BLUE)Formatting code...$(NC)"
	uv run ruff format .
	@echo "$(GREEN)✅ Code formatted$(NC)"

lint: ## Run linters with ruff
	@echo "$(BLUE)Running linters...$(NC)"
	uv run ruff check model/
	@echo "$(GREEN)✅ Linting complete$(NC)"

lint-go: ## Lint Go code with golangci-lint
	@echo "$(BLUE)Linting Go code...$(NC)"
	@if command -v golangci-lint >/dev/null 2>&1; then \
		golangci-lint run; \
	else \
		echo "$(YELLOW)⚠️  golangci-lint not found. Install with: go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)✅ Go linting complete$(NC)"

lint-frontend: ## Lint frontend code with ESLint
	@echo "$(BLUE)Linting frontend code...$(NC)"
	@if [ ! -d "src/frontend/node_modules" ]; then \
		echo "$(YELLOW)⚠️  Frontend dependencies not installed. Run 'make electron-install' first.$(NC)"; \
		exit 1; \
	fi
	@cd src/frontend && npm run lint
	@echo "$(GREEN)✅ Frontend linting complete$(NC)"

lint-frontend-fix: ## Lint and auto-fix frontend code with ESLint
	@echo "$(BLUE)Auto-fixing frontend code...$(NC)"
	@if [ ! -d "src/frontend/node_modules" ]; then \
		echo "$(YELLOW)⚠️  Frontend dependencies not installed. Run 'make electron-install' first.$(NC)"; \
		exit 1; \
	fi
	@cd src/frontend && npm run lint:fix
	@echo "$(GREEN)✅ Frontend auto-fix complete$(NC)"

lint-all: lint lint-go lint-frontend ## Run all linters (Python, Go, Frontend)
	@echo "$(GREEN)✅ All linting complete$(NC)"

typecheck: ## Run type checker with ruff
	@echo "$(BLUE)Running type checker...$(NC)"
	uv run ruff check model/ --select TYP
	@echo "$(GREEN)✅ Type checking complete$(NC)"

typecheck-frontend: ## Run TypeScript type checking
	@echo "$(BLUE)Running TypeScript type checker...$(NC)"
	@if [ ! -d "src/frontend/node_modules" ]; then \
		echo "$(YELLOW)⚠️  Frontend dependencies not installed. Run 'make electron-install' first.$(NC)"; \
		exit 1; \
	fi
	@cd src/frontend && npm run type-check
	@echo "$(GREEN)✅ TypeScript type checking complete$(NC)"

check: format lint typecheck ## Run Python code quality checks

check-all: format lint-all typecheck typecheck-frontend ## Run all code quality checks (Python, Go, Frontend)

ruff-fix: ## Auto-fix ruff issues
	@echo "$(BLUE)Auto-fixing ruff issues...$(NC)"
	uv run ruff check model/ --fix
	@echo "$(GREEN)✅ Auto-fix complete$(NC)"

ruff-all: ## Run all ruff checks (lint + format + typecheck)
	@echo "$(BLUE)Running all ruff checks...$(NC)"
	uv run ruff check model/ --fix
	uv run ruff format .
	@echo "$(GREEN)✅ All ruff checks complete$(NC)"

##@ Testing

test-python: ## Run Python tests
	@echo "$(BLUE)Running Python tests...$(NC)"
	uv run pytest tests/ -v
	@echo "$(GREEN)✅ Tests complete$(NC)"

test-go: ## Run Go tests
	@echo "$(BLUE)Running Go tests...$(NC)"
	go test ./... -v
	@echo "$(GREEN)✅ Go tests complete$(NC)"

test-all: test test-go ## Run all tests (Python, Go)
	@echo "$(GREEN)✅ All tests complete$(NC)"

##@ Cleanup

clean: ## Remove build artifacts and cache
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	find . -type d \( -name "__pycache__" -o -name "*.egg-info" -o -name ".pytest_cache" -o -name ".mypy_cache" \) -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf build/ *.egg-info
	@echo "$(GREEN)✅ Cleanup complete$(NC)"

clean-venv: ## Remove virtual environment
	@echo "$(BLUE)Removing virtual environment...$(NC)"
	rm -rf .venv
	@echo "$(GREEN)✅ Virtual environment removed$(NC)"

clean-all: clean clean-venv ## Remove everything (artifacts, cache, and venv)
	@echo "$(GREEN)✅ Full cleanup complete$(NC)"

##@ Electron

electron-install: ## Install Electron UI dependencies
	@echo "$(BLUE)Installing Electron UI dependencies...$(NC)"
	@cd src/frontend && npm install
	@echo "$(GREEN)✅ Electron dependencies installed$(NC)"

build-go: ## Build Go binary for development
	@echo "$(BLUE)Building Go binary for development...$(NC)"
	@mkdir -p build
	@CGO_ENABLED=1 \
	go build \
	  -ldflags="-extldflags '-L./build/tokenizers'" \
	  -o build/yaak-proxy \
	  ./src/backend
	@echo "$(GREEN)✅ Go binary built at build/yaak-proxy$(NC)"

electron-build: ## Build Electron app for production
	@echo "$(BLUE)Building Electron app...$(NC)"
	@cd src/frontend && npm run build:electron
	@echo "$(GREEN)✅ Electron app built$(NC)"

electron-run: electron-build ## Run Electron app (builds first)
	@echo "$(BLUE)Starting Electron app...$(NC)"
	@cd src/frontend && npm run electron

electron: electron-run ## Alias for electron-run

electron-dev: ## Run Electron app in development mode (assumes backend is running in debugger)
	@echo "$(BLUE)Building frontend for Electron...$(NC)"
	@cd src/frontend && npm run build:electron
	@echo "$(GREEN)✅ Frontend built$(NC)"
	@echo "$(BLUE)Starting Electron in development mode...$(NC)"
	@echo "$(YELLOW)Note: Assumes Go backend is running separately (e.g., in VSCode debugger)$(NC)"
	@echo "$(YELLOW)Note: Run 'npm run dev' in another terminal for hot reload$(NC)"
	@cd src/frontend && EXTERNAL_BACKEND=true npm run electron:dev

electron-dev-external: electron-dev ## Alias for electron-dev (for backwards compatibility)

##@ Build

build-dmg: ## Build DMG package with Go binary and Electron app
	@echo "$(BLUE)Building DMG package...$(NC)"
	@if [ ! -f "src/scripts/build_dmg.sh" ]; then \
		echo "$(YELLOW)⚠️  build_dmg.sh script not found$(NC)"; \
		exit 1; \
	fi
	@chmod +x src/scripts/build_dmg.sh
	@./src/scripts/build_dmg.sh
	@echo "$(GREEN)✅ DMG build complete$(NC)"
