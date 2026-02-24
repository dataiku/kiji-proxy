.PHONY: help install install-dev venv
.PHONY: format lint lint-go lint-frontend lint-frontend-fix lint-all typecheck typecheck-frontend check check-all ruff-fix ruff-all
.PHONY: test test-go test-all
.PHONY: clean clean-venv clean-all
.PHONY: build-dmg build-linux verify-linux
.PHONY: electron-build electron-run electron electron-dev electron-install
.PHONY: list show shell jupyter info quickstart
.PHONY: pr

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

# Go build flags
CGO_LDFLAGS := -L./build/tokenizers

# Version from package.json
VERSION := $(shell cd src/frontend && node -p "require('./package.json').version" 2>/dev/null || echo "0.0.0")

##@ General

help: ## Display this help message
	@echo "$(BLUE)Kiji PII Detection - Development Commands$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make $(GREEN)<target>$(NC)\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BLUE)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

info: ## Show project info
	@echo "$(BLUE)Project Information$(NC)"
	@echo "Name:    $(GREEN)kiji-pii-detection$(NC)"
	@echo -n "Version: $(GREEN)"
	@cd src/frontend && node -p "require('./package.json').version" 2>/dev/null || echo "unknown"
	@echo "$(NC)"
	@echo "Python:  $(GREEN)$(shell .venv/bin/python --version 2>&1 || echo "Not installed (run 'make venv')")$(NC)"
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
	@echo "$(BLUE)Version Information$(NC)"
	@echo "  Version is managed in src/frontend/package.json"
	@echo "  Backend receives version via ldflags during build"
	@echo "  Check version: ./build/kiji-proxy --version"
	@echo "  API endpoint: http://localhost:8080/version"
	@echo ""
	@echo "$(BLUE)Quick Commands$(NC)"
	@echo "  make install     - Install dependencies"
	@echo "  make test        - Run tests"
	@echo "  make help        - Show all commands"

##@ Git & PR

pr: ## Generate semantic PR title + summary with Claude Code and create the PR
	@if gh pr view --json number -q .number >/dev/null 2>&1; then \
		echo "$(YELLOW)⚠️  A PR already exists for this branch:$(NC)"; \
		gh pr view --json url -q .url; \
		exit 1; \
	fi; \
	CURRENT=$$(git branch --show-current); \
	if [ "$$CURRENT" = "main" ] || [ "$$CURRENT" = "master" ]; then \
		echo "$(YELLOW)⚠️  You are on $$CURRENT — switch to a feature branch first$(NC)"; \
		exit 1; \
	fi; \
	BASE=$$(gh repo view --json defaultBranchRef -q .defaultBranchRef.name 2>/dev/null || echo "main"); \
	COMMITS=$$(git log --oneline "refs/heads/$$BASE"..HEAD); \
	if [ -z "$$COMMITS" ]; then \
		echo "$(YELLOW)⚠️  No commits found between $$BASE and HEAD$(NC)"; \
		exit 1; \
	fi; \
	DIFF_STAT=$$(git diff --stat "refs/heads/$$BASE"...HEAD); \
	echo "$(BLUE)Analyzing $$(echo "$$COMMITS" | wc -l | tr -d ' ') commits against $$BASE...$(NC)"; \
	RESULT=$$(echo "$$COMMITS" | npx --yes @anthropic-ai/claude-code -p "You are analyzing commits for a GitHub pull request that uses squash-and-merge. \
The PR title becomes the final commit message and MUST follow the Conventional Commits format. \
\
Semantic types: \
- feat: new feature for the user \
- fix: bug fix for the user \
- docs: documentation changes \
- style: formatting, no code change \
- refactor: refactoring production code \
- test: adding/refactoring tests \
- chore: maintenance, deps, config \
- ci: CI/CD changes \
- perf: performance improvements \
\
Step 1: Classify each commit into exactly one semantic type. \
Step 2: Check if the commits span MULTIPLE semantic types. \
\
If commits span multiple types, output EXACTLY this format (no other text): \
MIXED \
<type1>: <list of commits> \
<type2>: <list of commits> \
... \
\
If all commits fit a single type, output EXACTLY this format (no other text): \
TITLE: <type>: <short summary in present tense, lowercase, under 70 chars> \
BODY: \
## Summary \
<2-5 bullet points describing the changes> \
\
## Changes \
<bulleted list of specific changes, grouped logically> \
\
Here are the commits: \
$$COMMITS \
\
Diff stat: \
$$DIFF_STAT"); \
	if echo "$$RESULT" | head -1 | grep -q "^MIXED"; then \
		echo ""; \
		echo "$(YELLOW)⚠️  This PR spans multiple semantic types — consider splitting into separate PRs:$(NC)"; \
		echo ""; \
		echo "$$RESULT" | tail -n +2; \
		echo ""; \
		echo "$(YELLOW)Tip: Use interactive rebase or cherry-pick to split commits into focused PRs.$(NC)"; \
		exit 1; \
	fi; \
	TITLE=$$(echo "$$RESULT" | head -1 | sed 's/^TITLE: //'); \
	BODY=$$(echo "$$RESULT" | tail -n +2 | sed 's/^BODY://'); \
	echo ""; \
	echo "$(GREEN)Title:$(NC) $$TITLE"; \
	echo "$(GREEN)Body:$(NC)"; \
	echo "$$BODY"; \
	echo ""; \
	printf "$(BLUE)Create PR? [y/N] $(NC)"; \
	read -r CONFIRM; \
	if [ "$$CONFIRM" = "y" ] || [ "$$CONFIRM" = "Y" ]; then \
		git push -u origin "$$CURRENT"; \
		gh pr create --title "$$TITLE" --body "$$BODY"; \
	else \
		echo "$(YELLOW)PR not created.$(NC)"; \
	fi

##@ Setup & Installation

venv: ## Create virtual environment with uv
	@echo "$(BLUE)Creating virtual environment with Python 3.13...$(NC)"
	uv venv --python 3.13
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

lint-frontend-fix: ## Lint and auto-fix frontend code with ESLint, then run type check
	@echo "$(BLUE)Auto-fixing frontend code...$(NC)"
	@if [ ! -d "src/frontend/node_modules" ]; then \
		echo "$(YELLOW)⚠️  Frontend dependencies not installed. Run 'make electron-install' first.$(NC)"; \
		exit 1; \
	fi
	@cd src/frontend && npm run lint:fix
	@echo "$(GREEN)✅ Frontend auto-fix complete$(NC)"
	@echo "$(BLUE)Running TypeScript type check...$(NC)"
	@cd src/frontend && npm run type-check
	@echo "$(GREEN)✅ TypeScript type check complete$(NC)"

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
	CGO_LDFLAGS="$(CGO_LDFLAGS)" go test ./... -v
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

setup-onnx: ## Set up ONNX Runtime library for development
	@echo "$(BLUE)Setting up ONNX Runtime library...$(NC)"
	@mkdir -p build
	@if [ -f "build/libonnxruntime.1.24.2.dylib" ]; then \
		echo "$(GREEN)✅ ONNX library already exists$(NC)"; \
	elif [ -f "src/frontend/resources/libonnxruntime.1.24.2.dylib" ]; then \
		ln -sf ../src/frontend/resources/libonnxruntime.1.24.2.dylib build/libonnxruntime.1.24.2.dylib; \
		echo "$(GREEN)✅ Linked existing ONNX library from resources$(NC)"; \
	else \
		if [ ! -d ".venv" ]; then \
			echo "$(YELLOW)Creating virtual environment with Python 3.13...$(NC)"; \
			uv venv --python 3.13; \
		fi; \
		uv pip install --quiet onnxruntime==1.24.2; \
		ONNX_LIB=$$(find .venv -name "libonnxruntime*.dylib" | head -1); \
		if [ -n "$$ONNX_LIB" ]; then \
			cp "$$ONNX_LIB" build/libonnxruntime.1.24.2.dylib; \
			echo "$(GREEN)✅ ONNX library installed$(NC)"; \
		else \
			echo "$(YELLOW)⚠️  Could not find ONNX library, continuing anyway$(NC)"; \
		fi; \
	fi

build-go: ## Build Go binary for development
	@echo "$(BLUE)Building Go binary for development...$(NC)"
	@mkdir -p build
	@CGO_ENABLED=1 \
	CGO_LDFLAGS="$(CGO_LDFLAGS)" \
	go build \
	  -ldflags="-X main.version=$(VERSION) -extldflags '$(CGO_LDFLAGS)'" \
	  -o build/kiji-proxy \
	  ./src/backend
	@echo "$(GREEN)✅ Go binary built at build/kiji-proxy (version $(VERSION))$(NC)"

electron-build: ## Build Electron app for production
	@echo "$(BLUE)Building Electron app...$(NC)"
	@cd src/frontend && npm run build:electron
	@echo "$(GREEN)✅ Electron app built$(NC)"

electron-run: setup-onnx build-go electron-build ## Run Electron app (builds Go binary and frontend first)
	@echo "$(BLUE)Preparing resources for Electron...$(NC)"
	@mkdir -p src/frontend/resources
	@cp build/kiji-proxy src/frontend/resources/kiji-proxy
	@chmod +x src/frontend/resources/kiji-proxy
	@if [ -f "build/libonnxruntime.1.24.2.dylib" ]; then \
		cp build/libonnxruntime.1.24.2.dylib src/frontend/resources/libonnxruntime.1.24.2.dylib; \
		echo "$(GREEN)✅ ONNX library copied to resources$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  ONNX library not found at build/libonnxruntime.1.24.2.dylib$(NC)"; \
	fi
	@echo "$(GREEN)✅ Resources prepared$(NC)"
	@echo "$(BLUE)Starting Electron app...$(NC)"
	@cd src/frontend && NODE_ENV=development npm run electron

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

go-backend-dev: ## Run Go backend in development mode
	@echo "$(BLUE)Running Go backend in development mode...$(NC)"
	CGO_LDFLAGS="$(CGO_LDFLAGS)" go run ./src/backend -config ./src/backend/config/config.development.json


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

build-linux: ## Build Linux standalone binary (without Electron)
	@echo "$(BLUE)Building Linux standalone binary...$(NC)"
	@if [ ! -f "src/scripts/build_linux.sh" ]; then \
		echo "$(YELLOW)⚠️  build_linux.sh script not found$(NC)"; \
		exit 1; \
	fi
	@chmod +x src/scripts/build_linux.sh
	@./src/scripts/build_linux.sh
	@echo "$(GREEN)✅ Linux build complete$(NC)"

verify-linux: ## Verify Linux build includes all required files (tokenizer, model, etc.)
	@echo "$(BLUE)Verifying Linux build...$(NC)"
	@if [ ! -f "src/scripts/verify_linux_build.sh" ]; then \
		echo "$(YELLOW)⚠️  verify_linux_build.sh script not found$(NC)"; \
		exit 1; \
	fi
	@chmod +x src/scripts/verify_linux_build.sh
	@./src/scripts/verify_linux_build.sh
