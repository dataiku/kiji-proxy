# Contributing to kiji-proxy

Thanks for your interest in contributing. This document covers how to get set up, run tests, and submit changes.

## Dev Setup

**Prerequisites:** Go 1.21+, Node.js 18+, make (optional but handy).

```bash
# Clone your fork
git clone https://github.com/<your-username>/kiji-proxy.git
cd kiji-proxy

# Install backend dependencies
cd src/backend && go mod download

# Install frontend dependencies
cd ../../src/frontend && npm install
```

Copy and edit the development config before running:

```bash
cp src/backend/config/config.development.json config.json
# Edit config.json with your API keys / settings
```

Start the backend:

```bash
go run ./src/backend/cmd/...
```

Start the frontend (separate terminal):

```bash
cd src/frontend && npm run dev
```

## Running Tests

```bash
# Backend unit tests
cd src/backend && go test ./...

# Backend tests with race detector
go test -race ./...

# Frontend tests
cd src/frontend && npm test
```

## Branching & PR Conventions

- Branch off `main`. Use short, descriptive names:
  - `feat/<topic>` for new features
  - `fix/<topic>` for bug fixes
  - `docs/<topic>` for documentation-only changes
- Keep commits focused; one logical change per commit.
- Reference the issue your PR addresses with `Closes #<issue>` in the PR description.
- All CI checks must pass before a PR is merged.
- PRs require at least one approving review.

## Code Style

**Go:**
- Run `gofmt -w .` and `go vet ./...` before committing.
- Follow standard Go naming conventions.
- Add comments for exported types, functions, and constants.

**TypeScript / Frontend:**
- Run `npm run lint` before committing.
- Prefer explicit types over `any`.

## Reporting Issues

Use the GitHub issue tracker. Please search for existing issues before opening a new one. Include reproduction steps, expected vs. actual behavior, and relevant logs.
