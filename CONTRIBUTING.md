# Contributing to kiji-proxy

Thank you for your interest in contributing!

## Prerequisites

- [Go](https://golang.org/) ≥ 1.21
- [Node.js](https://nodejs.org/) ≥ 18
- [npm](https://www.npmjs.com/)

## Dev Setup

```bash
git clone https://github.com/dataiku/kiji-proxy.git
cd kiji-proxy
npm install
```

## Running Tests

**Backend (Go):**
```bash
cd src/backend
go test ./...
```

**Frontend:**
```bash
cd src/frontend
npm test
```

## Branching & PR Conventions

- Branch names: `feat/<short-description>`, `fix/<short-description>`, `chore/<short-description>`
- One issue per PR — keeps reviews focused and merges fast
- Reference the issue in your PR description: `Closes #<number>`
- Keep commits atomic and descriptive

## Code Style

**Go:** Follow standard `gofmt` formatting. Run `go vet ./...` before pushing.

**TypeScript/React:** Follow the existing ESLint configuration in the project root.
