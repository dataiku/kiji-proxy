## Description

<!-- Describe what this PR does and why. -->

## Linked Issue

Closes #

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Refactor / code quality
- [ ] Documentation update
- [ ] CI / build change

## Testing Done

<!-- Describe how you tested the changes. Include commands run and results observed. -->

**Unit tests (Go):**
```
make test-go
```

**Unit tests (Python):**
```
make test
```

**Linting & type checks:**
```
make lint
make typecheck
```

**Manual testing (if applicable):**
- Describe the scenario tested (e.g. PII type detected, proxy request flow, Chrome extension behavior)
- Include before/after behavior if fixing a bug
- Note any edge cases verified (e.g. nested PII in JSON payloads, restoration accuracy)

## Checklist

- [ ] `make test-go` and `make test` pass locally
- [ ] `make lint` reports no new errors
- [ ] New PII detection changes are covered by unit tests in `tests/`
- [ ] Go proxy changes include test coverage in `src/` or `model/`
- [ ] Documentation updated if behavior or configuration changed
- [ ] No sensitive data or credentials included
