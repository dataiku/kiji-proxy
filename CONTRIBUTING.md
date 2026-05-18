# Contributing to Kiji Privacy Proxy

## Table of contents

- [Welcome](#welcome)
- [Code of conduct](#code-of-conduct)
- [How can I contribute?](#how-can-i-contribute)
  - [Reporting bugs](#reporting-bugs)
  - [Suggesting enhancements](#suggesting-enhancements)
  - [Your first code contribution](#your-first-code-contribution)
  - [Other ways to contribute](#other-ways-to-contribute)
- [Development setup](#development-setup)
- [Making changes](#making-changes)
  - [Branching](#branching)
  - [Commit messages](#commit-messages)
  - [Changesets](#changesets)
  - [Tests](#tests)
  - [Style guide](#style-guide)
- [Pull request process](#pull-request-process)
- [Code review](#code-review)
- [Contributor License Agreement (CLA)](#contributor-license-agreement-cla)
- [Security issues](#security-issues)
- [Communication](#communication)
- [Recognition](#recognition)
- [License](#license)

---

## Welcome

Thanks for your interest in contributing to **Kiji Privacy Proxy** — an intelligent privacy layer for AI APIs, built by [575 Lab](https://www.dataiku.com/company/dataiku-for-the-future/open-source/), Dataiku's Open Source Office. From commenting on and triaging issues, to reviewing and sending pull requests, all contributions are welcome. There is no contribution too small.

Please take a moment to review this document. Following these guidelines helps communicate that you respect the time of the people maintaining the project. In return, we'll do our best to respect yours when addressing your issue, reviewing your patch, or evaluating your idea.

---

## Code of conduct

We expect everyone participating in this project to be respectful, inclusive, and constructive. Harassment or abuse of any kind will not be tolerated. Please report unacceptable behaviour to **opensource@dataiku.com**.

---

## How can I contribute?

### Reporting bugs

Before opening a new issue:

1. **Check you're on the latest version.** Many issues are fixed in newer releases — see [GitHub Releases](https://github.com/dataiku/kiji-proxy/releases).
2. **Search existing issues** (open and closed) at [github.com/dataiku/kiji-proxy/issues](https://github.com/dataiku/kiji-proxy/issues).
3. **If the bug involves a security risk**, do **not** file a public issue — see [Security issues](#security-issues).

When opening a bug report, please use our [bug report template](.github/ISSUE_TEMPLATE/10_bug_report.yml) and include:

- **Kiji Privacy Proxy version** (e.g. `v1.0.5`) — visible in the desktop app or `./build/kiji-proxy --version`.
- **Operating system and architecture** (e.g. macOS Tahoe 26.2 on aarch64, Ubuntu 22.04 on x86_64).
- **Model provider details** — which provider (OpenAI, Anthropic, Gemini, Mistral) and model (e.g. GPT-5, Claude Sonnet 4.5).
- **Reproduction steps** from a clean install.
- **Current vs. expected behaviour** — screenshots and videos are very welcome.
- **Logs or stack traces**, copy-pasted as text (not screenshots).

The more context you provide, the easier it is to fix the problem fast.

### Suggesting enhancements

Feature requests live in [GitHub Discussions](https://github.com/dataiku/kiji-proxy/discussions/new/choose), not Issues. When proposing an enhancement, describe:

- **The problem you're trying to solve** (not just the solution you have in mind).
- **Who it affects** — end users, developers integrating with the proxy, model trainers, etc.
- **Alternatives you considered.**

For larger changes — new PII types, new LLM providers, architectural changes — please open a discussion *before* writing code so we can align on the design. See [Adding a New LLM Provider](docs/02-development-guide.md#adding-a-new-llm-provider) for the relevant interfaces and touchpoints.

### Your first code contribution

Unsure where to begin? Look for issues tagged:

- **`good first issue`** — small, well-scoped tasks for newcomers.
- **`help wanted`** — issues we'd love community help on.
- **`documentation`** — improvements to docs or examples are a great first PR.

If an issue interests you, leave a comment so we know you're working on it and can offer guidance. Don't hesitate to ask questions — in the issue thread, in [GitHub Discussions](https://github.com/dataiku/kiji-proxy/discussions), or on our [Slack community](https://join.slack.com/t/dataiku-opensource/shared_invite/zt-3o6yq14rp-FTtAHZYhyru~jLZ~S6xPLA).

### Other ways to contribute

You don't have to write code to be a valued contributor:

- **Triage issues** — reproduce bugs, ask for missing information, add labels.
- **Review pull requests** — even non-maintainer review is highly valuable.
- **Improve documentation** — the guides in [`docs/`](docs/), tutorials, examples.
- **Improve the ML model** — contribute training samples or new PII types to the [HuggingFace dataset](https://huggingface.co/datasets/DataikuNLP/kiji-pii-training-data). See [Customizing the PII Model](docs/07-customizing-pii-model.md).
- **Answer questions** in Discussions or on Slack.
- **Spread the word** — talks, blog posts, screencasts.

---

## Development setup

Kiji is a multi-language project: a **Go 1.25+** backend, a **TypeScript / React / Electron** desktop frontend, **Python 3.13+** (via `uv`) for the ML model and training pipeline, and a **Rust** toolchain for tokenizers. The desktop app on macOS and a standalone server on Linux are both supported.

```bash
# 1. Fork on GitHub, then clone your fork
git clone git@github.com:<your-username>/kiji-proxy.git
cd kiji-proxy

# 2. Add the upstream remote
git remote add upstream git@github.com:dataiku/kiji-proxy.git

# 3. Pull model files from Git LFS
git lfs pull

# 4. Install dependencies (Electron, tokenizers, ONNX Runtime)
make electron-install
make setup-tokenizers
make setup-onnx

# 5. Verify the setup
make check-all
```

Required tools:

- **Go 1.25+** with CGO enabled
- **Node.js 20+** and npm
- **Python 3.13+** managed by [`uv`](https://docs.astral.sh/uv/)
- **Rust** (latest stable)
- **Git LFS** (model files are LFS-tracked)

For step-by-step setup, debugging with VSCode, and a tour of the architecture, see the [Development Guide](docs/02-development-guide.md).

---

## Making changes

### Branching

- **Fork-and-pull model.** Push to a branch on your fork, then open a PR against `main`.
- Use **descriptive branch names**: `fix/restore-mapping-race`, `feat/add-mistral-provider`, `docs/clarify-pac-setup`.
- Rebase onto the latest `main` rather than merging it in — keeps history linear and reviewable.

### Commit messages

We follow [Conventional Commits](https://www.conventionalcommits.org/). The full guide lives in [`.github/semantic-commit-guide.md`](.github/semantic-commit-guide.md).

Format: `<type>(<scope>): <subject>`

Common types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`, `deps`.

Examples:

```
feat(providers): add Mistral support
fix(detector): handle empty token arrays from ONNX session
docs(getting-started): clarify PAC setup on macOS
deps(npm): bump electron-builder to 26.0.0
```

A `semantic-pr` workflow validates PR titles, so make sure your **PR title** also follows this format — that's what ends up in the changelog.

### Changesets

User-visible changes (features, fixes, breaking changes) require a [changeset](https://github.com/changesets/changesets). Changesets drive automated versioning and release notes.

```bash
# From the repo root
npm run changeset

# Pick the bump type (major / minor / patch) and write a short description.
# Commit the generated file under .changeset/ alongside your code changes.
```

Skip the changeset only for internal-only changes (CI, refactors with no behaviour change, dev-tooling updates). When in doubt, add one — patch-level is fine. See [Release Management](docs/04-release-management.md) for the full release flow.

### Tests

- **Add tests for new code.** If you fix a bug, add a regression test that fails before your fix and passes after.
- **Update tests when you change behaviour.** Update docs too if you change APIs or config.
- **Run the full suite locally** before pushing:

  ```bash
  make test-all       # Go + Python unit tests
  make check-all      # format + lint + type-check across Python, Go, frontend
  ```

- **CI must pass.** The [`Lint and Test`](.github/workflows/lint-and-test.yml) workflow runs on every PR — Python lint (`ruff`), Go lint (`golangci-lint`), Go tests, frontend ESLint + TypeScript check, and `actionlint` for workflow files. PRs with failing CI will not be merged.

The model evaluation harnesses (`make test-e2e`, `make test-benchmark`) are optional locally but useful when changing the detector or model.

### Style guide

- **Go:** `gofmt` formatting, `golangci-lint` clean. Run `make lint-go`.
- **TypeScript / React:** ESLint + Prettier conventions enforced by `npm run lint`. Type-check with `npm run type-check`. Run `make lint-frontend-fix` to auto-fix.
- **Python:** `ruff` for both linting and formatting. Run `make format` and `make lint`. **Always use `uv run python`** instead of bare `python` / `python3` — the project's Python env is managed by `uv`.
- **Rust:** `cargo fmt`, fix `cargo clippy` warnings.

We don't merge PRs that fail the formatter or linter — set up your editor to run them on save.

---

## Pull request process

Before submitting a PR, please confirm:

- [ ] You have forked the repo and branched from `main`.
- [ ] You've added tests for new code or bug fixes.
- [ ] You've updated relevant docs in [`docs/`](docs/).
- [ ] `make test-all` and `make check-all` pass locally.
- [ ] Your commit messages and **PR title** follow [Conventional Commits](#commit-messages).
- [ ] You've added a [changeset](#changesets) if the change is user-visible.
- [ ] You've signed the [CLA](#contributor-license-agreement-cla) (the bot will prompt you on your first PR).
- [ ] You've added yourself to [CONTRIBUTORS.md](CONTRIBUTORS.md) (first-time contributors).

When opening the PR:

1. **Link the issue** it addresses (`Closes #123`) in the description.
2. **Keep PRs focused.** One logical change per PR. Five small PRs review faster than one large one.
3. **Mark it as a draft** if it isn't ready for review yet.
4. **Watch the CI checks** — fix any failures before requesting review.

A maintainer will normally respond within a few business days. If you haven't heard back, feel free to ping the thread politely — we may simply have missed it.

---

## Code review

All submissions — including from maintainers — require review.

- **Be patient.** Maintainers review on a best-effort basis.
- **Address review comments in new commits** during the review so reviewers can see what changed. Squash at the end if asked.
- **Reviewers:** be kind, be specific, and explain *why*. Prefer asking questions over making demands.

---

## Contributor License Agreement (CLA)

Before we can accept your first contribution, you need to sign Dataiku's **Individual Contributor License Agreement**. The full text is available at [`.github/cla/CLA.md`](.github/cla/CLA.md).

This is a one-time, automated process:

1. Open your PR as normal.
2. The [CLA Assistant bot](.github/workflows/cla.yml) will comment on the PR with a link if you haven't signed yet.
3. Click the link and follow the instructions to sign (you sign by leaving a specific comment on the PR — there's no separate form).
4. Once signed, your signature is recorded and you won't need to sign again for future PRs.

If your contribution is made on behalf of a company that holds IP rights in your work, you'll need to ensure you have authorization to contribute — see section 6 of the CLA.

---

## Security issues

**Please do not file public GitHub issues for security vulnerabilities.**

Email **opensource@dataiku.com** with details. We'll acknowledge receipt and coordinate disclosure with you.

See [Security Best Practices](docs/05-advanced-topics.md#security-best-practices) for background on Kiji's security model.

---

## Communication

- **[GitHub Issues](https://github.com/dataiku/kiji-proxy/issues)** — bug reports.
- **[GitHub Discussions](https://github.com/dataiku/kiji-proxy/discussions)** — feature requests, questions, ideas, show-and-tell.
- **[Slack community](https://join.slack.com/t/dataiku-opensource/shared_invite/zt-3o6yq14rp-FTtAHZYhyru~jLZ~S6xPLA)** — real-time chat with maintainers and other contributors.

Keep project discussion **public** wherever possible. If you contact a maintainer privately with a non-sensitive question, we may politely redirect to a public channel so the answer benefits everyone.

---

## Recognition

Once your PR is merged, please add yourself to [CONTRIBUTORS.md](CONTRIBUTORS.md) — this includes contributions of every kind: code, docs, triage, model training data, and design. Open the PR with your changes, or include the `CONTRIBUTORS.md` update in the same PR as your contribution.

**Thank you in advance for your contribution!** Every fix, every doc tweak, every reproduction case helps make Kiji better for everyone.

---

## License

By contributing to Kiji Privacy Proxy, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE), and you confirm you have the right to submit the work under that licence (see the [CLA](#contributor-license-agreement-cla) for the formal terms).

Please **do not** add per-file copyright headers — they go stale as files accumulate contributors.
