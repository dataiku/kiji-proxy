# Changesets

This directory contains changesets - human-readable files that describe changes to the project.

## How to use changesets

When you make a change that should trigger a release:

1. Run `npx changeset` (or `npm run changeset` if configured)
2. Select the type of change (major, minor, patch)
3. Write a description of your change
4. Commit the generated changeset file

When a release PR is merged to main, the version will be automatically bumped and a DMG will be built.

## Versioning

We follow [Semantic Versioning](https://semver.org/):

- **Major** (x.0.0): Breaking changes
- **Minor** (0.x.0): New features (backwards compatible)
- **Patch** (0.0.x): Bug fixes

## More Information

- [Changesets Documentation](https://github.com/changesets/changesets)
- [Release Process](../docs/RELEASE.md)
