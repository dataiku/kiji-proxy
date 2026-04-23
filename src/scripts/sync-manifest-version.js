#!/usr/bin/env node
/**
 * Syncs chrome-extension/manifest.json `version` to src/frontend/package.json.
 * Runs after `changeset version` so the manifest always tracks the app.
 */
"use strict";

const fs = require("node:fs");
const path = require("node:path");

const repoRoot = path.resolve(__dirname, "..", "..");
const source = path.join(repoRoot, "src", "frontend", "package.json");
const target = path.join(repoRoot, "chrome-extension", "manifest.json");

const sourcePkg = JSON.parse(fs.readFileSync(source, "utf8"));
const version = sourcePkg.version;
if (!version) {
  console.error(`sync-manifest-version: no version in ${source}`);
  process.exit(1);
}

const manifestRaw = fs.readFileSync(target, "utf8");
const manifest = JSON.parse(manifestRaw);

if (manifest.version === version) {
  console.log(`sync-manifest-version: manifest already at ${version}`);
  process.exit(0);
}

const previous = manifest.version;
manifest.version = version;

// Preserve the source file's trailing newline (if any) and 2-space indentation
// used by the rest of the repo's JSON.
const trailing = manifestRaw.endsWith("\n") ? "\n" : "";
fs.writeFileSync(target, JSON.stringify(manifest, null, 2) + trailing);

console.log(
  `sync-manifest-version: ${target.replace(repoRoot + path.sep, "")} ${previous} → ${version}`
);
