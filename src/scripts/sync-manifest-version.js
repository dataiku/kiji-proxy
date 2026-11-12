#!/usr/bin/env node
/**
 * Syncs browser extension manifest versions to src/frontend/package.json.
 * Runs after `changeset version` so all extension packages track the app.
 */
"use strict";

const fs = require("node:fs");
const path = require("node:path");

const repoRoot = path.resolve(__dirname, "..", "..");
const source = path.join(repoRoot, "src", "frontend", "package.json");
const targets = [
  path.join(repoRoot, "chrome-extension", "manifest.json"),
  path.join(repoRoot, "chrome-extension", "manifest.firefox.json"),
];

const sourcePkg = JSON.parse(fs.readFileSync(source, "utf8"));
const version = sourcePkg.version;
if (!version) {
  console.error(`sync-manifest-version: no version in ${source}`);
  process.exit(1);
}

for (const target of targets) {
  const manifestRaw = fs.readFileSync(target, "utf8");
  const manifest = JSON.parse(manifestRaw);

  if (manifest.version === version) {
    console.log(
      `sync-manifest-version: ${target.replace(repoRoot + path.sep, "")} already at ${version}`
    );
    continue;
  }

  const previous = manifest.version;
  manifest.version = version;
  const trailing = manifestRaw.endsWith("\n") ? "\n" : "";
  fs.writeFileSync(target, JSON.stringify(manifest, null, 2) + trailing);
  console.log(
    `sync-manifest-version: ${target.replace(repoRoot + path.sep, "")} ${previous} → ${version}`
  );
}
