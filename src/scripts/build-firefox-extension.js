#!/usr/bin/env node
/** Package the shared extension with Firefox's Manifest V3 background page. */
"use strict";

const fs = require("node:fs");
const path = require("node:path");

const repoRoot = path.resolve(__dirname, "../..");
const sourceDir = path.join(repoRoot, "chrome-extension");
const args = process.argv.slice(2);
let outputDir = path.join(repoRoot, "dist/firefox-extension");
let version;

for (let i = 0; i < args.length; i += 2) {
  const name = args[i];
  const value = args[i + 1];
  if (!["--output", "--version"].includes(name) || !value || value.startsWith("--")) {
    throw new Error("Usage: build-firefox-extension.js [--output directory] [--version version]");
  }
  if (name === "--output") outputDir = path.resolve(value);
  else version = value;
}

// Never replace the source directory, one of its parents, or a source subdirectory.
const sourceRelative = path.relative(sourceDir, outputDir);
const outputRelative = path.relative(outputDir, sourceDir);
if (!sourceRelative.startsWith(`..${path.sep}`) || !outputRelative.startsWith(`..${path.sep}`)) {
  throw new Error("Output must be outside the extension source and its parent directories");
}
const manifest = JSON.parse(fs.readFileSync(path.join(sourceDir, "manifest.firefox.json"), "utf8"));
if (version) manifest.version = version;
if (!/^(0|[1-9]\d*)(\.(0|[1-9]\d*)){0,3}$/.test(manifest.version)) {
  throw new Error("Firefox extension version must contain one to four numeric components");
}

// Refuse to overwrite unrelated files in a caller-supplied directory.
if (fs.existsSync(outputDir) && fs.readdirSync(outputDir).length > 0) {
  const previousPath = path.join(outputDir, "manifest.json");
  const previous = fs.existsSync(previousPath) ? JSON.parse(fs.readFileSync(previousPath, "utf8")) : {};
  if (previous.browser_specific_settings?.gecko?.id !== manifest.browser_specific_settings.gecko.id) {
    throw new Error("Output directory is not an existing Kiji Firefox build");
  }
}
fs.mkdirSync(outputDir, { recursive: true });
fs.cpSync(sourceDir, outputDir, {
  recursive: true,
  filter: (entry) => !["manifest.json", "manifest.firefox.json", ".DS_Store"].includes(path.basename(entry)),
});
fs.writeFileSync(path.join(outputDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
console.log(`Firefox extension built at ${path.relative(repoRoot, outputDir)}`);
