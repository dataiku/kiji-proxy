#!/usr/bin/env node
/** Package the shared extension with Firefox's Manifest V3 background page. */
"use strict";

const fs = require("node:fs");
const path = require("node:path");

const repoRoot = path.resolve(__dirname, "../..");
const sourceDir = path.join(repoRoot, "browser-extension");
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
const releaseVersion = version || manifest.version;
// Firefox/AMO accepts up to four numeric components, each at most nine digits.
// Reserve the fourth component for the release channel so alpha < beta < rc
// < stable, including updates from the last prerelease to the stable release.
const semver = /^(0|[1-9]\d{0,8})\.(0|[1-9]\d{0,8})\.(0|[1-9]\d{0,8})(?:-(alpha|beta|rc)(?:\.(0|[1-9]\d{0,7}))?)?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$/;
const match = releaseVersion.match(semver);
if (match) {
  const [, major, minor, patch, channel, sequence] = match;
  // Unnumbered alpha precedes alpha.0. Leave room for numbered prereleases
  // without overlapping the next channel's range.
  const ordinal = sequence === undefined ? 0 : Number(sequence) + 1;
  if (ordinal >= 100_000_000) {
    throw new Error("Prerelease sequence must be between 0 and 99999998");
  }
  const channelBase = { alpha: 100_000_000, beta: 200_000_000, rc: 300_000_000 };
  const revision = channel ? channelBase[channel] + ordinal : 400_000_000;
  manifest.version = [major, minor, patch, revision].join(".");
} else if (/^(0|[1-9]\d{0,8})(\.(0|[1-9]\d{0,8})){0,3}$/.test(releaseVersion)) {
  // Preserve explicit one-, two-, or four-component Firefox version overrides.
  manifest.version = releaseVersion;
} else {
  throw new Error("Expected a numeric version or a SemVer alpha, beta, rc, or stable release");
}
manifest.version_name = releaseVersion;

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
