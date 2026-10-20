#!/usr/bin/env node
/**
 * Checks that the native binary pins in src/frontend/package.json
 * optionalDependencies (#583) match the version of the package that loads
 * them, as resolved in the root package-lock.json:
 *
 *   lightningcss-*        must equal node_modules/lightningcss
 *   @tailwindcss/oxide-*  must equal node_modules/@tailwindcss/oxide
 *
 * A pin at any other version is installed as an unused nested copy, and the
 * binary the build actually loads is no longer protected by the pin.
 *
 * Usage: check-native-pins.js [package.json] [package-lock.json]
 */
"use strict";

const fs = require("node:fs");
const path = require("node:path");

const repoRoot = path.resolve(__dirname, "..", "..");
const pkgPath = process.argv[2] || path.join(repoRoot, "src", "frontend", "package.json");
const lockPath = process.argv[3] || path.join(repoRoot, "package-lock.json");

const PARENTS = [
  { prefix: "lightningcss-", parent: "lightningcss" },
  { prefix: "@tailwindcss/oxide-", parent: "@tailwindcss/oxide" },
];

const pins = JSON.parse(fs.readFileSync(pkgPath, "utf8")).optionalDependencies || {};
const lockPackages = JSON.parse(fs.readFileSync(lockPath, "utf8")).packages || {};

const errors = [];
let checked = 0;

for (const [name, spec] of Object.entries(pins)) {
  const rule = PARENTS.find((r) => name.startsWith(r.prefix));
  if (!rule) continue;
  checked++;
  const parent = lockPackages[`node_modules/${rule.parent}`];
  if (!parent) {
    errors.push(`${name}: ${rule.parent} is not in ${path.basename(lockPath)}`);
  } else if (spec !== parent.version) {
    errors.push(`${name} is pinned to "${spec}" but ${rule.parent} resolves to ${parent.version}`);
  }
}

if (errors.length > 0) {
  console.error("check-native-pins: native binary pins are out of sync:");
  for (const e of errors) console.error(`  - ${e}`);
  console.error(
    `Set these optionalDependencies in ${path.relative(repoRoot, pkgPath)} to the ` +
      "version shown, then run `npm install` from the repo root."
  );
  process.exit(1);
}

console.log(`check-native-pins: ${checked} native binary pins match their parent packages`);
