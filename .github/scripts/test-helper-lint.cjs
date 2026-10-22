const assert = require("node:assert/strict");
const { spawnSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const repoRoot = path.resolve(__dirname, "../..");

test("the normal lint command checks Node helper scripts", () => {
  const fixtureDir = fs.mkdtempSync(path.join(repoRoot, "src/scripts/lint-fixture-"));
  const fixture = path.join(fixtureDir, "helper.js");
  const lint = () => spawnSync("npm", ["run", "lint"], {
    cwd: repoRoot,
    encoding: "utf8",
    timeout: 60000,
  });

  try {
    fs.writeFileSync(fixture, "module.exports = missingHelperIdentifier;\n");
    const undefinedName = lint();
    assert.ifError(undefinedName.error);
    assert.equal(undefinedName.status, 1, undefinedName.stdout + undefinedName.stderr);
    assert.match(undefinedName.stdout, /'missingHelperIdentifier' is not defined/);

    fs.writeFileSync(fixture, "module.exports = window.document;\n");
    const browserGlobal = lint();
    assert.ifError(browserGlobal.error);
    assert.equal(browserGlobal.status, 1, browserGlobal.stdout + browserGlobal.stderr);
    assert.match(browserGlobal.stdout, /'window' is not defined/);

    // Generated output must remain excluded even when it contains invalid names.
    fs.mkdirSync(path.join(fixtureDir, "dist"));
    fs.writeFileSync(path.join(fixtureDir, "dist/generated.js"), "generatedOnlyIdentifier;\n");
    fs.writeFileSync(fixture, 'const path = require("node:path");\nmodule.exports = path.join(__dirname, process.platform);\n');
    const commonJS = lint();
    assert.ifError(commonJS.error);
    assert.equal(commonJS.status, 0, commonJS.stdout + commonJS.stderr);
  } finally {
    fs.rmSync(fixtureDir, { recursive: true, force: true });
  }
});
