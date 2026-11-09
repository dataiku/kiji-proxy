const assert = require("node:assert/strict");
const { spawnSync } = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");

const repoRoot = path.resolve(__dirname, "../..");
const scriptsDir = path.join(repoRoot, "src/scripts");
const frontendDir = path.join(repoRoot, "src/frontend");
// Resolve ESLint the way the frontend workspace does (deps may be hoisted).
const eslintBin = path.join(
  path.dirname(require.resolve("eslint/package.json", { paths: [frontendDir] })),
  "bin/eslint.js",
);

test("the lint script also lints src/scripts", () => {
  const pkg = JSON.parse(fs.readFileSync(path.join(frontendDir, "package.json"), "utf8"));
  assert.match(pkg.scripts.lint, /\.\.\/scripts/);
  assert.match(pkg.scripts["lint:fix"], /\.\.\/scripts/);
});

test("the helper-script ESLint config catches real problems only", () => {
  const fixtureDir = fs.mkdtempSync(path.join(scriptsDir, "lint-fixture-"));
  const cleanup = () => fs.rmSync(fixtureDir, { recursive: true, force: true });
  // Make sure an interrupted run does not leave a lint-failing fixture behind.
  process.once("exit", cleanup);
  for (const signal of ["SIGINT", "SIGTERM"]) {
    process.once(signal, () => {
      cleanup();
      process.exit(1);
    });
  }

  const fixture = path.join(fixtureDir, "helper.js");
  // Lint only the fixture directory so unrelated repo lint errors cannot
  // change the outcome, and read structured output instead of formatter text.
  const lint = (source) => {
    fs.writeFileSync(fixture, source);
    const result = spawnSync(process.execPath, [eslintBin, "--format", "json", fixtureDir], {
      cwd: scriptsDir,
      encoding: "utf8",
      timeout: 60000,
    });
    assert.ifError(result.error);
    const report = JSON.parse(result.stdout);
    return report.flatMap((file) => file.messages);
  };

  try {
    assert.deepEqual(
      lint("module.exports = missingHelperIdentifier;\n").map((m) => [m.ruleId, m.message]),
      [["no-undef", "'missingHelperIdentifier' is not defined."]],
    );

    assert.deepEqual(
      lint("module.exports = window.document;\n").map((m) => m.ruleId),
      ["no-undef"],
    );

    // Generated output must remain excluded even when it contains invalid names.
    fs.mkdirSync(path.join(fixtureDir, "dist"));
    fs.writeFileSync(path.join(fixtureDir, "dist/generated.js"), "generatedOnlyIdentifier;\n");

    // Valid CommonJS using Node globals and modern syntax must pass.
    assert.deepEqual(
      lint(
        'const path = require("node:path");\n' +
          "let cache;\ncache ??= new URL(`file://${__dirname}`);\n" +
          "module.exports = [path.join(__dirname, process.platform), cache, fetch, 1_000];\n",
      ),
      [],
    );
  } finally {
    cleanup();
  }
});
