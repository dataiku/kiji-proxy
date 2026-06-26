#!/usr/bin/env node

/**
 * Dev orchestration for `make electron-dev`.
 *
 * Ensures a frontend webpack dev server is running on PORT, then launches
 * Electron against it. The dev server is only reused when it actually belongs
 * to this worktree — a bare HTTP 200 is not treated as proof, because a stale
 * server from a deleted git worktree keeps serving its last in-memory bundle
 * and would otherwise be silently adopted.
 *
 * Behaviour when something already listens on PORT:
 *   - cwd matches this worktree -> reuse it (announce pid + cwd)
 *   - cwd differs               -> refuse with guidance (exit 1)
 *   - cwd cannot be determined  -> reuse with a warning
 *   - FORCE_RESTART=1           -> kill the holder, wait, then start fresh
 *
 * Lifecycle: this process supervises both children. A dev server it spawned is
 * torn down when Electron exits or on Ctrl-C; a reused server is left running.
 */

"use strict";

const { spawn, execFileSync } = require("node:child_process");
const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");

const PORT = 3000;
const LOG = "/tmp/kiji-dev-server.log";
const repoRoot = path.resolve(__dirname, "..", "..");
const frontendDir = path.join(repoRoot, "src", "frontend");
const expectedCwd = frontendDir;
const force = process.env.FORCE_RESTART === "1";

const useColor = process.stdout.isTTY && !process.env.NO_COLOR;
const paint = (code, s) => (useColor ? `[${code}m${s}[0m` : s);
const blue = (s) => paint("34", s);
const green = (s) => paint("32", s);
const yellow = (s) => paint("33", s);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/**
 * Identify the process listening on `port`.
 * @returns {{pid: number, cwd: string|null}|null} null when nothing listens.
 */
function portHolder(port) {
  let out;
  try {
    out = execFileSync(
      "lsof",
      ["-nP", `-iTCP:${port}`, "-sTCP:LISTEN", "-t"],
      { encoding: "utf8" }
    );
  } catch {
    return null; // lsof exits non-zero when nothing matches
  }
  const pid = Number(out.split("\n").find((l) => l.trim())); // first listener
  if (!pid) return null;

  let cwd = null;
  try {
    const fields = execFileSync(
      "lsof",
      ["-a", "-p", String(pid), "-d", "cwd", "-Fn"],
      { encoding: "utf8" }
    );
    const nLine = fields.split("\n").find((l) => l.startsWith("n"));
    if (nLine) cwd = nLine.slice(1);
  } catch {
    cwd = null; // permissions, etc. — caller treats as unverifiable
  }
  return { pid, cwd };
}

/**
 * Decide what to do about whatever holds the port.
 * Pure function so the policy can be unit-tested in isolation.
 * @returns {'start'|'reuse'|'refuse'|'unverifiable'}
 */
function classifyHolder(holder, wantedCwd) {
  if (!holder) return "start";
  if (!holder.cwd) return "unverifiable";
  if (holder.cwd === wantedCwd) return "reuse";
  return "refuse";
}

/** Resolve true once the dev server answers HTTP on `port`. */
function probe(port) {
  return new Promise((resolve) => {
    const req = http.get(
      { host: "localhost", port, path: "/", timeout: 2000 },
      (res) => {
        res.resume();
        resolve(true);
      }
    );
    req.on("error", () => resolve(false));
    req.on("timeout", () => {
      req.destroy();
      resolve(false);
    });
  });
}

async function waitForExit(pid, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      process.kill(pid, 0); // still alive
    } catch {
      return true; // gone
    }
    await sleep(500);
  }
  return false;
}

async function main() {
  console.log(blue("Starting Electron development environment..."));
  console.log(
    yellow(
      "Note: Assumes Go backend is running separately (e.g., 'make go-backend-dev' or VSCode debugger)"
    )
  );

  let holder = portHolder(PORT);

  if (holder && force) {
    console.log(
      yellow(`FORCE_RESTART=1 — stopping server on :${PORT} (pid ${holder.pid})...`)
    );
    try {
      process.kill(holder.pid, "SIGTERM");
    } catch {
      /* already gone */
    }
    await waitForExit(holder.pid, 30000);
    holder = portHolder(PORT);
  }

  // `server` stays null when we reuse — we never tear down something we didn't start.
  let server = null;

  switch (classifyHolder(holder, expectedCwd)) {
    case "reuse":
      console.log(
        green(`✅ Reusing dev server (pid ${holder.pid}, cwd ${holder.cwd})`)
      );
      break;

    case "unverifiable":
      console.log(
        yellow(
          `⚠️  Reusing dev server on :${PORT} (pid ${holder.pid}); could not verify its working directory`
        )
      );
      break;

    case "refuse":
      console.error(
        yellow(`❌ Port ${PORT} is held by a server from another directory:`)
      );
      console.error(yellow(`     pid ${holder.pid}, cwd ${holder.cwd}`));
      console.error(yellow(`     expected:  ${expectedCwd}`));
      console.error(
        yellow(
          `   Run 'kill ${holder.pid}', or rerun with 'FORCE_RESTART=1 make electron-dev'.`
        )
      );
      process.exit(1);
      break;

    case "start":
      server = await startDevServer();
      break;
  }

  launchElectron(server);
}

/** Spawn the webpack dev server (its own process group) and wait until ready. */
async function startDevServer() {
  console.log(blue(`Starting webpack dev server (logs: ${LOG})...`));
  const logFd = fs.openSync(LOG, "a");
  const server = spawn("npm", ["run", "dev"], {
    cwd: frontendDir,
    stdio: ["ignore", logFd, logFd],
    detached: true, // own group, so we can kill the whole tree later
  });

  console.log(yellow(`Waiting for dev server on http://localhost:${PORT}...`));
  const deadline = Date.now() + 60000;
  while (Date.now() < deadline) {
    if (await probe(PORT)) {
      console.log(green(`✅ Dev server ready (pid ${server.pid})`));
      return server;
    }
    if (server.exitCode !== null) {
      console.error(
        yellow(`⚠️  Dev server exited unexpectedly — see ${LOG}`)
      );
      process.exit(1);
    }
    await sleep(1000);
  }
  console.error(yellow(`⚠️  Dev server didn't respond after 60s — see ${LOG}`));
  try {
    process.kill(-server.pid, "SIGTERM");
  } catch {
    /* ignore */
  }
  process.exit(1);
}

/** Run Electron in the foreground; tear down a spawned dev server on exit. */
function launchElectron(server) {
  console.log(blue("Starting Electron in development mode..."));
  const electron = spawn("npm", ["run", "electron:dev"], {
    cwd: frontendDir,
    stdio: "inherit",
    env: { ...process.env, EXTERNAL_BACKEND: "true" },
  });

  let toreDown = false;
  const teardown = () => {
    if (toreDown) return;
    toreDown = true;
    if (server && server.pid) {
      try {
        process.kill(-server.pid, "SIGTERM"); // kill the dev server's group
      } catch {
        /* already gone */
      }
    }
  };

  electron.on("exit", (code, signal) => {
    teardown();
    process.exit(code === null ? (signal ? 1 : 0) : code);
  });
  electron.on("error", (err) => {
    console.error(yellow(`Failed to launch Electron: ${err.message}`));
    teardown();
    process.exit(1);
  });

  // Forward interrupts to Electron and let its exit handler drive teardown.
  for (const sig of ["SIGINT", "SIGTERM"]) {
    process.on(sig, () => {
      if (electron.exitCode === null) {
        try {
          electron.kill(sig);
        } catch {
          /* ignore */
        }
      } else {
        teardown();
        process.exit(0);
      }
    });
  }
}

if (require.main === module) {
  main().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}

module.exports = { classifyHolder, portHolder };
