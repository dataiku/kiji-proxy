// Timed pause of the transparent proxy (see /api/proxy/pause in the Go
// backend). The backend owns the live pause; Electron only persists the
// deadline as a timestamp so a pause survives an app restart and still ends
// on time. Kept free of Electron imports so it can be unit tested in Node.

// Durations offered in the UI. Must match allowedPauseMinutes in
// src/backend/server/proxy_pause.go.
const PAUSE_DURATIONS_MINUTES = [1, 5, 15, 30, 60];

// Config key holding the RFC 3339 deadline of the current pause.
const PAUSED_UNTIL_CONFIG_KEY = "proxyPausedUntil";

// Env var read by the backend at startup (loadProxyConfig in main.go).
const PAUSED_UNTIL_ENV = "TRANSPARENT_PROXY_PAUSED_UNTIL";

const isValidPauseMinutes = (minutes) =>
  PAUSE_DURATIONS_MINUTES.includes(minutes);

// Returns the persisted pause deadline if it is still in the future, else null.
const activePausedUntil = (config, now = Date.now()) => {
  const value = config?.[PAUSED_UNTIL_CONFIG_KEY];
  if (typeof value !== "string") return null;
  const deadline = Date.parse(value);
  if (Number.isNaN(deadline) || deadline <= now) return null;
  return value;
};

// Sets or clears the backend env var for a pause that is still running, so a
// restarted backend picks up where the previous one left off. An inherited
// value is always overwritten so a stale shell variable can't pause the proxy.
const applyPausedUntilEnv = (env, config, now = Date.now()) => {
  const pausedUntil = activePausedUntil(config, now);
  if (pausedUntil) {
    env[PAUSED_UNTIL_ENV] = pausedUntil;
  } else {
    delete env[PAUSED_UNTIL_ENV];
  }
  return env;
};

// Mirrors a backend pause status into the persisted config.
const persistPauseStatus = (config, status) => {
  if (status?.paused && typeof status.paused_until === "string") {
    config[PAUSED_UNTIL_CONFIG_KEY] = status.paused_until;
  } else {
    delete config[PAUSED_UNTIL_CONFIG_KEY];
  }
  return config;
};

module.exports = {
  PAUSE_DURATIONS_MINUTES,
  PAUSED_UNTIL_CONFIG_KEY,
  PAUSED_UNTIL_ENV,
  isValidPauseMinutes,
  activePausedUntil,
  applyPausedUntilEnv,
  persistPauseStatus,
};
