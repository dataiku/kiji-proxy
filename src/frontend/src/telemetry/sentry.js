// Single source of truth for the frontend's Sentry wiring, shared by the
// Electron main process (@sentry/electron/main), the renderer
// (@sentry/electron/renderer), and the misclassification reporter.
//
// Telemetry is OPT-IN. Nothing is sent unless the user turned on
// "Crash & error reporting" in Settings, which persists as `telemetryEnabled`
// in the Electron config. Each process checks that flag before calling
// Sentry.init; when it is off, Sentry is never initialized, so every capture
// call is a harmless no-op and no data leaves the machine.
//
// CommonJS so the Electron main process can `require` it; webpack lets the
// renderer/TS code `import` from it just the same.

// Public (client-side) Sentry DSN. Safe to embed — it only permits sending
// events, not reading them. Kept here so the three init sites never duplicate it.
const SENTRY_DSN =
  "https://d7ad4213601549253c0d313b271f83cf@o4510660510679040.ingest.de.sentry.io/4510660556095568";

// Keep performance tracing light (10% of transactions) rather than the previous
// 100%, which was needlessly expensive.
const SENTRY_TRACES_SAMPLE_RATE = 0.1;

module.exports = { SENTRY_DSN, SENTRY_TRACES_SAMPLE_RATE };
