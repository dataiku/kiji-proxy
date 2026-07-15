import React from "react";
import ReactDOM from "react-dom/client";
import "./src/styles/styles.css";
import "./src/i18n";
import AppShell from "./src/components/AppShell.tsx";
import ErrorBoundary from "./src/components/ErrorBoundary.tsx";
import * as Sentry from "@sentry/electron/renderer";
import { SENTRY_DSN, SENTRY_TRACES_SAMPLE_RATE } from "./src/telemetry/sentry";

// Telemetry (Sentry) is OPT-IN: only initialize the renderer SDK when the user
// enabled "Crash & error reporting" in Settings (persisted in the Electron
// config, read here over IPC). When disabled, Sentry is never initialized, so
// every capture call elsewhere in the renderer is a harmless no-op.
async function initTelemetryRenderer() {
  try {
    const enabled = await window.electronAPI?.getTelemetryEnabled?.();
    if (!enabled) return;
    Sentry.init({
      dsn: SENTRY_DSN,
      environment: process.env.NODE_ENV || "production",
      tracesSampleRate: SENTRY_TRACES_SAMPLE_RATE,
    });
  } catch (error) {
    console.error("[Telemetry] Failed to initialize renderer Sentry:", error);
  }
}

// Fire-and-forget: rendering must not wait on the telemetry preference lookup.
initTelemetryRenderer();

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <ErrorBoundary>
    <AppShell />
  </ErrorBoundary>
);
