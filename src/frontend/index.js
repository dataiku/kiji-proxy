import React from "react";
import ReactDOM from "react-dom/client";
import "./src/styles/styles.css";
import PrivacyProxyUI from "./src/components/privacy-proxy-ui.tsx";
import ErrorBoundary from "./src/components/ErrorBoundary.tsx";
import * as Sentry from "@sentry/electron/renderer";

// Initialize Sentry for renderer process
Sentry.init({
  dsn: "https://d7ad4213601549253c0d313b271f83cf@o4510660510679040.ingest.de.sentry.io/4510660556095568",
  environment: process.env.NODE_ENV || "production",
  tracesSampleRate: 1.0,
});

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <ErrorBoundary>
    <PrivacyProxyUI />
  </ErrorBoundary>
);
