import React from "react";
import ReactDOM from "react-dom/client";
import "./styles.css";
import PrivacyProxyUI from "./privacy-proxy-ui.tsx";
import ErrorBoundary from "./ErrorBoundary.tsx";

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <ErrorBoundary>
    <PrivacyProxyUI />
  </ErrorBoundary>
);
