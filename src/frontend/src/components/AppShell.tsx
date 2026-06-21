import { useState } from "react";
import { isElectron } from "../utils/providerHelpers";
import { useServerHealth } from "../hooks/useServerHealth";
import Sidebar, { ViewId } from "./dashboard/Sidebar";
import DashboardView from "./dashboard/DashboardView";
import SettingsView from "./settings/SettingsView";
import ActivityView from "./activity/ActivityView";
import MappingsView from "./mappings/MappingsView";
import AboutView from "./about/AboutView";
import PrivacyProxyUI from "./privacy-proxy-ui";

/**
 * Top-level shell for the proxy app.
 *
 * The deep-forest sidebar is the always-on server (persistent nav + live
 * status); the light area is the workspace. The Dashboard is home; the original
 * masking tool lives under the "Playground" view; Settings, Activity, and
 * Mappings are each their own workspace view. Only the Playground is kept
 * mounted (so its in-progress state survives navigation); the other views mount
 * on demand and load their data fresh.
 */
export default function AppShell() {
  const [view, setView] = useState<ViewId>("dashboard");
  // Bumped after a successful provider save so the Playground re-reads its
  // cached provider config (the selector ✓ marks, active provider, etc.).
  const [settingsReloadN, setSettingsReloadN] = useState(0);

  const { serverStatus, serverHealth, modelSignature, version, uptimeSeconds } =
    useServerHealth(isElectron);

  const server = {
    status:
      serverStatus === "online"
        ? serverHealth.modelHealthy
          ? "online"
          : "degraded"
        : "offline",
    version,
    model: modelSignature,
    port: 8080,
    uptimeSeconds: uptimeSeconds ?? undefined,
  } as const;

  return (
    <div className="kiji-shell">
      <Sidebar active={view} onNavigate={setView} server={server} />
      <main className="kiji-main">
        {view === "dashboard" && (
          <DashboardView onShowActivity={() => setView("activity")} />
        )}
        {view === "activity" && (
          <ActivityView modelSignature={modelSignature} />
        )}
        {view === "mappings" && <MappingsView />}
        {view === "settings" && (
          <SettingsView
            onProvidersSaved={() => setSettingsReloadN((n) => n + 1)}
          />
        )}
        {view === "about" && <AboutView />}
        {/* Kept mounted so Playground state persists across navigation */}
        <div hidden={view !== "playground"}>
          <PrivacyProxyUI
            embedded
            onRequestSettings={() => setView("settings")}
            onRequestAbout={() => setView("about")}
            reloadSettingsSignal={settingsReloadN}
          />
        </div>
      </main>
    </div>
  );
}
