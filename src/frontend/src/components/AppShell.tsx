import { useEffect, useState } from "react";
import { apiUrl, isElectron } from "../utils/providerHelpers";
import { useServerHealth } from "../hooks/useServerHealth";
import Sidebar, { ViewId } from "./dashboard/Sidebar";
import DashboardView from "./dashboard/DashboardView";
import SettingsView from "./settings/SettingsView";
import ActivityView from "./activity/ActivityView";
import MappingsView from "./mappings/MappingsView";
import AboutView from "./about/AboutView";
import PrivacyProxyUI from "./privacy-proxy-ui";
import { ADMIN_ROLE_CHOSEN_EVENT } from "./onboarding/WelcomeModal";

/**
 * Top-level shell for the proxy app.
 *
 * The deep-forest sidebar is the always-on server (persistent nav + live
 * status); the light area is the workspace. The launch screen depends on the
 * role chosen during onboarding: admins land on the Dashboard, everyone else on
 * the Playground (the original masking tool). Settings, Activity, and Mappings
 * are each their own workspace view. Only the Playground is kept mounted (so its
 * in-progress state survives navigation); the other views mount on demand and
 * load their data fresh.
 */
export default function AppShell() {
  // The launch screen is role-dependent. `admin` is read async over IPC, so the
  // view starts unresolved (null) and the workspace stays empty until the role
  // arrives — this avoids briefly flashing the wrong screen on first paint.
  const [view, setView] = useState<ViewId | null>(null);

  useEffect(() => {
    let cancelled = false;
    const resolveInitialView = async () => {
      let isAdmin = false;
      if (isElectron && window.electronAPI) {
        // Desktop: the role chosen during onboarding.
        try {
          isAdmin = await window.electronAPI.getAdmin();
        } catch (error) {
          console.error("Failed to read admin preference:", error);
        }
      } else {
        // Web: a configured username + password (HTTP Basic Auth) is the admin
        // signal — whoever set the credentials and can load the gated UI is the
        // admin. /api/auth/status exposes only this boolean, never the secrets.
        try {
          const res = await fetch(apiUrl("/api/auth/status", isElectron));
          if (res.ok) {
            const data = await res.json();
            isAdmin = data.basicAuthActive === true;
          }
        } catch (error) {
          console.error("Failed to read auth status:", error);
        }
      }
      if (!cancelled) setView(isAdmin ? "dashboard" : "playground");
    };
    resolveInitialView();
    return () => {
      cancelled = true;
    };
  }, []);

  // The initial view is resolved once on mount, but onboarding can choose the
  // admin role afterwards (WelcomeModal lives under the Playground). When that
  // happens, send the new admin straight to the Dashboard for this session
  // instead of waiting for a restart to re-read the flag.
  useEffect(() => {
    const handleAdminChosen = () => setView("dashboard");
    window.addEventListener(ADMIN_ROLE_CHOSEN_EVENT, handleAdminChosen);
    return () => {
      window.removeEventListener(ADMIN_ROLE_CHOSEN_EVENT, handleAdminChosen);
    };
  }, []);
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
