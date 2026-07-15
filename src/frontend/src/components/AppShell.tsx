import { useEffect, useState } from "react";
import { isElectron } from "../utils/providerHelpers";
import { useIsAdmin } from "../hooks/useIsAdmin";
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
  // The launch screen is role-dependent. `isAdmin` is read async (IPC on the
  // desktop, /api/auth/status on the web), so it starts unresolved (null) and
  // the view stays empty until the role arrives — this avoids briefly flashing
  // the wrong screen on first paint.
  const isAdmin = useIsAdmin();
  const [view, setView] = useState<ViewId | null>(null);

  // Once the role resolves, admins default to the Dashboard and everyone else to
  // the Playground. This default is derived during render rather than synced via
  // an effect, so an explicit `view` — set by the user or the onboarding event
  // below — always wins and we never clobber a view already navigated to. Until
  // the role arrives (`isAdmin === null`) the view stays empty, which avoids
  // briefly flashing the wrong screen on first paint.
  const resolvedView: ViewId | null =
    view ?? (isAdmin === null ? null : isAdmin ? "dashboard" : "playground");

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
      <Sidebar active={resolvedView} onNavigate={setView} server={server} />
      <main className="kiji-main">
        {resolvedView === "dashboard" && (
          <DashboardView onShowActivity={() => setView("activity")} />
        )}
        {resolvedView === "activity" && (
          <ActivityView modelSignature={modelSignature} />
        )}
        {resolvedView === "mappings" && <MappingsView />}
        {resolvedView === "settings" && (
          <SettingsView
            onProvidersSaved={() => setSettingsReloadN((n) => n + 1)}
          />
        )}
        {resolvedView === "about" && <AboutView />}
        {/* Kept mounted so Playground state persists across navigation */}
        <div hidden={resolvedView !== "playground"}>
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
